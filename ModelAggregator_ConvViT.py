from backbone import convnext
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.conv_layer import CrossModelBlock, InnerModelBlock, TokenPredictHead, PatchPredictHead
# All possible students and teachers
# from backbone.resnet import resnet50
# from backbone.convnext import ConvNeXt
# from backbone.vit import clip_vit_b_16, clip_vit_l_14, dinov3_vit_b_16, dinov3_vit_l_16
# from backbone.vit_mae import mae_vit_b_16, mae_vit_l_16
from math import sqrt

"""
Student is convnet while teachers maybe combination of ViT, ConvNeXt, and ResNet
"""

# ImageNet stats
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# OpenAI stats
openai_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
openai_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

def imagenet_to_openai(x):
    """
    x: Tensor [B, C, H, W] normalized with ImageNet stats.
    Returns: Tensor normalized with OpenAI stats.
    """
    scale = imagenet_std / openai_std
    shift = (imagenet_mean - openai_mean) / openai_std
    return x * scale.to(x.device) + shift.to(x.device)

class CosineLoss(nn.Module):
    def __init__(self, ):
        super(CosineLoss, self).__init__()
    
    def forward(self, x1, x2):
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        sim = (x1 * x2).sum(dim=1)
        loss = 1 - sim.mean()
        return loss

class Aggergator(nn.Module):
    """
    Aggregate multiple pre-trained models
    """
    def __init__(self, student:nn.Module, teachers:dict, num_blocks=2, in_dim=2048, dim=512, out_dims=[2048,], token_dim=256, 
                 teacher_type=['conv_classifier', 'conv_mae', 'conv_dino', 'clip-b']):
        super(Aggergator, self).__init__()
        # intermediate block
        self.student = student
        self.teachers = nn.ModuleList()
        assert len(teacher_type) == len(teachers), "number of teachers should equal types"
        assert len(teacher_type) == len(out_dims), "number of teachers should equal out_dims"
        for tea in teachers:
            self.teachers.append(tea)
        self.teacher_type = [item.split("-")[0] for item in teacher_type]  # do not include -b, -l
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim_dict = dict(zip(self.teacher_type, out_dims))
        self.token_dim = token_dim
        assert len(teachers) == len(teacher_type), "number of teachers should equal types"
        self.stu_proj = nn.Conv2d(in_dim, dim*len(teacher_type), kernel_size=1, padding=0)
        basic_block = nn.Sequential(
            CrossModelBlock(dim+token_dim, num_models=len(teacher_type), groups=128),
            InnerModelBlock(dim+token_dim, num_models=len(teacher_type))
        )
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(basic_block)
        
        # heads
        self.model_head = TokenPredictHead(dim=token_dim, out_dim=max(out_dims))  # max dim
        # for ConvNets teachers
        if "classifier" in self.teacher_type:
            self.class_head = TokenPredictHead(dim=dim, out_dim=self.out_dim_dict["classifier"])
        if "mae_conv" in self.teacher_type:
            self.mae_head_conv = PatchPredictHead(dim=dim, out_dim=self.out_dim_dict["mae_conv"])
        if "dino_conv" in self.teacher_type:
            self.dino_head_conv = PatchPredictHead(dim=dim, out_dim=self.out_dim_dict["dino_conv"])
        # for ViT teachers
        if "clip" in self.teacher_type:
            self.clip_head = TokenPredictHead(dim=dim, out_dim=self.out_dim_dict["clip"])
        if "sam" in self.teacher_type:
            self.sam_head = PatchPredictHead(dim=dim, out_dim=self.out_dim_dict["sam"])
        if "dinov3_vit" in self.teacher_type:
            self.dino_head_vit = PatchPredictHead(dim=dim, out_dim=self.out_dim_dict["dinov3_vit"])
        if "mae_vit" in self.teacher_type:
            self.mae_head_vit = PatchPredictHead(dim=dim, out_dim=self.out_dim_dict["mae_vit"])
        
        self.l2_criterion = nn.MSELoss()
        self.cos_criterion = CosineLoss()
        self.criterion_table = {
            "classifier": self.l2_criterion,
            "mae_conv": self.l2_criterion,
            "dino_conv": self.cos_criterion,
            "sam": self.l2_criterion,
            "clip": self.cos_criterion,
            "dinov3_vit": self.cos_criterion,
            "mae_vit": self.l2_criterion
        }
        self.model_token = nn.Parameter(torch.empty(1, token_dim, 1, 1))
        nn.init.normal_(self.model_token, std=1e-6) # initialize model_token
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        x: [b, c, h, w]: images
        """
        # student branch
        feat = self.student(x)
        b, _, h, w = feat.shape
        projs = self.stu_proj(feat)
        num_teacher = len(self.teachers)
        projs = torch.chunk(projs, chunks=num_teacher, dim=1) # [[b,c,h,w],...]
        # cat token
        token = self.model_token.repeat(b, 1, h, w)
        projs = [torch.cat([item, token], dim=1) for item in projs]
        for block in self.blocks:
            projs = block(projs)
        # chunk projs
        model_token_out = [item[:, self.dim:, :, :].unsqueeze(1) for item in projs] # [b, 1, d, h, w]
        model_feat_out = [item[:, :self.dim, :, :] for item in projs]
        # teacher predict
        with torch.no_grad():
            teacher_feats = dict()
            for k, teacher in zip(self.teacher_type, self.teachers):
                if "clip" in k:
                    # CLIP re-normalization
                    x = imagenet_to_openai(x)
                teacher_feats[k] = teacher(x)
        # model token part
        model_token_out = torch.cat(model_token_out, dim=1).view(b*num_teacher, self.token_dim, h, w)
        model_token_out = self.model_head(model_token_out).view(b, num_teacher, -1) # [b, 3, d]
        # model predict part
        loss_dict = dict()
        model_idx = 0
        ## ConvNets
        if "classifier" in self.teacher_type:
            teacher_cls_feat = self.avg_pool(teacher_feats["classifier"]).squeeze(-1).squeeze(-1)
            loss_cls_token = self.l2_criterion(teacher_cls_feat, model_token_out[:, model_idx, :self.out_dim_dict["classifier"]])
            # feat predict
            stu_cls_feat = self.class_head(model_feat_out[model_idx])
            loss_cls_patch = self.criterion_table["classifier"](stu_cls_feat, teacher_cls_feat)
            loss_dict["cls_model_conv"] = loss_cls_token
            loss_dict["cls_patch_conv"] = loss_cls_patch
            model_idx += 1
        if "mae_conv" in self.teacher_type:
            teacher_mae_feat = teacher_feats["mae_conv"]
            loss_mae_token = self.l2_criterion(self.avg_pool(teacher_mae_feat).squeeze(-1).squeeze(-1), 
                                               model_token_out[:, model_idx, :self.out_dim_dict["mae_conv"]])
            # feat predict
            stu_mae_feat = self.mae_head_conv(model_feat_out[model_idx])
            loss_mae_patch = self.criterion_table["mae_conv"](stu_mae_feat, teacher_mae_feat)
            loss_dict["mae_model_conv"] = loss_mae_token
            loss_dict["mae_patch_conv"] = loss_mae_patch
            model_idx += 1
        if "dino_conv" in self.teacher_type:
            teacher_dino_feat = teacher_feats["dino_conv"]
            loss_dino_token = self.l2_criterion(self.avg_pool(teacher_dino_feat).squeeze(-1).squeeze(-1), 
                                               model_token_out[:, model_idx, :self.out_dim_dict["dino_conv"]])
            # feat predict
            stu_dino_feat = self.dino_head_conv(model_feat_out[model_idx])
            loss_dino_patch = self.criterion_table["dino_conv"](stu_dino_feat, teacher_dino_feat)
            loss_dict["dino_model_conv"] = loss_dino_token
            loss_dict["dino_patch_conv"] = loss_dino_patch
            model_idx += 1
        ## ViT
        if "clip" in self.teacher_type:
            # select model token from teacher
            teacher_cls_feat = teacher_feats["clip"] # [b, d]
            loss_cls_token = self.l2_criterion(teacher_cls_feat, model_token_out[:, model_idx, :self.out_dim_dict["clip"]])
            # pool feat predict
            std_pool_feat = model_feat_out[model_idx]  # [b, d, 1, 1] -> [b, d]
            std_pool_feat = self.clip_head(std_pool_feat)
            loss_cls_patch = self.criterion_table["clip"](std_pool_feat, teacher_cls_feat) # [b, d] 
            loss_dict["clip_cls"] = loss_cls_token
            loss_dict["clip_pool"] = loss_cls_patch
            model_idx += 1
        if "dinov3_vit" in self.teacher_type:
            teacher_cls_feat = teacher_feats["dinov3_vit"][:, 0, :] # [b, d]
            loss_dino_token = self.l2_criterion(teacher_cls_feat, model_token_out[:, model_idx, :self.out_dim_dict["dinov3_vit"]])
            # feat predict
            teacher_dino_feat = teacher_feats["dinov3_vit"][:, 5:, :]  # [N, N] must interporlate
            b, L, d = teacher_dino_feat.shape
            teacher_dino_feat = teacher_dino_feat.view(b, int(sqrt(L)), int(sqrt(L)), d).permute(0, 3, 1, 2).contiguous()
            stu_dino_feat = self.dino_head_vit(model_feat_out[model_idx])
            b, c, h, w = stu_dino_feat
            # ViT has a larger spatial resolution, resize ViT feature
            teacher_dino_feat = F.interpolate(teacher_dino_feat, size=(h, w), mode="bilinear")
            loss_dino_patch = self.criterion_table["dinov3_vit"](stu_dino_feat, teacher_dino_feat)
            loss_dict["dino_cls_vit"] = loss_dino_token
            loss_dict["dino_patch_vit"] = loss_dino_patch
            model_idx += 1
        if "sam" in self.teacher_type:
            # SAM model does not have cls token
            teacher_sam_feat = teacher_feats["sam"][:, 1:, :] # [b, l, d]
            teacher_cls_feat = teacher_sam_feat.mean(dim=1) # [b, d]
            loss_sam_token = self.l2_criterion(teacher_cls_feat, model_token_out[:, model_idx, :self.out_dim_dict["sam"]])
            # feat predict
            stu_sam_feat = self.sam_head(model_feat_out[model_idx]) # [b, c, h, w]
            b, L, d = teacher_sam_feat.shape
            teacher_sam_feat = teacher_sam_feat.view(b, int(sqrt(L)), int(sqrt(L)), d).permute(0, 3, 1, 2).contiguous()
            b, c, h, w = stu_sam_feat.shape
            teacher_sam_feat = F.interpolate(teacher_sam_feat, size=(h, w), mode="bilinear")
            loss_sam_patch = self.criterion_table["sam"](stu_sam_feat, teacher_sam_feat)
            loss_dict["sam_pool"] = loss_sam_token
            loss_dict["sam_patch"] = loss_sam_patch
            model_idx += 1
        if "mae_vit" in self.teacher_type:
            teacher_cls_feat = teacher_feats["mae_vit"][:, 0, :]
            teacher_mae_feat = teacher_feats["mae_vit"][:, 1:, :] # [b, l, d]
            loss_mae_token = self.l2_criterion(teacher_cls_feat, model_token_out[:, model_idx, :self.out_dim_dict["mae_vit"]])
            # feat predict
            b, L, d = teacher_mae_feat.shape
            teacher_mae_feat = teacher_mae_feat.view(b, int(sqrt(L)), int(sqrt(L)), d).permute(0, 3, 1, 2).contiguous()
            b, c, h, w = model_feat_out[model_idx].shape
            teacher_mae_feat = F.interpolate(teacher_mae_feat, size=(h, w), mode="bilinear")
            stu_mae_feat = self.mae_head_vit(model_feat_out[model_idx])
            loss_mae_patch = self.criterion_table["mae_vit"](stu_mae_feat, teacher_mae_feat)
            loss_dict["mae_cls_vit"] = loss_mae_token
            loss_dict["mae_patch_vit"] = loss_mae_patch
            model_idx += 1
        return loss_dict


if __name__ == "__main__":
    stu = resnet50()
    tea1 = resnet50()
    tea2 = resnet50()
    tea3 = resnet50()
    tea3 = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    tea4 = mae_vit_b_16(ckpt_path="pretrain/mae_pretrain_vit_base.pth")
    tea5, _ = clip_vit_b_16(ckpt_path=None)

    model_agg = Aggergator(stu, teachers=[tea4], teacher_type=["mae_vit-b"], out_dims=[768])
    x = torch.rand(1, 3, 224, 224)
    loss_dict = model_agg(x)
    print(loss_dict)