import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.conv_layer import CrossModelBlock, InnerModelBlock, TokenPredictHead, PatchPredictHead
import os
from gram_loss import GramLoss


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
                 teacher_type=['classifier', 'mae', 'dino']):
        super(Aggergator, self).__init__()
        # intermediate block
        self.student = student
        self.teachers = nn.ModuleList()
        for tea in teachers:
            self.teachers.append(tea)
        self.teacher_type = [item.split("-")[0] for item in teacher_type]
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim_dict = dict(zip(self.teacher_type, out_dims))
        self.token_dim = token_dim
        assert len(teachers) == len(teacher_type), "number of teachers should equal types"
        self.stu_proj = nn.Conv2d(in_dim, dim*len(teacher_type), kernel_size=1, padding=0)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
            CrossModelBlock(dim+token_dim, num_models=len(teacher_type), groups=64),
            InnerModelBlock(dim+token_dim, num_models=len(teacher_type))
        ))
        
        # heads
        self.model_head = TokenPredictHead(dim=token_dim, out_dim=max(out_dims))  # max dim
        if "classifier" in self.teacher_type:
            self.class_head = TokenPredictHead(dim=dim, out_dim=self.out_dim_dict["classifier"])
        if "mae" in self.teacher_type:
            self.mae_head = PatchPredictHead(dim=dim, out_dim=self.out_dim_dict["mae"])
        if "dino" in self.teacher_type:
            self.dino_head = PatchPredictHead(dim=dim, out_dim=self.out_dim_dict["dino"])
        
        self.l2_criterion = nn.MSELoss()
        self.cos_criterion = CosineLoss()
        self.gram_criterion = GramLoss()

        self.criterion_table = {
            "classifier": self.cos_criterion,
            "mae": self.l2_criterion,
            "dino": self.gram_criterion, # use l2 for dinov3 
        }

        self.model_token = nn.Parameter(torch.empty(1, token_dim, 1, 1))
        nn.init.normal_(self.model_token, std=1e-6) # initialize model_token
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    
    def convert_student(self):
        del self.teachers

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
        model_token_out = [item[:, self.dim:, :, :] for item in projs]
        model_feat_out = [item[:, :self.dim, :, :] for item in projs]
        # teacher predict
        with torch.no_grad():
            teacher_feats = dict()
            for k, teacher in zip(self.teacher_type, self.teachers):
                teacher_feats[k] = teacher(x)
        # model token part
        model_token_out = torch.cat(model_token_out, dim=0) # .view(b*num_teacher, self.token_dim, h, w)
        model_token_out = self.model_head(model_token_out).view(num_teacher, b, -1).permute(1, 0, 2) # .view(b, num_teacher, -1)
        # model predict part
        loss_dict = dict()
        model_idx = 0
        if "classifier" in self.teacher_type:
            teacher_cls_feat = self.avg_pool(teacher_feats["classifier"]).squeeze(-1).squeeze(-1)
            loss_cls_token = self.cos_criterion(teacher_cls_feat, model_token_out[:, model_idx, :self.out_dim_dict["classifier"]])
            # feat predict
            stu_cls_feat = self.class_head(model_feat_out[model_idx])
            loss_cls_patch = self.criterion_table["classifier"](stu_cls_feat, teacher_cls_feat)
            loss_dict["cls_model"] = loss_cls_token
            loss_dict["cls_patch"] = loss_cls_patch
            model_idx += 1
        if "mae" in self.teacher_type:
            teacher_mae_feat = teacher_feats["mae"]
            loss_mae_token = self.cos_criterion(self.avg_pool(teacher_mae_feat).squeeze(-1).squeeze(-1), 
                                               model_token_out[:, model_idx, :self.out_dim_dict["mae"]])
            # feat predict
            stu_mae_feat = self.mae_head(model_feat_out[model_idx])
            loss_mae_patch = self.criterion_table["mae"](stu_mae_feat, teacher_mae_feat)
            loss_dict["mae_model"] = loss_mae_token
            loss_dict["mae_patch"] = loss_mae_patch
            model_idx += 1
        if "dino" in self.teacher_type:
            teacher_dino_feat = teacher_feats["dino"]
            loss_dino_token = self.cos_criterion(self.avg_pool(teacher_dino_feat).squeeze(-1).squeeze(-1), 
                                               model_token_out[:, model_idx, :self.out_dim_dict["dino"]])
            # feat predict
            stu_dino_feat = self.dino_head(model_feat_out[model_idx])
            loss_dino_patch = self.criterion_table["dino"](stu_dino_feat, teacher_dino_feat)
            loss_dict["dino_model"] = loss_dino_token
            loss_dict["dino_patch"] = loss_dino_patch
            model_idx += 1
        return loss_dict
    
class BackboneStuConv(nn.Module):
    def __init__(self, student, num_blocks=2, in_dim=2048, dim=512, out_dims=[2048,], token_dim=256, 
                 teacher_type=['classifier', 'mae', 'dino'], pretrain_ckpt="", num_classes=100):
        super().__init__()  
        self.student = student
        self.teacher_type = teacher_type
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim_dict = dict(zip(self.teacher_type, out_dims))
        self.token_dim = token_dim
        self.stu_proj = nn.Conv2d(in_dim, dim*len(teacher_type), kernel_size=1, padding=0)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
            CrossModelBlock(dim+token_dim, num_models=len(teacher_type), groups=128),
            InnerModelBlock(dim+token_dim, num_models=len(teacher_type))
        ))
        
        # heads
        self.model_head = TokenPredictHead(dim=token_dim, out_dim=max(out_dims))  # max dim
        if "classifier" in self.teacher_type:
            self.class_head = TokenPredictHead(dim=dim, out_dim=self.out_dim_dict["classifier"])
        if "mae" in self.teacher_type:
            self.mae_head = PatchPredictHead(dim=dim, out_dim=self.out_dim_dict["mae"])
        if "dino" in self.teacher_type:
            self.dino_head = PatchPredictHead(dim=dim, out_dim=self.out_dim_dict["dino"])
        
        self.model_token = nn.Parameter(torch.empty(1, token_dim, 1, 1))
        nn.init.normal_(self.model_token, std=1e-6) # initialize model_token
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if os.path.exists(pretrain_ckpt):
            print("==> Load checkpoint ...")
            ckp = torch.load(pretrain_ckpt, map_location="cpu", weights_only=False)
            for name, _ in self.named_parameters():
                if name not in ckp:
                    print(f"Warning: {name} not in checkpoint ... ")
            self.load_state_dict(ckp, strict=False)  
        
        # for name, param in self.student.named_parameters():
        #     print(name, param.mean())
    
    def forward(self, x):
        """
        x: [b, c, h, w]: images
        """
        # student branch
        feat = self.student(x)
        return feat
        b, _, h, w = feat.shape
        projs = self.stu_proj(feat)
        num_teacher = len(self.teacher_type)
        projs = torch.chunk(projs, chunks=num_teacher, dim=1) # [[b,c,h,w],...]
        # cat token
        token = self.model_token.repeat(b, 1, h, w)
        projs = [torch.cat([item, token], dim=1) for item in projs]
        for block in self.blocks:
            projs = block(projs)
        # chunk projs
        model_token_out = [item[:, self.dim:, :, :] for item in projs]
        model_feat_out = [item[:, :self.dim, :, :] for item in projs]
        # model_token_out = self.model_head(model_token_out[-1])
        cls_feat_out = self.class_head.norm(model_feat_out[0])
        dino_feat_out = self.dino_head(model_feat_out[-1])
        dino_token = self.model_head.norm(model_token_out[-1])
        mae_feat_out = self.mae_head(model_feat_out[1])
        return dino_feat_out # model_token_out, torch.cat([dino_feat_out, dino_token], dim=1)

if __name__ == "__main__":
    from backbone.convnext import ConvNeXt, DINOv3ConvNeXt, convnext_b, convnext_b_norm
    from PIL import Image
    from torchvision import transforms

    imagenet_mean_ = [0.485, 0.456, 0.406]
    imagenet_std_ = [0.229, 0.224, 0.225]

    def cnt_params(model):
        cnt = 0
        for p in model.parameters():
            if p.requires_grad:
                cnt += p.numel()
        return cnt

    # stu = resnet50()
    stu = convnext_b_norm(drop_path_rate=0.0)
    tea3 = convnext_b_norm()
    tea4 = convnext_b()
    tea5 = DINOv3ConvNeXt("pretrain/dinov3_convnext_b").cpu()

    # model_agg = Aggergator(stu, teachers=[tea1, tea2, tea3], teacher_type=["classifier-resnet", "mae-resnet", "dino-resnet"], out_dims=[2048, 2048, 2048], num_blocks=2, in_dim=2048, dim=2048, token_dim=1024)
    model_agg = Aggergator(stu, teachers=[tea3, tea4, tea5], teacher_type=["classifier-convnext_b_norm", "mae-convnext_b", "dino-dinov3_convnext_b"], out_dims=[1024, 1024, 1024], num_blocks=2, in_dim=1024, dim=1024, token_dim=512)
    # ckp = torch.load("../../VLM/logs/classifier_mae_dino2048_token1024_out2048/checkpoint.pth", map_location="cpu", weights_only=False)
    ckp = torch.load("logs/convnext_b_norm_cls_mae_dinov3_1024_dinov3l2/classifier-convnext_b_norm_mae-convnext_b_dino-dinov3_convnext_b/checkpoint.pth", map_location="cpu", weights_only=False)

    model_agg.load_state_dict(ckp["model"], strict=True)
    # model_agg.convert_student()
    # state_dict = model_agg.state_dict()
    # torch.save(state_dict, "../../VLM/distills/convnext_b_norm_allconv_student.pth")

    img = Image.open("cat.jpg")
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean_, std=imagenet_std_),
    ])
    x = transform(img).unsqueeze(0)
    
    loss_dict = model_agg(x)
    print(loss_dict)