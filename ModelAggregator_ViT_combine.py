import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.vit_layer_channel import CrossModelBlock, InnerModelBlock, TokenPredictHead, PatchPredictHead
from layers.linear_dispart import LinearDispartLoss, isotropy_score, off_diagonal_cov_norm
from math import sqrt
from backbone.vanilla_vit import VanillaViT
import os
from backbone.vit_mae import mae_vit_b_16, mae_vit_l_16
from backbone.vit import DINOv3ViT, clip_vit_b_16, clip_vit_l_14


def linear_CKA(X, Y):
    # X, Y: [N, d]
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (X @ Y.T).pow(2).sum()
    norm = (X @ X.T).pow(2).sum().sqrt() * (Y @ Y.T).pow(2).sum().sqrt()
    return hsic / (1e-7+norm)

import torch

def cosine_CKA(X, Y, eps=1e-7):
    # X, Y: [N, d]
    # Normalize each row to unit length (cosine similarity basis)
    X = X / (X.norm(dim=1, keepdim=True) + eps)
    Y = Y / (Y.norm(dim=1, keepdim=True) + eps)
    # Center the features (like in linear CKA)
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    # HSIC numerator
    hsic = (X @ Y.T).pow(2).sum()
    # Normalization terms
    norm = (X @ X.T).pow(2).sum().sqrt() * (Y @ Y.T).pow(2).sum().sqrt()
    return hsic / (eps + norm)


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
        # [b, l, d] shape
        x1 = F.normalize(x1, dim=-1)
        x2 = F.normalize(x2, dim=-1)
        sim = (x1 * x2).sum(dim=-1)
        loss = 1 - sim.mean()
        return loss


class RelationalCosineLoss(nn.Module):
    def __init__(self):
        super(RelationalCosineLoss, self).__init__()

    def forward(self, student_feat, teacher_feat):
        """
        student_feat: [B, D] 或 [B, L, D] (如果是 3D 则默认取 [CLS] 或全局池化)
        teacher_feat: [B, D] 或 [B, L, D]
        """
        # 1. 预处理：确保输入是 [B, D] 维度
        # 如果是 ViT 的输出 [B, L, D]，通常提取 [CLS] token (即第一个 token)
        if student_feat.dim() == 3:
            student_feat = student_feat[:, 0]
        if teacher_feat.dim() == 3:
            teacher_feat = teacher_feat[:, 0]

        # 2. 特征归一化 (L2 Normalization)
        # 这是计算余弦相似度的前提
        s_norm = F.normalize(student_feat, p=2, dim=-1)
        t_norm = F.normalize(teacher_feat, p=2, dim=-1)

        # 3. 计算样本间的 [B, B] 相似度矩阵
        # 利用矩阵乘法：[B, D] @ [D, B] -> [B, B]
        # matrix_sim[i, j] 表示第 i 个样本和第 j 个样本的余弦相似度
        sim_s = torch.mm(s_norm, s_norm.t())
        sim_t = torch.mm(t_norm, t_norm.t())

        # 4. 移除对角线元素 (Self-similarity)
        # 对角线上的值永远是 1 (样本与自身的相似度)，不包含蒸馏信息且会干扰 Loss
        batch_size = sim_s.shape[0]
        mask = torch.eye(batch_size, device=sim_s.device).bool()
        
        # 提取非对角线元素，形状变为 [B * (B-1)]
        sim_s_offdiag = sim_s[~mask]
        sim_t_offdiag = sim_t[~mask]

        # 5. 计算学生和教师相似度矩阵之间的差异
        # 通常使用 MSE (均方误差)，让学生的矩阵逼近教师的矩阵
        loss = F.mse_loss(sim_s_offdiag, sim_t_offdiag)
        return loss

class Aggergator(nn.Module):
    """
    Aggregate multiple pre-trained models
    """
    def __init__(self, student:nn.Module, teachers:dict, num_blocks=2, in_dim=768, dim=768, out_dims=[768,], token_dim=512, 
                 teacher_type=['sam', 'clip', 'dinov3']):
        super(Aggergator, self).__init__()
        # intermediate block
        self.student = student
        self.teachers = nn.ModuleList()
        assert len(teacher_type) == len(out_dims), "number of teacher types must match number of out dims"
        assert len(teachers) == len(teacher_type), "number of teachers should equal types"
        for tea in teachers:
            self.teachers.append(tea)
        self.teacher_type = [item.split("-")[0] for item in teacher_type]  # do not include -b, -l
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim_dict = dict(zip(self.teacher_type, out_dims))
        self.token_dim = token_dim
        self.stu_proj = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(in_dim, dim*len(teacher_type))
        )
        print("==> number of blocks: ", num_blocks)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
            CrossModelBlock(dim+token_dim, num_models=len(teacher_type), heads=8*len(teacher_type)),
            InnerModelBlock(dim+token_dim, num_models=len(teacher_type), heads=8)
        ))
        for name, param in self.blocks.named_parameters():
            print(name, param.numel())
        # heads
        self.model_head = TokenPredictHead(dim=token_dim+dim, out_dim=max(out_dims))  # max dim
        # if "clip" in self.teacher_type:
        #     self.clip_head = TokenPredictHead(dim=dim, out_dim=self.out_dim_dict["clip"])
        if "sam" in self.teacher_type:
            self.sam_head = PatchPredictHead(dim=dim, out_dim=self.out_dim_dict["sam"])
        if "dinov3" in self.teacher_type:
            self.dino_head = PatchPredictHead(dim=dim+token_dim, out_dim=self.out_dim_dict["dinov3"])
            self.dino_ld_head = PatchPredictHead(dim=dim+token_dim, out_dim=self.out_dim_dict["dinov3"])
        if "mae" in self.teacher_type:
            self.mae_head = PatchPredictHead(dim=dim+token_dim, out_dim=self.out_dim_dict["mae"])
            self.mae_ld_head = PatchPredictHead(dim=dim+token_dim, out_dim=self.out_dim_dict["mae"])
        # Shared LD criterion — teacher-agnostic, one instance covers all patch teachers.
        # energy_thresh=0.90 means K captures ≥90% of teacher variance;
        # stable-rank clause additionally excludes low-variance tail dimensions.
        self.ld_criterion = LinearDispartLoss(
            lambda_align=25.0, energy_thresh=0.90, min_k=16, whitening_eps=1e-3,
        )

        self.l2_criterion = nn.MSELoss()
        self.cos_criterion = CosineLoss()
        self.relation_criterion = RelationalCosineLoss()
        self.criterion_table = {
            "cls": self.cos_criterion,
            "sam": self.l2_criterion,
            "clip": self.cos_criterion,
            "dinov3": self.cos_criterion,
            "mae": self.l2_criterion
        }
        self.model_token = nn.Parameter(torch.empty(1, 1, token_dim))
        nn.init.normal_(self.model_token, std=1e-6) # initialize model_token
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        x: [b, c, h, w]: images
        """
        # student branch
        feat = self.student(x)
        b, l, d = feat.shape
        projs = self.stu_proj(feat)
        num_teacher = len(self.teachers)
        projs = torch.chunk(projs, chunks=num_teacher, dim=-1) # [[b,c,h,w],...]
        # print("Student proj: ", projs[0].shape)
        # cat token
        token = self.model_token.repeat(b, l, 1)
        projs = [torch.cat([item, token], dim=-1) for item in projs]
        for block in self.blocks:
            projs = block(projs)
        # chunk projs
        model_token_out = [item[:, [0], :] for item in projs] # [b, 1, d]
        model_feat_out =  [item[:, 1:, :] for item in projs] # [b, l, d]
        # teacher predict
        with torch.no_grad():
            teacher_feats = dict()
            for k, teacher in zip(self.teacher_type, self.teachers):
                if "clip" in k:
                    x = imagenet_to_openai(x)
                teacher_feats[k] = teacher(x)
        # model token part
        # For ViT archs, we use its cls token for model token prediction
        model_cls_token_out = torch.cat(model_token_out, dim=0)# [3b, l, d] # .view(b*num_teacher, num_tokens, self.token_dim) # [b*3, d]
        model_cls_token_out = self.model_head(model_cls_token_out).view(num_teacher, b, -1).permute(1, 0, 2) # [b, 3, d]
        # model predict part
        loss_dict = dict()
        model_idx = 0
        if "cls" in self.teacher_type:
            # select model token from teacher
            teacher_cls_feat = teacher_feats["cls"][:,0,:] # [b, d]
            loss_cls_token = self.cos_criterion(teacher_cls_feat, model_cls_token_out[:, model_idx, :self.out_dim_dict["cls"]]) 
            loss_dict["cls_cls"] = loss_cls_token
            loss_dict['cls_rel'] = self.relation_criterion(model_cls_token_out[:, model_idx, :self.out_dim_dict["clip"]], teacher_cls_feat) * 0.5
            model_idx += 1
        if "clip" in self.teacher_type:
            # select model token from teacher
            teacher_cls_feat = teacher_feats["clip"] # [b, d]
            loss_cls_token = self.cos_criterion(teacher_cls_feat, model_cls_token_out[:, model_idx, :self.out_dim_dict["clip"]]) 
            
            loss_dict["clip_cls"] = loss_cls_token
            loss_dict['clip_rel'] = self.relation_criterion(model_cls_token_out[:, model_idx, :self.out_dim_dict["clip"]], teacher_cls_feat) * 0.5
            # loss_dict["clip_pool"] = loss_cls_patch
            model_idx += 1
        if "dinov3" in self.teacher_type:
            teacher_cls_feat = teacher_feats["dinov3"][:, 0, :] # [b, d]
            loss_dino_token = self.cos_criterion(teacher_cls_feat, model_cls_token_out[:, model_idx, :self.out_dim_dict["dinov3"]])
            # feat predict
            teacher_dino_feat = teacher_feats["dinov3"][:, 5:, :] # have 4 register tokens
            stu_dino_feat = self.dino_head(model_feat_out[model_idx])
            if stu_dino_feat.shape[1] != teacher_dino_feat.shape[1]:
                L_stu, L_dino = stu_dino_feat.shape[1], teacher_dino_feat.shape[1]
                b, c = stu_dino_feat.shape[0], stu_dino_feat.shape[2]
                stu_dino_feat = stu_dino_feat.reshape(b, int(sqrt(L_stu)), int(sqrt(L_stu)), c).permute(0, 3, 1, 2)
                stu_dino_feat = F.interpolate(stu_dino_feat, size=(int(sqrt(L_dino)), int(sqrt(L_dino))), mode="bilinear") # [b, c, h, w]
                stu_dino_feat = stu_dino_feat.view(b, c, -1).permute(0, 2, 1).contiguous()
            loss_dino_patch = self.criterion_table["dinov3"](stu_dino_feat, teacher_dino_feat)
            loss_dict["dinov3_cls"] = loss_dino_token
            loss_dict["dinov3_rel"] = self.relation_criterion(model_cls_token_out[:, model_idx, :self.out_dim_dict["dinov3"]], teacher_cls_feat) * 0.5
            loss_dict["dinov3_patch"] = loss_dino_patch
            # --- Linear disentanglement alignment ---
            # Project student via the dedicated LD head, then align with DINO
            # patch features in their ZCA-whitened space.  Variance + covariance
            # terms are applied to the student projection to structurally enforce
            # the property, not just mimic the teacher's output values.
            stu_ld_feat = self.dino_ld_head(model_feat_out[model_idx])
            if stu_ld_feat.shape[1] != teacher_dino_feat.shape[1]:
                L_stu2 = stu_ld_feat.shape[1]
                L_dino2 = teacher_dino_feat.shape[1]
                c2 = stu_ld_feat.shape[2]
                stu_ld_feat = stu_ld_feat.reshape(b, int(sqrt(L_stu2)), int(sqrt(L_stu2)), c2).permute(0, 3, 1, 2)
                stu_ld_feat = F.interpolate(stu_ld_feat, size=(int(sqrt(L_dino2)), int(sqrt(L_dino2))), mode="bilinear")
                stu_ld_feat = stu_ld_feat.view(b, c2, -1).permute(0, 2, 1).contiguous()
            Z_s = stu_ld_feat.view(b * stu_ld_feat.shape[1], -1)
            Z_t = teacher_dino_feat.view(b * teacher_dino_feat.shape[1], -1)
            ld_losses = self.ld_criterion(Z_s, Z_t)
            loss_dict.update(ld_losses)
            model_idx += 1
        if "sam" in self.teacher_type:
            # SAM model does not have cls token
            teacher_sam_feat = teacher_feats["sam"][:, 1:, :] # [b, l, d]
            teacher_cls_feat = teacher_sam_feat.mean(dim=1) # [b, d]
            loss_sam_token = self.l2_criterion(teacher_cls_feat, model_cls_token_out[:, model_idx, :self.out_dim_dict["sam"]])
            # feat predict
            stu_sam_feat = self.sam_head(model_feat_out[model_idx])
            loss_sam_patch = self.criterion_table["sam"](stu_sam_feat, teacher_sam_feat)
            loss_dict["sam_pool"] = loss_sam_token
            loss_dict["sam_patch"] = loss_sam_patch
            model_idx += 1
        if "mae" in self.teacher_type:
            # print(teacher_feats["mae"].shape)
            teacher_cls_feat = teacher_feats["mae"][:, 0, :]
            teacher_mae_feat = teacher_feats["mae"][:, 1:, :] # [b, l, d]
            loss_mae_token = self.cos_criterion(teacher_cls_feat, model_cls_token_out[:, model_idx, :self.out_dim_dict["mae"]])
            # feat predict
            stu_mae_feat = self.mae_head(model_feat_out[model_idx])
            if stu_mae_feat.shape[1] != teacher_mae_feat.shape[1]:
                L_stu, L_mae = stu_mae_feat.shape[1], teacher_mae_feat.shape[1]
                b, c = stu_mae_feat.shape[0], stu_mae_feat.shape[2]
                stu_mae_feat = stu_mae_feat.reshape(b, int(sqrt(L_stu)), int(sqrt(L_stu)), c).permute(0, 3, 1, 2)
                stu_mae_feat = F.interpolate(stu_mae_feat, size=(int(sqrt(L_mae)), int(sqrt(L_mae))), mode="bilinear") # [b, c, h, w]
                stu_mae_feat = stu_mae_feat.view(b, c, -1).permute(0, 2, 1).contiguous()
            loss_mae_patch = self.criterion_table["mae"](stu_mae_feat, teacher_mae_feat)
            loss_dict["mae_cls"] = loss_mae_token
            loss_dict["mae_rel"] = self.relation_criterion(model_cls_token_out[:, model_idx, :self.out_dim_dict["mae"]], teacher_cls_feat) * 0.5
            loss_dict["mae_patch"] = loss_mae_patch
            # --- MAE linear disentanglement alignment ---
            # Raw MAE features are anisotropic: the top eigenvectors encode low-frequency
            # reconstruction patterns and dominate MSE gradients.  ZCA whitening inside
            # ld_criterion equalises gradient magnitude across all eigendirections, forcing
            # the student to capture the full MAE representation rather than only its
            # dominant spatial frequencies.
            stu_mae_ld_feat = self.mae_ld_head(model_feat_out[model_idx])
            if stu_mae_ld_feat.shape[1] != teacher_mae_feat.shape[1]:
                L_stu2, L_mae2 = stu_mae_ld_feat.shape[1], teacher_mae_feat.shape[1]
                c2 = stu_mae_ld_feat.shape[2]
                stu_mae_ld_feat = stu_mae_ld_feat.reshape(b, int(sqrt(L_stu2)), int(sqrt(L_stu2)), c2).permute(0, 3, 1, 2)
                stu_mae_ld_feat = F.interpolate(stu_mae_ld_feat, size=(int(sqrt(L_mae2)), int(sqrt(L_mae2))), mode="bilinear")
                stu_mae_ld_feat = stu_mae_ld_feat.view(b, c2, -1).permute(0, 2, 1).contiguous()
            Z_s_mae = stu_mae_ld_feat.view(b * stu_mae_ld_feat.shape[1], -1)
            Z_t_mae = teacher_mae_feat.view(b * teacher_mae_feat.shape[1], -1)
            mae_ld_losses = self.ld_criterion(Z_s_mae, Z_t_mae)
            loss_dict.update({f"mae_{k}": v for k, v in mae_ld_losses.items()})
            model_idx += 1
        return loss_dict


class BackboneStuViT(nn.Module):
    """
    Aggregate multiple pre-trained models
    """
    def __init__(self, student="vit_b", num_blocks=2, in_dim=768, dim=768, out_dims=[768,], token_dim=512, 
                 teacher_type=['sam', 'clip', 'dinov3'], pretrain_ckpt="", img_size=224):
        super(BackboneStuViT, self).__init__()
        # intermediate block
        if student == "vit_b":
            self.student = VanillaViT("vit_base_patch16_224", pretrained=False, img_size=img_size)
        else:
            self.student = VanillaViT("vit_large_patch16_224", pretrained=False, img_size=img_size)
        
        self.teacher_type = [item.split("-")[0] for item in teacher_type]  # do not include -b, -l
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim_dict = dict(zip(self.teacher_type, out_dims))
        self.token_dim = token_dim
        self.stu_proj = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(in_dim, dim*len(teacher_type))
        )
        print("==> number of blocks: ", num_blocks)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
            CrossModelBlock(dim+token_dim, num_models=len(teacher_type), heads=8*len(teacher_type)),
            InnerModelBlock(dim+token_dim, num_models=len(teacher_type), heads=8)
        ))
        self.l2_criterion = nn.MSELoss()
        
        self.model_head = TokenPredictHead(dim=token_dim+dim, out_dim=max(out_dims))  # max dim
        # if "clip" in self.teacher_type:
        #     self.clip_head = TokenPredictHead(dim=dim+token_dim, out_dim=self.out_dim_dict["clip"])
        if "sam" in self.teacher_type:
            self.sam_head = PatchPredictHead(dim=dim, out_dim=self.out_dim_dict["sam"])
        if "dinov3" in self.teacher_type:
            self.dino_head = PatchPredictHead(dim=dim+token_dim, out_dim=self.out_dim_dict["dinov3"])
            self.dino_ld_head = PatchPredictHead(dim=dim+token_dim, out_dim=self.out_dim_dict["dinov3"])
        if "mae" in self.teacher_type:
            self.mae_head = PatchPredictHead(dim=dim+token_dim, out_dim=self.out_dim_dict["mae"])
            self.mae_ld_head = PatchPredictHead(dim=dim+token_dim, out_dim=self.out_dim_dict["mae"])

        self.model_token = nn.Parameter(torch.empty(1, 1, token_dim))
        nn.init.normal_(self.model_token, std=1e-6) # initialize model_token
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        if os.path.exists(pretrain_ckpt):
            ckp = torch.load(pretrain_ckpt, map_location="cpu", weights_only=False)
            ckp = ckp.get('model', ckp)
            for name, _ in self.named_parameters():
                if name not in ckp:
                    print(f"Warning: {name} not in checkpoint ... ")
            self.load_state_dict(ckp, strict=False)
        self.student.interp_pos_embed()

    @torch.no_grad()
    def forward(self, x):
        """
        x: [b, c, h, w]: images
        """
        # student branch
        # loss_dict = self.model_agg(x)
        # print(loss_dict)
        # student branch
        feat = self.student(x)
        b, l, d = feat.shape
        return feat
        projs = self.stu_proj(feat)
        num_teacher = len(self.teacher_type)
        projs = torch.chunk(projs, chunks=num_teacher, dim=-1) # [[b,c,h,w],...]
        # print("Student proj: ", projs[0].shape)
        # cat token
        token = self.model_token.repeat(b, l, 1)
        projs = [torch.cat([item, token], dim=-1) for item in projs]
        for block in self.blocks:
            projs = block(projs)
        # chunk projs
        # model_cls_tokens_raw = [item[:, 0, :self.dim] for item in projs]
        # model_token_out = [item[:, [0], :] for item in projs] # [b, 1, d]
        model_feat_out =  [item[:, 1:, :] for item in projs] # [b, l, d]
        # clip_cls_token = self.model_head(model_token_out[0])[:, :self.out_dim_dict["clip"]]
        # dino_cls_token = self.model_head(model_token_out[1])
        # mae_cls_token = self.model_head(model_token_out[-1])
        # cls_token = dino_cls_token
        # clip_feat = self.clip_head.norm(model_feat_out[0])
        # clip_feat = model_feat_out[0]
        dino_feat = self.dino_head(model_feat_out[1])
        # mae_feat = self.mae_head(model_feat_out[2])
        # tea feats
        # clip_feat_tea = self.clip(x)
        # mae_feat_tea = self.mae(x)
        # dino_feat_tea = self.dinov3(x)
        # compute linear cka
        # print(clip_feat_tea.shape, clip_cls_token.shape)
        # clip_token_cka = linear_CKA(clip_cls_token, clip_feat_tea)
        # dino_token_cka = linear_CKA(dino_cls_token, dino_feat_tea[:,0,:])
        # mae_token_cka = linear_CKA(mae_cls_token, mae_feat_tea[:, 0, :])

        # dino_feat_cka = linear_CKA(dino_feat[:, 1:, :].view(b, -1), dino_feat_tea[:, 5:, :].view(b, -1))
        # mae_feat_cka = linear_CKA(mae_feat[:, 1:, :].view(b, -1), mae_feat_tea[:, 1:, :].view(b, -1))
        # print("clip token cka: ", clip_token_cka)
        # print("dino token cka: ", dino_token_cka)
        # print("mae token cka: ", mae_token_cka)
        # print("dino_feat_cka: ", dino_feat_cka)
        # print("mae_feat_cka: ", mae_feat_cka)
        return dino_feat # self.model_head.norm(projs[1])

if __name__ == "__main__":
    from backbone.vit import ViT, clip_vit_b_16
    from backbone.vit_mae import mae_vit_b_16
    from feat_visualization import make_transform
    from PIL import Image

    # pretrain_ckpt = "distills/vit_b_distill.pth"
    pretrain_ckpt = "distills/vit_l_distill.pth"
    # pretrain_ckpt = "logs/vit_l_cls_mae_dinov3_1024/clip-clip_l_mae-mae_l_dinov3-dinov3_l/checkpoint.pth"

    model = BackboneStuViT(student="vit_l", num_blocks=1, teacher_type=["clip", "mae", "dinov3"], out_dims=[768, 1024, 1024], 
        token_dim=768, dim=1024, in_dim=1024, pretrain_ckpt=pretrain_ckpt, img_size=14*16)

    # model = BackboneStuViT(student="vit_b", num_blocks=1, teacher_type=["clip", "mae", "dinov3"], out_dims=[512, 768, 768], 
    #     token_dim=512, dim=768, in_dim=768, pretrain_ckpt=pretrain_ckpt, img_size=14*16)

    image1 = Image.open("cute-furry.jpg")
    image2 = Image.open("cat.jpg")
    image3 = Image.open("cats_ship.jpg")

    transform = make_transform(224)

    image1 = transform(image1).unsqueeze(0)
    image2 = transform(image2).unsqueeze(0)
    image3 = transform(image3).unsqueeze(0)
    x = torch.cat([image1, image2, image3], dim=0)
    model(x)

    # stu = ViT(image_size=224, patch_size=16, num_classes=1, dim=768, depth=12, heads=12, mlp_dim=3072)
    # tea1 = mae_vit_b_16(ckpt_path="pretrain/mae_pretrain_vit_base.pth")
    # tea2 = ViT(image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072)
    # tea3 = ViT(image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072)

    # model_agg = Aggergator(stu, teachers=[tea1, tea2], teacher_type=["mae", "dinov3"], out_dims=[768, 768])
    # x = torch.rand(1, 3, 224, 224)
    # loss_dict = model_agg(x)
    # print(loss_dict)