import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from backbone.vanilla_vit import VanillaViT
import os


def linear_CKA(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (X @ Y.T).pow(2).sum()
    norm = (X @ X.T).pow(2).sum().sqrt() * (Y @ Y.T).pow(2).sum().sqrt()
    return hsic / (1e-7 + norm)


def cosine_CKA(X, Y, eps=1e-7):
    X = X / (X.norm(dim=1, keepdim=True) + eps)
    Y = Y / (Y.norm(dim=1, keepdim=True) + eps)
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (X @ Y.T).pow(2).sum()
    norm = (X @ X.T).pow(2).sum().sqrt() * (Y @ Y.T).pow(2).sum().sqrt()
    return hsic / (eps + norm)


imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
openai_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
openai_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def imagenet_to_openai(x):
    scale = imagenet_std / openai_std
    shift = (imagenet_mean - openai_mean) / openai_std
    return x * scale.to(x.device) + shift.to(x.device)


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, x1, x2):
        x1 = F.normalize(x1, dim=-1)
        x2 = F.normalize(x2, dim=-1)
        sim = (x1 * x2).sum(dim=-1)
        return 1 - sim.mean()


class RelationalCosineLoss(nn.Module):
    def __init__(self):
        super(RelationalCosineLoss, self).__init__()

    def forward(self, student_feat, teacher_feat):
        if student_feat.dim() == 3:
            student_feat = student_feat[:, 0]
        if teacher_feat.dim() == 3:
            teacher_feat = teacher_feat[:, 0]

        s_norm = F.normalize(student_feat, p=2, dim=-1)
        t_norm = F.normalize(teacher_feat, p=2, dim=-1)

        sim_s = torch.mm(s_norm, s_norm.t())
        sim_t = torch.mm(t_norm, t_norm.t())

        batch_size = sim_s.shape[0]
        mask = torch.eye(batch_size, device=sim_s.device).bool()

        sim_s_offdiag = sim_s[~mask]
        sim_t_offdiag = sim_t[~mask]

        return F.mse_loss(sim_s_offdiag, sim_t_offdiag)


class Aggregator(nn.Module):
    """
    Direct student-teacher feature alignment.
    No cross-model or inner-model interaction modules — per-teacher projection heads
    directly align student features to each teacher's feature space.
    """

    def __init__(self, student: nn.Module, teachers: dict, in_dim=768,
                 teacher_type=['sam', 'clip', 'dinov3'], out_dims=[768, ]):
        super(Aggregator, self).__init__()
        self.student = student
        self.teachers = nn.ModuleList()
        assert len(teacher_type) == len(out_dims), "number of teacher types must match number of out dims"
        assert len(teachers) == len(teacher_type), "number of teachers should equal types"
        for tea in teachers:
            self.teachers.append(tea)
        self.teacher_type = [item.split("-")[0] for item in teacher_type]
        self.in_dim = in_dim
        self.out_dim_dict = dict(zip(self.teacher_type, out_dims))

        # Per-teacher projection heads (student dim -> teacher dim)
        self.cls_projs = nn.ModuleDict()
        self.patch_projs = nn.ModuleDict()
        for t_type, out_dim in zip(self.teacher_type, out_dims):
            self.cls_projs[t_type] = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, out_dim),
            )
            if t_type in ["sam", "dinov3", "mae"]:
                self.patch_projs[t_type] = nn.Sequential(
                    nn.Linear(in_dim, in_dim),
                    nn.GELU(),
                    nn.Linear(in_dim, out_dim),
                )

        self.l2_criterion = nn.MSELoss()
        self.cos_criterion = CosineLoss()
        self.relation_criterion = RelationalCosineLoss()
        self.criterion_table = {
            "cls": self.cos_criterion,
            "sam": self.l2_criterion,
            "clip": self.cos_criterion,
            "dinov3": self.cos_criterion,
            "mae": self.l2_criterion,
        }

    def _interpolate_patches(self, stu_feat, teacher_feat):
        if stu_feat.shape[1] != teacher_feat.shape[1]:
            L_stu, L_tea = stu_feat.shape[1], teacher_feat.shape[1]
            b, c = stu_feat.shape[0], stu_feat.shape[2]
            stu_feat = stu_feat.reshape(b, int(sqrt(L_stu)), int(sqrt(L_stu)), c).permute(0, 3, 1, 2)
            stu_feat = F.interpolate(stu_feat, size=(int(sqrt(L_tea)), int(sqrt(L_tea))), mode="bilinear")
            stu_feat = stu_feat.view(b, c, -1).permute(0, 2, 1).contiguous()
        return stu_feat

    def forward(self, x):
        # Student forward
        feat = self.student(x)
        stu_cls = feat[:, 0, :]
        stu_patches = feat[:, 1:, :]

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_feats = dict()
            for k, teacher in zip(self.teacher_type, self.teachers):
                inp = imagenet_to_openai(x) if "clip" in k else x
                teacher_feats[k] = teacher(inp)

        loss_dict = dict()

        if "cls" in self.teacher_type:
            proj = self.cls_projs["cls"](stu_cls)
            tea_cls = teacher_feats["cls"][:, 0, :]
            loss_dict["cls_cls"] = self.cos_criterion(proj, tea_cls)
            loss_dict["cls_rel"] = self.relation_criterion(proj, tea_cls) * 0.5

        if "clip" in self.teacher_type:
            proj = self.cls_projs["clip"](stu_cls)
            tea_feat = teacher_feats["clip"]
            loss_dict["clip_cls"] = self.cos_criterion(proj, tea_feat)
            loss_dict["clip_rel"] = self.relation_criterion(proj, tea_feat) * 0.5

        if "dinov3" in self.teacher_type:
            proj_cls = self.cls_projs["dinov3"](stu_cls)
            tea_cls = teacher_feats["dinov3"][:, 0, :]
            loss_dict["dinov3_cls"] = self.cos_criterion(proj_cls, tea_cls)
            loss_dict["dinov3_rel"] = self.relation_criterion(proj_cls, tea_cls) * 0.5
            proj_patch = self.patch_projs["dinov3"](stu_patches)
            tea_patch = teacher_feats["dinov3"][:, 5:, :]
            proj_patch = self._interpolate_patches(proj_patch, tea_patch)
            loss_dict["dinov3_patch"] = self.criterion_table["dinov3"](proj_patch, tea_patch)

        if "sam" in self.teacher_type:
            proj_cls = self.cls_projs["sam"](stu_cls)
            tea_sam = teacher_feats["sam"][:, 1:, :]
            tea_pool = tea_sam.mean(dim=1)
            loss_dict["sam_pool"] = self.l2_criterion(tea_pool, proj_cls)
            proj_patch = self.patch_projs["sam"](stu_patches)
            loss_dict["sam_patch"] = self.criterion_table["sam"](proj_patch, tea_sam)

        if "mae" in self.teacher_type:
            proj_cls = self.cls_projs["mae"](stu_cls)
            tea_cls = teacher_feats["mae"][:, 0, :]
            loss_dict["mae_cls"] = self.cos_criterion(proj_cls, tea_cls)
            loss_dict["mae_rel"] = self.relation_criterion(proj_cls, tea_cls) * 0.5
            proj_patch = self.patch_projs["mae"](stu_patches)
            tea_patch = teacher_feats["mae"][:, 1:, :]
            proj_patch = self._interpolate_patches(proj_patch, tea_patch)
            loss_dict["mae_patch"] = self.criterion_table["mae"](proj_patch, tea_patch)

        return loss_dict


class DirectAlignViT(nn.Module):
    """
    Direct alignment student ViT for inference.
    """

    def __init__(self, student="vit_b", in_dim=768,
                 teacher_type=['sam', 'clip', 'dinov3'], out_dims=[768, ],
                 pretrain_ckpt="", img_size=224):
        super(DirectAlignViT, self).__init__()
        if student == "vit_b":
            self.student = VanillaViT("vit_base_patch16_224", pretrained=False, img_size=img_size)
        else:
            self.student = VanillaViT("vit_large_patch16_224", pretrained=False, img_size=img_size)

        self.teacher_type = [item.split("-")[0] for item in teacher_type]
        self.in_dim = in_dim
        self.out_dim_dict = dict(zip(self.teacher_type, out_dims))

        self.cls_projs = nn.ModuleDict()
        self.patch_projs = nn.ModuleDict()
        for t_type, out_dim in zip(self.teacher_type, out_dims):
            self.cls_projs[t_type] = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, out_dim),
            )
            if t_type in ["sam", "dinov3", "mae"]:
                self.patch_projs[t_type] = nn.Sequential(
                    nn.Linear(in_dim, in_dim),
                    nn.GELU(),
                    nn.Linear(in_dim, out_dim),
                )

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
        return self.student(x)


if __name__ == "__main__":
    from backbone.vit import clip_vit_b_16
    from backbone.vit_mae import mae_vit_b_16

    pretrain_ckpt = ""

    model_agg = DirectAlignAggregator(
        student=VanillaViT("vit_base_patch16_224", pretrained=False, img_size=224),
        teachers=[mae_vit_b_16(ckpt_path="pretrain/mae_pretrain_vit_base.pth"),
                  clip_vit_b_16(),
                  ],
        teacher_type=["mae", "dinov3"],
        out_dims=[768, 768],
        in_dim=768,
    )

    x = torch.rand(2, 3, 224, 224)
    loss_dict = model_agg(x)
    print(loss_dict)
