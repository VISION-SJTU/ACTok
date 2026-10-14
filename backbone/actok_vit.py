import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_layer_old import CrossModelBlock, InnerModelBlock, TokenPredictHead, PatchPredictHead
from math import sqrt
import os
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

def linear_CKA(X, Y):
    # X, Y: [N, d]
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (X @ Y.T).pow(2).sum()
    norm = (X @ X.T).pow(2).sum().sqrt() * (Y @ Y.T).pow(2).sum().sqrt()
    return hsic / norm

# ImageNet stats
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# OpenAI stats
openai_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
openai_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

class ACTokViTTower(nn.Module):
    """
    Aggregate multiple pre-trained models
    """
    def __init__(self, vision_tower, args, delay_load=False):
        super(ACTokViTTower, self).__init__()
        # intermediate block
        self.student = VanillaViT("vit_large_patch16_224", pretrained=False)
        teacher_type = ['clip-clip_b', 'mae-mae_b', 'dinov3-dinov3_b']
        out_dims = [768, 1024, 1024]
        self.teacher_type = [item.split("-")[0] for item in teacher_type]  # do not include -b, -l
        self.dim = 1024
        self.in_dim = 1024
        self.out_dim_dict = dict(zip(self.teacher_type, out_dims))
        self.token_dim = 768
        self.stu_proj = nn.Linear(self.in_dim, self.dim*len(teacher_type))
        num_blocks = 1
        pretrain_ckpt = ''
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        print("==> number of blocks: ", num_blocks)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
            CrossModelBlock(self.dim+self.token_dim, num_models=len(teacher_type), heads=8*len(teacher_type)),
            InnerModelBlock(self.dim+self.token_dim, num_models=len(teacher_type), heads=8)
        ))
        
        self.model_head = TokenPredictHead(dim=self.token_dim, out_dim=max(out_dims))  # max dim
        self.cls_token_head = TokenPredictHead(dim=self.dim, out_dim=max(out_dims))  # max dim
        if "clip" in self.teacher_type:
            self.clip_head = TokenPredictHead(dim=self.dim, out_dim=self.out_dim_dict["clip"])
        if "sam" in self.teacher_type:
            self.sam_head = PatchPredictHead(dim=self.dim, out_dim=self.out_dim_dict["sam"])
        if "dinov3" in self.teacher_type:
            self.dino_head = PatchPredictHead(dim=self.dim, out_dim=self.out_dim_dict["dinov3"])
        if "mae" in self.teacher_type:
            self.mae_head = PatchPredictHead(dim=self.dim, out_dim=self.out_dim_dict["mae"])
        
        self.model_token = nn.Parameter(torch.empty(1, 1, self.token_dim))
        nn.init.normal_(self.model_token, std=1e-6) # initialize model_token
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        if os.path.exists(pretrain_ckpt):
            ckp = torch.load(pretrain_ckpt, map_location="cpu", weights_only=False)
            ckp = ckp.get('model', ckp)
            for name, _ in self.named_parameters():
                if name not in ckp:
                    print(f"Warning: {name} not in checkpoint ... ")
            self.load_state_dict(ckp, strict=False)
        for param in self.parameters():
            param.requires_grad_(False)
    
    @torch.no_grad()
    def _forward_features(self, x):
        """
        x: [b, c, h, w]: images
        """
        feat = self.student(x)
        b, l, d = feat.shape
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
        model_token_out = projs[0][:, [0], self.dim:]# [b, 1, d]
        model_feat_out = projs[0][:, 1:, :self.dim]
        clip_cls_token = self.model_head(model_token_out[:, :, :768])
        clip_patch_token = self.clip_head(model_feat_out)
        if self.select_feature == 'patch':
            return clip_patch_token
        else:
            return clip_cls_token

    @torch.no_grad()
    def forward(self, images):
        """
        x: [b, c, h, w]: images
        """
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self._forward_features(image.to(device=self.device))
                image_features.append(image_feature)
        else:
            image_features = self._forward_features(image.to(device=self.device))
        return image_features
    
    def load_model(self, device_map=None):
        if isinstance(device_map, dict):
            self.to(device_map[device_map.keys[0]])
        else:
            self.to(device_map)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.dtype

    @property
    def device(self):
        return self.device

    # @property
    # def config(self):
    #     if self.is_loaded:
    #         return self.vision_tower.config
    #     else:
    #         return self.cfg_only

    @property
    def hidden_size(self):
        return 1024

    @property
    def num_patches_per_side(self):
        return 24

    @property
    def num_patches(self):
        return 576
    