# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from email.policy import strict
from functools import partial

import torch
import torch.nn as nn
import os
import timm.models.vision_transformer


class MAEViT(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(MAEViT, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        # if self.global_pool:
        #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        #     outcome = self.fc_norm(x)
        # else:
        #     x = self.norm(x)
        #     outcome = x # x[:, 0]
        outcome = x[:, 1:]
        return outcome
    
    def forward(self, x):
        return self.forward_features(x)

def mae_vit_b_16(ckpt_path, pretrained=True):
    model = MAEViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=True)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        for name, _ in model.named_parameters():
            if name not in ckpt:
                print(f"Warning: {name} not found in checkpoint")
        model.load_state_dict(ckpt, strict=False)
    # freeze
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

def mae_vit_l_16(ckpt_path, pretrained=True):
    model = MAEViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=True)
        if "model" in ckpt.keys():
            ckpt = ckpt["model"]
        for name, _ in model.named_parameters():
            if name not in ckpt:
                print(f"Warning: {name} not found in checkpoint")
        model.load_state_dict(ckpt, strict=False)
    # freeze
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model