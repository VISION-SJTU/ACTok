import torch
import torch.nn.functional as F
import os


def linear_CKA_matrix(features1, features2, eps=1e-7):
    """
    Compute NxN linear CKA matrix from a list of feature maps.

    Args:
        features: list of tensors, each [1, H, W, C]
        eps: small constant for numerical stability

    Returns:
        cka_mat: [N, N] symmetric tensor of linear CKA similarities
    """
    N = len(features1)
    M = len(features2)
    cka_mat = torch.zeros(N, M)

    for i in range(N):
        Xi = features1[i]  # [1, H, W, C]
        Xi = Xi - Xi.mean(dim=0, keepdim=True)  # center features
        Ki = Xi @ Xi.T                         # linear kernel
        Ki2 = (Ki ** 2).sum().sqrt()           # Frobenius norm

        for j in range(M):
            Yj = features2[j]
            Yj = Yj - Yj.mean(dim=0, keepdim=True)

            Kj = Yj @ Yj.T
            Kj2 = (Kj ** 2).sum().sqrt()

            # Cross-similarity
            hsic = (Ki * Kj).sum()
            cka = hsic / (Ki2 * Kj2 + eps)

            cka_mat[i, j] = cka

    return cka_mat

def forbenious_product_matrix(X, Y):
    # X, Y: [N, H, W, C]
    # output: [N, N]
    print("shape of X: ", X.shape)
    print("shape of Y: ", Y.shape)
    N, H, W, C = X.shape
    X, Y = X.view(N, -1), Y.view(N, -1)
    X, Y = X - X.mean(dim=-1, keepdim=True), Y - Y.mean(dim=-1, keepdim=True)
    X, Y = F.normalize(X, dim=-1), F.normalize(Y, dim=-1)
    sim = X @ Y.T
    sim = (sim - sim.min()) / (sim.max() - sim.min())
    return sim

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np


def visualize_cka(cka_mat, layer_names=None, figsize=(6,5), cmap='viridis'):
    """
    Visualize an NxN CKA similarity matrix.

    Args:
        cka_mat: torch.Tensor or np.ndarray [N, N]
        layer_names: optional list of N strings for axis labels
        figsize: tuple, figure size
        cmap: matplotlib colormap
    """
    if isinstance(cka_mat, torch.Tensor):
        cka_mat = cka_mat.cpu().numpy()
    mask = np.tril(np.ones_like(cka_mat, dtype=bool))

    N = cka_mat.shape[0]
    # if layer_names is None:
    #     layer_names = [f'F{i+1}' for i in range(N)]
    layer_names = ['{}'.format(i+1) for i in range(N)]
    plt.figure(figsize=figsize)
    sns.heatmap(
        cka_mat, 
        xticklabels=layer_names, 
        yticklabels=layer_names[::-1], 
        annot=False,
        square=True, 
        # cmap=cmap, 
        cbar=True, 
        vmin=0.0, vmax=1.0,
        linewidths=0.5
    )
    plt.title("Cosine CKA Similarity Map", fontsize=13, pad=10)
    plt.xlabel("pwconv2 layers")
    plt.ylabel("pwconv1 layers")
    plt.tight_layout()
    # plt.show()
    plt.savefig("pwconv_layers.png")

def visualize_grouped_cka(cka_mat, group_size=2, figsize=(6,5), cmap='magma'):
    """
    Visualize grouped NxN CKA matrix highlighting within-group similarity.

    Args:
        cka_mat: [N, N] tensor or array (N = num_groups * group_size)
        group_size: number of features per group (default=2)
        figsize: figure size
        cmap: colormap for sns.heatmap
    """
    if isinstance(cka_mat, torch.Tensor):
        cka_mat = cka_mat.cpu().numpy()

    N = cka_mat.shape[0]
    M = N // group_size  # number of groups

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cka_mat, vmin=0, vmax=1, cmap=cmap,
        square=True, cbar=True, linewidths=0.5,
        xticklabels=False, yticklabels=False, ax=ax
    )

    # Draw lines to mark group boundaries
    for g in range(1, M):
        ax.axhline(g * group_size, color='white', lw=2)
        ax.axvline(g * group_size, color='white', lw=2)

    ax.set_title("Linear CKA Similarity: Within-group Pairs Highlighted", fontsize=13, pad=10)
    plt.tight_layout()
    plt.show()

# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from transformers import pipeline
import os
from safetensors import safe_open
from PIL import Image
import torchvision.transforms as transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
print(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

input_size = 224

trans_train = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        pw_res = [x.mean(dim=[1,2])]
        x = self.act(x)
        x = self.pwconv2(x)
        pw_res.append(x.mean(dim=[1,2]))
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x, pw_res

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                 last_norm=False):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.last_norm = last_norm
        if self.last_norm: 
            self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def load_model(self, pretrain_ckpt):
        if "safetensors" not in pretrain_ckpt:
            state = torch.load(pretrain_ckpt, map_location="cpu", weights_only=True)
            state = state.get("module", state)
            state = state.get("model", state)
            for name, _ in self.named_parameters():
                if name not in state:
                    print(f"Warning: {name} not in pretrain checkpoint")
            self.load_state_dict(state, strict=False)
        else:
            with safe_open(pretrain_ckpt, framework="pt") as state:
                tran_state = dict()
                for k in state.keys():
                    v = state.get_tensor(k)
                    if "downsample" in k:
                        new_k = k.replace("downsample_layers.", "")
                        new_k = new_k.replace("stages", "downsample_layers")
                        tran_state[new_k] = v
                    elif "stages" in k:
                        k = k.replace("layers.", "")
                        if "depthwise" in k:
                            tran_state[k.replace("depthwise_conv", "dwconv")] = v
                        elif "pointwise" in k:
                            tran_state[k.replace("pointwise_conv", "pwconv")] = v
                        elif "gamma" in k:
                            tran_state[k] = v
                        elif "layer_norm" in k:
                            tran_state[k.replace("layer_norm", "norm")] = v
                    elif "layer_norm" in k:
                        tran_state[k.replace("layer_norm", "norm")] = v
                        
                for name, _ in self.named_parameters():
                    if name not in tran_state.keys():
                        print(f"Warning: {name} not in pretrain checkpoint")
                self.load_state_dict(tran_state, strict=False)
        print("==> Load pretrain checkpoint from: {}".format(pretrain_ckpt))

    @torch.no_grad()
    def forward(self, x):
        pw_res_all = []
        for i in range(4):
            res = []
            x = self.downsample_layers[i](x)
            for stage in self.stages[i]:
                x, pw_res = stage(x)
                res.extend(pw_res)
            # x = self.stages[i](x)
            pw_res_all.append(res)
        if self.last_norm:
            x = self.norm(x)
        return x, pw_res_all
    
    def freeze(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.requires_grad_(False)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def convnext_b(pretrain_ckpt="", pretrained=False):
    model = ConvNeXt(depths=[3, 3, 27, 3], drop_path_rate=0.0, dims=[128, 256, 512, 1024])
    if os.path.exists(pretrain_ckpt):
        model.load_model(pretrain_ckpt)
    return model


if __name__ == "__main__":
    from backbone.vanilla_vit import VanillaViT

    # vit_model = VanillaViT()

    sub_classes = os.listdir("cka_imgs")
    batch = []
    for sub_cls in sub_classes[:100]:
        image_names = os.listdir("cka_imgs/{}".format(sub_cls))
        img_name = image_names[0]
        img = Image.open(os.path.join("cka_imgs", sub_cls, img_name)).convert('RGB')
        img = trans_train(img).unsqueeze(0)
        batch.append(img)
        print(img.mean(), img.std(), img_name)
    batch = torch.cat(batch, dim=0)
    print("batch size: ", batch.shape)
    pretrain_ckpt = "pretrain/convnext_base_22k_224.pth"
    model = convnext_b(pretrain_ckpt=pretrain_ckpt)
    _, outs = model(batch)
    # with torch.no_grad():
    #     _, outs = vit_model(batch)
    # feats = outs
    feats = []
    feats.extend(outs[0])
    feats.extend(outs[1])
    feats.extend(outs[2])
    feats.extend(outs[3])
    # sim = forbenious_product_matrix(out, out)
    sim = linear_CKA_matrix(feats[::2][::-1], feats[1::2])
    visualize_cka(sim)
    print(sim) 
    
