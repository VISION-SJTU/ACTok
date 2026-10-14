"""
We use PCA for feature visualization
"""
from sklearn.decomposition import PCA, KernelPCA
from PIL import Image
# from test_dinov3 import feat
import torchvision.transforms as transforms
import torch
import numpy as np


# ImageNet stats
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# OpenAI stats
openai_mean = [0.48145466, 0.4578275, 0.40821073]
openai_std = [0.26862954, 0.26130258, 0.27577711]


def make_transform(input_size, use_clip_stats=False):
    if use_clip_stats:
        mean = imagenet_mean
        std = imagenet_std
    else:
        mean = openai_mean
        std = openai_std
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform


def zca_whiten(X: torch.Tensor, eps: float = 1e-3, alpha: float = 1.0) -> torch.Tensor:
    """
    Soft ZCA whitening: X_w = X_c @ U diag(S^{-alpha/2}) U^T.
    alpha=1.0 → full whitening (Cov ≈ I).
    alpha=0.5 → half-whitening (Cov ≈ S^{0.5}, boosts weak dims mildly).
    alpha=0.0 → no-op (raw centered features).
    """
    X = X - X.mean(0, keepdim=True)
    C = X.T @ X / X.shape[0]
    U, S, _ = torch.linalg.svd(C)
    scale = (S + eps).pow(-alpha / 2)
    W = U @ torch.diag(scale) @ U.T
    return X @ W


@torch.no_grad()
def extract_patch_feats(model, image, input_size, use_clip_stats=False, downscale_factor=16, device=None):
    """
    Run `model` on `image` and return the patch-token features as [N, D]
    (N = input_size**2 spatial patches).  Shared by all visualizers below.
    """
    real_size = input_size * downscale_factor # since we downscale the image by a factor of 16 or 32
    transform = make_transform(real_size, use_clip_stats)
    model.eval()
    image = Image.open(image).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    if use_clip_stats:
        patch_features = []

        def hook_fn(module, input, output):
            # output shape is [L, N, D]
            patch_features.append(output.permute(1, 0, 2))

        # 注册钩子到 transformer 最后一层
        handle = model.transformer.register_forward_hook(hook_fn)
        # interpolate position embedding
        new_pos_embed = interp_pos_embed(model.positional_embedding, input_size)
        model.positional_embedding.data = new_pos_embed
        with torch.no_grad():
            model(image)
        # index 0 是 CLS, 1: 是 patches
        features = patch_features[0]
        handle.remove()
    else:
        features = model(image)
    if not isinstance(features, torch.Tensor):
        features = features[-1]
    if features.ndim == 4:
        ## conv feature b, c, h, w
        b, c, h, w = features.shape
        features = features.view(b, c, h * w).permute(0, 2, 1) # [b, h*w, c]
    else:
        features = features[:, 5:, :]   # drop CLS + register tokens (DINOv3: 5)
    return features.squeeze(0)  # [N, D]


@torch.no_grad()
def PCAVis(model, image, input_size, use_clip_stats=False, downscale_factor=16, device=None, whiten=False, whiten_alpha=0.5):
    patch_feats = extract_patch_feats(model, image, input_size, use_clip_stats, downscale_factor, device)
    if whiten:
        patch_feats = zca_whiten(patch_feats, alpha=whiten_alpha)
    # config pca
    pca = KernelPCA(n_components=3, kernel='cosine')
    pca_features = pca.fit_transform(patch_feats.cpu().numpy())
    # norm pca
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    # reshape
    pca_features = pca_features.reshape(input_size, input_size, 3)[:,:,:3]
    return pca_features


@torch.no_grad()
def ClusterSeg(model, image, input_size, n_clusters=6, use_clip_stats=False, downscale_factor=16,
               device=None, whiten=False, whiten_alpha=0.5, seed=0):
    """
    Label-free "feature layering" visualization.

    Cluster the patch features with KMeans on L2-normalized features (= cosine
    KMeans) and paint each patch by its cluster id.  KMeans partitions space with
    *linear* (Voronoi) boundaries, so clean, contiguous, semantically coherent
    segments are direct visual evidence that the features are linearly separable
    — unlike PCA-RGB, which instead relies on anisotropy.

    Returns
    -------
    seg   : [input_size, input_size, 3] float RGB segmentation map
    labels: [N] int cluster ids (for computing silhouette etc. if desired)
    """
    from sklearn.cluster import KMeans
    from matplotlib import colormaps
    patch_feats = extract_patch_feats(model, image, input_size, use_clip_stats, downscale_factor, device)
    if whiten:
        patch_feats = zca_whiten(patch_feats, alpha=whiten_alpha)
    X = torch.nn.functional.normalize(patch_feats, dim=-1).cpu().numpy()  # cosine KMeans
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X)
    palette = colormaps['tab10'].resampled(n_clusters)(np.arange(n_clusters))[:, :3]
    seg = palette[labels].reshape(input_size, input_size, 3)
    return seg, labels


def interp_pos_embed(pos_embed, input_size, mode='bicubic'):
    pos_embed = pos_embed.unsqueeze(0)
    b, L, d = pos_embed.shape
    cls_token = pos_embed[:, :1, :]    # keep [CLS] token
    pos_tokens = pos_embed[:, 1:, :]   # remove [CLS] token
    
    # infer original grid size
    orig_size = int(L**0.5)
    assert orig_size * orig_size == L - 1, "pos_embed has unexpected shape"
    # target size
    new_size = (input_size, input_size)
    # reshape to [B, C, H, W]
    pos_tokens = pos_tokens.view(b, orig_size, orig_size, d).permute(0, 3, 1, 2)
    # interpolate
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=new_size, mode=mode, align_corners=False)
    # back to [B, HW, C]
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).view(b, -1, d)
    # concat cls + patch tokens
    new_pos_embed = torch.nn.Parameter(torch.cat([cls_token, pos_tokens], dim=1))
    return new_pos_embed.squeeze(0)