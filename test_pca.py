from backbone.convnext import DINOv3ConvNeXt, convnext_b, convnext_l, convnext_b_norm
from backbone.resnet import resnet50
from backbone.vit_mae import mae_vit_b_16
from backbone.vit import DINOv3ViT, clip_vit_b_16
from feat_visualization import PCAVis
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.utils as tv_utils
from PIL import Image
import torch
import pdb
from ModelAggregator_Conv import BackboneStuConv
from ModelAggregator_ViT import BackboneStuViT
from backbone.vanilla_vit import VanillaViT
from backbone.radio import Radio
from backbone.proteus import vit_base

import inspect
# import transformers
import pdb

from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel

print(DINOv3ViTModel.__module__)  # module name
print(inspect.getfile(DINOv3ViTModel))  # file path


device = "cpu"

def make_transform(input_size):
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(input_size),
        transforms.ToTensor()
    ])
    return transform

image = "imgs/cute-furry.jpg"
image_name = image.split("/")[-1].split(".")[0]

print(image_name)
# model = DINOv3ConvNeXt("pretrain/base").cpu()
# print(model.model.__class__)
# model = resnet50()
# model.load_model("pretrain/resnet_distill.pth")
# model = vit_base(img_size=224, patch_size=14, init_values=1.0, ffn_layer='mlp', block_chunks=0,
#             num_register_tokens=0,
#             interpolate_antialias=False,
#             interpolate_offset=0.1)
# ckp = torch.load('pretrain/proteus_vit_b.pth')
# state = dict()
# for k, v in ckp.items():
#     if 'student' in k:
#         state[k.replace('student.backbone.', '')] = v
# model.load_state_dict(state)
# model = mae_vit_b_16(ckpt_path="pretrain/mae_pretrain_vit_base.pth")
# help(model)
# pdb.set_trace()
# model = convnext_b_norm(pretrain_ckpt="pretrain/convnext_base_22k_224.pth")
# model = convnext_b_norm(pretrain_ckpt="pretrain/convnext_b_norm_vits.pth")
# model = BackboneStuConv(student=resnet50(), teacher_type=["classifier", "mae", "dino"], out_dims=[2048, 2048, 2048], num_blocks=2, in_dim=2048, dim=2048, token_dim=1024)
# model = BackboneStuConv(student=convnext_b_norm(), teacher_type=["clip-clip_b", "mae_vit-mae_b", "dinov3_vit-dinov3_b"], out_dims=[512, 768, 768], num_blocks=2, in_dim=1024, dim=768, token_dim=512,
                        # pretrain_ckpt="pretrain/convnext_b_norm_distill_allvit.pth")
# model = BackboneStuConv(student=convnext_b_norm(), teacher_type=["classifier", "mae", "dino"], out_dims=[1024, 1024, 1024], num_blocks=2, in_dim=1024, dim=1024, token_dim=512,
#                         pretrain_ckpt="pretrain/convnext_b_norm_distill.pth")
# model.load_state_dict(torch.load("pretrain/resnet_distill_student.pth", map_location="cpu"), strict=True)
# model = VanillaViT(model_name="vit_base_patch16_224", pretrained=False, img_size=64*16)
# model.interp_pos_embed()
# ckp = torch.load("pretrain/mae_pretrain_vit_base.pth", weights_only=False, map_location="cpu")
# model.model.load_state_dict(ckp['model'], strict=False)
# model.interp_pos_embed()
# ckp = torch.load("pretrain/vit_b_distill.pth", weights_only=False, map_location="cpu")
# model.load_state_dict(ckp)
model = DINOv3ViT("pretrain/dinov3/dinov3-vitb16-pretrain-lvd1689m").to(device)
# pretrain_ckpt = "" # "pretrain/vit_b_distill.pth"
# # pretrain_ckpt = 'pretrain/checkpoint.pth'
# model = BackboneStuViT(student="vit_b", num_blocks=1, teacher_type=["clip-clip_b", "mae-mae_b", "dinov3-dinov3_b"], out_dims=[512, 768, 768], 
        # token_dim=512, dim=768, in_dim=768, pretrain_ckpt=pretrain_ckpt, img_size=64*16)
# print(model)
# model, _ = clip_vit_b_16()
# model = Radio()

# print(model.__class__)
# print(dir(model))
# pdb.set_trace()
# model, _ = clip_vit_b_16()
# print(dir(model))
# model.eval()
# x = torch.rand(1, 3, 224, 224)
# o1 = model(x)
# o2 = model.forward_unfold(x)

# diff = (o1 - o2).abs().sum()
# print(f"Max diff between forward and forward_unfold: {diff}")

# NOTE: whiten_alpha controls the whitening strength via S^{-alpha/2}.
#   alpha=0.0 -> identity (NO whitening, just centering) -- the previous value,
#               which is why the two panels looked identical.
#   alpha=0.5 -> soft ZCA: rebalances the spectrum, surfaces low-variance (tail)
#               directions WITHOUT amplifying pure noise -- the sweet spot.
#   alpha=1.0 -> full ZCA (Cov=I): for already-isotropic teachers (DINOv3) this
#               mostly amplifies noise and degrades the PCA visualization.
WHITEN_ALPHA = 1.0
pca_raw     = PCAVis(model, image, 64, downscale_factor=16, device=device, use_clip_stats=False, whiten=False)
pca_whitened = PCAVis(model, image, 64, downscale_factor=16, device=device, use_clip_stats=False, whiten=True, whiten_alpha=WHITEN_ALPHA)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(pca_raw)
axes[0].set_title("Raw features (PCA)")
axes[0].axis("off")
axes[1].imshow(pca_whitened)
axes[1].set_title(f"Soft ZCA-whitened (α={WHITEN_ALPHA}, PCA)")
axes[1].axis("off")
plt.tight_layout()
plt.savefig(f"visualizations/{image_name}_ld_compare.png", bbox_inches='tight', pad_inches=0.1)
plt.close()

# also save individual images
fig, ax = plt.subplots()
ax.imshow(pca_raw)
ax.axis("off")
plt.savefig(f"visualizations/{image_name}.png", bbox_inches='tight', pad_inches=0)
plt.close()

# save original image
original_image = Image.open(image).convert("RGB")
img = make_transform(64*16)(original_image)
tv_utils.save_image(img, f"visualizations/{image_name}_original.png")
