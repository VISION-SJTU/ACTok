from backbone.convnext import DINOv3ConvNeXt, convnext_b, convnext_l, convnext_b_norm
from backbone.resnet import resnet50
from backbone.vit_mae import mae_vit_b_16
from backbone.vanilla_vit import VanillaViT
from backbone.vit import DINOv3ViT, clip_vit_b_16
from feat_visualization import PCAVis
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.utils as tv_utils
from PIL import Image
import torch
import pdb
from ModelAggregator_Conv import BackboneStuConv
import os

import inspect
import transformers

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

# model = DINOv3ConvNeXt("pretrain/base").cpu()
# print(model.model.__class__)
# model = resnet50()
# model.load_model("pretrain/resnet_distill.pth")
# model = vit_base(pretrain="pretrain/proteus_vit_b.pth")
# model = mae_vit_b_16(ckpt_path="pretrain/mae_pretrain_vit_base.pth")
# help(model)
# pdb.set_trace()
# model = convnext_b_norm(pretrain_ckpt="pretrain/convnext_base_22k_224.pth")
# model = convnext_b_norm(pretrain_ckpt="pretrain/convnext_b_norm_distill.pth")
# model = BackboneStuConv(student=resnet50(), teacher_type=["classifier", "mae", "dino"], out_dims=[2048, 2048, 2048], num_blocks=2, in_dim=2048, dim=2048, token_dim=1024)
# model.load_state_dict(torch.load("pretrain/resnet_distill_student.pth", map_location="cpu"), strict=True)
# model = VanillaViT(model_name="vit_base_patch16_224", pretrained=False)
# ckp = torch.load("pretrain/vit_b_distill.pth", weights_only=False, map_location="cpu")
# model.load_state_dict(ckp)
model = DINOv3ViT("pretrain/dinov3/dinov3-vitb16-pretrain-lvd1689m").to(device)
# print(model)

imgs = os.listdir("cats")
for image in imgs:
    if image.endswith("png") or image.endswith("jpg"):
        image_name = image.split(".")[0]
        image = f"cats/{image}"
        pca_features = PCAVis(model, image, 128, downscale_factor=16, device=device)
        plt.imshow(pca_features)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"cats/dinov3/{image_name}.jpg", bbox_inches='tight', pad_inches=0)

        # save original image
        original_image = Image.open(image).convert("RGB")
        img = make_transform(128*16)(original_image)
        tv_utils.save_image(img, f"cats/original/{image_name}_original.jpg")
