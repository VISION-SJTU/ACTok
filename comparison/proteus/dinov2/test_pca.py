from imp import init_builtin
from feat_visualization import PCAVis
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.utils as tv_utils
from PIL import Image
from vision_transformer import vit_base


def make_transform(input_size):
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(input_size),
        transforms.ToTensor()
    ])
    return transform

image = "../../../steel_new.jpg"
image_name = image.split("/")[-1].split(".")[0]
# model = DINOv3ConvNeXt("pretrain/base").cpu()
# model = resnet50()
# model.load_model("pretrain/resnet_distill.pth")
model = vit_base(pretrain="../../../pretrain/proteus_vit_b.pth", patch_size=14, init_values=0.1)
# model = mae_vit_b_16(ckpt_path="pretrain/mae_pretrain_vit_base.pth")
model.eval()
pca_features = PCAVis(model, image, 128, downscale_factor=14)
plt.imshow(pca_features)
plt.xticks([])
plt.yticks([])
plt.savefig(f"visualizations/{image_name}.png", bbox_inches='tight', pad_inches=0)

# save original image
original_image = Image.open(image).convert("RGB")
img = make_transform(128*14)(original_image)
tv_utils.save_image(img, f"visualizations/{image_name}_original.png")