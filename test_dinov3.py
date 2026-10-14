import torch
from backbone.convnext import ConvNeXt, dinov3_convnext_b, DINOv3ConvNeXt
from transformers import pipeline
from transformers.image_utils import load_image
from PIL import Image
from torchvision.transforms import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


trans_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)
print(image)

feature_extractor = pipeline(
    model="pretrain/base",
    task="image-feature-extraction", 
)

features = feature_extractor(image)
features = torch.Tensor(features).squeeze(0)[1:].view(7, 7, 1024)

convnext_b = dinov3_convnext_b("pretrain/base/model.safetensors")
convnext_b.freeze()
image = trans_train(image)
image = image.unsqueeze(0)
print(image.shape)
output = convnext_b(image)
output = output.view(1024, 7, 7).permute(1, 2, 0)

diff = features - output
print(diff.mean())
print(features.mean(), output.mean(), features.shape, output.shape)

model = DINOv3ConvNeXt("pretrain/base").cpu()
feat = model.forward(image)
print(feat.mean())

# dinov3_vit_b = torch.hub.load("dinov3", 'dinov3_vitb16', source='local', weights="pretrain/dinov3_vit_b.pth")

