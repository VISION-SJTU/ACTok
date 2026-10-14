"""
We use PCA for feature visualization
"""
from sklearn.decomposition import PCA
from PIL import Image
import torchvision.transforms as transforms
import torch


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


@torch.no_grad()
def PCAVis(model, image, input_size, use_clip_stats=False, downscale_factor=16):
    real_size = input_size * downscale_factor # since we downscale the image by a factor of 16 or 32
    transform = make_transform(real_size, use_clip_stats)
    model.eval()
    image = Image.open(image).convert("RGB")
    image = transform(image).unsqueeze(0)
    features = model(image)
    print(features.shape)
    if features.ndim == 4:
        ## conv feature b, c, h, w
        b, c, h, w = features.shape
        features = features.view(b, c, h * w).permute(0, 2, 1) # [b, h*w, c]
    else:
        features = features[:,:,:]
    # config pca
    pca = PCA(n_components=3, whiten=True)
    print(features.shape)
    pca_features = pca.fit_transform(features.squeeze(0).cpu().numpy())
    # norm pca
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    # reshape
    pca_features = pca_features.reshape(input_size, input_size, 3)
    return pca_features