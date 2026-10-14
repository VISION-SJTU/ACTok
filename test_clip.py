from backbone.vit import clip_vit_b_16
import torch
import clip
from PIL import Image
from torchvision.transforms import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ModelAggregator_ViT_new import imagenet_to_openai

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

print(preprocess)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image = image_features @ text_features.T
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs, logits_per_image)  # prints: [[0.9927937  0.00421068 0.00299572]]

print("="*60)
print("Use our own model inference")

# ImageNet transform
clip_model, _ = clip_vit_b_16()

trans_train = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])

image = Image.open("cat.jpg")
img = trans_train(image).unsqueeze(0)
img = imagenet_to_openai(img)
# img = preprocess(image).unsqueeze(0)
# DO not transform to openai
with torch.no_grad():
    image_features = clip_model(img)
    logits_per_image = image_features @ text_features.T
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
print("Label probs:", probs, logits_per_image)  # prints: [[0.9927937  0.00421068 0.00299572]]

