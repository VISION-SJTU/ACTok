"""Reproduce historical backbone PCA without overwriting existing figures.

Example:
    python tools/validate_checkpoint_pca.py --architecture resnet50 \
        --checkpoint pretrain/resnet_distill.pth --image imgs/oranges.jpeg \
        --input-size 3072 --pca linear-whiten \
        --reference path/to/historical.png --output outputs/resnet_pca.png

The legacy OpenAI normalization is intentional: it matches the old visualization
scripts. Use --normalization imagenet for a separate ImageNet-normalized run.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import KernelPCA, PCA
from threadpoolctl import threadpool_limits
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backbone.convnext import convnext_b_norm
from backbone.resnet import resnet50


def load_backbone(architecture, checkpoint):
    """Load every backbone parameter strictly; aggregation heads are not used."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    if any(key.startswith("student.") for key in state):
        state = {key.removeprefix("student."): value for key, value in state.items()
                 if key.startswith("student.")}
    model = resnet50() if architecture == "resnet50" else convnext_b_norm()
    model.load_state_dict(state, strict=True)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["resnet50", "convnext_b"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--pca", choices=["linear-whiten", "cosine-kernel"], required=True)
    parser.add_argument("--normalization", choices=["legacy-openai", "imagenet"], default="legacy-openai")
    parser.add_argument("--reference")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    if args.normalization == "legacy-openai":
        mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(args.input_size), transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    model = load_backbone(args.architecture, args.checkpoint)
    inputs = transform(Image.open(args.image).convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        features = model(inputs)
    if features.ndim != 4 or not torch.isfinite(features).all():
        raise ValueError("Expected finite BCHW backbone features")
    _, _, height, width = features.shape
    values = features.flatten(2).transpose(1, 2).squeeze(0).numpy()
    reducer = (PCA(n_components=3, whiten=True, random_state=0)
               if args.pca == "linear-whiten" else
               KernelPCA(n_components=3, kernel="cosine", eigen_solver="arpack", random_state=0))
    with threadpool_limits(limits=args.threads):
        rgb = reducer.fit_transform(values)
    span = rgb.max() - rgb.min()
    if not np.isfinite(rgb).all() or span <= 0:
        raise ValueError("PCA produced invalid or constant colors")
    rgb = ((rgb - rgb.min()) / span).reshape(height, width, 3)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    protected = {Path(args.image).resolve()}
    if args.reference:
        protected.add(Path(args.reference).resolve())
    if output.resolve() in protected:
        raise ValueError("Output must not overwrite the source or reference image")
    plt.figure()
    plt.imshow(rgb)
    if args.pca == "linear-whiten":
        plt.xticks([])
        plt.yticks([])
    else:
        plt.axis("off")
    plt.savefig(output, bbox_inches="tight", pad_inches=0)
    plt.close()
    report = {**vars(args), "strict_backbone_loading": True,
              "feature_source": "backbone", "feature_shape": list(features.shape),
              "finite_features": True, "pca_seed": 0}
    if args.reference:
        a = np.asarray(Image.open(args.reference).convert("RGB"), dtype=float)[3:-3, 3:-3] / 255
        b = np.asarray(Image.open(output).convert("RGB"), dtype=float)[3:-3, 3:-3] / 255
        if a.shape != b.shape:
            raise ValueError("Rendered image sizes differ; compare at the same Matplotlib figure settings")
        correlations = np.corrcoef(a.reshape(-1, 3).T, b.reshape(-1, 3).T)[:3, 3:]
        report.update(pixel_exact=bool(np.array_equal(a, b)),
                      pixel_rmse=float(np.sqrt(np.mean((a - b) ** 2))),
                      rgb_channel_correlations=np.diag(correlations).tolist(),
                      mean_rgb_channel_correlation=float(np.diag(correlations).mean()))
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
