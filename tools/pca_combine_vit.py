"""Visualize DINO patch features from the channel-concat combine model.

Run from the repository root, for example:
    python tools/pca_combine_vit.py --image imgs/aerial.jpeg --input-size 2048 \
        --output outputs/combine/aerial.png

The default checkpoint is the locally available combine ViT-B / Large-teacher
variant. It differs from the original vit_b_distill.pth release.
PCAVis retains its historical OpenAI image normalization.
"""

import argparse
import json
from pathlib import Path
import sys
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ModelAggregator_ViT_combine import BackboneStuViT
from feat_visualization import PCAVis
from layers.vit_layer_channel import Attention


def memory_efficient_attention(self, x):
    q, k, v = self.to_qkv(x).chunk(3, dim=-1)
    batch, length, _ = q.shape
    q, k, v = [value.reshape(batch, length, self.heads, -1).transpose(1, 2)
               for value in (q, k, v)]
    output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, dropout_p=self.dropout.p if self.training else 0.0, scale=self.scale,
    )
    return self.to_out(output.transpose(1, 2).reshape(batch, length, -1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="pretrain/vit_b_clsl_mael_dinov3l.pth")
    parser.add_argument("--image", required=True)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.input_size <= 0 or args.input_size % 16:
        parser.error("--input-size must be a positive multiple of 16")
    if not Path(args.checkpoint).is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    output = Path(args.output)
    if output.resolve() == Path(args.image).resolve():
        parser.error("Output must not overwrite the source image")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    model = BackboneStuViT(
        student="vit_b", num_blocks=1, in_dim=768, dim=768,
        token_dim=512, out_dims=[768, 1024, 1024],
        teacher_type=["clip", "dinov3", "mae"],
        pretrain_ckpt=args.checkpoint, img_size=224,
    ).eval()
    # Verify numerical equivalence before using bounded-memory attention at high resolution.
    probe = torch.zeros(1, 3, 224, 224)
    with torch.inference_mode():
        native = model(probe)
        for module in model.modules():
            if isinstance(module, Attention):
                module.forward = types.MethodType(memory_efficient_attention, module)
        efficient = model(probe)
    torch.testing.assert_close(efficient, native, rtol=1e-4, atol=1e-5)
    model.student.interp_pos_embed(new_size=args.input_size // 16)
    np.random.seed(0)
    with threadpool_limits(limits=args.threads):
        rgb = PCAVis(model, args.image, input_size=args.input_size // 16,
                     downscale_factor=16, device="cpu", whiten=False)
    if not np.isfinite(rgb).all():
        raise ValueError("PCA produced non-finite colors")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(rgb)
    plt.axis("off")
    plt.savefig(output, bbox_inches="tight", pad_inches=0)
    plt.close()
    report = {**vars(args), "implementation": "ModelAggregator_ViT_combine.BackboneStuViT",
              "feature_source": "DINO head after cross-model and inner-model interactions",
              "cross_model_blocks": 1, "inner_model_blocks": 1,
              "feature_shape": [1, (args.input_size // 16) ** 2, 1024],
              "pca_shape": list(rgb.shape), "normalization": "legacy-openai",
              "attention_check_max_abs_difference": float((efficient - native).abs().max())}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
