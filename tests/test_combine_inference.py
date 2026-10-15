import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch
from torch import nn

from ModelAggregator_ViT_combine import BackboneStuViT
from backbone.vanilla_vit import VanillaViT
from feat_visualization import extract_patch_feats
from layers.vit_layer_channel import CrossModelBlock, InnerModelBlock


class FixedFeatures(nn.Module):
    def __init__(self, tokens, channels):
        super().__init__()
        self.register_buffer("features", torch.arange(tokens * channels).float().reshape(1, tokens, channels))

    def forward(self, images):
        return self.features.expand(images.shape[0], -1, -1)


class CombineInferenceTests(unittest.TestCase):
    def test_forward_runs_both_interactions_and_returns_dino_patches(self):
        # A small real interaction stack catches the former backbone early return.
        model = BackboneStuViT.__new__(BackboneStuViT)
        nn.Module.__init__(model)
        model.student = FixedFeatures(5, 2)
        model.teacher_type = ["clip", "dinov3", "mae"]
        model.stu_proj = nn.Linear(2, 6)
        model.model_token = nn.Parameter(torch.zeros(1, 1, 1))
        model.blocks = nn.ModuleList([nn.Sequential(
            CrossModelBlock(3, num_models=3, heads=3),
            InnerModelBlock(3, num_models=3, heads=1),
        )])
        model.dino_head = nn.Linear(3, 4)
        calls = []
        handles = [module.register_forward_hook(lambda module, args, result: calls.append(type(module).__name__))
                   for module in model.blocks[0]]
        model.eval()
        inputs = torch.zeros(1, 3, 32, 32)
        output = model(inputs)
        self.assertEqual(tuple(output.shape), (1, 4, 4))
        self.assertEqual(calls, ["CrossModelBlock", "InnerModelBlock"])
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            model.dino_head.bias.add_(1)
        torch.testing.assert_close(model(inputs), output + 1)

    def test_pca_accepts_patches_cls_and_dinov3_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (32, 32)).save(path)
            for prefix in (0, 1, 5):
                with self.subTest(prefix=prefix):
                    model = FixedFeatures(4 + prefix, 3)
                    actual = extract_patch_feats(model, path, input_size=2, device="cpu")
                    torch.testing.assert_close(actual, model.features[0, prefix:])
            with self.assertRaisesRegex(ValueError, "prefix tokens"):
                extract_patch_feats(FixedFeatures(6, 3), path, input_size=2, device="cpu")

    def test_vit_accepts_interpolated_input_resolution(self):
        model = VanillaViT("vit_tiny_patch16_224", pretrained=False, img_size=32).eval()
        model.interp_pos_embed()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (1, 5, 192))


if __name__ == "__main__":
    unittest.main()
