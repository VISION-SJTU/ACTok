# ACTok

Research code for the ACTok ECCV project. This repository explores aggregating multiple pretrained visual teachers into a shared student through model tokens, cross-model interaction, and teacher-specific prediction heads.

The current release contains training code, backbone wrappers, aggregation variants, feature visualization utilities, and fine-grained classification evaluation. Pretrained teacher weights and trained ACTok checkpoints are **not included**. Paper metadata, links, and citation information will be added when available.

## Approach

The ViT training entry point uses a student ViT with frozen CLIP, DINOv3, and MAE teachers. Student features are projected into teacher-specific branches, augmented with learnable model tokens, and processed by cross-model and inner-model blocks. Prediction heads provide feature and relational distillation losses. The repository also includes convolutional and mixed-backbone experiments, direct alignment, and gradient-conflict diagnostics.

`main_ViT.py` currently selects `ModelAggregator_ViT_combine.py`. Other aggregation files are experimental variants; their checkpoint layouts and inference wrappers should not be assumed interchangeable.

## Installation

Use a Python environment with Python 3.10 or later. Training requires a CUDA-enabled PyTorch environment; install matching PyTorch and torchvision builds for your CUDA installation, then install the remaining dependencies:

```bash
git clone https://github.com/VISION-SJTU/ACTok.git
cd ACTok
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` lists dependencies found in the source; it is not a tested version lock. The installed Transformers version must support DINOv3, and PyTorch must support the AMP and distributed APIs used here. OpenAI CLIP is installed from its source repository.

## Data and teacher checkpoints

Training uses images without class labels from an ImageNet-style directory:

```text
/path/to/imagenet/
└── train/
    ├── class_0001/
    │   ├── image_0001.jpg
    │   └── ...
    └── ...
```

Pass the dataset root with `-data_path`. `datasets/dataset.py` loads the training split and applies random resized crops, horizontal flips, and ImageNet normalization. The current loader uses the entire training split; it does not implement dataset-ratio subsampling.

Prepare teacher weights locally:

| Teacher | Base configuration | Large configuration |
| --- | --- | --- |
| CLIP | OpenAI `ViT-B/16` | OpenAI `ViT-L/14` |
| DINOv3 | Local Hugging Face ViT-B/16 model directory | Local Hugging Face ViT-L/16 model directory |
| MAE | `pretrain/mae_pretrain_vit_base.pth` | `pretrain/mae_pretrain_vit_large.pth` |

CLIP is loaded through `clip.load` and may download weights on first use. Its corresponding `-teacher_pretrain` value is `none`; the local CLIP files in `pretrain/` are not used by this loader. A DINOv3 model directory must contain the model configuration and weights required by the Transformers loader. Obtain each teacher from its original distribution and comply with its access requirements and license.

Example local layout (weights remain ignored by Git):

```text
pretrain/
├── mae_pretrain_vit_base.pth
├── mae_pretrain_vit_large.pth
├── dinov3-vitb16-pretrain-lvd1689m/
│   ├── config.json
│   └── model.safetensors
└── dinov3-vitl16-pretrain-lvd1689m/
    ├── config.json
    └── model.safetensors
```

Verify the MAE path before training: the current MAE wrapper skips checkpoint loading if the file is missing.

## Training

Run commands from the repository root. The example below configures a ViT-B student and three teachers on one GPU. Adjust the batch size to fit GPU memory.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 main_ViT.py \
  -distributed -backbone vit_b \
  -teacher_type clip-clip_b dinov3-dinov3_b mae-mae_b \
  -teacher_pretrain none pretrain/dinov3-vitb16-pretrain-lvd1689m pretrain/mae_pretrain_vit_base.pth \
  -data_path /path/to/imagenet \
  -batch_size 32 -epoch 300 -num_workers 8 \
  -in_dim 768 -dim 768 -out_dims 512 768 768 -token_dim 512 \
  -lr 0.0001 -weight_decay 0.01 -clip_grad 0.05 \
  -output_dir logs/actok_vit_b
```

Teacher order matters: use **CLIP, DINOv3, MAE** for this aggregation implementation, and keep `-teacher_pretrain` and `-out_dims` in the same order. The loss branches consume teacher indices in this order.

For multiple GPUs, set `CUDA_VISIBLE_DEVICES=0,1,2,3` and `--nproc_per_node=4`. `-batch_size` is per process; `-grad_accu` controls gradient accumulation. For ViT-L, use `vit_l`, `clip-clip_l dinov3-dinov3_l mae-mae_l`, the large teacher checkpoints, `-in_dim 1024 -dim 1024 -out_dims 768 1024 1024 -token_dim 768`.

These are launch examples derived from the code, not verified paper reproduction settings. The existing `train_*.sh` files retain local experiment paths and resume arguments; adapt them before use.

### Outputs and resuming

The entry point appends the joined teacher names to `-output_dir`. For the example above, outputs are stored under:

```text
logs/actok_vit_b/clip-clip_b_dinov3-dinov3_b_mae-mae_b/
├── checkpoint.pth       # full aggregation model, optimizer, scheduler, AMP state
├── backbone.pth         # student backbone state dictionary only
├── log.txt              # per-epoch training statistics
└── conflict_history.csv # gradient-conflict measurements
```

Resume by adding the following to the same training command:

```bash
-resume logs/actok_vit_b/clip-clip_b_dinov3-dinov3_b_mae-mae_b/checkpoint.pth
```

Keep the model configuration and teacher order consistent with the checkpoint. `backbone.pth` contains only the student; the aggregation prediction heads require the full checkpoint and its matching aggregation variant.

## Evaluation and visualization

- `downstream/fine_grained/` provides linear evaluation on 12 classification datasets. See its [README](downstream/fine_grained/README.md), configure the model/checkpoint in `run_all_datasets.sh`, and run it from that directory. The evaluator additionally imports RAPIDS cuML; install a CUDA-compatible cuML environment separately. Its supplied model choices and checkpoint loading must be adapted for the ACTok representation being evaluated.
- `feat_visualization.py`, `test_pca.py`, and `video_pca.py` provide feature visualization utilities. Update their image, checkpoint, and output paths before running. The `test_*.py` files are experiment scripts, not an automated test suite.
- `gradient_conflict_tracker.py` and `measure_gradient_conflict.py` inspect disagreement between teacher objectives. Plotting scripts support experimental analysis.

The ADE20K, depth, and ImageNet evaluation folders currently have no implementation in this checkout.

## Code structure

```text
backbone/                       # ViT, MAE, CLIP, DINOv3, ConvNeXt, and ResNet wrappers
datasets/                       # image loading and training transforms
layers/                         # cross-model blocks, heads, and auxiliary losses
main_ViT.py                     # current ViT training entry point
ModelAggregator_ViT_combine.py  # aggregation implementation selected by main_ViT.py
ModelAggregator_ViT*.py         # alternative ViT aggregation experiments
DirectAlign_ViT.py              # direct-alignment experiment
main_conv.py, main_ConvViT.py    # convolutional and mixed-backbone training
downstream/fine_grained/        # classification evaluation
comparison/                    # comparison-model utilities
```

## Release status

This is a source-code release. End-to-end training and evaluation have not been validated in a pinned environment for this release. Checkpoints, datasets, exact paper settings, result tables, and citation metadata are not bundled.

## Acknowledgments

This repository includes or adapts components from CLIP, MAE, DINOv3, timm, and other research implementations. Fine-grained evaluation builds on SynCLR. Original attribution and license notices are retained in the relevant source files; third-party components and weights remain subject to their respective licenses.
