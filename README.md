# Endo4DWAM-Fast


> Pipeline updated 2026-09-05: aspect-preserving crop, held-out episodes, strict geometry labels and checkpoint-config evaluation. See [training guide](scripts/TRAINING.md) for current commands and limitations.

A **video + action world model baseline for endoscopy**, built on top of
[FastWAM](https://github.com/yuantianyuan01/FastWAM) (Wan2.2-TI2V-5B video DiT + ActionDiT,
trained with joint flow matching) and adapted to the EndoWAM endoscope dataset.

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

For a step-by-step Chinese walkthrough of training, see [`scripts/TRAINING.md`](./scripts/TRAINING.md).

## Index

- [Overview](#overview)
- [File Structure](#file-structure)
- [Environment Setup](#environment-setup)
- [Model Preparation](#model-preparation)
- [Dataset](#dataset)
- [One-time Preprocessing](#one-time-preprocessing)
- [Training](#training)
- [Training Outputs](#training-outputs)
- [Evaluation](#evaluation)
- [Key Design Notes](#key-design-notes)
- [Acknowledgements](#acknowledgements)
- [BibTeX](#bibtex)

## Overview

The model jointly denoises a **video latent stream** and an **action stream** with two
experts (a Mixture-of-Transformers, "MoT") that share attention:

- **video expert** — Wan2.2-TI2V-5B `WanVideoDiT` (30 layers, hidden 3072)
- **action expert** — `ActionDiT` (30 layers, hidden 1024), initialised from a
  layer-interpolated Wan2.2 DiT backbone

Three variants are available, selected by the Hydra `model` group:

| Variant | Class | Model config | Behaviour |
|---|---|---|---|
| base (uncond) | `Endo4DWAM` | `configs/model/endo4dwam.yaml` | Action tokens attend **only the first-frame** video latent, so video K/V can be cached at inference |
| joint | `Endo4DWAMJoint` | `configs/model/endo4dwam_joint.yaml` | Action tokens attend the **full** video sequence — a stronger but slower baseline |
| IDM | `Endo4DWAMIDM` | `configs/model/endo4dwam_idm.yaml` | Two-stage inference: denoise the video first, then condition the action on it |

**Recommended starting point:** the base variant, via `scripts/train_endowam_lora_uncond.sh`.

What differs from upstream FastWAM:

- Retargeted from LIBERO / RoboTwin manipulation to **monocular endoscope video** with a
  3-DoF discrete pseudo-action.
- Added a **LoRA** path on the video expert (upstream only does full fine-tuning of the DiT),
  mirroring the EndoWAM Cosmos LoRA setup (rank 16 / alpha 32 / dropout 0.05).
- Added `save_total_limit` so only the latest checkpoint is kept.

## File Structure

```text
Endo4DWAM-Fast/
├── configs/
│   ├── train.yaml                        # Global training defaults
│   ├── data/
│   │   └── endowam_endoscope.yaml        # EndoWAM endoscope dataset config
│   ├── model/
│   │   ├── endo4dwam.yaml                # Base model config (incl. LoRA defaults)
│   │   ├── endo4dwam_joint.yaml
│   │   └── endo4dwam_idm.yaml
│   └── task/
│       ├── endowam_fastwam_baseline_1cam_1e-4.yaml # Pinned original FastWAM control
│       ├── endowam_uncond_1cam_1e-4.yaml # Base + LoRA task config
│       └── endowam_joint_1cam_1e-4.yaml  # Joint task config
├── scripts/
│   ├── TRAINING.md                       # Detailed training guide (Chinese)
│   ├── train_endowam_lora_uncond.sh      # Base + LoRA launcher
│   ├── train_endowam_lora_joint.sh       # Joint + LoRA launcher
│   ├── train_zero1.sh / train_zero2.sh   # Generic DeepSpeed ZeRO-1 / ZeRO-2 launchers
│   ├── train.py                          # Hydra training entrypoint
│   ├── build_endowam_episodes_stats.py   # One-time: generate meta/episodes_stats.jsonl
│   ├── precompute_text_embeds.py         # One-time: cache T5 text embeddings
│   ├── preprocess_action_dit_backbone.py # One-time: build the ActionDiT backbone
│   └── val_chunk_endowam.py              # Offline per-axis action accuracy + video metrics
├── src/endo4dwam/                        # Core package
├── runs/                                 # Training outputs (checkpoints, logs, eval videos)
├── checkpoints/                          # Pretrained / external checkpoints
└── data/                                 # Text-embedding cache and other local data
```

`experiments/{libero,robotwin}/` and `third_party/RoboTwin/` are inherited from upstream
FastWAM. They are kept so upstream benchmarks stay runnable, but they are **not** part of
the endoscope pipeline and are not validated here.

## Environment Setup

```bash
conda create -n endo4dwam python=3.10 -y
conda activate endo4dwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## Model Preparation

Required before training. Step 1 — point the Wan model cache at `./checkpoints`
(optional, this is the default):

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

The video DiT base model (`Wan-AI/Wan2.2-TI2V-5B`) and the tokenizer
(`Wan-AI/Wan2.1-T2V-1.3B`) are fetched into that directory on first use.

Step 2 — pre-generate the ActionDiT backbone (layer-interpolated from the Wan2.2 DiT):

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/endo4dwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

The training config loads this file via `model.action_dit_pretrained_path`.

## Dataset

Training uses the EndoWAM endoscope **pseudo-action** dataset
(`endowam_pseudo_z60`), laid out as LeRobot v2.1 roots:

```text
endowam_pseudo_z60/
├── ercp/          201 episodes / 407,617 frames   # 3 procedures = 3 LeRobot-v2.1
├── esophagus/     163 episodes / 165,335 frames   #   roots, merged via
└── ureter/        151 episodes / 406,612 frames   #   MultiLeRobotDataset
```

| Field | Value |
|---|---|
| Camera | single, `observation.images.endoscope` |
| Video | 360x480 (HxW, aspect 0.750) raw, 30 fps; aspect-preserving resize then centre crop to 256x320 (~21px of width cropped, no distortion) |
| Action | 3-DoF discrete pseudo-action (m2/m3/m4 target rpm), values in `{-1, 0, 1}` |
| State | previous-step action, also 3-D |

Two consequences of the actions being **discrete direction commands** rather than
end-effector poses — both already set in `configs/data/endowam_endoscope.yaml`:

- `delta_action_dim_mask.default: [false, false, false]` — actions must **not** be differenced.
- `norm_default_mode: min/max` — keeps the already-`[-1, 1]` values near identity instead
  of letting z-scoring blow up the discrete levels.

Point `dataset_dirs` in that config at your own copy of the dataset if the path differs.

## One-time Preprocessing

### 1) Generate `meta/episodes_stats.jsonl`

The LeRobot loader requires per-episode stats on v2.1 roots:

```bash
python scripts/build_endowam_episodes_stats.py \
  --data_root /path/to/endowam_pseudo_z60
```

### 2) Precompute the T5 text-embedding cache

The text encoder is frozen and never loaded onto the GPU during training, so prompt
embeddings must be cached to disk first:

```bash
python scripts/precompute_text_embeds.py task=endowam_fastwam_baseline_1cam_1e-4 +overwrite=false
```

The cache lands in `./data/text_embeds_cache/endowam/` and is **shared by the uncond and
joint variants** — running it once is enough. For multi-GPU:

```bash
torchrun --standalone --nproc_per_node=2 scripts/precompute_text_embeds.py task=endowam_fastwam_baseline_1cam_1e-4 +overwrite=false
```

## Training

Pinned original FastWAM control (joint video/action training, action-only validation and deployment):

```bash
bash scripts/train_zero1.sh 2 task=endowam_fastwam_baseline_1cam_1e-4
```

See [`scripts/TRAINING.md`](scripts/TRAINING.md) for orthogonal history, geometry and persistent-memory ablations.

Dedicated launchers (GPU ids, LoRA settings and hyperparameters are exposed as variables
at the top of each script):

```bash
bash scripts/train_endowam_lora_uncond.sh    # Endo4DWAM base + LoRA
bash scripts/train_endowam_lora_joint.sh     # Endo4DWAMJoint + LoRA
```

Resume — the script scans `<output_dir>/checkpoints/state/step_*/` for the highest step and
restores optimizer, scheduler and dataloader progress:

```bash
bash scripts/train_endowam_lora_uncond.sh --resume
```

Generic launchers, driven by Hydra overrides:

```bash
bash scripts/train_zero1.sh <nproc_per_node> task=<task_name> [overrides...]

bash scripts/train_zero1.sh 2 task=endowam_uncond_1cam_1e-4
bash scripts/train_zero1.sh 2 task=endowam_uncond_1cam_1e-4 learning_rate=5e-5
bash scripts/train_zero2.sh 2 task=endowam_uncond_1cam_1e-4   # ZeRO-2, lower memory
```

> The generic launchers do not enable LoRA by themselves — it is controlled by
> `model.lora.enable`, which both `endowam_*_1cam_1e-4` task configs already set to `true`.

`configs/data/endowam_endoscope.yaml` deliberately does not set `pretrained_norm_stats`, so
the **first** run computes action/state normalisation stats from the data and writes them to
`runs/<...>/dataset_stats.json`. To keep normalisation fixed across later runs, add
`pretrained_norm_stats: <path to that file>` under `train:` (and `val:`) in the data config —
see `configs/data/robotwin.yaml` for the pattern.

Approximate single-GPU memory, base variant at `batch_size=4`, ZeRO-1, bf16:
~40–48 GB without gradient checkpointing, ~28–35 GB with
`model.mot_checkpoint_mixed_attn=true`.

## Training Outputs

```text
runs/<run_root>/<run_id>/
├── config.yaml                   # Full resolved config snapshot
├── dataset_stats.json            # Action/state normalisation stats (auto-generated)
├── train_endowam_lora_uncond.sh  # Copy of the launcher, for reproducibility
├── checkpoints/
│   ├── weights/step_001000.pt    # MoT state_dict (includes LoRA params)
│   └── state/step_001000/        # Optimizer + scheduler + RNG state
└── eval/step_000500_rank_000.mp4 # Prediction | VAE reconstruction | ground truth
```

With `save_total_limit=1` only the most recent checkpoint is kept; older `weights/` files
and `state/` directories are deleted automatically.

## Evaluation

`scripts/val_chunk_endowam.py` runs offline validation on a single episode:

```bash
python scripts/val_chunk_endowam.py \
  --ckpt runs/endowam_uncond_lora/<run_id>/checkpoints/weights/step_080000.pt \
  --task endowam_uncond_1cam_1e-4 \
  --dataset_root /path/to/endowam_pseudo_z60/esophagus \
  --episode 144 --execution_horizon 8 --max_windows 4000 \
  --num_video_saves 0 --gpu 0
```

It reports:

- **Per-axis action accuracy.** The model is a continuous flow-matching model, not a
  classifier, so predictions are denormalised and rounded to the nearest of `{-1, 0, +1}`
  before comparison — valid here because the dataset actions are discrete by construction.
- **Training-equivalent diffusion losses** (`loss_video` / `loss_action`).
- **Joint video rollouts** for a few windows (`--num_video_saves > 0`): prediction vs VAE
  reconstruction vs ground truth, side by side, with PSNR/SSIM.

## Key Design Notes

**What trains, what stays frozen.** LoRA is injected into the video expert only; the action
expert is fully fine-tuned:

| Component | Mode |
|---|---|
| Video expert base weights (Wan2.2 pretrained) | frozen (LoRA adapters bypass them) |
| Video expert LoRA params (`lora_A` / `lora_B`) | trained |
| Action expert (all params) | trained, no LoRA |
| VAE | frozen |
| Text encoder (UMT5-XXL) | frozen, precomputed offline |

**Resolution constraint.** Wan2.2 uses `WanVideoVAE38`, whose spatial compression is **16x**
(2x patchify, then an 8x encoder), and the DiT patchifies by another 2x. Video height and
width must therefore both be divisible by **32**. The default 256x320 satisfies this while
close to the native 360x480 aspect ratio (0.750 vs 0.800); the loader resizes
preserving aspect and centre-crops, so no distortion is introduced.

## Acknowledgements

This codebase is derived from [FastWAM](https://github.com/yuantianyuan01/FastWAM)
("Fast-WAM: Do World Action Models Need Test-time Future Imagination?"). We thank the
authors for releasing it. It also builds on
[Wan2.2](https://github.com/Wan-Video/Wan2.2), [LeRobot](https://github.com/huggingface/lerobot),
and the [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) evaluation code vendored
under `third_party/`.

## BibTeX

If you use this codebase, please cite the upstream FastWAM paper:

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026},
  url={https://arxiv.org/abs/2603.16666}
}
```
