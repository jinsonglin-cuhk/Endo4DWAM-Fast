# FastWAM 训练说明

本文档说明如何在 EndoWAM 内镜数据集上训练 FastWAM 系列模型。

> **工作目录**：以下所有命令均在仓库根目录 `/mnt/data2/ljs/FastWAM` 下执行。

---

## 目录

1. [模型变体说明](#1-模型变体说明)
2. [目录结构](#2-目录结构)
3. [前置准备（一次性）](#3-前置准备一次性)
4. [启动训练](#4-启动训练)
5. [Resume 续训](#5-resume-续训)
6. [通用训练脚本](#6-通用训练脚本)
7. [训练输出结构](#7-训练输出结构)
8. [关键配置说明](#8-关键配置说明)

---

## 1. 模型变体说明

| 变体 | 脚本 | Task Config | 说明 |
|---|---|---|---|
| **FastWAM** (base) | `train_endowam_lora_uncond.sh` | `endowam_uncond_1cam_1e-4` | action tokens 只 attend **第一帧** video latent；推理时可缓存视频 K/V 加速 |
| **FastWAMJoint** | `train_endowam_lora_joint.sh` | `endowam_joint_1cam_1e-4` | action tokens attend **完整** video 序列；更强的 baseline |
| **FastWAMIDM** | `train_zero1.sh` + IDM task | `endowam_idm_*` | 两阶段推理：先去噪视频再 condition 动作 |

**推荐起点**：`FastWAM` (base) — 对应 `train_endowam_lora_uncond.sh`。

---

## 2. 目录结构

```
FastWAM/
├── checkpoints/
│   ├── ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt  # ActionDiT 预训练 backbone
│   └── Wan-AI/Wan2.2-TI2V-5B/                                # 视频 DiT 基础模型
├── configs/
│   ├── train.yaml              # 全局训练默认值
│   ├── data/
│   │   └── endowam_endoscope.yaml   # 数据集配置
│   ├── model/
│   │   ├── fastwam.yaml        # FastWAM base 模型配置（含 LoRA 默认值）
│   │   ├── fastwam_joint.yaml  # FastWAMJoint 模型配置
│   │   └── fastwam_idm.yaml    # FastWAMIDM 模型配置
│   └── task/
│       ├── endowam_uncond_1cam_1e-4.yaml   # EndoWAM base 任务配置
│       └── endowam_joint_1cam_1e-4.yaml    # EndoWAM joint 任务配置
├── scripts/
│   ├── TRAINING.md                       # 本文档
│   ├── train_endowam_lora_uncond.sh      # FastWAM base + LoRA 训练脚本
│   ├── train_endowam_lora_joint.sh       # FastWAMJoint + LoRA 训练脚本
│   ├── train_zero1.sh                    # 通用训练启动脚本（ZeRO-1）
│   ├── train_zero2.sh                    # 通用训练启动脚本（ZeRO-2，更省显存）
│   ├── build_endowam_episodes_stats.py   # 一次性：生成 episodes_stats.jsonl
│   ├── precompute_text_embeds.py         # 一次性：预计算 T5 文本 embedding
│   └── preprocess_action_dit_backbone.py # 一次性：预处理 ActionDiT backbone
└── src/fastwam/                          # 模型代码
```

---

## 3. 前置准备（一次性）

### 3.1 生成 episodes_stats.jsonl

FastWAM 的 LeRobot 数据加载器需要每个子集下存在 `meta/episodes_stats.jsonl`。

```bash
# ✅ EndoWAM z60_rot45 数据集已预先生成，无需重新执行
# 若换了新数据集或文件缺失，才需要运行：
python scripts/build_endowam_episodes_stats.py \
    --data_root /mnt/data2/ljs/EndoWAM/dataset/endowam_pseudo_z60_rot45
```

**当前状态**：`endowam_pseudo_z60_rot45` 下全部 24 个子集（3 procedures × 8 rot）的 `episodes_stats.jsonl` 均已存在。

### 3.2 预计算 T5 文本 Embedding 缓存

训练时文本 encoder 被冻结且不加载到 GPU，所以需要提前把 prompt 编码结果缓存到磁盘。

```bash
# 单 GPU（推荐，endowam 场景 unique prompt 极少，耗时 < 1 分钟）
CUDA_VISIBLE_DEVICES=6 python scripts/precompute_text_embeds.py \
    task=endowam_uncond_1cam_1e-4
```

- 缓存写入 `./data/text_embeds_cache/endowam/`
- `uncond` 和 `joint` 共用同一缓存目录，**只需运行一次**，两个变体均可使用
- 多 GPU 加速（可选）：

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 \
    scripts/precompute_text_embeds.py task=endowam_uncond_1cam_1e-4
```

### 3.3 确认 ActionDiT Backbone 存在

```bash
ls checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

此文件是 ActionDiT 的预训练 backbone（LoRA 注入前的基础权重），训练时通过 `action_dit_pretrained_path` 加载。

---

## 4. 启动训练

### 4.1 FastWAM base（推荐起点）

训练**基础 FastWAM** 模型（action 只 attend 第一帧）：

```bash
bash scripts/train_endowam_lora_uncond.sh
```

### 4.2 FastWAMJoint

训练 **FastWAMJoint**（action attend 完整视频序列）：

```bash
bash scripts/train_endowam_lora_joint.sh
```

### 4.3 训练脚本关键变量说明

以 `train_endowam_lora_uncond.sh` 为例，脚本顶部暴露了所有关键参数：

```bash
# GPU
export CUDA_VISIBLE_DEVICES=6,7
NPROC_PER_NODE=2

# LoRA 配置（与 EndoWAM train_endowam_causal_gaze_latent.sh 参数一致）
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TRAIN_BASE=false        # false = 只训 LoRA 参数，冻结视频 DiT 基础权重
LORA_TARGET_MODULES="self_attn.q,self_attn.k,..."   # 注入层

# 训练超参
MAX_STEPS=40000              # = EndoWAM max_train_steps
BATCH_SIZE=4                 # 单 GPU
LEARNING_RATE=1e-4
SAVE_EVERY=1000              # = EndoWAM save_interval
SAVE_TOTAL_LIMIT=1           # 只保留最新 1 个 checkpoint（= EndoWAM save_total_limit）
EVAL_EVERY=500               # = EndoWAM eval_interval

# 输出目录（固定，方便 resume 时自动定位）
RUN_ROOT=./runs/endowam_uncond_lora
RUN_ID=fastwam_uncond_lora_endowam_z60_rot45
```

修改上述变量即可调整训练配置，无需改动 YAML。

---

## 5. Resume 续训

```bash
# FastWAM base
bash scripts/train_endowam_lora_uncond.sh --resume

# FastWAMJoint
bash scripts/train_endowam_lora_joint.sh --resume
```

脚本会自动扫描 `<OUTPUT_DIR>/checkpoints/state/step_*/` 找最大 step 的目录续训，optimizer、scheduler、dataloader 进度全部恢复。

---

## 6. 通用训练脚本

`train_zero1.sh` 和 `train_zero2.sh` 是通用启动器，通过 Hydra task override 选择任务：

```bash
# 基础用法
bash scripts/train_zero1.sh <nproc_per_node> task=<task_name> [hydra_overrides...]

# 示例：2 GPU 训练 FastWAM base
bash scripts/train_zero1.sh 2 task=endowam_uncond_1cam_1e-4

# 示例：覆盖单个超参
bash scripts/train_zero1.sh 2 task=endowam_uncond_1cam_1e-4 learning_rate=5e-5

# ZeRO-2（显存不够时用，通讯开销略大）
bash scripts/train_zero2.sh 2 task=endowam_uncond_1cam_1e-4
```

> ⚠️ 通用脚本不会自动开启 LoRA；LoRA 由 task config 中 `model.lora.enable=true` 控制。
> `endowam_uncond_1cam_1e-4` 和 `endowam_joint_1cam_1e-4` 已在 YAML 中默认开启 LoRA。

---

## 7. 训练输出结构

```
runs/endowam_uncond_lora/fastwam_uncond_lora_endowam_z60_rot45/
├── config.yaml                   # 训练时完整配置快照
├── dataset_stats.json            # 首次训练自动生成的 action/state 归一化统计
├── train_endowam_lora_uncond.sh  # 启动脚本副本（复现用）
├── checkpoints/
│   ├── weights/
│   │   └── step_001000.pt        # 模型权重（MoT state_dict，含 LoRA 参数）
│   └── state/
│       └── step_001000/          # 完整训练状态（optimizer + scheduler + rng）
│           ├── pytorch_model.bin
│           ├── scheduler.bin
│           ├── rng_state.pth
│           └── trainer_state.json
└── eval/
    └── step_000500_rank_000.mp4  # 每 eval_every 步生成的推理对比视频
                                  # （左: 模型预测, 中: VAE重建, 右: GT）
```

**checkpoint 保留策略**：`SAVE_TOTAL_LIMIT=1` 时只保留最新的 1 个，旧 checkpoint（weights + state 目录）自动删除。

---

## 8. 关键配置说明

### LoRA 注入策略

FastWAM 的 LoRA 只注入到 **视频 expert（WanVideoDiT 5B）**，Action expert 保持全参数微调：

| 组件 | 训练方式 |
|---|---|
| 视频 expert 基础权重（Wan2.2 预训练） | **冻结**（LoRA adapter 旁路） |
| 视频 expert LoRA 参数（lora_A / lora_B） | **训练** |
| Action expert 全部参数 | **训练**（无 LoRA） |
| VAE | 冻结 |
| Text encoder（UMT5-XXL） | 冻结（离线预计算缓存） |

### 与 EndoWAM 的参数对照

| EndoWAM 参数 | FastWAM 等价配置 |
|---|---|
| `cosmos_lora_rank=16` | `LORA_RANK=16` |
| `cosmos_lora_alpha=32` | `LORA_ALPHA=32` |
| `cosmos_lora_dropout=0.05` | `LORA_DROPOUT=0.05` |
| `cosmos_lora_train_base=false` | `LORA_TRAIN_BASE=false` |
| `trainer.max_train_steps=40000` | `MAX_STEPS=40000` |
| `trainer.save_interval=1000` | `SAVE_EVERY=1000` |
| `trainer.save_total_limit=1` | `SAVE_TOTAL_LIMIT=1` |
| `trainer.eval_interval=500` | `EVAL_EVERY=500` |
| `per_device_batch_size=4` | `BATCH_SIZE=4` |

### 显存估算（参考）

| 配置 | 显存需求（单 GPU） |
|---|---|
| FastWAM base，batch=4，ZeRO-1，bf16，无梯度检查点 | ~40–48 GB |
| 开启梯度检查点（`mot_checkpoint_mixed_attn=true`） | ~28–35 GB |

### 视频分辨率约束

Wan2.2 使用 `WanVideoVAE38`，空间压缩 **16×**（内部先 patchify 2×，再编码器 8×）。DiT patch 额外做 2×。因此视频分辨率必须同时被 **32（= 16 × 2）** 整除：

| 分辨率 | H/32 | W/32 | 是否可用 |
|---|---|---|---|
| 240×320 | 7.5 | 10 | ❌ |
| **256×320** | 8 | 10 | ✅ |
| 288×384 | 9 | 12 | ✅ |
| 224×288 | 7 | 9 | ✅ |

EndoWAM 数据集当前使用 `[256, 320]`（从原始 270×360 下采样，近似 4:3）。
| ZeRO-2（参数 shard 到多卡） | 线性降低 |

> task config 里默认 `model.mot_checkpoint_mixed_attn: false`，若 OOM 改为 `true`。
