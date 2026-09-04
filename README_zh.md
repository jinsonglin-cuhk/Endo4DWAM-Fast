# Endo4DWAM-Fast

面向**内镜**的视频 + 动作世界模型 baseline，基于
[FastWAM](https://github.com/yuantianyuan01/FastWAM)（Wan2.2-TI2V-5B 视频 DiT + ActionDiT，
联合 flow matching 训练）改造，适配 EndoWAM 内镜数据集。

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

更详细的分步训练说明见 [`scripts/TRAINING.md`](./scripts/TRAINING.md)。

## 目录

- [总览](#总览)
- [目录结构](#目录结构)
- [环境安装](#环境安装)
- [模型准备](#模型准备)
- [数据集](#数据集)
- [一次性预处理](#一次性预处理)
- [训练](#训练)
- [训练输出](#训练输出)
- [评估](#评估)
- [关键设计说明](#关键设计说明)
- [致谢](#致谢)
- [BibTeX](#bibtex)

## 总览

模型用两个共享 attention 的 expert（Mixture-of-Transformers，简称 MoT）同时对
**视频 latent 流**和**动作流**做去噪：

- **视频 expert** —— Wan2.2-TI2V-5B `WanVideoDiT`（30 层，hidden 3072）
- **动作 expert** —— `ActionDiT`（30 层，hidden 1024），由 Wan2.2 DiT 逐层线性插值得到的
  backbone 初始化

共三个变体，通过 Hydra 的 `model` group 选择：

| 变体 | 类名 | 模型配置 | 行为 |
|---|---|---|---|
| base（uncond） | `Endo4DWAM` | `configs/model/endo4dwam.yaml` | action token 只 attend **第一帧** video latent，推理时可缓存视频 K/V 加速 |
| joint | `Endo4DWAMJoint` | `configs/model/endo4dwam_joint.yaml` | action token attend **完整** video 序列，更强但更慢的 baseline |
| IDM | `Endo4DWAMIDM` | `configs/model/endo4dwam_idm.yaml` | 两阶段推理：先去噪视频，再以其为条件预测动作 |

**推荐起点**：base 变体，对应 `scripts/train_endowam_lora_uncond.sh`。

与上游 FastWAM 的差异：

- 任务从 LIBERO / RoboTwin 机械臂操作换成**单目内镜视频** + 3 维离散伪动作。
- 在视频 expert 上新增了 **LoRA** 路径（上游只做 DiT 全参数微调），参数与 EndoWAM 的
  Cosmos LoRA 对齐（rank 16 / alpha 32 / dropout 0.05）。
- 新增 `save_total_limit`，只保留最新 checkpoint。

## 目录结构

```text
Endo4DWAM-Fast/
├── configs/
│   ├── train.yaml                        # 全局训练默认值
│   ├── data/
│   │   └── endowam_endoscope.yaml        # EndoWAM 内镜数据集配置
│   ├── model/
│   │   ├── endo4dwam.yaml                # base 模型配置（含 LoRA 默认值）
│   │   ├── endo4dwam_joint.yaml
│   │   └── endo4dwam_idm.yaml
│   └── task/
│       ├── endowam_uncond_1cam_1e-4.yaml # base 任务配置
│       └── endowam_joint_1cam_1e-4.yaml  # joint 任务配置
├── scripts/
│   ├── TRAINING.md                       # 详细训练文档
│   ├── train_endowam_lora_uncond.sh      # base + LoRA 训练脚本
│   ├── train_endowam_lora_joint.sh       # joint + LoRA 训练脚本
│   ├── train_zero1.sh / train_zero2.sh   # 通用 DeepSpeed ZeRO-1 / ZeRO-2 启动脚本
│   ├── train.py                          # Hydra 训练入口
│   ├── build_endowam_episodes_stats.py   # 一次性：生成 meta/episodes_stats.jsonl
│   ├── precompute_text_embeds.py         # 一次性：预计算 T5 文本 embedding 缓存
│   ├── preprocess_action_dit_backbone.py # 一次性：生成 ActionDiT backbone
│   └── val_chunk_endowam.py              # 离线 per-axis 动作精度 + 视频指标
├── src/endo4dwam/                        # 核心代码
├── runs/                                 # 训练输出（checkpoint、日志、eval 视频）
├── checkpoints/                          # 预训练 / 外部权重
└── data/                                 # 文本 embedding 缓存等本地数据
```

`experiments/{libero,robotwin}/` 和 `third_party/RoboTwin/` 继承自上游 FastWAM，保留下来是为了
上游 benchmark 仍可运行，但它们**不属于**内镜流程，本项目也未做验证。

## 环境安装

```bash
conda create -n endo4dwam python=3.10 -y
conda activate endo4dwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## 模型准备

训练前必做。第 1 步：把 Wan 模型缓存目录指到 `./checkpoints`（可选，这是默认值）：

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

视频 DiT 基础模型（`Wan-AI/Wan2.2-TI2V-5B`）和 tokenizer（`Wan-AI/Wan2.1-T2V-1.3B`）会在
首次使用时下载到该目录。

第 2 步：预生成 ActionDiT backbone（由 Wan2.2 DiT 逐层插值得到）：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/endo4dwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

训练时通过 `model.action_dit_pretrained_path` 加载该文件。

## 数据集

训练使用 EndoWAM 内镜**伪动作**数据集（`endowam_pseudo_z60`），按 LeRobot v2.1 组织：

```text
endowam_pseudo_z60/
├── ercp/          201 episodes / 407,617 帧    # 3 个术式 = 3 个 LeRobot-v2.1
├── esophagus/     163 episodes / 165,335 帧    #   root，通过
└── ureter/        151 episodes / 406,612 帧    #   MultiLeRobotDataset 合并
```

| 字段 | 取值 |
|---|---|
| 相机 | 单目，`observation.images.endoscope` |
| 视频 | 原始 360×480（H×W，比例 0.750），30 fps；等比缩放后中心裁剪到 256×320（裁掉约 21px 宽，无形变） |
| 动作 | 3 维离散伪动作（m2/m3/m4 目标转速），取值 `{-1, 0, 1}` |
| 状态 | 上一步的动作，同样 3 维 |

动作是**离散方向指令**而非末端位姿，由此有两个必须注意的设置（`configs/data/endowam_endoscope.yaml`
中已配好）：

- `delta_action_dim_mask.default: [false, false, false]` —— 动作**不能**做差分。
- `norm_default_mode: min/max` —— 把本来就在 `[-1, 1]` 的值映射为近似恒等，避免 z-score
  把离散取值放大。

如果数据集路径不同，改该配置里的 `dataset_dirs` 即可。

## 一次性预处理

### 1）生成 `meta/episodes_stats.jsonl`

LeRobot 加载器要求 v2.1 的每个 root 下有 per-episode 统计：

```bash
python scripts/build_endowam_episodes_stats.py \
  --data_root /path/to/endowam_pseudo_z60
```

### 2）预计算 T5 文本 embedding 缓存

训练时文本编码器被冻结且不加载到 GPU，所以 prompt 的编码结果需要提前缓存到磁盘：

```bash
python scripts/precompute_text_embeds.py task=endowam_uncond_1cam_1e-4
```

缓存写入 `./data/text_embeds_cache/endowam/`，**uncond 与 joint 共用同一份**，只需运行一次。
多 GPU 加速：

```bash
torchrun --standalone --nproc_per_node=2 scripts/precompute_text_embeds.py task=endowam_uncond_1cam_1e-4
```

## 训练

专用启动脚本（GPU 编号、LoRA 配置和超参都以变量形式暴露在脚本顶部）：

```bash
bash scripts/train_endowam_lora_uncond.sh    # Endo4DWAM base + LoRA
bash scripts/train_endowam_lora_joint.sh     # Endo4DWAMJoint + LoRA
```

续训 —— 脚本会自动扫描 `<output_dir>/checkpoints/state/step_*/` 找 step 最大的目录，
optimizer、scheduler、dataloader 进度全部恢复：

```bash
bash scripts/train_endowam_lora_uncond.sh --resume
```

通用启动脚本，通过 Hydra override 驱动：

```bash
bash scripts/train_zero1.sh <nproc_per_node> task=<task_name> [overrides...]

bash scripts/train_zero1.sh 2 task=endowam_uncond_1cam_1e-4
bash scripts/train_zero1.sh 2 task=endowam_uncond_1cam_1e-4 learning_rate=5e-5
bash scripts/train_zero2.sh 2 task=endowam_uncond_1cam_1e-4   # ZeRO-2，更省显存
```

> 通用脚本不会自动开启 LoRA；LoRA 由 `model.lora.enable` 控制，两个 `endowam_*_1cam_1e-4`
> task config 中已默认设为 `true`。

`configs/data/endowam_endoscope.yaml` 有意没有设置 `pretrained_norm_stats`，所以**首次**训练
会从数据现算 action/state 归一化统计，并写到 `runs/<...>/dataset_stats.json`。想让后续训练复用
同一份统计，在数据配置的 `train:`（以及 `val:`）下加上 `pretrained_norm_stats: <该文件路径>`
即可，写法可参考 `configs/data/robotwin.yaml`。

单卡显存参考（base 变体，`batch_size=4`，ZeRO-1，bf16）：不开梯度检查点约 40–48 GB，
开启 `model.mot_checkpoint_mixed_attn=true` 后约 28–35 GB。

## 训练输出

```text
runs/<run_root>/<run_id>/
├── config.yaml                   # 训练时完整配置快照
├── dataset_stats.json            # 自动生成的 action/state 归一化统计
├── train_endowam_lora_uncond.sh  # 启动脚本副本（复现用）
├── checkpoints/
│   ├── weights/step_001000.pt    # MoT state_dict（含 LoRA 参数）
│   └── state/step_001000/        # optimizer + scheduler + RNG 状态
└── eval/step_000500_rank_000.mp4 # 左：模型预测 | 中：VAE 重建 | 右：GT
```

`save_total_limit=1` 时只保留最新的 checkpoint，旧的 `weights/` 文件和 `state/` 目录会自动删除。

## 评估

`scripts/val_chunk_endowam.py` 对单个 episode 做离线验证：

```bash
python scripts/val_chunk_endowam.py \
  --ckpt runs/endowam_uncond_lora/<run_id>/checkpoints/weights/step_080000.pt \
  --task endowam_uncond_1cam_1e-4 \
  --dataset_root /path/to/endowam_pseudo_z60/esophagus/rot045 \
  --episode 144 --execution_horizon 8 --max_windows 4000 \
  --num_video_saves 0 --gpu 0
```

输出包括：

- **Per-axis 动作精度**。模型是连续 flow-matching 模型而非分类器，所以是把反归一化后的
  预测四舍五入到 `{-1, 0, +1}` 再比较 —— 该数据集的动作本来就是离散的，这样做是合理的。
- **与训练同款的 diffusion loss**（`loss_video` / `loss_action`）。
- **联合视频推理**（`--num_video_saves > 0` 时对若干窗口执行）：预测视频 / VAE 重建 / GT
  三路拼接成 mp4，并给出 PSNR / SSIM。

## 关键设计说明

**哪些参数在训、哪些冻结。** LoRA 只注入视频 expert，动作 expert 做全参数微调：

| 组件 | 训练方式 |
|---|---|
| 视频 expert 基础权重（Wan2.2 预训练） | 冻结（LoRA adapter 旁路） |
| 视频 expert LoRA 参数（`lora_A` / `lora_B`） | 训练 |
| 动作 expert 全部参数 | 训练（无 LoRA） |
| VAE | 冻结 |
| 文本编码器（UMT5-XXL） | 冻结，离线预计算 |

**分辨率约束。** Wan2.2 用的是 `WanVideoVAE38`，空间压缩为 **16×**（先 2× patchify，再 8× 编码器），
DiT 还会再 patchify 2×。因此视频的高和宽都必须被 **32** 整除。默认的 256×320 满足该约束，
接近原始 360×480 的宽高比（0.750 vs 0.800）；加载器等比缩放后中心裁剪，不引入形变。

## 致谢

本代码库基于 [FastWAM](https://github.com/yuantianyuan01/FastWAM)
（"Fast-WAM: Do World Action Models Need Test-time Future Imagination?"）改造，感谢原作者开源。
同时也基于 [Wan2.2](https://github.com/Wan-Video/Wan2.2)、
[LeRobot](https://github.com/huggingface/lerobot)，以及 `third_party/` 下 vendored 的
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 评估代码。

## BibTeX

如果本代码库对你有帮助，请引用上游的 FastWAM 论文：

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026},
  url={https://arxiv.org/abs/2603.16666}
}
```
