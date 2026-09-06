# Endo4DWAM-Fast 训练与评估

更新：2026-09-06。本文描述当前实现；几何标签约定见 [geometry_motion_distillation.md](../docs/geometry_motion_distillation.md)。

## 环境与准备

```bash
cd /mnt/data2/ljs/Endo4DWAM/Endo4DWAM-Fast
conda activate fastwam
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install -e .
```

Shell 启动器会自动切换到本仓库并设置 `PYTHONPATH`；直接运行 Python 脚本时使用上述环境，或先 `pip install -e .`。机器上旧 fastwam 的 editable 安装不能代替本仓库。

```bash
python scripts/build_endowam_episodes_stats.py \
  --data_root /mnt/data2/ljs/Endo4DWAM/Endo4DWAM/dataset/endowam_pseudo_z60
CUDA_VISIBLE_DEVICES=6 python scripts/precompute_text_embeds.py \
  task=endowam_fastwam_baseline_1cam_1e-4 +overwrite=false
ls checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

统计脚本默认跳过已有文件。uncond/joint 共用文本缓存。不要把文档中的某次“文件已生成/缺失”记录当成当前机器状态。

## 默认数据与空间变换

- 三个 root：ercp、esophagus、ureter；原视频 360×480。
- RGB：先等比 resize 到 **256×341**，再中心裁剪到 **256×320**。processor 执行此变换；外层同尺寸变换不再改变图像。
- `num_frames=33`，原始帧步长 1，`action_video_freq_ratio=2`：32 个动作、17 帧视频、5 帧 latent。
- baseline 默认 K=1（33帧窗口）；公平的 K=2 消融使用41帧窗口。K=2虽裁掉已观察的前8步动作，但仍预测32步，并与K=1同为4个future latent，避免history和horizon同时变化。
- 按每个 root 的 episode 做固定 seed=42 的 90%/10% 划分，训练统计只用训练集，验证复用这些统计。若多个 episode 来自同一个原始视频，还需在数据生成侧按源视频分组划分。

旧 pipeline 使用直接拉伸且没有留出验证集。旧 checkpoint 的结果不能作为新划分下未见数据的泛化结果；要做严格对照，需重新训练。

## Baseline 训练

严格原始 FastWAM control 固定为：全量 MoT、K=1、无geometry、无persistent memory、无curriculum；训练联合生成future video/action，held-out评估与部署只调用`infer_action()`，不会扩散、解码或保存未来视频：

```bash
bash scripts/train_zero1.sh 2 task=endowam_fastwam_baseline_1cam_1e-4
```

下面两个是保留的 LoRA / Joint 比较任务，不等同于严格原始 baseline：

```bash
bash scripts/train_endowam_lora_uncond.sh
bash scripts/train_endowam_lora_joint.sh
```

| 配置 | uncond | joint |
|---|---:|---:|
| max_steps | 80000 | 40000 |
| batch / GPU | 6 | 4 |
| gradient accumulation | 4 | 8 |
| 默认 GPU 数 | 2 | 2 |
| 有效 batch | 48 | 64 |
| learning rate | 1e-4 | 1e-4 |
| save / eval interval | 1000 / 500 | 1000 / 500 |
| checkpoint 保留数 | 1 | 1 |

视频 expert 训练 LoRA（rank=16、alpha=32、dropout=0.05），动作 expert 全参数训练。默认 task 关闭 mixed-attention checkpointing，显存不足可开启。

新 run ID 为 `endo4dwam_{uncond,joint}_lora_z60_crop_split`，避免覆盖旧的拉伸/全量训练实验。可以用环境变量和末尾 Hydra overrides 调整：

```bash
CUDA_VISIBLE_DEVICES=6 NPROC_PER_NODE=1 RUN_ID=debug_crop \
  bash scripts/train_endowam_lora_uncond.sh \
  batch_size=1 max_steps=10 model.mot_checkpoint_mixed_attn=true
```

## 续训与 warm start

```bash
bash scripts/train_endowam_lora_uncond.sh --resume
```

`--resume` 选择带 `trainer_state.json` 的最大 step 目录。没有状态时明确失败，不会重新训练。完整状态的 model/data 配置必须与 run 的 `config.yaml` 一致；启动时还应使用原来的 batch、GPU 数和梯度累积配置。

已有 run 不允许不带 `--resume` 直接重用。改变数据变换、划分或模型结构时使用新 RUN_ID；如需 warm start，用 `.pt` 权重路径（仅恢复权重，不恢复 optimizer/step）：

```bash
RUN_ID=new_experiment bash scripts/train_endowam_lora_uncond.sh \
  resume=/absolute/path/to/step_040000.pt
```

`save_total_limit=1` 会轮换删除旧权重和训练状态；需要保留的实验权重应存放在新 run 之外。

## 通用及多机启动

```bash
bash scripts/train_zero1.sh 2 task=endowam_fastwam_baseline_1cam_1e-4
bash scripts/train_zero2.sh 2 task=endowam_fastwam_baseline_1cam_1e-4
```

通用启动器使用task YAML超参；严格baseline为batch=4、max_steps=40000、gradient_accumulation_steps=1。这与LoRA专用脚本的overrides不同。

多机时每台执行同一命令，设置各自 `NODE_RANK`。参数 2 表示每台 GPU 数；启动器把总进程数设为 `2 * NNODES`：

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 RUN_ID=cluster_run \
  bash scripts/train_zero1.sh 2 task=endowam_fastwam_baseline_1cam_1e-4
# 另一台使用 NODE_RANK=1，其余一致。
```

各节点必须能访问相同的代码、数据和输出路径。未设置 RUN_ID 时会通过 TCPStore 同步。

## 几何监督

Baseline 默认 `geometry.enable=false`。启用后在分配 5B 模型之前检查完整性。2026-09-06 对 ercp 201、esophagus 163、ureter 151，共515个 episode 的 EdGE depth 与 stride-2 RAFT flow 严格预检通过。当前 EdGE 使用 sidecar 的 teacher/quality 状态验收；`depth_qc_path` 只用于兼容旧标签。

默认几何readout直接使用EdGE源码中的原生DPT depth head。EdGE源码和权重默认读取本机生产教师路径，也可通过`EDGE_SRC`、`EDGE_WEIGHTS`覆盖。DA3源码仅在运行`head.type=da3`消融或对应测试时需要；`addict==2.4.0` 已加入项目依赖：

```bash
python -m pip install -e .
export PYTHONPATH="/mnt/data2/ljs/Depth-Anything-3/src:$PYTHONPATH"
export EDGE_SRC="/mnt/data2/ljs/Endo4DWAM/EdGE-review"
export EDGE_WEIGHTS="/mnt/data2/ljs/Endo4DWAM/EdGE-review/weights/model.safetensors"
python -c 'from depth_anything_3.model.dpt import DPT; print("DPT import OK")'
```

完整 K=2 persistent-memory 课程训练（S0 memory/registers/head → S1 video LoRA → S2 action/proprio，两条γ_p=0.05）：

```bash
bash scripts/train_zero1.sh 2 task=endowam_geometry_k2_1cam_1e-4
```

S0/S1/S2 阈值分别是0/2000/6000，可用 Hydra override 调整。Reference task 启用64×1024 visual-only memory、depth gradient loss权重0.1和masked Charbonnier flow；公平K=2协议使用4个future depth/flow目标，不额外混入observed-flow。S3需要另行接入真实机器人数据，不由该task假定完成。

增强配置默认`model.geometry.head.type=edge`。DA3 head对照使用：

```bash
bash scripts/train_zero1.sh 2 task=endowam_geometry_k2_1cam_1e-4 \
  model.geometry.head.type=da3
```

随机训练batch不会跨窗口保存状态：memory在每个窗口从零状态开始，按2个clean history latent依次更新。`infer_action`则返回`memory_state`，在线/离线调用方必须在同一episode的下个chunk传回，并在episode reset时调用`init_memory()`。不要把video KV cache当成persistent memory；KV cache只在一次action diffusion内复用。

## 正交消融

三个配置组分别只控制history、训练辅助监督、persistent memory。以严格baseline为底：

| 实验 | Hydra overrides | 说明 |
|---|---|---|
| B0 | 无 | K1、mixed、无geometry/memory |
| H | `history=k2` | 只增加短期历史；action=32、future latent=4保持不变 |
| C | `history=k2 persistent_memory=cached_control` | 无memory但走相同cached训练路径 |
| G | `history=k2 auxiliary=geometry persistent_memory=cached_control` | geometry-only |
| M | `history=k2 persistent_memory=on` | memory-only |
| GM | `history=k2 auxiliary=geometry persistent_memory=on` | 完整模型 |

例如：

```bash
bash scripts/train_zero1.sh 2 \
  task=endowam_fastwam_baseline_1cam_1e-4 \
  history=k2 auxiliary=geometry persistent_memory=on
```

比较memory时优先使用C↔M或G↔GM，避免把原始mixed attention与cached execution差异错误归因给memory。`model.memory.action_read`和`geometry_read`还能分别关闭memory到动作/几何readout的边。

## 离线评估

```bash
python scripts/val_chunk_endowam.py \
  --ckpt runs/endowam_uncond_lora/endo4dwam_uncond_lora_z60_crop_split/checkpoints/weights/step_080000.pt \
  --dataset_root /mnt/data2/ljs/Endo4DWAM/Endo4DWAM/dataset/endowam_pseudo_z60/esophagus \
  --episode 144 --execution_horizon 8 --num_video_saves 2
```

评估自动读取 run 下的 `config.yaml` 和 `dataset_stats.json`，也可显式传 `--config` / `--dataset_stats`。`--task` 仅作标签，不覆盖训练快照。Base persistent memory在episode开头重置并跨成功窗口传递；Joint自动传入采样后的视频帧数。辅助头在纯policy离线评估时不构建；memory updater保留。严格baseline的训练期held-out evaluation只跑action；增强task若开启视频诊断，可另行设置`eval_generate_video=true`并报告video/geometry指标。

`--episode` 使用该 root 内的 episode 序号，不会自动过滤为留出集；正式报告必须选择训练未见的 episode。`execution_horizon` 必须在 1 到动作 horizon 之间。

失败窗口会写入 summary，记录实际完成数、计划数和 `status=failed`，进程非零退出。验证/离线评估读取失败不再随机替换为别的样本。视频/action 指标与 auxiliary-depth/flow 指标分开。

## 验证

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -v
```

CPU 回归覆盖RGB裁剪、EdGE/RAFT协议、公平K2窗口、mixed/cached等价控制、两条γ_p、显式memory状态/detach、geometry读取memory、边界mask、带符号flow、术式权重、Joint参数、配置快照、续训及多机启动参数。完整5B GPU backward和跨机器分布式运行仍需在目标训练环境验证。

可选的真实 DPT 测试在具有 DA3 依赖的环境单独运行：

```bash
PYTHONPATH=src /home/user/miniconda3/envs/DAv3/bin/python \
  -m unittest discover -s tests -p test_da3_head.py -v
```

本次验证：fastwam环境26项pipeline回归全部通过，DAv3环境真实DA3/EdGE DPT的2项测试通过；EdGE 62个depth-head tensor严格加载，depth/motion形状与反向梯度通过。baseline实样本为video17/action32，公平K2实样本为video21/action32/depth4/flow4。唯一prompt的真实T5缓存已验证为`[128,4096]` bf16，`+overwrite=false`会在加载UMT5前直接返回。尚未运行5B完整训练步。
