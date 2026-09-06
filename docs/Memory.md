# Persistent visual memory 决策与实现记录（2026-09-06）

本文把“记忆”分成两类，避免此前把历史帧、KV cache 与跨 chunk 状态混为一谈：

- `K` 帧 history：当前决策窗口的短时原始观测。K=2对应5个VAE输入像素帧、2个clean latent frame。
- `M_t`：固定64×1024 token的长时压缩状态，在action chunk之间显式传递。
- video KV cache：一次 action diffusion 的计算缓存，每次 replan 重建，不是跨chunk记忆。
- geometry registers：训练期查询 token，每个sample重建，部署时删除，也不是persistent state。

## 当前 reference 方案

```text
previous M ───────────────┐
                         v
RGB history -> video expert L18 -> visual-only memory updater -> next M
       │                                      │
       ├──────── history K/V ─────────────────┼──> action expert
       └── layers 12/14/16/18 ──> depth/flow registers
                                              ^
                                      registers also read M
```

实现位置：

- `persistent_memory.py`：零初始化显式状态、learned slot identity、逐history latent更新、共享1024→3072 action adapter。
- `mot.py`：cached action attention加入memory K/V；video query永远没有action K/V；Base action只看到history video而非future video。
- `endo4dwam.py`：训练窗口内memory unroll、geometry读取memory、`init_memory()`与`infer_action(memory_state=...)`协议。
- `trainer.py`：`memory_unfreeze_step`及独立`action_to_memory_grad_scale_*`课程。
- `val_chunk_endowam.py`：episode开始重置，成功窗口间传递state。

Memory writer 的唯一输入是previous memory和clean history hidden state。Action token、伪action标签和noisy future video都不能进入写接口。Action loss仍可沿“读取memory”反向影响writer，因此另外使用straight-through gradient scale；S0/S1为0，S2 reference为0.05。Geometry loss不经过这道门，负责先把memory锚定到深度/运动结构。

训练batch是随机窗口，当前不保存跨batch状态。每个窗口从M0开始并按K个history latent frame unroll；显式外部state进入时先detach。这样训练/部署的更新算子一致，但训练看到的memory时间跨度仍短于真正rollout。严格的跨窗口truncated BPTT是后续实验项，需要episode顺序sampler、每rank state表、断点恢复和episode边界同步，不能只在model里缓存一个tensor来冒充。

## 已实现、保留消融与未实现

| 项目 | 状态 |
|---|---|
| Base action读取history+memory，不读future video | 已实现 |
| Memory只由history视觉写入 | 已实现 |
| Geometry registers读取history+memory | 已实现 |
| `memory.action_read` / `geometry_read`独立开关 | 已实现 |
| mixed baseline / cached-no-memory execution control | 已实现 |
| K1/K2固定32步action、4步future latent的公平窗口 | 已实现（33/41 raw frames） |
| Episode级显式reset/carry | 离线chunk与RobotWin policy已实现 |
| Depth clip-affine + gradient loss | 已实现，可配置 |
| Flow masked Charbonnier | 已实现，保留SmoothL1消融 |
| Joint读取future video | 有意保留的消融；Base `infer_joint`默认仍返回memory-aware action |
| Video expert读取persistent memory | 当前未实现；reference只让action/register读取 |
| 跨训练batch/episode BPTT | 未实现 |
| S3真实动作γ=1对齐 | 缺真实动作训练与独立验证 |

为什么保留历史帧：memory是有损压缩，无法替代最近连续帧中直接可见的器械速度、组织形变和遮挡变化；反过来，K=2只覆盖局部时间，无法替代跨多个action chunk的状态。是否对控制有效仍必须做等预算四组消融：K1/no-memory、K2/no-memory、K1/memory、K2/memory，并以action/rollout和快速运动/遮挡子集为主，不能只看辅助loss。

---

# Pipeline 修复记录（2026-09-06）

当前可执行协议以 [TRAINING.md](../scripts/TRAINING.md) 和 [几何设计](geometry_motion_distillation.md) 为准。下面 2026-09-04 的记录保留为历史实验，不应直接作为当前运行步骤。

- 修复 RGB 直接拉伸：360×480 → 256×341 → 中心裁剪256×320；旧“本来就在裁剪”的结论错误。
- 增加固定 seed 的 episode 90/10 划分；新 run ID 带 `_crop_split`，避免覆盖旧实验。
- 几何训练已对齐现有 EdGE causal-streaming 与 RAFT stride=2 sidecar；515/515 episode 预检通过。保留 legacy QC 兼容，ureter depth 仍降权0.4。
- Motion DPT 修正为 u/v/confidence 三个 logits，u/v 使用线性激活；最终投影重新初始化。Trainer 保留 head.freeze 设置。
- Geometry readout默认从DA3 DPT切换为EdGE原生DPT：62个depth-head权重完整加载；四个`[I;I]`初始化的1024→2048 adapter对齐register与EdGE frame/global维度。Depth保留EdGE projection，motion只复用neck并重置有符号u/v projection；同形DA3类经数值对比并不等价，未作为替代实现；`geometry.head.type=da3`保留为消融。
- 评估使用训练 config 快照；Joint 传视频长度；异常记入 summary 并非零退出；验证读取失败不随机替换样本。
- 修复空目录续训和多机 Accelerate 参数，保留 TCPStore 主节点至所有节点读取 run ID。
- K=2使用5帧history，并从action/proprio/pad target裁掉已观察的8步。旧33帧smoke得到`[24,3]`；当前公平配置把窗口扩为41帧，目标保持`[32,3]`。
- observed flow、action→video K/V 的 γ_p straight-through 梯度门、S0→S2 自动渐进解冻、训练期 depth/flow 验证指标已实现；IDM 和依赖真实动作数据的 S3 未实现。
- 新增 K2 task `endowam_geometry_k2_1cam_1e-4`；最新CPU回归覆盖persistent memory和启动前配置契约，完整5B GPU backward仍需在目标训练环境执行。

---

# Endo4DWAM-Fast 建库 + 全量改名 + 几何监督方案 改动记录（2026-09-04）

## 1. 仓库建立：由 FastWAM 工作区快照播种

**来源**：`/mnt/data2/ljs/FastWAM` 的**工作区快照**（不是 HEAD），共 825 个文件 / 4.5M
= 817 个 tracked + 8 个未跟踪的 EndoWAM 适配文件。69G 里其余全是 gitignore 的
`checkpoints/`(33G)、`runs/`(36G)、`data/`、训练日志。

**不保留上游 git 历史**，单个 Initial commit —— 与 EndoWAM / EndoGuard / EndoGrasp 的建库方式一致。
本地 git identity 设为 `jinsonglin-cuhk`（全局是 `xiaoyuanzi22333`，未动）。

远端：`git@github.com:jinsonglin-cuhk/Endo4DWAM-Fast.git`。

---

## 2. 全量改名：fastwam → endo4dwam

`third_party/` 之外共 272 处 / 47 个文件。

| 类别 | 旧 | 新 |
|------|-----|-----|
| 包 | `src/fastwam/` | `src/endo4dwam/` |
| 模块 | `models/wan22/fastwam{,_joint,_idm}.py` | `endo4dwam{,_joint,_idm}.py` |
| | `processors/fastwam_processor.py` | `endo4dwam_processor.py` |
| 配置 | `configs/model/fastwam{,_joint,_idm}.yaml` | `endo4dwam{,_joint,_idm}.yaml` |
| 目录 | `experiments/robotwin/fastwam_policy/` | `endo4dwam_policy/` |
| 脚本 | `scripts/val_chunk_endowam_fastwam.py` | [`val_chunk_endowam.py`](../scripts/val_chunk_endowam.py) |
| 类 | `FastWAM{,Joint,IDM,Processor}` | `Endo4DWAM{,Joint,IDM,Processor}` |
| 工厂 | `runtime.create_fastwam*` | `create_endo4dwam*` |

同步更新：`pyproject.toml`（name + `packages.find`）、hydra `_target_`、task 的
`override /model:`、wandb project（`fast-wam` → `endo4dwam-fast`，连字符写法第一遍 sed 没匹配到）。

**有意保留的三处**（不是漏改）：
- [LICENSE](../LICENSE) 第 3 行上游 `The FastWAM Authors` 版权声明 —— MIT 第 12–13 行要求分发时保留；
  第 4 行**追加**了 `The Endo4DWAM-Fast Authors`，而非替换。
- [.gitignore](../.gitignore):238-239 的 `data/endo4dwam_model_hf_release/`（目录名已改，
  内含文件仍是上游 LIBERO/RoboTwin release 权重）。
- conda env 名 `fastwam`（机器上真实存在的 env，写成新名字命令会报错）。

**验证**：`compileall` 覆盖 src/scripts/experiments；`bash -n` 覆盖 4 个启动脚本；
8 个模块 import + 4 个类 + 3 个 `runtime.create_endo4dwam*` 工厂；hydra 组合 5 个 task config
且每个 `model._target_` 解析到真实 callable。

---

## 3. README 重写：内镜为主线，中英双语

[README.md](../README.md) / [README_zh.md](../README_zh.md) 从上游论文的官方 README
（讲 LIBERO / RoboTwin）重写为内镜主线：总览（三变体对照表 + 与上游差异）→ 目录结构 → 环境 →
模型准备 → 数据集 → 一次性预处理 → 训练/resume → 输出结构 → 评估 → LoRA 与分辨率约束 →
致谢 + 上游 BibTeX。

LIBERO/RoboTwin 章节删除，但代码仍在树里，用一行标注为「继承自上游、本项目未验证」。

写作时对着配置核出两处按常识写错的说法，已改正：
`configs/data/endowam_endoscope.yaml` 里**没有** `pretrained_norm_stats` 键（走默认 None，首跑现算）；
`delta_action_dim_mask` 嵌在 `default:` 下。

---

## 4. 新增设计文档：WAM4D 式几何 + 运动蒸馏

[docs/geometry_motion_distillation.md](geometry_motion_distillation.md)（v2，经审阅修订）。

**核心思路**（仿 [WAM4D](/mnt/data2/ljs/Endo4DWAM/reference/)）：训练期专用的 spatial register token
查询中间层 history 视频特征、解码未来深度，把预训练几何基础模型的先验反向蒸馏进 backbone；
部署时整条 readout 摘掉，推理开销为零。我们额外扩展一路运动监督（论文没有）。

**与论文的三处结构差异**：

| # | 论文 | 我们 | 应对 |
|---|------|------|------|
| 1 | history 1/5/9 帧 | 只有 1 帧 clean history latent | 提升为**核心方法变量**，主版本 ≥2 帧 |
| 2 | RoboTwin 有仿真 GT 深度 | 单目相对深度，尺度歧义 | clip-wise shared affine alignment |
| 3 | 只有深度 | 深度 + 光流 | **独立** motion registers |

**审阅后修正的关键点**：`geo_dim` 用 1024 而非 3072（否则辅助分支 959M 参数反客为主）；
深度对齐必须 `clamp s>0` 且 `stopgrad(s*,b*)`；flow 不能存 H.264（yuv420p 会把色度通道空间降采样，
而 u,v 是带符号连续量）；resize 必须变换 **flow 向量本身**（`u'=u·W'/W`，翻转 `u→-u`）；
RAFT 只当 teacher，不复用其 decoder（依赖 correlation volume / iterative state，接不上）。

**工程阶段 P0–P4 与训练课程 S0–S3 分开**，后者是 auxiliary warm-up → clinical world grounding →
pseudo-action alignment（含梯度门 γ_p）→ real embodiment alignment。

---

## 5. P0 验证：rot 复用不成立（否掉 8 倍捷径）

**假设**：24 个 root 是 3 术式 × 8 旋转角，伪标签只需在 rot000 上算一次，其余解析旋转得到。

**实测否定**：把 `rot000` 第 0 帧旋转 45°/90° 去匹配 `rot045`/`rot090`，MAE 分别 53.67 / 67.20，
**比完全不旋转的 39.44 / 66.80 还差**。尺度参照：同视频相邻两帧 MAE 仅 0.75。
三个变体都无黑角、直方图 L1 距离 0.31–0.33。

**结论**：rot### 之间是真正不同的视觉内容（推测与 `z60` 透视重投影有关），没有捷径。
（该数据集后已弃用，见第 8 节。）

---

## 6. P0 验证：DA3 深度符号问题 → 换教师

**判据**：内镜光照随距离衰减 ⇒ **亮 = 近**，语义正确的深度必须满足 `corr(depth, 亮度) < 0`。

在 `endowam_pseudo_z60` 上、每术式 15 个 episode：

| 术式 | GIANT 均值 / 不可用 | MONO 均值 / 不可用 | 亮度 std | 对比度预测 corr | 判定 |
|------|------|------|------|------|------|
| esophagus | −0.360 / 2/15 | **−0.839** / 0/15 | 35.2 | — | 好 |
| ercp | +0.289 / **14/15** | **−0.479** / 2/15 | 17.8 (0.51×) | ≈−0.43 | 吻合，MONO 下正常 |
| ureter | −0.012 / 9/15 | **−0.324** / 3/15 | 27.2 (0.77×) | ≈−0.65 | **只有一半，真差** |

**两个易混淆点已分离**：
1. 原始 corr 跨术式不可直接比较 —— 判据强度取决于亮度对比度。按对比度折算后 **ercp 在 MONO 下是正常的**
   （实测 −0.479 vs 预测 −0.43），低数值是判据被削弱的假象；**只有 ureter 是真差**。
2. **根因不是多视图位姿退化** —— GIANT 逐帧（6/15）与多视图（6/15）一样差，是 GIANT-nested 模型本身失效。

**对已产出数据的检查**：对当时正在生成的 ercp 深度直接跑 QC，**8/10 符号反转**，
另 2 个相关性 −0.048 / −0.042（等于无信号）。

**决定**：深度教师从论文的 `DA3NESTED-GIANT-LARGE-1.1` 改为 **`DA3MONO-LARGE`**；
`corr(depth, 亮度)` 从一次性检查提升为**流水线内的 per-clip QC 门控**。

**与损失的关系**：符号反转正是 `s>0` 约束会**暴露**而非修复的情形 —— 放开 `s<0` 会静默地教模型学反的几何。

---

## 7. P0 验证：吞吐与显存实测

| 批大小 | 吞吐 | 峰值显存 |
|--------|------|----------|
| B=1 | 3.03 fps | 7.18 GiB |
| **B=16（最佳）** | **6.60 fps** | 10.36 GiB |
| B=32 | 5.11 fps（回落） | 11.88 GiB |

方案原假设 15 fps，**乐观了 2.3 倍**。加载 6.36 GiB，在 A6000 每卡仅剩 16–19GB 空闲下宽裕。

其他：`is_metric=1` 但深度范围只有 0.43–1.2，对毫米级内镜场景无意义，**再次确认仿射不变损失是对的**；
`conf` 图可作损失权重，但在**暗/远区域塌陷** —— 而那恰是管腔所在。

---

## 8. 数据集切换：`endowam_pseudo_z60`（3 root，360×480）

路径 `/mnt/data2/ljs/Endo4DWAM/Endo4DWAM/dataset/endowam_pseudo_z60`
（软链到 `/mnt/data2/ljs/EndoWAM/dataset/endowam_pseudo_z60`，同一 inode）。

| root | episodes | frames | 占比 |
|------|----------|--------|------|
| ercp | 201 | 407,617 | 42% |
| ureter | 151 | 406,612 | **41%** |
| esophagus | 163 | 165,335 | 17% |

**改动**：[configs/data/endowam_endoscope.yaml](../configs/data/endowam_endoscope.yaml)
`dataset_dirs` 24 → 3、`raw_shape` `[3,270,360]` → `[3,360,480]`。

**`video_size` 保持 `[256,320]`**：源比例 0.750 而目标 0.800，加载器
（`ResizeSmallestSideAspectPreserving` + `CenterCrop`）等比缩放后中心裁剪，
**无形变，裁掉约 21px 宽**。0.750 且能被 32 整除的备选已写进配置注释：
192×256（0.60× token）、288×384（1.35×）。

同时清理 9 个文件里的旧路径 / 旧分辨率引用（2 个 task config、4 个脚本、TRAINING.md、2 个 README）。

### ureter 处置：按分支拆分，而非整体取舍

- **光流监督：全权重保留。** RAFT 不依赖单目深度先验，光流从图像对直接可观测，ureter 的运动与其他术式无异。
- **深度监督：per-episode QC 门控 + 术式级降权。** MONO 下 12/15 通过门控；通过的也确实更弱，
  λ_depth 按 ≈0.4×（`|corr_ureter| / |corr_esophagus|`）缩放。

**理由**：ureter 占 41%。整体丢弃 = 为了只影响深度分支的问题，让两个分支各损失 41% 数据。

---

## 9. `build_endowam_episodes_stats.py` 支持扁平布局（修 bug）

**问题**：脚本按 `<procedure>/rot###/` 两层遍历，`d.name.startswith("rot")` 筛不到就 `continue`。
新数据集是扁平布局（`<procedure>/meta/info.json`），会**静默什么都不写、还报告成功**。

**改动**：[scripts/build_endowam_episodes_stats.py](../scripts/build_endowam_episodes_stats.py)
同时支持两种布局；对既非扁平也无 rot 子目录的目录发 WARN；找不到任何 root 时报错而非假装完成。

**验证**：3/3 roots，写出 201 / 163 / 151 个 episode 的 `episodes_stats.jsonl`，`dp_cache` 正确跳过。

---

## 10. RUN_ID 改名与环境坑

**RUN_ID**：`fastwam_{uncond,joint}_lora_endowam_rot45` → `endo4dwam_{uncond,joint}_lora_z60`。
原先保留旧名是为了 `--resume` 能找到现有 checkpoint；数据集切换后该理由失效
（旧 checkpoint 在新数据上本就无法续训），且名字里的 `rot45` 已是错的。

**环境坑**（[scripts/TRAINING.md](../scripts/TRAINING.md) 开头已记录）：
1. `fastwam` env 的 editable 安装可能仍指向旧仓库 `/mnt/data2/ljs/FastWAM/src/fastwam`，
   本仓库的 `endo4dwam` 未安装 ⇒ `import endo4dwam` 报错，而 `import fastwam` **静默跑旧代码**。
   解法：在本仓库执行`pip install -e .`或设置`PYTHONPATH=src`。项目依赖现已显式包含`addict==2.4.0`，
   2026-09-06也已安装进本机`fastwam`环境。
2. 别用 `openpi` 等其它 venv —— 没有 hydra。

---

## 11. P0 补完 + history 帧数参数化（当日晚）

### 11.1 pixel ↔ latent 映射（实测，推翻原假设）

扰动单个像素帧再重编码 VAE，17 帧 256×320：

```
lat0 ← pix0(仅此一帧)  lat1 ← pix1-4  lat2 ← pix5-8  lat3 ← pix9-12  lat4 ← pix13-16
```

严格因果但**感受野向前泄漏 2–4 步**（pix0 仍影响 lat0..lat3）；**组内 argmax 是第一帧而非最后一帧**
（lat1: pix1=0.850 vs pix4=0.756）⇒ 监督时刻应取 **pix 1/5/9/13**，方案原写的 4/8/12/16 是错的。
K 个 clean latent 帧 = 前 4(K−1)+1 个像素帧；在当时固定17个视频帧的旧协议下，**K=2会让future监督步从4降到3**；当前41帧公平协议已消除该混杂。

### 11.2 history 帧数参数化（解掉「不动 attention mask」的矛盾）

**问题**：`wan_video_dit.py` 的 `first_frame_causal` 把 history 硬编码成只有第一帧。

**改动**：`WanVideoDiT` 新增 `num_history_latent_frames`（默认 1），mask 泛化为 K 帧 history 块
（别名 `first_k_frames_causal`）；[endo4dwam.py](../src/endo4dwam/models/wan22/endo4dwam.py) 的
action→history mask、`build_inputs` 切片、`training_loss` 丢弃步数，以及
[endo4dwam_idm.py](../src/endo4dwam/models/wan22/endo4dwam_idm.py) 两条分支同步参数化。
`_compute_video_loss_per_sample` 的 `include_initial_video_step: bool` 改为
`num_dropped_latent_steps: int`（bool 表达不了 K>1）。
**当时状态：推理仍只支持单帧history。该限制现已对Base/Joint解除**：两者接受与配置一致的K帧输入，Base还能更新persistent memory；IDM仍是K=1旧路径。此句仅保留用于说明当日修复顺序。

**验证**：K=1 掩码与旧实现**逐位一致**；K=2 结构正确；pad 折叠重构在 200 组随机输入上与旧逻辑等价；
键通过 `video_dit_config` 的签名校验；5 个 task config 仍全部 compose。

### 11.3 几何头改用 MONO 的 `DPT`

教师换 MONO 后头应同源。MONO 用 **`DPT`**（`DualDPT` 是 GIANT-1.1 anyview 分支的）。

| 项 | 结果 |
|---|---|
| 参数量 | **29.8M**（DualDPT 约 47M/头） |
| 权重加载 | **missing=0 / unexpected=0**，抽样键与文件逐位一致 |
| `dim_in` | **1024**，正好等于选定的 `geo_dim` |
| `patch_size=32` | 正好产出 256×320，对齐 register 网格 8×10 |
| 显存(fwd+bwd, bf16) | batch4×3步 **1.086 GiB**；batch4×4步 1.481 GiB |

### 11.4 光流：两个影响设计的发现

- **69% 的帧几乎静止**（p50 幅度 0.00083，p99 0.0589）。运动监督目标绝大多数是零，
  模型预测全零即可低 loss ⇒ λ_flow 与运动加权采样需据此设计。
- **当时仅靠光度warp无法判定方向/步长**：高运动帧上`t→t+1`在4/6情形最优、`t→t+2`在2/6，而sidecar写`stride: 2`。后续已由生成器源码确认数组语义为`t→t+stride`，当前即`t→t+2`。
- **纠正方案中的一处错误**：StereoMIS 的 RAFT 不是内镜微调版，只是官方仓库副本 + 标准
  `raft-things.pth`。在用的 `C_T_SKHT_V2` 训练更充分，无可换项。

---

## 设计决策与遗留（2026-09-04）

**已定**
- 建库不保留上游历史，单 Initial commit；LICENSE 追加而非替换版权行。
- 深度教师 `DA3MONO-LARGE`；`corr(depth,亮度)` 作为流水线内 per-clip QC 门控。
- ureter 保留，按分支拆分（flow 全权重 / depth 门控 + 0.4× 降权）。
- `video_size` 保持 `[256,320]`（内镜训练输入**不是** 224×224；224 是 DA3 伪标签侧的 `process_res`，
  以及上游 LIBERO 配置的分辨率）。
- geo_dim 1024；深度对齐 `clamp s>0` + `stopgrad(s*,b*)`；flow 存 fp16 无损。

**遗留 / 待办**
- ~~**P0 未完**~~ —— **已全部完成**，见第 11 节。
- ~~flow 方向/步长约定~~ —— 已直接核对生成器：数组第`t`项由帧`t`与`t+stride`生成；当前sidecar的`stride: 2`即 **`t→t+2`**，无需因这一点重跑。
- ~~深度几何变换~~ —— **已改存原图 360×480**，对齐歧义消失；换 MONO 后 ercp 符号在实际产出上翻正
  （ep0 +0.481→−0.517，ep1 +0.316→−0.624）。
- **flow 归一化不能免除几何变换**：360×480先等比resize到256×341再中心裁剪256×320，
  所以 `u` 必须乘 **341/320=1.065625**、`v` 不变。漏掉不报错，只会让水平运动目标系统性偏小约6.6%。
- **历史问题（已被原图深度替代）**：早期深度是方形 `224×224`，源 360×480（0.750）、训练管线 256×320（0.800）。
  用相关性区分「整幅压扁」与「中心裁剪」得 0.6444 vs 0.6423，**差异太小判不了**，
  须由生成侧把确切变换写进 `depth_meta`。
- ~~history ≥2 帧推翻「不动 attention mask」~~ —— **已实现并验证**（第 11.2 节）。
  旧33帧K2会让future监督从4降到3；当前公平41帧K2保持4步。~~推理路径尚未支持K>1~~ 已完成：infer接受5个history像素帧，Base action path还支持跨chunk memory state。
- **光流头没有预训练先验**：论文 Table 8 显示随机初始化的几何头比不加还差。λ_flow 需从小值起步并单独消融。
- **算力**：全量 979,564 帧；A6000 8 张当前全部满载，每卡仅剩 16–19GB。
- 训练前必做：`precompute_text_embeds.py task=endowam_fastwam_baseline_1cam_1e-4 +overwrite=false`。
  新数据集9条task记录去重后只有一条prompt；缓存状态应以目录实查为准，不再在记忆文档硬编码“为空”。
