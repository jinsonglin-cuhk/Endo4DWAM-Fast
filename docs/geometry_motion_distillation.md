# Endo4DWAM-Fast：仿 WAM4D 的几何 + 运动蒸馏（v2，已按审阅意见修订）

## Context

Endo4DWAM-Fast 目前只有两路监督：视频 latent 的 flow-matching loss 和 action 的 flow-matching loss。模型对场景几何和运动结构没有任何显式约束，而动作预测又完全依赖这套视频特征。

WAM4D（论文 PDF 在仓库之外：`/mnt/data2/ljs/Endo4DWAM/reference/WAM4D - Fast 4D World Action Model via Spatial Register Tokens.pdf`）的核心：**用训练期专用的 spatial register token 查询 history 视频特征、解码未来深度，把预训练几何基础模型的先验反向蒸馏进 backbone**；部署时整条 readout 摘掉，推理开销为零。

> 引用口径（修订）：**在 RoboTwin 2.0 的 10-task ablation split 上**，可训练预训练几何头把 clean success 从 71.7% 提到 80.1%（Table 6 / Table 8）。全量 50-task 主表（Table 1）里 WAM4D 是 93.8。**不要拿 71.7→80.1 当全量主结果引用。**

本方案把 WAM4D 适配到内镜 pipeline，并扩展一路运动监督（论文没有）。

---

## v2 变更清单（审阅意见落地）

| # | 变更 | 性质 |
|---|---|---|
| 3 | **history 提升为核心方法变量**，主版本 ≥2 个 clean latent 帧 | 方法 |
| 4 | 运动监督拆成 **observed flow + future flow** | 方法 |
| 5 | 新增 **pseudo-action → Video Expert 的梯度门 γ_p** | 方法 |
| 6 | **工程阶段 P0–P4 与训练课程 S0–S3 分开** | 结构 |
| 7 | 深度对齐 **clamp s>0 + detach (s*,b*)**（原方案写的"默认不 detach"是错的） | 修正 |
| 8 | **flow 不存 H.264**，改 fp16 无损；depth 改 uint16/fp16 | 修正 |
| 9 | resize 必须**变换 flow 向量本身**，不只是变换图像 | 修正 |
| 12 | DA3 head **渐进解冻**，不是 step 0 就 trainable | 修正 |
| 13 | RAFT **只当 teacher**，不复用其 decoder 当预训练头 | 修正 |
| 14 | λ 下调至 λ_d=0.25–0.5、λ_f=0.05–0.2，并监控梯度范数比 | 修正 |
| 15 | 主对照组换成 **stop-gradient geometry control** | 修正 |
| 16 | 修正 71.7→80.1 的引用口径 | 修正 |
| 17 | 深度指标必须是 **aligned relative-depth metrics** | 修正 |
| 18 | 新增 P0：实测 **pixel frame ↔ latent timestep** 真实映射 | 新增 |
| 19 | "逐比特一致" → **"zero auxiliary inference overhead"** | 措辞 |
| 2 | register 分支是**实现简化**，不是方法差异 | 措辞 |

### #3 的连带后果（原方案漏了，很重要）

要 ≥2 帧 clean history latent，就必须改 `video_attention_mask_mode: "first_frame_causal"`——`wan_video_dit.py:501-505` 现在把"history"硬编码成**只有第一帧**：

```python
if self.video_attention_mask_mode == "first_frame_causal":
    video_mask = torch.ones((video_seq_len, video_seq_len), ...)
    first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
    video_mask[:first_frame_tokens, first_frame_tokens:] = False
```

需要泛化成 `first_k_frames_causal`，同时 `Endo4DWAM._build_mot_attention_mask`（`endo4dwam.py:404-406`，action → 前 K 帧）和 `training_loss` 里的 `latents[:,:,0:1] = first_frame_latents`（`:467`）都要跟着改成前 K 帧。**这推翻了 v1"不动 attention mask"的说法。**

代价：latent 共 5 帧，2 帧 history ⇒ future 只剩 3 帧（监督时刻从 4 个降到 3 个）。这个取舍要在 P0.18 拿到真实 pixel↔latent 映射后再定死。

---

## 已验证的事实（P0 已完成部分）

- **rot### 之间没有解析复用捷径**（实测）。把 rot000 旋转 45°/90° 匹配 rot045/rot090，MAE 53.67/67.20，**比不旋转的 39.44/66.80 还差**；同视频相邻帧 MAE 仅 0.75。伪标签必须逐 root 算。
- **24 roots 合计 7,836,512 帧**。**实测** DA3 最佳 6.60 fps（B=16）⇒ 全量 **330 GPU-h**；rot000 子集+步长4 为 10.3 GPU-h。8 张 A6000 当前全部 100% 占用、每卡仅剩 16–19GB。
- **开关必须用 `self.mot.training`**（实测）：训练中 `model.training=False` 而 `model.mot.training=True`（`trainer.py:291-293` 先 `model.eval()` 再 `model.dit.train()`）。用 `self.training` 会让分支整轮静默不训练。
- **`loss_dict` 新 key 必须每 rank 每步无条件发出**，否则 `trainer.py:750-754` 的 all-gather 因 key 集合不一致而死锁。
- **模块挂 `self.mot` 下且名字不含 `mixtures.video.`** → 自动可训练 + 自动进 checkpoint，trainer 零改动。挂顶层会被静默冻结，挂 video expert 内会被 LoRA 规则静默冻结。
- **`save_total_limit: 1` 会删掉现有 checkpoint**（`trainer._rotate_checkpoints`）。开工前必须先把 warm-start 权重复制走，且 `resume` 只能给 `.pt` 文件路径（目录路径走 DeepSpeed 严格加载必挂）。
- **env 分工**：`DAv3`（有 DA3 + cv2/decord/imageio）跑离线伪标签；`fastwam`（有 av）跑训练。`DAv3` 的 `depth_anything_3` 指向 `/mnt/data2/ljs/Depth-Anything-3`，**不是** beilei 那份，API 不同，行号需重新对齐。
- 本地已缓存 `DA3-BASE` / `DA3MONO-LARGE` / `DA3NESTED-GIANT-LARGE` / `DA3NESTED-GIANT-LARGE-1.1`。

---

## P0 实测结果（已完成，2026-09-04）

### 结论摘要

| 项 | 结果 |
|---|---|
| DA3 显存 | 加载 6.4 GiB，B=16 峰值 10.4 GiB —— 在每卡 16–19GB 空闲里宽裕 |
| **真实吞吐** | 最佳 B=16 @ **6.60 fps**（方案原假设 15 fps，**乐观了 2.3 倍**） |
| **全量代价** | 7,836,512 帧 → **330 GPU-hours**（原估 145） |
| rot000 子集 + 步长4 | 245k 帧 → **10.3 GPU-hours**，可行 |
| **深度教师** | **必须从 GIANT-1.1 换成 `DA3MONO-LARGE`**（见下） |

### 深度符号 QC：一个必须进流水线的检验

内镜有个可靠的客观判据：**光照随距离衰减 ⇒ 亮 = 近**。因此语义正确的深度应满足 `corr(depth, 亮度) < 0`。实测 3 术式 × 5 episode × 4 帧：

| 配置 | 符号反转 | 平均相关 |
|---|---|---|
| DA3 GIANT-1.1 多视图 | **6/15** | −0.129 |
| DA3 GIANT-1.1 逐帧 | **6/15** | −0.133 |
| **DA3MONO-LARGE 逐帧** | **3/15** | **−0.457** |

逐术式（GIANT 多视图 → MONO）：

- **ercp：5/5 符号反转（+0.43~+0.63）→ MONO 后 0/5 反转（−0.69~−0.81）**。这是换教师的决定性理由。
- esophagus：两者都好（GIANT −0.84~−0.90 略优于 MONO −0.72~−0.85）。
- **ureter：所有配置都不可靠**（MONO 下 3/5 为正或近零：+0.29/+0.26/+0.08）。

**根因不是多视图位姿退化**——GIANT 逐帧与多视图同为 6/15，是 GIANT-nested 模型本身在 ercp 内容上失效。

### 这对损失函数的意义

符号反转正是 `s>0` 约束会**暴露**而非修复的情形：翻转的 clip 在 `s>0` 下根本对不齐；若放开 `s<0`，则会静默地教模型学反的几何。所以：

1. **`corr(depth, 亮度)` 必须作为 per-clip QC 门控内建进伪标签流水线**，对 `corr > 0` 的 clip 直接拒绝或标记。
2. **ureter 需要单独处置**：或排除出深度监督、或换内镜专用教师（Endo3R / EndoDAC）。不要假设它能用。

### 跨帧尺度一致性（clip-wise 共享仿射的前提）

MONO 逐帧推理下，一个 clip 内共享一组 $(s,b)$ 的残差是逐帧最优的 **1.29–2.33 倍**（esophagus 1.29–1.46、ercp 1.37–2.33、ureter 1.91）。可用，但有可测的漂移。注意这是**上界**——测法把真实场景变化也计入了残差。

作为对比，GIANT 多视图模式的跨帧尺度非常稳（各帧范围都在 ~0.47–0.66），而 GIANT/MONO 逐帧模式下各帧范围乱跳（0.49–0.97）。若漂移成为问题，退路是滑窗多视图 + 跨窗尺度缝合。

### 其他观察

- `is_metric=1`，但深度范围只有 0.43–1.2。若当米解释，内镜场景应是毫米级——**该"metric"标注对我们无意义**，再次确认仿射不变损失是对的。
- `conf` 图可用作损失权重 $w$，但它在**暗/远区域塌陷**——而那恰是管腔所在、深度最重要的地方。
- DA3 内部把 270×360 重采样到 378×504。

### 数据集切换到 `endowam_pseudo_z60`（无 rot 增广）

3 个 root，共 **979,564 帧**、40G，源分辨率 **360×480**（rot45 那份是 270×360）：

| root | episodes | frames | 占比 |
|---|---|---|---|
| ercp | 201 | 407,617 | 42% |
| ureter | 151 | 406,612 | **41%** |
| esophagus | 163 | 165,335 | 17% |

⚠️ `configs/data/endowam_endoscope.yaml` 仍指向 `endowam_pseudo_z60_rot45` 且 `raw_shape: [3, 270, 360]`，需改为新数据集与 360×480。

### ureter 处置决定（按术式拆分监督，而不是整体取舍）

在 `endowam_pseudo_z60` 上、每术式 15 个 episode 实测（判据 corr(depth,亮度)，"不可用" = corr > −0.15）：

| 术式 | GIANT 均值 / 不可用 | MONO 均值 / 不可用 | 亮度 std | 对比度预测 corr | 判定 |
|---|---|---|---|---|---|
| esophagus | −0.360 / 2/15 | **−0.839** / 0/15 | 35.2 | — | 好 |
| ercp | +0.289 / **14/15** | **−0.479** / 2/15 | 17.8 (0.51×) | ≈−0.43 | **吻合，MONO 下质量与 esophagus 相当** |
| ureter | −0.012 / 9/15 | **−0.324** / 3/15 | 27.2 (0.77×) | ≈−0.65 | **只有预测值一半，真差** |

关键区分：**ercp 在 MONO 下数值低是判据被低对比度削弱的假象**（实测 −0.479 vs 预测 −0.43，吻合）；**ureter 是真的差**——对比度只能解释到 −0.65，实测 −0.324。

**决定：保留 ureter，但按分支拆分**

- **光流监督：全权重保留。** RAFT 不依赖单目深度先验，光流直接从图像对可观测，ureter 的运动和别的术式没有区别。没有理由排除。
- **深度监督：per-episode QC 门控 + 术式级降权。** MONO 下 12/15 通过门控（拒掉约 20%）；通过的那些也真的更弱，所以 ureter 的 λ_depth 按可靠度缩放（≈ 0.4×，即 |corr_ureter| / |corr_esophagus|）。

理由：ureter 是 40.6 万帧、占 41%。整体丢弃等于为了**只有深度分支有问题**而牺牲两个分支各 41% 的数据。

### 待确认：几何对齐

已产出的深度是方形 `224×224`，而源是 360×480（比例 0.750）、训练管线是 256×320（比例 0.800）。我用 corr 区分"整幅压扁"与"中心裁剪"得到 0.6444 vs 0.6423，**差异太小，无法判定**。这个变换必须由生成侧明确记录进 `depth_meta`，否则监督对齐无从保证（见 v2 变更清单 #9）。

### P0 补完：latent 映射、DPT 头、光流（2026-09-04 晚）

**1. pixel ↔ latent 映射（实测，推翻原假设）**

扰动单个像素帧再重编码，17 帧 256×320：

```
lat0 ← pix0 (仅此一帧)   lat1 ← pix1-4   lat2 ← pix5-8   lat3 ← pix9-12   lat4 ← pix13-16
```

- 严格因果（无像素帧影响更早的 latent 步），但**感受野向前泄漏 2–4 步**（pix0 仍影响 lat0..lat3）。
- **每组内 argmax 是该组第一帧而非最后一帧**（lat1: pix1=0.850 vs pix4=0.756）。
  ⇒ 监督时刻应取 **pix 1/5/9/13**，原方案写的 4/8/12/16 是错的。组内各帧贡献相当（0.76–0.95），取任一帧都可辩护。
- K 个 clean latent 帧 = 前 4(K−1)+1 个像素帧；**K=2 ⇒ 5 帧 history，future 监督步从 4 降到 3**。

**2. history 帧数已参数化（代码已落地）**

`WanVideoDiT` 新增 `num_history_latent_frames`（默认 1，与旧实现逐位一致），
`first_frame_causal` 泛化为 K 帧 history 块（别名 `first_k_frames_causal`）。
`Endo4DWAM` 的 action→history mask、`build_inputs` 的 history 切片、`training_loss` 的丢弃步数，
以及 IDM 两条分支都已跟着参数化。`_compute_video_loss_per_sample` 的
`include_initial_video_step: bool` 改为 `num_dropped_latent_steps: int`（bool 表达不了 K>1）。
**推理仍只支持单帧 history，K>1 时四条 infer 路径显式报错**，避免训练/推理不一致被静默引入。

**3. 几何头：改用 MONO 的 `DPT`，不是论文的 `DualDPT`**

教师换成 MONO-LARGE 后，头也应同源。实测：

| 项 | 结果 |
|---|---|
| MONO head 类 | **`DPT`**（`DualDPT` 是 GIANT-1.1 anyview 分支用的） |
| 配置 | `dim_in=1024, output_dim=1, features=256, out_channels=[256,512,1024,1024]` |
| 参数量 | **29.8M**（DualDPT 约 47M/头） |
| 权重加载 | **missing=0 / unexpected=0**，抽样键与文件逐位一致 |
| `patch_size=32` | 正好产出 256×320，与 register 网格 8×10 对齐 |
| 显存（前向+反向，bf16） | batch4×3步 **1.086 GiB**；batch4×4步 1.481 GiB |

**`dim_in=1024` 正好等于选定的 `geo_dim=1024`**，register 输出可直接接入。
输出含一个用不到的 `sky` 头，可关掉再省。

**4. 光流：两个影响设计的发现**

- **69% 的帧几乎静止**（幅度 < 0.002 归一化 ≈ 0.2px@112；p50=0.00083、p90=0.0153、p99=0.0589）。
  运动监督的目标**绝大多数是零**，模型预测全零即可拿到低 loss。λ_flow 的调法与是否按运动幅度采样需要据此设计。
- **方向/步长约定从数据判不明确**：在高运动帧上用光度 warp 检验，`t→t+1` 在 4/6 情形最优（+7.2%/+28.3%/+18.0%/+22.0%），
  `t→t+2` 在 2/6 最优（+3.3%/+0.9%），而 `flow_meta` 写的是 `stride: 2`。**须由生成侧确认**，dataloader 依赖这个约定。
- **StereoMIS 的 RAFT 不是内镜微调版**（只是官方 RAFT 仓库副本 + 标准 `raft-things.pth`）。
  原方案说它「可能更贴内镜域」是错的；已在用的 `C_T_SKHT_V2` 反而训练更充分，**无可换项**。

### 生成侧两项已定（2026-09-04 晚，生成侧确认）

**1. flow 约定：`t → t+1`**（已定）。已产出的 201 个 ercp flow 文件 `flow_meta` 记的是 `stride: 2`，
与该约定不符，**需重跑**。dataloader 按 `t → t+1` 实现。

**2. 深度改存原图尺寸 `360×480`**（已改），`depth_meta` 现记 `height/width`，
方形 `224×224` 的对齐歧义随之消失。实测新产出 `(N, 360, 480) float16`。

**3. 换教师在实际产出上验证通过**：同一批 ercp episode，GIANT → MONO 后符号从错翻正。

| episode | 旧（GIANT） | 新（MONO, 360×480） |
|---|---|---|
| episode_000000 | +0.481（反转） | **−0.517** ✓ |
| episode_000001 | +0.316（反转） | **−0.624** ✓ |

### ⚠️ flow 归一化不能免除几何变换（具体数值）

flow 存的是 `u/W, v/H`。RGB 从 360×480 走等比缩放 + 中心裁剪到 256×320：

```
s = max(256/360, 320/480) = 0.711111
缩放后 256.0 × 341.3  →  中心裁剪到 256 × 320（裁掉 21.3 px 宽）
```

因为**裁剪只发生在宽度方向**：

| 分量 | 归一化基准 | 换算系数 |
|---|---|---|
| `v`（垂直） | 360 → 256 | **1.000000**（不变） |
| `u`（水平） | 480 → 320 | **1.066667**（必须乘） |

即使 flow 已按图像尺寸归一化，**`u` 仍必须乘 1.0667，`v` 不变**。漏掉这一步不会报错，
只会让水平运动的监督目标系统性偏小 6.7%。

### P0 剩余项

~~RAFT 两版对比~~、~~几何头加载 + 显存实测~~、~~pixel↔latent 映射~~ —— **均已完成，见上一节**。

仍未定：
- ureter 深度的 per-clip 门控阈值需在完整数据上标定
- 推理路径尚未支持 K>1 history（目前显式报错）
- 已产出的 201 个 ercp flow 需按 `t→t+1` 重跑

---

## 与 WAM4D 的关系（措辞修正，#2）

WAM4D 的 spatial register 本身就是一条**辅助 depth extraction path**：register 作 Q，`[register, history video]` 作 K/V，并明确禁止 action 访问 register。它并没有把 register 当普通 token 混进主序列。

所以我们的做法准确表述是：

> We implement the register readout as an **external unidirectional auxiliary branch** attached to intermediate Video-Expert features, preserving the Fast-WAM main attention graph unchanged.

即**实现上的简化**，保留了 WAM4D 的 unidirectional readout 原则，而非方法上的偏离。

---

## 架构

### A. 离线伪标签

**深度教师**：**`DA3MONO-LARGE`**（P0 实测后从论文的 GIANT-1.1 改过来，理由见 P0 章节）。几何头初始化应与之保持同源。**运动教师**：RAFT（torchvision `raft_large` 已验证可用；`/mnt/data2/beilei/repository/StereoMIS-Dataset-in-Pytorch/RAFT` 为内镜域备选）。RAFT **只作 teacher**（见 #13）。

**存储格式（#8 修正）**——不用 H.264：

| 模态 | 格式 | 理由 |
|---|---|---|
| depth | `uint16`（或 fp16）无损 | 8-bit + 块效应对 affine-invariant loss 勉强可接受，但没必要冒险 |
| flow (u,v) | **`fp16` 无损** | u,v 是带符号连续量；H.264 的量化、块效应、RGB↔YUV、**chroma subsampling（yuv420p 会把两个色度通道空间降采样）**对 flow 监督是灾难 |
| mask | `uint8` / bool | — |

容器用 npz / zarr / HDF5 / lossless chunk 均可，按 episode 分片。**代价**：不能再复用现有的视频解码路径，dataloader 侧需要写一个并行的读取器，与 `video_sample_indices` 对齐。这是为数值正确性付出的必要复杂度。

**几何变换必须共享参数，不是共享函数（#9 修正）**：空间位置一致只是一半。resize 后 flow 向量本身要变换：

$$u' = u\cdot\frac{W'}{W},\qquad v' = v\cdot\frac{H'}{H}$$

水平翻转 $u\to-u$，垂直翻转 $v\to-v$，旋转要转向量。所以 RGB / depth / flow 必须共享**同一组几何变换参数**，由一个统一的 transform 对象分发，而不是各自调同一个 `resize` 函数。**这条极易静默训练错且不会报错。**

**算力分级**：先只做 3 个 `rot000` root + 时间步长 4（约 4–6 GPU-h），验证方法有效后再决定是否铺满 24 roots。

### B. 模型分支（训练期专用）

在 `MoT.forward` 捕获 layer ∈ {12,14,16,18} 的 video hidden states，切出 history 段（前 K 帧的 token）。

```
Z_hist^ℓ ← Linear(3072→1024)          # geo_dim=1024（#11 认可；可另做 512 的轻量 check）
R⁰ = Repeat_τ(R★)                     # R★ 可学习 [8×10=80, 1024]
R^{ℓ+1} = Block_ℓ(Q=R^ℓ, K=V=[R^ℓ, Z_hist^ℓ])   # 4 blocks, num_heads=8 (head_dim 128)
G = P(R^final) → Head(G)
```

两套**独立** register（#10 认可）：$R_D \neq R_F$，$Block_D \neq Block_F$。显存不够时的退化顺序是 **separate query banks + shared block weights**（$R_D\neq R_F$, $Block_D=Block_F$），**而不是删掉 flow 分支**。

- **深度头**：预训练 DA3 `DualDPT`，**渐进解冻**（#12）：S0 完全冻结，只训 projection + register；S1 early 起以 $LR_{DA3}\approx0.05\text{–}0.1\times LR_{register}$ 解冻。理由：projection/register 一开始是随机的，让随机特征直接强更新 DA3 会迅速破坏预训练 prior。终态仍是论文的 trainable pretrained head。
- **运动头**：轻量可训 decoder（DPT 式 multi-scale），最后一层 flow projection 随机初始化 + warm-up + 小 λ_F。**不复用 RAFT decoder**——RAFT 的 update head 依赖 correlation volume、context features、iterative state，$R_F\to$ RAFT decoder 继承不到预训练语义（#13）。

### C. 损失

$$L = L_V + \lambda_a L_a + \lambda_D L_D + \lambda_F L_F$$

**λ 起始值（#14 修正，不照抄 WAM4D）**：$\lambda_v=1$、$\lambda_D=0.25\text{–}0.5$、$\lambda_F=0.05\text{–}0.2$。理由：我们的 depth 是 domain-shifted 伪标签、flow 是 RAFT 伪标签、action 本身也是伪动作，raw loss scale 与论文完全不同。

**必须监控共享 history 特征 $H$ 上的梯度范数比**，比看 loss 数值有意义得多：

$$r_D=\frac{\|\nabla_H L_D\|}{\|\nabla_H L_V\|},\qquad r_F=\frac{\|\nabla_H L_F\|}{\|\nabla_H L_V\|}$$

#### C.1 深度：clip-wise shared affine alignment（#7 修正）

整个 future clip 共享**一组** $(s^*,b^*)$：

$$(s^*,b^*)=\arg\min_{s,b}\sum_{\tau}\sum_{p} w_{\tau,p}\big(s\hat D_{\tau,p}+b-D^*_{\tau,p}\big)^2$$

两处修正：
1. **约束 $s>0$**：至少 clamp $s\leftarrow\max(s,\epsilon)$。否则"近远完全反过来"的预测可以靠负 scale 被对齐回来，这不是我们要的 invariance。
2. **detach**：$(s^*,b^*)=\operatorname{stopgrad}(\text{LeastSquares}(\cdot))$。否则网络会同时用"改预测"和"改最优对齐"两条路降 loss。**v1 写的"默认不 detach"是错的。**

$$L_D=\frac1N\sum_{\tau,p} w_{\tau,p}\,\mathrm{SmoothL1}\big(s^*\hat D_{\tau,p}+b^*,\;D^*_{\tau,p}\big)$$

全 clip 共享 $(s,b)$ ⇒ 模型无法逐帧作弊，跨帧时序几何被保留。$w$ 屏蔽黑边、过曝高光、pad 帧。

#### C.2 运动：observed + future（#4 新增）

单帧 history 下 $p(F_{future}|I_t)$ 高度多模态——同一张内镜图像，操作者可以前进/后退/左右/旋转。future depth 没这么严重（单张 RGB 已含大量 scene geometry prior），但 **future flow 对 camera action 的依赖强得多**。

所以拆成两项：

$$L_{motion}=L_{flow}^{obs}+\lambda_{future}L_{flow}^{future},\qquad \lambda_{future}<1 \text{ 起步}$$

- $F_{t-4\to t}$（**已发生，确定的**）= current motion state
- $F_{t\to t+4}$（**待预测，多模态的**）= future motion evolution

物理语义上比单纯"depth+flow 双监督"更干净：**Depth = geometry state，Observed Flow = motion state，Future Flow = motion dynamics**。这也是 history ≥2 帧的另一个理由——没有 2 帧就没有 observed flow。

#### C.3 pseudo-action 梯度门 γ_p（#5 新增）

我们的 $a^{pseudo}\neq a^{robot}$，而 Fast-WAM 的特点恰恰是 Video Expert 与 Action Expert 通过 MoT 深度耦合——$L_{pseudo\text{-}action}$ 会经由 action←video 的注意力反向污染 Video Expert。

在 action 分支读取 video K/V 处插梯度门：

$$K_V^{action}=\operatorname{sg}(K_V)+\gamma_p\big(K_V-\operatorname{sg}(K_V)\big)$$

前向恒等，反向按 $\gamma_p$ 缩放。$\gamma_p=0$ 时 Video Expert 完全不受伪动作污染；$\gamma_p=1$ 退化为现状。实现位置：`mot.py` 里 action expert 取 `k_cat`/`v_cat` 的 video 段处。

---

## 工程阶段 P0–P4

**P0 — 验证（不写训练代码）**
1. ~~rot### 旋转关系~~ **已完成：无捷径**
2. **DA3 在内镜上的伪深度质量目视检查** ← 全案最大科学风险，未做
3. RAFT 两版内镜对比
4. DualDPT 权重加载 + 显存实测
5. DA3/RAFT 真实吞吐（替换假设值）
6. **（#18 新增）实测 pixel frame ↔ latent timestep 映射**：只扰动单个 pixel frame $k$，看哪个 latent timestep 变化最大。Wan VAE 有 causal temporal conv、first-frame 特殊处理、时间感受野偏移，**不能只按 stride=4 推断**。这直接决定 register 的时间坐标和 temporal RoPE。

**P1 — 伪标签流水线（先子集）**：`scripts/build_endowam_depth_flow.py`，fp16 无损输出。
**P2 — 模型分支**：MoT 逐层捕获 → register/block/head → 损失 → 配置接线 → first-K-frames causal mask。默认 `geometry.enable=false` 与今天等价。
**P3 — 训练**：见下方 S0–S3。
**P4 — 消融**：register 层位 / history 帧数 / 几何头初始化与解冻策略。

## 训练课程 S0–S3（#6 新增，与工程阶段分开）

| Stage | Video Expert | Registers | Action Expert | 主要 Loss |
|---|---|---|---|---|
| **S0** auxiliary warm-up | Frozen | Train | Frozen | $L_D+L_F$ |
| **S1** clinical world grounding | Low LR | Train | Frozen | $L_V+\lambda_D L_D+\lambda_F L_F$ |
| **S2** pseudo-action alignment | Frozen → very low LR | keep | Train | $L_{pseudo}$，$\gamma_p:0\to0.1$ |
| **S3** real embodiment alignment | Frozen → low LR | optional | Train | $L_{real\text{-}action}+\eta L_{world}$ |

对应的主线是：**Clinical Video → Geometry/Motion World Representation → Pseudo Motion → Real Robot Control**。

---

## 验证与对照组

**主对照组换成 stop-gradient geometry control（#15）**，比参数量对齐更有说服力：

```
History Feature ──detach──► Depth/Flow Registers
```

辅助头照训、depth/flow loss 照降，但 $\nabla_{VideoExpert}L_D=\nabla_{VideoExpert}L_F=0$。若 detached 组几何指标好但 action 不涨、full distillation 组 action 涨，就直接证明收益来自 **physical priors being distilled into the policy representation**，而非多了参数 / 多了辅助网络 / optimizer 行为改变。参数量对齐组作为次要对照保留。

**其他验证**
- `geometry.enable=false` 时 loss 与当前实现逐位一致。
- **推理侧口径（#19 修正）**：说 **zero auxiliary inference overhead** —— readout 在推理时整条移除，Fast-WAM 原始 policy 架构不变。**不要说"与 baseline 逐比特一致"**：几何训练会改变 backbone 权重，训练后的输出当然不可能和 baseline bitwise identical。
- **深度指标必须 aligned（#17）**：我们没有 metric depth，训练用了 affine-invariant 对齐，评估也必须先做同样的 $(s,b)$ 对齐再算 AbsRel/δ₁，并明确命名为 **aligned relative-depth metrics**，否则无法与 WAM4D 的 metric/simulation 指标横向比较。运动指标用 EPE，分 observed / future 两项分别报。

---

## 主要风险

1. **future flow 欠定**（已降级但未消除）：靠 history≥2 + observed flow 打底 + $\lambda_{future}<1$ 缓解。
2. **DA3 在内镜域外**：P0.2 不过关就换教师（Endo3R / EndoDAC 机器上有 env 痕迹）。
3. **显存**：A6000 每卡仅剩 16–19GB。退化顺序：shared block weights → 降 head 分辨率 → 砍监督时刻数。
4. **算力**：全量 784 万帧不可行，首轮只能在 1/8 数据上出结论，跨角度泛化要等铺满。
5. **flow 存储改无损后 dataloader 复杂度上升**：不能再白蹭视频解码路径，需要新读取器 + 几何变换参数共享机制，这是新增的 bug 面。

## 不做的事

- 不动推理路径、不动 RoboTwin/LIBERO 相关代码。
- 第一轮不做 joint / IDM 变体。
- 不做 point-cloud / F-score-T 指标（无可靠 intrinsics 与 metric scale）。
- ~~不动 attention mask~~ —— **已作废**，history≥2 必须改 mask（见 v2 变更清单）。
