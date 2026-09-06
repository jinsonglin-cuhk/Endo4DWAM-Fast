# Endo4DWAM-Fast 几何与运动蒸馏：设计与实现状态

更新：2026-09-06。执行命令见 [TRAINING.md](../scripts/TRAINING.md)。历史实验记录见 [Memory.md](Memory.md)，其中过去的路径、标签尺寸和机器资源状态不代表当前协议。

## 目标与当前边界

通过训练专用 registers 查询视频 backbone 的 history 特征与 persistent visual memory，用未来深度和运动伪标签提供辅助监督。部署时删除 depth/flow readout，但保留已经学到的 video expert、memory updater 与 action expert。

当前实现：base/Joint Endo4DWAM、K≥1、video/action 加 EdGE depth 与 stride-2 RAFT flow 辅助监督；K≥2 的历史动作裁剪、action→video 与 action→memory 两条独立梯度门、S0→S2 渐进解冻、验证集 aligned-depth/flow EPE 指标均已接线。Base 还实现了64×1024的显式 persistent visual memory。可执行 K=2 配置为 `task=endowam_geometry_k2_1cam_1e-4`。Joint 保留为读取未来视频的消融，IDM 仍明确拒绝几何训练。

严格baseline由`task=endowam_fastwam_baseline_1cam_1e-4`固定：训练仍联合预测future video/action，验证和部署走`infer_action()`且不生成future video。`history`、`auxiliary`、`persistent_memory`三个Hydra组可独立组合；memory消融另有`persistent_memory=cached_control`，控制split/cache执行路径的混杂。

代码提供可重复的消融入口，但“正式消融结果”必须由实际训练产生，不能由实现本身宣称完成。S3 真实机器人对齐也仍取决于尚未放入当前数据配置的真实动作数据。

## Persistent memory 与非对称信息流

Base 模型的决策路径是：clean RGB history → video expert → 中层 history hidden state → recurrent memory updater → action expert。默认状态 `M0` 为零，形状 `[B,64,1024]`。learned slot embedding 区分64个槽位；每个 clean latent frame 按时间顺序更新两层 gated cross-attention/MLP。更新接口只接受视觉 hidden state，没有 action 或 noisy future video 参数。

训练 DataLoader 会随机采样窗口，不能把相邻 batch 假装成同一 episode。因此当前训练在每个窗口从 `M0` 开始，对 K 个 clean history latent frame 做窗口内 unroll；若调用者显式传入 `sample["memory_state"]`，上一状态先 detach，实现 truncated boundary。当前**没有**顺序 episode sampler，也没有跨 batch BPTT。

部署时 `infer_action(..., memory_state=...)` 返回新的 `memory_state`。调用方在同一 episode 的下个 action chunk 回传它，在新 episode 调用 `init_memory()` 重置。`val_chunk_endowam.py` 已按这个协议跨窗口传递；RobotWin policy 的 `reset()` 也会重置。状态不保存在 module 成员里，因此并行环境不会互相污染。

每层 action query 的 key/value 顺序为 `[history video, persistent memory, action]`：

| Query | 可见 K/V | 不可见 | 状态 |
|---|---|---|---|
| video | causal video | action、memory | 已实现；video 不读 memory |
| action | history video、memory、action | future video | Base 已实现 |
| memory writer | previous memory、history video | action、future video | 已实现 |
| geometry registers | registers、history video、memory | action、future video | 已实现 |

Memory 先用一个共享1024→3072 adapter，再复用 video expert 每层已有的 K/V projection，避免增加约1.9亿个逐层 adapter 参数；整个 memory module 约3150万参数。`memory.action_read`和`memory.geometry_read`分别控制两个消费者；`action_to_memory_grad_scale`只缩放action loss通过memory K/V回到writer的梯度，forward数值不变。真实action可设1，伪action推荐从0升到0.05–0.2。

没有实现的部分必须明确区分：Joint仍能读取future video，是消融而非reference policy；Base的`infer_joint()`默认用无memory的联合分支生成视频，但返回由`infer_action()`计算的memory-aware action和新state，若显式设置`test_action_with_infer_action=false`则action也不使用memory。部署reference path仍应直接调用`infer_action()`；严格跨训练窗口的episode BPTT仍需新增顺序sampler和分布式状态管理。

## 统一坐标与时间协议

### RGB

原始 360×480 RGB → 等比 resize 到 256×341 → torchvision center crop 到 256×320。

2026-09-05 前 processor 的 `Resize([256,320])` 实际直接拉伸；外层 resize/crop 对已变换图像无效。此前“无形变、裁掉约21px”的描述不符合旧代码。新配置已修复。新旧实验使用不同 run ID。

### 时间

K=1 baseline读取33个原始帧、32步动作，视频每两帧采一帧：17个VAE输入帧、5个latent。公平K=2使用41个原始帧：21个VAE输入帧、6个latent；5个history像素帧覆盖前8步动作，裁剪后仍有32步action和4个future latent。这样K只改变可观测历史与总计算量，不改变预测horizon。

P0 观察到 latent0 对应输入帧0，后续每个 latent 聚合4帧。当前选择每组第一帧作为监督代表点；这是一项明确约定，不意味着 latent 只编码这一帧。

| K | 窗口 | 采样视频中的future监督帧 | 原始 episode 中的相对帧（ratio=2，global stride=1） |
|---|---|---|---|
| 1 | 33 raw / 17 video | 1,5,9,13 | 2,10,18,26 |
| 2 | 41 raw / 21 video | 5,9,13,17 | 10,18,26,34 |

通式：`raw_index = frame_index + (4*j-3)*action_video_freq_ratio*global_sample_stride`，`j=K..T_lat-1`。

标签必须按真实 `dataset_index / episode_index / frame_index` 获取，不能根据请求索引推测重试后实际读到的样本。

RAFT 数组第 `t` 项是原始帧 `t→t+2`，stride 必须等于 `action_video_freq_ratio*global_sample_stride`。进入未来代表帧 `(4j-3)` 的局部 flow 从前一个采样视频帧 `(4j-4)` 开始；K=1 的默认 raw offsets 是 `0,8,16,24`。`motion.include_observed=true` 时，K≥2还加入`j=1..K-1`的历史局部流；公平K=2窗口因此会有5个flow、4个depth监督步。默认消融关闭observed flow，使两者都保持4步。

K=2不是把窗口开头仍当作动作预测起点：5个历史视频帧覆盖8个原始动作步，这8步会从action/proprio/pad target一并裁掉。当前公平协议把窗口从33扩到41帧，因此裁剪后action horizon保持32；旧33帧K2协议得到24步，只适合复现实验，不能作为纯history消融。

## 标签磁盘协议

每个 LeRobot root：

```text
geometry/
  depth/episode_000000.npy
  depth_meta/episode_000000.json
  depth_mask/episode_000000.npy       # 可选；缺省使用 finite/pad/FOV/QC mask
  flow/episode_000000.npy
  flow_meta/episode_000000.json
  flow_mask/episode_000000.npy        # 必须
  fov_mask.png                       # 可选，原图网格
```

所有数组时间长度必须等于 episode RGB 帧数 T，元数据 `num_frames` 也必须一致。读取时只取所需帧，不把整段大数组载入 RAM。

### Depth

`[T,360,480]`，float16/float32，无损保存，数值越大表示越远。元数据至少有：

```json
{"height":360,"width":480,"num_frames":2340,"model":"BaymaxShao/EdGE@e4e16c1-streaming","teacher_mode":"causal_streaming","quality_status":"selected_after_video_ab_test"}
```

depth/mask 按与 RGB 相同的 resize 尺寸和裁剪窗口变换；depth 双线性、mask 最近邻。NaN/Inf 置零并屏蔽；越界 pad 帧不贡献损失。

当前标签由 `BaymaxShao/EdGE@e4e16c1-streaming` 生成。loader 要求 `teacher_mode=causal_streaming` 且 `quality_status=selected_after_video_ab_test`。`depth_qc_path` 仅作为旧标签的兼容覆盖：一旦提供，仍严格检查每个 episode、帧数、mtime 与 passes。

亮度相关性和暗亮分位数深度差是筛查启发式，不能当成几何真值或证明教师正确。阈值仍需结合目视检查与术式标定。

### Flow

`[T,2,Hs,Ws]`，通道为 `(u/W,v/H)`，允许正负值。最后 `stride` 帧没有配对，存零并 mask=0；loader 还会按 episode 边界再次屏蔽。mask 为 `[T,Hs,Ws]`。

当前生成器在原始分辨率估计 RAFT，再把按图像尺寸归一化的 flow 降采样存储。元数据必须明确：

```json
{
  "num_frames":2340,
  "stride":2,
  "normalized_by_image_size":true,
  "flow_height":360,
  "flow_width":480,
  "store_height":120,
  "store_width":160
}
```

当前 RAFT 在原图估计并降采样存储；loader 先恢复到原图网格，再统一 resize/crop，归一化向量乘以实际 resize 尺寸与 crop 尺寸之比：

- `u *= 341/320 = 1.065625`；
- `v *= 256/256 = 1`。

旧文档中的 1.066667 是忽略整数 resize 四舍五入的近似值。若在训练裁剪网格计算，归一化 flow 仅改变存储分辨率时不额外缩放向量。

早期 ercp sidecar 使用 `flow_res/size`，由于向量按两轴各自归一化，loader 可无歧义恢复到原图比例；新标签必须使用 `flow_height/flow_width/store_height/store_width`。stride 不匹配会在加载模型前失败。

## 模型与损失

捕获 MoT layers `[12,14,16,18]` 的 history tokens，投影到 geo_dim=1024。Depth/motion 使用独立 registers 和 blocks；默认8头注意力、8×10空间网格、4个未来监督步。

默认深度readout直接使用EdGE源码中的原生DPT实现，并加载checkpoint中的完整depth head。EdGE head原本读取2048维的frame/global拼接token；当前四层geometry registers为1024维，因此每层经过一个以`[I;I]`初始化、随后可训练的1024→2048 adapter。这样保留EdGE内镜域decoder先验，而不把EdGE encoder、camera/point head或streaming bank带进训练模型。DA3与EdGE的参数键虽然同形，但内部融合/位置编码并不数值等价，因此没有用DA3类冒充EdGE实现；`geometry.head.type=da3`仍保留为严格head消融。

Motion复用EdGE DPT特征解码neck，最后投影重新初始化；它不是预训练光流头，也不复用RAFT decoder。EdGE depth projection为depth/confidence两个logit；depth readout使用depth并丢弃confidence。Motion则构造u/v/confidence三个logit，u/v使用 **linear** 激活并转成`[B,S,2,H,W]`。

EdGE depth head共32.65M参数；四个1024→2048 adapter增加约8.39M参数，每个readout合计约41.04M。Depth和motion使用独立readout。DA3对照head为29.8M且不需要维度adapter，因此比较两者时应同时报告参数量/显存，不能把差异只归因于预训练来源。

```text
L = L_video + lambda_action*L_action
    + lambda_depth*L_depth + lambda_flow*L_flow
```

Depth 用一个 clip 共享的正 affine scale/shift，拟合参数 detach，再算 masked SmoothL1，并可通过 `depth.gradient_weight` 加入对齐后深度的水平/垂直梯度 SmoothL1。K2 reference task 使用0.1。ureter 的0.4权重在每个 sample 的归一化损失后乘，不能只乘到像素 mask 上，否则会被分母抵消。

Flow 默认使用 masked Charbonnier：`((e²+eps²)^alpha-eps^(2alpha))`，reference task 为 `eps=1e-3, alpha=0.5`；仍保留 `loss_type=smooth_l1` 的兼容消融入口。无效像素和 episode 边界屏蔽。所有 rank 的 loss_dict 都保留相同 key，并额外记录 `memory_norm`。

`geometry.head.freeze=true` 会在普通训练模式下持续冻结 DPT。启用 curriculum 后，geometry/head/memory/video/action 分别按 `*_unfreeze_step` 自动切换；将来才解冻的参数会预先注册进 optimizer，而不会出现只改 `requires_grad`、实际没有 optimizer state 的假解冻。启用的分支必须有正 loss 权重及完整标签，否则启动直接报错。启动预检还会检查history长度一致性、capture layer范围/唯一性、register readout尺寸、future监督步数以及memory/head维度可整除性；这些错误不会再等到5B模型加载后才暴露。

γ_p 分别实现在每层 action query 读取 video K/V 与 memory K/V 的边上：forward 恒等，只缩放 action loss 回到视觉表征和记忆写入器的梯度；video 自身 loss与geometry loss的梯度不受影响。不能用整体缩放 action loss 代替，否则 action expert 也会一起被削弱。

## 数据现状与实验计划

2026-09-06 严格预检通过：ercp 201、esophagus 163、ureter 151，共515个 episode 的 depth/flow/meta/mask 均可读取。公平K=2目标形状为video `[3,21,256,320]`、action/proprio `[32,3]`、depth `[4,256,320]`、flow `[4,2,256,320]`。默认按episode留出10%验证集；源视频被切成多个episode时，仍建议进一步按源视频分组以防泄漏。

参考 task 的课程配置：

| 阶段 | 目标 | 当前状态 |
|---|---|---|
| S0 | 冻结 backbone/action，训练 memory/register/readout 对齐 | K2 task: step 0–1999 |
| S1 | 加入 video LoRA，保持 action→video/memory detach | K2 task: step 2000–5999 |
| S2 | 加入 action/proprio，两条 γ_p=0.05 | K2 task: step 6000 起 |
| S3 | 真实机器人动作对齐 | 代码可另配阈值；当前缺真实数据与独立验证 |

增强task可同时报告video/action、`depth_abs_rel`、`depth_rmse`与`flow_epe`；严格baseline的held-out评估只运行action policy并报告loss/action指标。建议消融仍包括无辅助监督、history detach control、不同head初始化/冻结、K=1/2和γ_p；每个对照必须独立run ID并保留resolved config。

## 历史帧是否必要

K=1只有当前锚点图像，不是通常意义上的“历史序列”。K=2才提供5个连续的采样视频帧，使模型能直接观察局部相机运动、组织形变和器械速度；这对单帧存在尺度/运动歧义的内镜场景有明确作用，也让按时间更新memory成为可定义的训练项。Persistent memory解决跨action chunk的长时压缩，K帧history解决当前决策点附近的短时可观测性，两者互补，不能互相替代。公平协议的代价是输入窗口由33增至41帧以及额外history VAE/attention开销，而不是缩短预测horizon。

因此历史帧不是 pipeline 正确运行的必要条件，K=1 应保留为成本更低的强基线；它是否提升策略必须用 K=1/K=2 同预算消融判断。优先观察 action L1/L2、rollout 指标及遮挡/快速运动子集，而不能仅凭辅助 flow EPE 下降得出“控制更好”。
