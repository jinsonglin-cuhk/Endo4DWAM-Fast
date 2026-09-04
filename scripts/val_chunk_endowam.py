"""
Endo4DWAM 离线开环 chunk 评测（EndoWAM 内窥镜数据集）。

对齐 EndoWAM/tests/val_chunk_endowam_casual.py 的评测方式：加载一个训练好的
Endo4DWAM checkpoint，对某个 LeRobot episode 做开环（GT state / 真实首帧锚定）
分段推理——每段只“执行”`--execution_horizon` 步就丢弃剩余预测、从真实数据
重新起播（receding-horizon rollout），累积整段 episode 的预测/GT action 曲线，
保存对比图，并计算与训练同款的 loss。

复用点（不重复实现）：
  - `endo4dwam.trainer.Wan22Trainer._to_batched_eval_sample`（staticmethod）：把
    `RobotVideoDataset[idx]` 的单样本 dict 打包成 `model.infer()` /
    `model.training_loss()` 需要的 batch-of-1 张量，和训练/评估用的完全一致。
  - `model.infer_action` / `model.infer` / `model.training_loss`：与
    `experiments/robotwin/endo4dwam_policy/deploy_policy.py`（闭环部署）和
    `Wan22Trainer.evaluate()`（训练期评估）完全相同的模型接口。
  - action 反归一化：照搬 `Wan22Trainer.evaluate()` 里
    `processor.action_state_merger.backward` + `processor.normalizer.backward`
    的组合逻辑（`ConcatLeftAlign` 只有和 state 一起走 backward 才能正确按
    per-key 统计量还原物理量）。
  - 数据/模型构建全部走 Hydra，用 `task=<task_name>` 复用训练时的
    `configs/task/*.yaml`，避免像 EndoWAM 脚本那样手工拼 vla_data 配置。

与 EndoWAM 参考脚本的差异：
  - Endo4DWAM 是 video+action 联合 flow-matching 模型，不是离散分类模型，所以
    “accuracy”是把反归一化后的连续预测四舍五入到 {-1,0,+1} 再比较（该数据集
    的动作本来就是离散方向指令，反归一化后应恰好落在这三个值附近）。
  - 额外算了训练同款的 diffusion loss（loss_video/loss_action，来自
    `model.training_loss`），以及若干窗口的 `model.infer()` 联合视频推理
    （预测视频 vs VAE 重建 vs GT 拼接 mp4 + PSNR/SSIM），这是 EndoWAM 因果分类
    模型没有的部分。

用法：
    conda activate fastwam   （机器上现有的 env，上游 FastWAM 时期建的）
    cd /mnt/data2/ljs/Endo4DWAM/Endo4DWAM-Fast
    python scripts/val_chunk_endowam.py \
        --ckpt runs/endowam_uncond_lora/fastwam_uncond_lora_endowam_rot45/checkpoints/weights/step_080000.pt \
        --task endowam_uncond_1cam_1e-4 \
        --dataset_root /mnt/data2/ljs/Endo4DWAM/Endo4DWAM/dataset/endowam_pseudo_z60/esophagus \
        --episode 144 --execution_horizon 8 --max_windows 4000 --num_video_saves 0 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from endo4dwam.trainer import Wan22Trainer  # noqa: E402
from endo4dwam.utils import misc  # noqa: E402
from endo4dwam.utils.video_io import save_mp4  # noqa: E402
from endo4dwam.utils.video_metrics import (  # noqa: E402
    pil_frames_to_video_tensor,
    video_psnr,
    video_ssim,
)

DEFAULT_DIM_LABELS = ["m2_target_rpm", "m3_target_rpm", "m4_target_rpm"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--ckpt", type=str,
        default="runs/endowam_uncond_lora/fastwam_uncond_lora_endowam_rot45/checkpoints/weights/step_080000.pt",
        help="model.save_checkpoint() 保存的 .pt 权重文件",
    )
    p.add_argument("--task", type=str, default="endowam_uncond_1cam_1e-4",
                    help="configs/task/<task>.yaml，用于复现训练时的 model/data 配置")
    p.add_argument("--dataset_stats", type=str, default=None,
                    help="dataset_stats.json 路径；缺省从 ckpt 的上级 run 目录自动查找")
    p.add_argument("--dataset_root", type=str,
                    default="/mnt/data2/ljs/Endo4DWAM/Endo4DWAM/dataset/endowam_pseudo_z60/ureter",
                    help="单个 LeRobot root（procedure/rotXXX），只在这一个 root 内选 episode")
    p.add_argument("--episode", type=int, default=0, help="root 内的 episode 序号（0-based）")
    p.add_argument("--execution_horizon", type=int, default=8,
                    help="每段只执行/打分前 N 步就丢弃剩余预测、从真实数据重新起播（receding horizon）")
    p.add_argument("--max_windows", type=int, default=40,
                    help="最多跑多少个窗口（<=0 表示跑完整个 episode，可能很慢）")
    p.add_argument("--num_video_saves", type=int, default=2,
                    help="额外做完整 model.infer()（视频+动作）并保存对比视频/算 loss 的窗口数")
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--text_cfg_scale", type=float, default=1.0)
    p.add_argument("--action_cfg_scale", type=float, default=1.0, help="仅用于 --num_video_saves 的联合推理")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dim_labels", type=str, default=",".join(DEFAULT_DIM_LABELS))
    p.add_argument("--out_dir", type=str, default=None,
                    help="缺省为 evaluate_results/endowam_offline/<ckpt_tag>/<dataset_tag>_ep<episode>_<timestamp>")
    p.add_argument("--json_out", type=str, default=None, help="缺省写到 <out_dir>/summary.json")
    return p.parse_args()


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(f"Unsupported mixed_precision: {mixed_precision}")
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    key = _normalize_mixed_precision(mixed_precision)
    if key == "no":
        return torch.float32
    if key == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 2 < len(parts):
            return f"{parts[idx + 1]}_{parts[idx + 2]}"
    return ckpt_path.stem


def _resolve_dataset_stats_path(ckpt_path: Path, explicit: Optional[str]) -> Path:
    if explicit:
        resolved = Path(explicit).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"--dataset_stats not found: {resolved}")
        return resolved
    for parent in list(ckpt_path.resolve().parents)[:4]:
        candidate = parent / "dataset_stats.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not auto-locate dataset_stats.json near {ckpt_path}. Pass --dataset_stats explicitly."
    )


def denormalize_action_chunk(processor, action_btd: torch.Tensor, proprio_btd: torch.Tensor) -> np.ndarray:
    """物理空间 action，[T, D]。照搬 Wan22Trainer.evaluate() 的 denorm 组合逻辑：

    ConcatLeftAlign.backward 需要 action 和 state 一起走，才能按每个子键各自的
    归一化统计量正确切分/还原（这里只有一个子键 "default"，但流程要保持一致）。
    """
    if action_btd.ndim == 2:
        action_btd = action_btd.unsqueeze(0)
    if proprio_btd.ndim == 2:
        proprio_btd = proprio_btd.unsqueeze(0)
    action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)
    proprio_btd = proprio_btd.detach().to(device="cpu", dtype=torch.float32)

    batch = {"action": action_btd, "state": proprio_btd}
    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)

    action_meta = processor.shape_meta["action"]
    state_meta = processor.shape_meta["state"]
    merged = {
        "action": {m["key"]: batch["action"][m["key"]].squeeze(0) for m in action_meta},
        "state": {m["key"]: batch["state"][m["key"]].squeeze(0) for m in state_meta},
    }
    merged = processor.action_state_merger.forward(merged)
    return merged["action"].unsqueeze(0)[0].numpy()  # [T, D]


def plot_chunk_results(gt: np.ndarray, pred: np.ndarray, exp_tag: str, save_path: Path,
                        dim_labels: list[str], execution_horizon: int) -> None:
    n_dim = min(gt.shape[1], pred.shape[1], len(dim_labels))
    chunk_indices = np.arange(0, min(len(gt), len(pred)), max(int(execution_horizon), 1))

    fig, axes = plt.subplots(n_dim, 1, figsize=(12, 3 * n_dim), squeeze=False)
    fig.suptitle(f"{exp_tag}\nEndo4DWAM chunk validation (receding-horizon, GT-anchored)", fontsize=14)
    for i in range(n_dim):
        ax = axes[i, 0]
        ax.plot(gt[:, i], "b-", label="GT", linewidth=2)
        ax.plot(pred[:, i], "orange", linestyle="-", label="Pred", linewidth=2)
        ax.plot(chunk_indices, pred[chunk_indices, i], "ro", markersize=4, zorder=3)
        ax.set_title(dim_labels[i])
        ax.grid(True, alpha=0.5)
        if i == 0:
            ax.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"已保存对比图: {save_path}")


def run_detailed_window(model, batched: dict, processor, args, out_dir: Path, window_idx: int) -> dict:
    """在一个窗口上做完整联合推理（视频+动作）+ 训练同款 loss，照搬
    Wan22Trainer.evaluate() 的对应逻辑，返回一份可 JSON 序列化的 metrics dict。
    """
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=model.torch_dtype):
        loss_total, loss_dict = model.training_loss(batched)
    loss_total = float(loss_total.detach().float().item())

    video0 = batched["video"][0]  # [3, T, H, W] in (-1, 1)
    action = batched["action"][0] if batched.get("action") is not None else None
    proprio_full = batched["proprio"][0] if batched.get("proprio") is not None else None  # [T, d]
    proprio0 = proprio_full[0] if proprio_full is not None else None  # [d]
    input_image = video0[:, 0].unsqueeze(0)
    _, num_frames, _, _ = video0.shape

    infer_kwargs = {
        "input_image": input_image,
        "num_frames": num_frames,
        "action": action,
        "action_horizon": batched["action_horizon"],
        "proprio": proprio0,
        "prompt": None,
        "context": batched["context"][0],
        "context_mask": batched["context_mask"][0],
        "text_cfg_scale": args.text_cfg_scale,
        "action_cfg_scale": args.action_cfg_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "tiled": False,
    }
    with torch.no_grad():
        pred = model.infer(**infer_kwargs)
    pred_video = pred["video"]

    pred_video_tensor = pil_frames_to_video_tensor(pred_video)
    gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

    gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
    with torch.no_grad():
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
    vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

    metrics = {
        "loss_total": loss_total,
        "loss_video": float(loss_dict["loss_video"]),
        "loss_action": float(loss_dict["loss_action"]),
        "psnr_rollout_vs_gt": video_psnr(pred=pred_video_tensor, target=gt_video_tensor),
        "ssim_rollout_vs_gt": video_ssim(pred=pred_video_tensor, target=gt_video_tensor),
        "psnr_decode_vs_gt": video_psnr(pred=vae_video_tensor, target=gt_video_tensor),
        "ssim_decode_vs_gt": video_ssim(pred=vae_video_tensor, target=gt_video_tensor),
    }

    if action is not None and proprio_full is not None:
        pred_action_denorm = denormalize_action_chunk(processor, pred["action"], proprio_full)
        gt_action_denorm = denormalize_action_chunk(processor, action, proprio_full)
        diff = pred_action_denorm - gt_action_denorm
        metrics["action_l1"] = float(np.abs(diff).mean())
        metrics["action_l2"] = float(np.square(diff).mean())

    stitched = torch.cat([pred_video_tensor, vae_video_tensor, gt_video_tensor], dim=2).contiguous()
    frames = []
    for t in range(stitched.shape[1]):
        frame = (stitched[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
        from PIL import Image
        frames.append(Image.fromarray(frame))
    video_path = out_dir / f"window_{window_idx:06d}_pred_vae_gt.mp4"
    save_mp4(frames, str(video_path), fps=8)
    metrics["video_path"] = str(video_path)
    return metrics


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    dataset_stats_path = _resolve_dataset_stats_path(ckpt_path, args.dataset_stats)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    ckpt_tag = _resolve_ckpt_tag(ckpt_path)
    dataset_tag = "_".join(dataset_root.parts[-2:])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "evaluate_results" / "endowam_offline" / ckpt_tag
        / f"{dataset_tag}_ep{args.episode:06d}_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    dim_labels = [s.strip() for s in args.dim_labels.split(",") if s.strip()]
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    print(f"[config] ckpt={ckpt_path}")
    print(f"[config] dataset_stats={dataset_stats_path}")
    print(f"[config] dataset_root={dataset_root} episode={args.episode}")
    print(f"[config] out_dir={out_dir}")

    misc.register_work_dir(str(out_dir))

    configs_root = str(PROJECT_ROOT / "configs")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=configs_root):
        cfg = compose(config_name="train", overrides=[f"task={args.task}"])

    model_dtype = _mixed_precision_to_model_dtype(cfg.mixed_precision)
    print(f"[model] building Endo4DWAM (dtype={model_dtype}, device={device}) ...")
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)

    payload = torch.load(str(ckpt_path), map_location="cpu")
    if "mot" not in payload:
        raise ValueError(f"Checkpoint missing `mot` key: {ckpt_path}")
    model_keys = set(model.mot.state_dict().keys())
    ckpt_keys = set(payload["mot"].keys())
    missing, unexpected = model_keys - ckpt_keys, ckpt_keys - model_keys
    if missing or unexpected:
        print(f"[warn] load_checkpoint key mismatch: missing={len(missing)} unexpected={len(unexpected)} "
              "(检查 --task 的 LoRA 配置是否与训练时一致)")
    model.load_checkpoint(str(ckpt_path))
    model = model.to(device).eval()

    print("[data] building single-episode dataset ...")
    # 用单一 root + 覆盖 stats 路径重建训练时的 24-root mixture 配置（其它字段如
    # shape_meta / action_video_freq_ratio / processor / norm 不变）。像
    # `runtime.build_datasets` 对 val 数据集那样，通过 instantiate() 的关键字参数
    # 覆盖，而不是直接改 cfg 属性——Hydra compose 出来的 DictConfig 是 struct 模式，
    # 只能覆盖已存在的键，`pretrained_norm_stats` 在 yaml 里没有默认值、不在 struct 里。
    dataset = instantiate(
        cfg.data.train,
        dataset_dirs=[str(dataset_root)],
        is_training_set=False,
        val_set_proportion=0.0,
        pretrained_norm_stats=str(dataset_stats_path),
    )
    dataset.lerobot_dataset.processor.eval()
    processor = dataset.lerobot_dataset.processor

    ep_from_all = dataset.lerobot_dataset.episode_data_index["from"]
    ep_to_all = dataset.lerobot_dataset.episode_data_index["to"]
    if not (0 <= args.episode < len(ep_from_all)):
        raise ValueError(f"--episode {args.episode} out of range [0, {len(ep_from_all)})")
    ep_from = int(ep_from_all[args.episode].item())
    ep_to = int(ep_to_all[args.episode].item())
    num_frames = dataset.num_frames
    action_horizon = num_frames - 1
    print(f"[data] episode {args.episode}: frames [{ep_from}, {ep_to}) = {ep_to - ep_from} raw frames, "
          f"window={num_frames} raw frames -> action_horizon={action_horizon}")

    all_starts = list(range(ep_from, ep_to - num_frames + 1, args.execution_horizon))
    if not all_starts:
        raise ValueError(
            f"Episode too short for one window: needs >= {num_frames} raw frames, has {ep_to - ep_from}."
        )
    if args.max_windows > 0 and len(all_starts) > args.max_windows:
        print(f"[data] {len(all_starts)} windows available, capping to --max_windows={args.max_windows} "
              f"(covers first {args.max_windows * args.execution_horizon} of {ep_to - ep_from} raw frames)")
        starts = all_starts[: args.max_windows]
    else:
        starts = all_starts

    save_video_at = set()
    if args.num_video_saves > 0 and starts:
        save_idx = np.linspace(0, len(starts) - 1, num=min(args.num_video_saves, len(starts)), dtype=int)
        save_video_at = set(int(i) for i in save_idx)

    exp_tag = f"{ckpt_tag}_{dataset_tag}_ep{args.episode:06d}"
    stitched_pred, stitched_gt = [], []
    correct_per_axis = np.zeros(len(dim_labels), dtype=np.int64)
    total_per_axis = np.zeros(len(dim_labels), dtype=np.int64)
    detailed_metrics = []

    print(f"[run] {len(starts)} windows, execution_horizon={args.execution_horizon}, "
          f"num_inference_steps={args.num_inference_steps}, {len(save_video_at)} detailed(video+loss) windows")

    for w, start_idx in enumerate(starts):
        try:
            raw_sample = dataset[start_idx]
            batched = Wan22Trainer._to_batched_eval_sample(raw_sample)

            first_frame = batched["video"][:, :, 0].to(device=model.device, dtype=model.torch_dtype)
            proprio0 = batched["proprio"][:, 0, :].to(device=model.device, dtype=model.torch_dtype)

            t0 = time.time()
            with torch.no_grad():
                pred = model.infer_action(
                    prompt=None,
                    input_image=first_frame,
                    action_horizon=action_horizon,
                    proprio=proprio0,
                    context=batched["context"][0],
                    context_mask=batched["context_mask"][0],
                    text_cfg_scale=args.text_cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed,
                )
            dt_ms = (time.time() - t0) * 1000.0

            pred_action_norm = pred["action"].unsqueeze(0)  # [1, T, D]
            gt_action_norm = batched["action"]  # [1, T, D]
            proprio_full = batched["proprio"]  # [1, T, d]

            pred_phys = denormalize_action_chunk(processor, pred_action_norm, proprio_full)
            gt_phys = denormalize_action_chunk(processor, gt_action_norm, proprio_full)

            n_exec = min(args.execution_horizon, pred_phys.shape[0], gt_phys.shape[0])
            print(f"window {w} (raw_idx={start_idx}): infer {dt_ms:.1f} ms, executing {n_exec} steps")

            for t in range(n_exec):
                stitched_pred.append(pred_phys[t])
                stitched_gt.append(gt_phys[t])
                pred_cls = np.rint(np.clip(pred_phys[t], -1.0, 1.0)).astype(np.int64)
                gt_cls = np.rint(np.clip(gt_phys[t], -1.0, 1.0)).astype(np.int64)
                n = min(len(pred_cls), len(dim_labels))
                correct_per_axis[:n] += (pred_cls[:n] == gt_cls[:n]).astype(np.int64)
                total_per_axis[:n] += 1

            if w in save_video_at:
                print(f"window {w}: running detailed model.infer() + training_loss() ...")
                metrics = run_detailed_window(model, batched, processor, args, out_dir, w)
                metrics["window"] = w
                metrics["raw_idx"] = start_idx
                detailed_metrics.append(metrics)
                print(f"  loss_total={metrics['loss_total']:.4f} loss_video={metrics['loss_video']:.4f} "
                      f"loss_action={metrics['loss_action']:.4f} "
                      f"psnr_rollout_vs_gt={metrics['psnr_rollout_vs_gt']:.2f}")
        except Exception as e:  # noqa: BLE001
            print(f"window {w} (raw_idx={start_idx}) 出错: {e}")
            traceback.print_exc()
            break

    if not stitched_pred:
        print("没有成功执行任何窗口，退出。")
        return

    gt_arr = np.array(stitched_gt)
    pred_arr = np.array(stitched_pred)
    plot_path = out_dir / f"{exp_tag}_actions.png"
    plot_chunk_results(gt_arr, pred_arr, exp_tag, plot_path, dim_labels, args.execution_horizon)

    print("\n=== Per-axis discrete accuracy (denorm, rounded to {-1,0,+1}) ===")
    per_axis = {}
    for i, label in enumerate(dim_labels):
        tot = int(total_per_axis[i])
        acc = float(correct_per_axis[i]) / max(tot, 1)
        per_axis[label] = {"correct": int(correct_per_axis[i]), "total": tot, "acc": acc}
        print(f"  {label}: {correct_per_axis[i]}/{tot} = {acc:.4f}")
    overall = float(correct_per_axis.sum()) / max(int(total_per_axis.sum()), 1)
    print(f"  overall: {int(correct_per_axis.sum())}/{int(total_per_axis.sum())} = {overall:.4f}")

    diff = pred_arr[:, : gt_arr.shape[1]] - gt_arr
    action_l1_rollout = float(np.abs(diff).mean())
    action_l2_rollout = float(np.square(diff).mean())
    print(f"  rollout action_l1={action_l1_rollout:.4f} action_l2={action_l2_rollout:.4f}")

    if detailed_metrics:
        mean_loss_total = float(np.mean([m["loss_total"] for m in detailed_metrics]))
        mean_loss_video = float(np.mean([m["loss_video"] for m in detailed_metrics]))
        mean_loss_action = float(np.mean([m["loss_action"] for m in detailed_metrics]))
        print(f"\n=== Detailed windows (n={len(detailed_metrics)}) ===")
        print(f"  mean loss_total={mean_loss_total:.4f} loss_video={mean_loss_video:.4f} "
              f"loss_action={mean_loss_action:.4f}")

    summary = {
        "ckpt": str(ckpt_path),
        "task": args.task,
        "dataset_root": str(dataset_root),
        "episode": args.episode,
        "execution_horizon": args.execution_horizon,
        "num_windows": len(starts),
        "num_windows_available": len(all_starts),
        "per_axis": per_axis,
        "overall_acc": overall,
        "rollout_action_l1": action_l1_rollout,
        "rollout_action_l2": action_l2_rollout,
        "detailed_windows": detailed_metrics,
        "plot": str(plot_path),
    }
    json_out = Path(args.json_out) if args.json_out else out_dir / "summary.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n已保存 JSON 摘要: {json_out}")


if __name__ == "__main__":
    main()
