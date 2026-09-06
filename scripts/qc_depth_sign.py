"""Heuristic quality gate for original-grid DA3 pseudo-depth labels.

Brightness/depth correlation and dark-vs-bright decile separation can identify
suspicious labels, but reflectance, lighting, occlusion and exposure also affect
these scores. They are not geometric ground truth or a calibrated cross-procedure
quality scale. Inspect examples and calibrate thresholds before using the gate.

Usage:
    python scripts/qc_depth_sign.py --data_root <dataset root> --out qc_depth.json
    python scripts/qc_depth_sign.py --calibrate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image


def brightness_like(frame_rgb: np.ndarray, shape_hw) -> np.ndarray:
    """Grey-scale brightness resampled (nearest) onto the depth grid."""
    grey = frame_rgb.mean(axis=2).astype(np.float32)
    h, w = shape_hw
    if grey.shape != (h, w):
        yi = np.linspace(0, grey.shape[0] - 1, h).astype(int)
        xi = np.linspace(0, grey.shape[1] - 1, w).astype(int)
        grey = grey[yi][:, xi]
    return grey


def score_episode(depth_path: Path, video_path: Path, num_samples: int):
    depth = np.load(depth_path, mmap_mode="r")
    n = depth.shape[0]
    if n == 0 or num_samples < 1:
        raise ValueError("Depth must be nonempty and num_samples positive")
    idxs = sorted(set(np.linspace(0, n - 1, num_samples).astype(int).tolist()))

    container = av.open(str(video_path))
    frames = {}
    try:
        for i, frame in enumerate(container.decode(video=0)):
            if i in set(idxs):
                frames[i] = frame.to_ndarray(format="rgb24")
            if i >= idxs[-1]:
                break
    finally:
        container.close()
    if len(frames) != len(idxs):
        raise ValueError(f"Video shorter than depth labels: {video_path}")

    fov_path = depth_path.parent.parent / "fov_mask.png"
    fov = np.asarray(Image.open(fov_path).convert("L")) > 0 if fov_path.exists() else None
    pearson, decile, contrast = [], [], []
    for i, rgb in frames.items():
        d = np.asarray(depth[i], dtype=np.float32)
        if rgb.shape[:2] != d.shape:
            raise ValueError(f"Depth QC requires original RGB grid: {depth_path}")
        g = brightness_like(rgb, d.shape)
        valid = np.isfinite(d) & (g > 5) & (g < 250)
        if fov is not None:
            if fov.shape != d.shape:
                raise ValueError(f"FOV mask shape mismatch: {fov_path}")
            valid &= fov
        dv, gv = d[valid], g[valid]
        if dv.size < 100:
            continue
        if dv.std() < 1e-8 or gv.std() < 1e-8:
            continue
        pearson.append(float(np.corrcoef(dv, gv)[0, 1]))
        contrast.append(float(gv.std()))

        lo, hi = np.percentile(gv, 10), np.percentile(gv, 90)
        dark, bright = dv[gv <= lo], dv[gv >= hi]
        rng = float(np.percentile(dv, 99) - np.percentile(dv, 1))
        if dark.size and bright.size and rng > 1e-8:
            decile.append(float((np.median(dark) - np.median(bright)) / rng))

    if not pearson:
        return None
    return {
        "episode": depth_path.stem,
        "num_frames": int(n),
        "depth_mtime_ns": depth_path.stat().st_mtime_ns,
        "num_scored": len(pearson),
        "pearson": float(np.mean(pearson)),
        "decile": float(np.mean(decile)) if decile else float("nan"),
        "brightness_std": float(np.mean(contrast)),
    }


def iter_roots(data_root: Path, proc_filter):
    for proc in sorted(d for d in data_root.iterdir()
                       if d.is_dir() and not d.name.startswith("_")
                       and (not proc_filter or d.name in proc_filter)):
        if (proc / "meta" / "info.json").is_file():
            yield proc.name, proc
            continue
        for rot in sorted(d for d in proc.iterdir() if d.is_dir() and d.name.startswith("rot")):
            yield proc.name, rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/mnt/data2/ljs/Endo4DWAM/Endo4DWAM/dataset/endowam_pseudo_z60")
    ap.add_argument("--procedures", default="")
    ap.add_argument("--num-samples", type=int, default=12, help="frames scored per episode")
    ap.add_argument("--decile-thresh", type=float, default=0.05,
                    help="minimum decile separation for a clip to pass the gate")
    ap.add_argument("--calibrate", action="store_true", help="sweep thresholds per procedure")
    ap.add_argument("--out", default="", help="write per-episode records to this json")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    proc_filter = {s for s in args.procedures.split(",") if s}

    records = []
    for proc_name, root in iter_roots(data_root, proc_filter):
        depth_dir = root / "geometry" / "depth"
        info = json.loads((root / "meta/info.json").read_text())
        if not depth_dir.is_dir():
            continue
        for depth_path in sorted(depth_dir.glob("episode_*.npy")):
            ep = int(depth_path.stem.split("_")[-1])
            video_path = root / info["video_path"].format(
                episode_chunk=ep // int(info.get("chunks_size", 1000)),
                episode_index=ep, video_key="observation.images.endoscope")
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            rec = score_episode(depth_path, video_path, args.num_samples)
            if rec is None:
                continue
            rec["procedure"] = proc_name
            rec["root"] = str(root)
            rec["passes"] = bool(rec["decile"] > args.decile_thresh)
            records.append(rec)
            print(f"  {proc_name:10s} {rec['episode']:18s} pearson={rec['pearson']:+.3f} "
                  f"decile={rec['decile']:+.3f} bstd={rec['brightness_std']:5.1f} "
                  f"{'PASS' if rec['passes'] else 'REJECT'}")

    if not records:
        raise RuntimeError(f"No depth scored under {data_root}")

    print(f"\n{'procedure':12s} {'n':>4s} {'pearson':>9s} {'decile':>9s} {'bstd':>7s} {'pass':>9s}")
    procs = sorted({r["procedure"] for r in records})
    for p in procs:
        rs = [r for r in records if r["procedure"] == p]
        npass = sum(r["passes"] for r in rs)
        print(f"{p:12s} {len(rs):4d} {np.mean([r['pearson'] for r in rs]):+9.3f} "
              f"{np.mean([r['decile'] for r in rs]):+9.3f} "
              f"{np.mean([r['brightness_std'] for r in rs]):7.1f} {npass}/{len(rs):<4d}")

    if args.calibrate:
        print("\n阈值扫描（decile separation）:")
        header = "  thresh " + "".join(f"{p:>12s}" for p in procs)
        print(header)
        for th in (0.0, 0.02, 0.05, 0.08, 0.12, 0.20):
            row = f"  {th:6.2f} "
            for p in procs:
                rs = [r for r in records if r["procedure"] == p]
                k = sum(1 for r in rs if r["decile"] > th)
                row += f"{k}/{len(rs)}".rjust(12)
            print(row)
        print("\n  两项分数都是启发式；跨术式阈值需结合目视检查标定，不能直接解释为深度准确率。")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(records, indent=2))
        print(f"\n写出 {len(records)} 条记录 -> {args.out}")


if __name__ == "__main__":
    main()
