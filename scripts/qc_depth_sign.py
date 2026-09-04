"""Per-clip quality gate for the DA3 pseudo-depth labels.

Endoscopy gives an objective check on depth *semantics*: illumination falls off
with distance, so bright means near and a correct depth map must put larger
depth values on darker pixels.

Two statistics are computed, because the obvious one is misleading:

  pearson   corr(depth, brightness) over all pixels. Must be negative. This is
            what was used to catch DA3-GIANT inverting ercp, but it is NOT
            comparable across procedures: its magnitude is attenuated by the
            brightness contrast of the scene. Measured brightness std is 35.2
            for esophagus, 27.2 for ureter and 17.8 for ercp, so the same depth
            quality scores very differently.

  decile    (median depth of the darkest 10% - median depth of the brightest
            10%) / (p99 - p1 of depth). Positive means far pixels are dark, i.e.
            correct. Only the two extremes are used and the result is divided by
            the depth range, so it is far less sensitive to overall contrast and
            is the statistic the gate should key on.

Usage:
    python scripts/qc_depth_sign.py --data_root <dataset root>
    python scripts/qc_depth_sign.py --calibrate          # threshold sweep per procedure
    python scripts/qc_depth_sign.py --out qc_depth.json  # per-episode records
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np


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
    if not frames:
        return None

    pearson, decile, contrast = [], [], []
    for i, rgb in frames.items():
        d = np.asarray(depth[i], dtype=np.float32)
        g = brightness_like(rgb, d.shape)
        dv, gv = d.ravel(), g.ravel()
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
        video_dir = root / "videos" / "chunk-000" / "observation.images.endoscope"
        if not depth_dir.is_dir():
            continue
        for depth_path in sorted(depth_dir.glob("episode_*.npy")):
            video_path = video_dir / f"{depth_path.stem}.mp4"
            if not video_path.is_file():
                continue
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
        print(f"[ERR] no depth found under {data_root}. Generation may still be running.")
        return

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
        print("\n  按术式选阈值时注意: pearson 跨术式不可比（被亮度对比度衰减），decile 才是可比的。")

    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=2))
        print(f"\n写出 {len(records)} 条记录 -> {args.out}")


if __name__ == "__main__":
    main()
