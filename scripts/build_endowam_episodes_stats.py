#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate `meta/episodes_stats.jsonl` for EndoWAM LeRobot-v2.1 datasets so that
Endo4DWAM's vendored LeRobot loader can read them.

Endo4DWAM's `LeRobotDatasetMetadata.load_metadata()` (v2.1 branch) calls
`load_episodes_stats()` which reads `meta/episodes_stats.jsonl`. The EndoWAM
datasets only ship `meta/stats_gr00t.json` (GR00T format), so the loader fails
with FileNotFoundError. This script computes per-episode min/max/mean/std/count
over the numeric features (action, observation.state) directly from the parquet
files and writes them in the exact schema LeRobot expects:

    {"episode_index": i, "stats": {"<feat>": {"min": [...], "max": [...],
                                              "mean": [...], "std": [...],
                                              "count": [N]}}}

Video / string features are intentionally skipped (LeRobot's own
`compute_episode_stats` skips them when image stats are disabled, and Endo4DWAM's
normalizer only consumes action + state).

The EndoWAM dataset is laid out as:
    <data_root>/<procedure>/rot###/
        meta/{info.json, episodes.jsonl, ...}
        data/chunk-000/episode_xxxxxx.parquet

This script handles both layouts: a flat <procedure>/ that itself holds
meta/info.json (endowam_pseudo_z60), and <procedure>/rot### subdirectories
(the rot-augmented variants).

Run (use an env with pyarrow, e.g. the `fastwam` env created for the upstream repo):
    /home/user/miniconda3/envs/fastwam/bin/python scripts/build_endowam_episodes_stats.py \
        --data_root /mnt/data2/ljs/Endo4DWAM/Endo4DWAM/dataset/endowam_pseudo_z60

    # restrict to some procedures / angles, or overwrite existing files:
    /home/user/miniconda3/envs/fastwam/bin/python scripts/build_endowam_episodes_stats.py \
        --data_root /mnt/data2/ljs/Endo4DWAM/Endo4DWAM/dataset/endowam_pseudo_z60 \
        --procedures ureter --angles rot000,rot045 --overwrite
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Numeric features to compute stats for. Everything else (video/string/index
# columns) is skipped, matching LeRobot's own episode-stats behavior.
NUMERIC_FEATURES = ["action", "observation.state"]


def feature_stats(arr: np.ndarray) -> dict:
    """Mirror lerobot.datasets.compute_stats.get_feature_stats for a [T, D] array."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def episode_index_from_parquet(path: Path, info: dict) -> int:
    """Recover episode_index from the parquet filename using info.json template."""
    # data_path looks like "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    stem = path.stem  # episode_000000
    return int(stem.split("_")[-1])


def process_rot_dir(rot_dir: Path, overwrite: bool) -> int:
    meta_dir = rot_dir / "meta"
    info_path = meta_dir / "info.json"
    if not info_path.exists():
        return 0

    out_path = meta_dir / "episodes_stats.jsonl"
    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path} already exists (use --overwrite)")
        return 0

    info = json.loads(info_path.read_text())
    parquet_paths = sorted((rot_dir / "data").rglob("episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet under {rot_dir / 'data'}")

    lines = []
    for pp in parquet_paths:
        ep_idx = episode_index_from_parquet(pp, info)
        table = pq.read_table(pp, columns=NUMERIC_FEATURES)
        cols = table.to_pydict()
        stats = {}
        for feat in NUMERIC_FEATURES:
            # Each cell is a fixed-size list; stack into [T, D].
            arr = np.stack([np.asarray(v, dtype=np.float32) for v in cols[feat]])
            stats[feat] = feature_stats(arr)
        lines.append({"episode_index": ep_idx, "stats": stats})

    lines.sort(key=lambda x: x["episode_index"])
    if len(lines) != int(info["total_episodes"]):
        raise ValueError(f"Incomplete parquet set under {rot_dir}: {len(lines)} vs {info['total_episodes']}")
    temporary = out_path.with_suffix(".jsonl.tmp")
    with open(temporary, "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    temporary.replace(out_path)
    print(f"  [OK] {out_path.relative_to(rot_dir.parents[1])}  ({len(lines)} episodes)")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",
                    default="/mnt/data2/ljs/Endo4DWAM/Endo4DWAM/dataset/endowam_pseudo_z60")
    ap.add_argument("--procedures", default="",
                    help="comma list to restrict procedures, e.g. 'ureter'; empty = all")
    ap.add_argument("--angles", default="",
                    help="comma list to restrict angles, e.g. 'rot000,rot045'; empty = all")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if episodes_stats.jsonl already exists")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    proc_filter = {s for s in args.procedures.split(",") if s}
    angle_filter = {s for s in args.angles.split(",") if s}

    procedures = sorted(d for d in data_root.iterdir()
                        if d.is_dir() and not d.name.startswith("_")
                        and (not proc_filter or d.name in proc_filter))
    if not procedures:
        raise RuntimeError(f"No procedure dirs under {data_root}")

    total = 0
    n_roots = 0
    for proc in procedures:
        # Two layouts are supported:
        #   flat     <data_root>/<procedure>/meta/info.json          (endowam_pseudo_z60)
        #   angled   <data_root>/<procedure>/rot###/meta/info.json   (…_rot45 variants)
        if (proc / "meta" / "info.json").is_file():
            if angle_filter:
                print(f"[{proc.name}] flat layout, ignoring --angles")
            print(f"[{proc.name}] flat layout")
            total += process_rot_dir(proc, args.overwrite)
            n_roots += 1
            continue

        rot_dirs = sorted(d for d in proc.iterdir()
                          if d.is_dir() and d.name.startswith("rot")
                          and (not angle_filter or d.name in angle_filter))
        if not rot_dirs:
            print(f"[{proc.name}] [WARN] neither meta/info.json nor rot### dirs; skipped")
            continue
        print(f"[{proc.name}] {len(rot_dirs)} angle dirs")
        for rot in rot_dirs:
            total += process_rot_dir(rot, args.overwrite)
            n_roots += int((rot / "meta/info.json").is_file())

    if n_roots == 0:
        raise RuntimeError(f"No LeRobot roots found under {data_root}")
    print(f"\nDone. wrote episodes_stats.jsonl for {total}/{n_roots} roots under {data_root}")


if __name__ == "__main__":
    main()
