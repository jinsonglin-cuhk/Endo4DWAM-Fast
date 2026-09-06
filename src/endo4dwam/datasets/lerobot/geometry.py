"""Strict, frame-indexed auxiliary labels for the endoscope training pipeline.

The reader implements the protocol emitted by the current label generators:
EdGE depth on the native RGB grid and RAFT ``t -> t + stride`` flow stored on a
smaller grid but normalised by the native image size.  Missing or temporally
ambiguous labels fail during dataset construction, before the 5B model loads.
"""
from pathlib import Path
import json

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from ..dataset_utils import CenterCrop


class GeometryLabels:
    def __init__(self, datasets, cfg, *, source_hw, target_hw, num_frames,
                 video_stride, sample_stride):
        self.roots = [Path(ds.root).resolve() for ds in datasets]
        self.source_hw = tuple(source_hw)
        self.target_hw = tuple(target_hw)
        self.k = int(cfg["num_history_latent_frames"])
        latent_steps = ((num_frames - 1) // video_stride) // 4 + 1
        self.steps = latent_steps - self.k
        if self.k < 1 or self.steps <= 0 or self.steps != int(cfg["num_supervised_steps"]):
            raise ValueError("geometry.num_supervised_steps must equal T_latent - K > 0")
        # Depth representatives in sampled-video coordinates: 1,5,9,13 for K=1.
        self.depth_offsets = np.array([4 * j - 3 for j in range(self.k, latent_steps)])
        self.depth_offsets *= int(video_stride) * int(sample_stride)
        # A flow label at raw index t describes t -> t + label_stride.  For the
        # representative at sampled index (4j-3), its one-video-step incoming
        # flow therefore starts at (4j-4).  With K>1, observed flow optionally
        # supervises the history transitions j=1..K-1 as well.
        self.flow_include_observed = bool(cfg.get("flow_include_observed", False))
        flow_start_j = 1 if self.flow_include_observed else self.k
        self.flow_offsets = np.array([4 * j - 4 for j in range(flow_start_j, latent_steps)])
        self.flow_offsets *= int(video_stride) * int(sample_stride)
        self.flow_steps = len(self.flow_offsets)
        self.expected_flow_stride = int(video_stride) * int(sample_stride)
        self.modalities = [key for key in ("depth", "flow") if cfg.get(key, False)]
        if not self.modalities:
            raise ValueError("Geometry enabled but neither depth nor flow is enabled")
        self.depth_weights = dict(cfg.get("depth_procedure_weights", {}))
        self.qc = {}
        self.use_legacy_depth_qc = False
        if "depth" in self.modalities:
            qc_path = cfg.get("depth_qc_path")
            if qc_path:
                self.use_legacy_depth_qc = True
                for record in json.loads(Path(qc_path).read_text()):
                    self.qc[(str(Path(record["root"]).resolve()), record["episode"])] = record
        args = {"img_h": self.target_hw[0], "img_w": self.target_hw[1]}
        self.crop = CenterCrop(args)
        self.entries = {}
        self.fov = {}
        passing_depth = 0
        for dataset_idx, ds in enumerate(datasets):
            root = self.roots[dataset_idx]
            fov_path = root / "geometry/fov_mask.png"
            if fov_path.exists():
                fov = np.asarray(Image.open(fov_path).convert("L")) > 0
                if fov.shape != self.source_hw:
                    raise ValueError(f"FOV mask shape mismatch: {fov_path}")
                self.fov[dataset_idx] = torch.from_numpy(fov.copy()).float()
            episodes = ds.episodes if ds.episodes is not None else list(ds.meta.episodes)
            for ep in episodes:
                stem = f"episode_{ep:06d}"
                length = int(ds.meta.episodes[ep]["length"])
                for kind in self.modalities:
                    base = root / "geometry"
                    path = base / kind / f"{stem}.npy"
                    meta_path = base / f"{kind}_meta" / f"{stem}.json"
                    if not path.is_file() or not meta_path.is_file():
                        raise FileNotFoundError(f"Missing {kind} labels or metadata: {path}")
                    meta = json.loads(meta_path.read_text())
                    array = np.load(path, mmap_mode="r")
                    if len(array) != length or meta.get("num_frames") != length:
                        raise ValueError(f"Label/RGB frame count mismatch: {path}")
                    if kind == "depth":
                        if array.shape != (length, *self.source_hw):
                            raise ValueError(f"Depth must be on the original RGB grid: {path}")
                        if (meta.get("height"), meta.get("width")) != self.source_hw:
                            raise ValueError(f"Depth metadata grid mismatch: {meta_path}")
                        if self.use_legacy_depth_qc:
                            record = self.qc.get((str(root), stem))
                            if record is None or record.get("num_frames") != length:
                                raise ValueError(f"Missing/mismatched depth QC for {path}")
                            if record.get("depth_mtime_ns") != path.stat().st_mtime_ns:
                                raise ValueError(f"Stale depth QC; rerun qc_depth_sign.py for {path}")
                            if not isinstance(record.get("passes"), bool):
                                raise ValueError(f"Invalid depth QC decision for {path}")
                            passing_depth += int(record["passes"])
                        else:
                            model = str(meta.get("model", "")).lower()
                            if "edge" not in model or meta.get("teacher_mode") != "causal_streaming":
                                raise ValueError(
                                    f"Depth must declare the EdGE causal-streaming teacher: {meta_path}"
                                )
                            if meta.get("quality_status") != "selected_after_video_ab_test":
                                raise ValueError(f"Depth label has not passed EdGE video A/B selection: {meta_path}")
                            passing_depth += 1
                    else:
                        if array.ndim != 4 or array.shape[1] != 2:
                            raise ValueError(f"Flow must be [T,2,H,W]: {path}")
                        if int(meta.get("stride", -1)) != self.expected_flow_stride:
                            raise ValueError(
                                "Flow stride must equal action_video_freq_ratio*global_sample_stride "
                                f"({self.expected_flow_stride}): {meta_path}"
                            )
                        if meta.get("normalized_by_image_size") is not True:
                            raise ValueError(f"Flow must use (u/W,v/H): {meta_path}")
                        # Current generator metadata records the native RAFT grid
                        # as flow_height/flow_width and the storage grid separately.
                        # Explicit legacy fields remain supported.
                        source_value = meta.get("source_hw")
                        if source_value is None and meta.get("flow_height") is not None:
                            source_value = (meta.get("flow_height"), meta.get("flow_width"))
                        # The first RAFT batch used a square ``flow_res`` grid.
                        # Per-axis normalisation makes its vectors and coordinates
                        # invariant when that grid is restored to the native aspect.
                        if source_value is None and meta.get("flow_res") is not None and meta.get("size") is not None:
                            source_value = self.source_hw
                        declared_source = tuple(source_value or ())
                        if declared_source != self.source_hw:
                            raise ValueError(f"Missing/mismatched native flow grid: {meta_path}")
                        store_hw = tuple(meta.get("target_hw", (meta.get("store_height"), meta.get("store_width"))))
                        if None in store_hw:
                            size = meta.get("size")
                            store_hw = (size, size)
                        if tuple(array.shape[-2:]) != store_hw:
                            raise ValueError(f"Flow storage grid metadata mismatch: {meta_path}")
                        space = meta.get("spatial_transform", "original")
                        if space not in ("original", "resize_center_crop"):
                            raise ValueError(f"Ambiguous flow spatial_transform: {meta_path}")
                        if space == "resize_center_crop" and tuple(meta.get("target_hw", [])) != self.target_hw:
                            raise ValueError(f"Flow crop grid differs from RGB: {meta_path}")
                    mask_path = base / f"{kind}_mask" / f"{stem}.npy"
                    if mask_path.exists():
                        mask = np.load(mask_path, mmap_mode="r")
                        if mask.shape != (length, *array.shape[-2:]):
                            raise ValueError(f"Label mask shape mismatch: {mask_path}")
                    elif kind == "flow":
                        raise FileNotFoundError(f"Flow requires a validity mask: {mask_path}")
                    self.entries[(dataset_idx, ep, kind)] = (path, meta, length, mask_path)
        if "depth" in self.modalities and passing_depth == 0:
            raise ValueError("No selected training episode passes depth acceptance")

    def _transform(self, value, kind, meta, *, mask=False):
        interpolation = TF.InterpolationMode.NEAREST if mask else TF.InterpolationMode.BILINEAR
        space = "original" if kind == "depth" else meta.get("spatial_transform", "original")
        if space == "original":
            value = TF.resize(value, list(self.source_hw), interpolation=interpolation)
            scale = max(t / s for t, s in zip(self.target_hw, self.source_hw))
            resized_hw = [int(s * scale + 0.5) for s in self.source_hw]
            value = TF.resize(value, resized_hw, interpolation=interpolation, antialias=True)
            value = self.crop(value)
            if kind == "flow" and not mask:
                # Normalised vectors gain the crop factor, including integer resize rounding.
                value[:, 0] *= resized_hw[1] / self.target_hw[1]
                value[:, 1] *= resized_hw[0] / self.target_hw[0]
        else:
            value = TF.resize(value, list(self.target_hw), interpolation=interpolation, antialias=True)
        return value

    def read(self, sample):
        dataset_idx, ep, frame = (int(sample[k]) for k in ("dataset_index", "episode_index", "frame_index"))
        result = {}
        for kind in self.modalities:
            path, meta, length, mask_path = self.entries[(dataset_idx, ep, kind)]
            offsets = self.depth_offsets if kind == "depth" else self.flow_offsets
            indices = frame + offsets
            flow_stride = int(meta.get("stride", 1)) if kind == "flow" else 0
            valid_time = indices + flow_stride < length if kind == "flow" else indices < length
            indices = indices.clip(0, length - 1)
            value = torch.from_numpy(np.array(np.load(path, mmap_mode="r")[indices], dtype=np.float32))
            if kind == "depth":
                value = value.unsqueeze(1)
            finite = torch.isfinite(value).all(dim=1, keepdim=True)
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            mask = finite.float()
            if mask_path.exists():
                mask *= torch.from_numpy(np.array(np.load(mask_path, mmap_mode="r")[indices], dtype=np.float32)).unsqueeze(1)
            mask *= torch.from_numpy(valid_time).view(-1, 1, 1, 1)
            value = self._transform(value, kind, meta)
            mask = self._transform(mask, kind, meta, mask=True).squeeze(1)
            if dataset_idx in self.fov:
                fov = self._transform(self.fov[dataset_idx][None, None], "depth", {}, mask=True)[0, 0]
                mask *= fov
            if kind == "depth":
                value = value.squeeze(1)
                root = self.roots[dataset_idx]
                if self.use_legacy_depth_qc:
                    record = self.qc[(str(root), path.stem)]
                    mask *= float(record["passes"])
                # Kept separate from the fit weights: uniform weights cancel in normalised loss.
                result["depth_weight"] = torch.tensor(float(self.depth_weights.get(root.name, 1.0)))
            result[kind] = value
            result[f"{kind}_mask"] = mask
        return result
