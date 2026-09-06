"""Load the pretrained EdGE DPT decoder as a geometry-register readout.

EdGE's decoder consumes four 2048-D frame/global feature levels. Endo4DWAM's
register branch emits four 1024-D levels, so each level gets a learned adapter
initialised as ``[identity, identity]``. The native EdGE DPT implementation is
loaded lazily from the configured source tree; using the superficially similar
DA3 DPT class is not numerically equivalent even though all parameter keys fit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import torch
from torch import nn

from endo4dwam.utils.logging_config import get_logger

logger = get_logger(__name__)

_EDGE_PREFIX = "depth_head."


class EdgeDPTReadout(nn.Module):
    """Adapt four register levels to the pretrained EdGE dense decoder."""

    def __init__(self, head: nn.Module, *, register_dim: int, edge_dim: int = 2048):
        super().__init__()
        self.head = head
        self.patch_size = int(head.patch_size)
        self.adapters = nn.ModuleList(
            nn.Linear(register_dim, edge_dim) for _ in range(4)
        )
        self._init_adapters(register_dim, edge_dim)

    def _init_adapters(self, register_dim: int, edge_dim: int) -> None:
        # EdGE concatenates same-width frame/global tokens. Duplicating the
        # register feature into both halves is a stable identity-like start;
        # training can subsequently specialise the two halves independently.
        if edge_dim != 2 * register_dim:
            return
        eye = torch.eye(register_dim)
        with torch.no_grad():
            for adapter in self.adapters:
                adapter.weight.zero_()
                adapter.bias.zero_()
                adapter.weight[:register_dim].copy_(eye)
                adapter.weight[register_dim:].copy_(eye)

    def forward(self, feats, *, H: int, W: int, patch_start_idx: int = 0, **kwargs):
        if len(feats) != len(self.adapters):
            raise ValueError(
                f"EdGE readout requires four feature levels, got {len(feats)}"
            )
        adapted = []
        for adapter, level in zip(self.adapters, feats):
            value = level[0] if isinstance(level, (tuple, list)) else level
            adapted.append(adapter(value))
        batch, steps = adapted[0].shape[:2]
        # Native EdGE DPT only reads image shape, not RGB values. Use a zero
        # shape-carrier to satisfy its public interface without the encoder.
        images = adapted[0].new_zeros((batch, steps, 3, H, W))
        prediction, confidence = self.head(
            adapted, images=images, patch_start_idx=patch_start_idx, **kwargs
        )
        output = {"depth": prediction, "depth_conf": confidence}
        # The pretrained EdGE depth head predicts depth + confidence. Geometry
        # loss uses depth only and expects [B,S,H,W], matching the DA3 wrapper.
        depth = output.get("depth")
        if depth is not None and depth.ndim == 5 and depth.shape[-1] == 1:
            output["depth"] = depth.squeeze(-1)
        return output


def build_edge_head(
    weights_path: str,
    *,
    source_path: str,
    register_dim: int = 1024,
    patch_size: int = 32,
    output_dim: int | None = None,
    device=None,
    dtype=None,
    freeze: bool = False,
) -> tuple[Any, dict]:
    """Build an EdGE-initialised depth or signed-motion DPT readout."""
    from safetensors import safe_open

    path = Path(weights_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"EdGE weights not found: {path}")
    source = Path(source_path).expanduser()
    if not (source / "edge/models/components/heads/dpt_head.py").is_file():
        raise FileNotFoundError(f"EdGE source tree not found: {source}")
    source_string = str(source.resolve())
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    from edge.models.components.heads.dpt_head import DPTHead
    module_path = Path(sys.modules[DPTHead.__module__].__file__).resolve()
    if source.resolve() not in module_path.parents:
        raise ImportError(
            f"Imported EdGE DPT from {module_path}, expected it under {source.resolve()}"
        )

    head_cfg = {
        "dim_in": 2048,
        "patch_size": int(patch_size),
        "output_dim": 2 if output_dim is None else int(output_dim) + 1,
        "activation": "exp" if output_dim is None else "linear",
        "conf_activation": "expp1",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "pos_embed": True,
        "intermediate_layer_idx": [0, 1, 2, 3],
    }
    head = DPTHead(**head_cfg)
    with safe_open(str(path), framework="pt", device="cpu") as state:
        head_state = {
            key[len(_EDGE_PREFIX):]: state.get_tensor(key)
            for key in state.keys()
            if key.startswith(_EDGE_PREFIX)
        }
    if not head_state:
        raise ValueError(f"No `{_EDGE_PREFIX}*` tensors in EdGE checkpoint: {path}")

    projection_prefix = f"scratch.output_conv2.{len(head.scratch.output_conv2) - 1}."
    if output_dim is not None:
        # Flow is signed and two-channel. Keep EdGE's endoscopic decoder neck,
        # but do not pretend its depth/confidence projection is a flow prior.
        head_state = {
            key: value for key, value in head_state.items()
            if not key.startswith(projection_prefix)
        }
    missing, unexpected = head.load_state_dict(head_state, strict=False)
    allowed_missing = output_dim is not None and all(
        key.startswith(projection_prefix) for key in missing
    )
    if unexpected or (missing and not allowed_missing):
        raise ValueError(
            "EdGE head checkpoint is incompatible: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    readout = EdgeDPTReadout(head, register_dim=int(register_dim), edge_dim=2048)
    if device is not None or dtype is not None:
        readout = readout.to(device=device, dtype=dtype)
    if freeze:
        readout.requires_grad_(False)
    logger.info(
        "EdGE head %s: loaded %d tensors, output_dim=%s, missing=%d",
        path, len(head_state), 1 if output_dim is None else output_dim, len(missing),
    )
    return readout, head_cfg
