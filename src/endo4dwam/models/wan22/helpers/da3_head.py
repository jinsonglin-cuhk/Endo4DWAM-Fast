"""Load a pretrained Depth-Anything-3 dense head as the geometry readout.

Kept in its own lazy module so `depth_anything_3` is never a hard dependency:
a geometry-disabled run must import and train without it installed.

Which head: the depth teacher is DA3MONO-LARGE, and MONO uses `DPT`, not the
`DualDPT` that WAM4D took from GIANT-1.1's any-view branch. Same-source matters
more than matching the paper's class -- and DPT is 29.8M params against
DualDPT's ~47M, with no auxiliary fusion chain we would never read.

Measured: all 64 `model.head.*` keys load with missing=0 and unexpected=0, and
`dim_in` is 1024, which is exactly the register branch's geo_dim, so register
outputs feed the head with no adapter. At patch_size=32 an 8x10 token grid
decodes to 256x320 -- the training resolution.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from endo4dwam.utils.logging_config import get_logger

logger = get_logger(__name__)

_HEAD_PREFIX = "model.head."


def _find_snapshot(model_id: str) -> Path:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    folder = "models--" + model_id.replace("/", "--")
    snapshots = sorted((cache / folder / "snapshots").glob("*"))
    if not snapshots:
        raise FileNotFoundError(
            f"No local snapshot for {model_id} under {cache}. "
            "The geometry head is loaded from the HF cache; download it first."
        )
    return snapshots[-1]


def build_da3_head(model_id: str = "depth-anything/DA3MONO-LARGE", *, patch_size: int = 32,
                   output_dim: int | None = None, device=None, dtype=None,
                   freeze: bool = False, max_missing: int = 0) -> Any:
    """Returns a DPT head initialised from `model_id`.

    `max_missing` guards the failure mode that matters: `load_state_dict` with
    strict=False will happily leave the whole head randomly initialised, and
    WAM4D's Table 8 found a randomly-initialised geometric head to be *worse*
    than no depth supervision at all. Loading fewer keys than expected raises.
    """
    from depth_anything_3.model.dpt import DPT  # lazy: not a hard dependency
    from safetensors import safe_open

    snapshot = _find_snapshot(model_id)
    head_cfg = {k: v for k, v in json.loads((snapshot / "config.json").read_text())["config"]["head"].items()
                if k != "__object__"}
    if output_dim is not None:
        # DPT reserves the last channel for confidence when output_dim > 1.
        head_cfg["output_dim"] = int(output_dim) + 1
        head_cfg["activation"] = "linear"

    head = DPT(patch_size=patch_size, **head_cfg)
    with safe_open(str(snapshot / "model.safetensors"), framework="pt", device="cpu") as state:
        head_state = {k[len(_HEAD_PREFIX):]: state.get_tensor(k)
                      for k in state.keys() if k.startswith(_HEAD_PREFIX)}
    if not head_state:
        raise ValueError(f"No `{_HEAD_PREFIX}*` keys in {snapshot / 'model.safetensors'}")

    projection_prefix = f"scratch.output_conv2.{len(head.scratch.output_conv2) - 1}."
    if output_dim is not None:
        # Reinitialise the modality-specific final projection, retaining the pretrained neck.
        head_state = {k: v for k, v in head_state.items() if not k.startswith(projection_prefix)}
    missing, unexpected = head.load_state_dict(head_state, strict=False)
    # output_dim changes legitimately alter the final conv, so allow those.
    hard_missing = [k for k in missing if not (output_dim is not None and k.startswith(projection_prefix))]
    if unexpected:
        raise ValueError(f"Unexpected DA3 head keys: {unexpected}")
    logger.info("DA3 head %s: loaded %d keys, missing=%d (hard=%d), unexpected=%d",
                model_id, len(head_state), len(missing), len(hard_missing), len(unexpected))
    if len(hard_missing) > max_missing:
        raise ValueError(
            f"DA3 head load left {len(hard_missing)} keys uninitialised (max {max_missing}): "
            f"{hard_missing[:8]}. Refusing to train against a randomly-initialised geometric head."
        )

    if device is not None or dtype is not None:
        head = head.to(device=device, dtype=dtype)
    if freeze:
        head.requires_grad_(False)
    return head, head_cfg
