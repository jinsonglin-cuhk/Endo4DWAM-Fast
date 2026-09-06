"""Training-only geometry / motion readout (WAM4D-style spatial registers).

Learnable register tokens query the video expert's *history* hidden states at a
few intermediate layers and decode a dense prediction, so the geometric prior of
a pretrained foundation head is back-propagated into the causal video features
the policy reads. The whole branch is removed at inference.

Placement matters and is not free to choose. The branch must hang off `MoT`
(i.e. inside `model.dit`) with names that do not contain `mixtures.video.`:

  * `trainer._apply_dit_only_train_mode` freezes everything, re-enables
    `model.dit`, and under LoRA freezes only params whose name contains
    `mixtures.video.` -- so modules here become trainable with no trainer change.
  * `trainer` builds the optimizer from `model.dit.parameters()`.
  * `Endo4DWAM.save_checkpoint` persists `self.mot.state_dict()` only.

Attaching to the top-level model instead leaves it silently frozen and unsaved;
attaching inside the video expert leaves it silently frozen under LoRA.

Attention names deliberately avoid `self_attn.q` / `cross_attn.q` / `ffn.0`, the
suffixes `runtime._maybe_apply_lora` targets, so a future MoT-wide LoRA cannot
wrap these by accident.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GeoBlock(nn.Module):
    """Registers attend to [registers, history video tokens]. No RoPE, no AdaLN.

    Deliberately not a reused `DiTBlock`: that needs `t_mod` and RoPE frequencies
    the register grid has no meaning for.
    """

    def __init__(self, geo_dim: int, num_heads: int, ffn_mult: int = 4):
        super().__init__()
        if geo_dim % num_heads != 0:
            raise ValueError(f"`geo_dim` {geo_dim} must be divisible by `num_heads` {num_heads}")
        self.num_heads = num_heads
        self.norm_q = nn.LayerNorm(geo_dim)
        self.norm_kv = nn.LayerNorm(geo_dim)
        self.to_q = nn.Linear(geo_dim, geo_dim)
        self.to_k = nn.Linear(geo_dim, geo_dim)
        self.to_v = nn.Linear(geo_dim, geo_dim)
        self.to_out = nn.Linear(geo_dim, geo_dim)
        self.norm_mlp = nn.LayerNorm(geo_dim)
        self.mlp = nn.Sequential(
            nn.Linear(geo_dim, ffn_mult * geo_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_mult * geo_dim, geo_dim),
        )

    def forward(
        self,
        registers: torch.Tensor,
        history: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_in = self.norm_q(registers)
        kv_parts = [registers, history]
        if memory is not None:
            kv_parts.append(memory)
        kv_in = self.norm_kv(torch.cat(kv_parts, dim=1))

        def split(x):
            b, s, _ = x.shape
            return x.view(b, s, self.num_heads, -1).transpose(1, 2)

        attn = F.scaled_dot_product_attention(split(self.to_q(q_in)),
                                              split(self.to_k(kv_in)),
                                              split(self.to_v(kv_in)))
        b, _, s, _ = attn.shape
        attn = attn.transpose(1, 2).reshape(b, s, -1)
        registers = registers + self.to_out(attn)
        return registers + self.mlp(self.norm_mlp(registers))


class RegisterStack(nn.Module):
    """One modality's register bank. Depth and motion each get their own."""

    def __init__(self, *, num_time: int, num_spatial: int, video_dim: int,
                 geo_dim: int, num_heads: int, num_blocks: int,
                 memory_dim: int | None = None):
        super().__init__()
        self.num_time = int(num_time)
        self.num_spatial = int(num_spatial)
        self.geo_dim = int(geo_dim)
        # A single spatial grid, repeated over the supervised timesteps, so a
        # register means "this 32x32 image cell" independent of when.
        self.grid = nn.Parameter(torch.randn(1, num_spatial, geo_dim) * geo_dim ** -0.5)
        self.time_embed = nn.Parameter(torch.randn(1, num_time, 1, geo_dim) * geo_dim ** -0.5)
        self.kv_down = nn.ModuleList([nn.Linear(video_dim, geo_dim) for _ in range(num_blocks)])
        self.memory_down = nn.ModuleList([
            nn.Identity() if memory_dim in (None, geo_dim) else nn.Linear(memory_dim, geo_dim)
            for _ in range(num_blocks)
        ])
        self.blocks = nn.ModuleList([_GeoBlock(geo_dim, num_heads) for _ in range(num_blocks)])

    def forward(
        self,
        captures: List[torch.Tensor],
        memory: torch.Tensor | None = None,
    ) -> List[torch.Tensor]:
        """`captures[i]` is the history slice of the video hidden state at capture
        layer i, shaped [B, N_hist, video_dim]. Returns one pyramid level per
        block, each [B, num_time, num_spatial, geo_dim].
        """
        if len(captures) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} captures, got {len(captures)}")
        batch = captures[0].shape[0]
        regs = (self.grid.unsqueeze(1) + self.time_embed).reshape(1, self.num_time * self.num_spatial, self.geo_dim)
        regs = regs.expand(batch, -1, -1).to(dtype=captures[0].dtype)

        levels = []
        for block, down, memory_down, hist in zip(
            self.blocks, self.kv_down, self.memory_down, captures
        ):
            memory_level = None if memory is None else memory_down(memory)
            regs = block(regs, down(hist), memory_level)
            levels.append(regs.view(batch, self.num_time, self.num_spatial, self.geo_dim))
        return levels


class GeometryBranch(nn.Module):
    """Holds the capture-layer list and the independent depth / motion stacks."""

    def __init__(self, *, capture_layers: Sequence[int], depth: RegisterStack | None,
                 motion: RegisterStack | None):
        super().__init__()
        self.capture_layers = tuple(int(i) for i in capture_layers)
        self.depth = depth
        self.motion = motion

    def history_slices(self, captures: Dict[int, torch.Tensor], history_tokens: int) -> List[torch.Tensor]:
        missing = [i for i in self.capture_layers if i not in captures]
        if missing:
            raise ValueError(f"missing captures for layers {missing}")
        return [captures[i][:, :history_tokens] for i in self.capture_layers]


def build_geometry_branch(config: dict, *, video_dim: int, num_spatial: int, num_time: int,
                          num_history: int = 1,
                          memory_dim: int | None = None,
                          device=None, dtype=None) -> GeometryBranch:
    capture_layers = config.get("capture_layers", (12, 14, 16, 18))
    geo_dim = int(config.get("geo_dim", 1024))
    num_heads = int(config.get("num_heads", 8))

    def stack(enabled: bool, steps: int) -> RegisterStack | None:
        if not enabled:
            return None
        return RegisterStack(num_time=steps, num_spatial=num_spatial, video_dim=video_dim,
                             geo_dim=geo_dim, num_heads=num_heads, num_blocks=len(capture_layers),
                             memory_dim=memory_dim)

    motion_cfg = config.get("motion", {})
    motion_steps = num_time + (int(num_history) - 1 if motion_cfg.get("include_observed", False) else 0)

    branch = GeometryBranch(
        capture_layers=capture_layers,
        depth=stack(bool(config.get("depth", {}).get("enable", True)), num_time),
        motion=stack(bool(motion_cfg.get("enable", True)), motion_steps),
    )
    if device is not None or dtype is not None:
        branch = branch.to(device=device, dtype=dtype)
    return branch
