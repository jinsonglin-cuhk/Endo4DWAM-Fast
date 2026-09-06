"""Persistent, visual-only world memory for the base Endo4DWAM policy.

The state is deliberately external to the module: callers pass ``memory_state``
between action chunks and reset it at an episode boundary.  The only write path
accepts history-video hidden states.  Actions and noisy future-video tokens have
no API through which they can modify the state.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MemoryUpdateBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_mult: int = 4):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"`memory.dim` {dim} must be divisible by `memory.num_heads` {num_heads}")
        self.num_heads = int(num_heads)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        # Start conservatively: the memory is useful from step one but cannot
        # immediately dominate a pretrained video/action representation.
        self.attn_gate = nn.Parameter(torch.tensor(-2.0))
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_mult * dim, dim),
        )
        self.mlp_gate = nn.Parameter(torch.tensor(-2.0))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, -1).transpose(1, 2)

    def forward(
        self,
        memory: torch.Tensor,
        visual_tokens: torch.Tensor,
        slot_embedding: torch.Tensor,
    ) -> torch.Tensor:
        q = self._split_heads(self.to_q(self.norm_q(memory + slot_embedding)))
        kv = self.norm_kv(torch.cat([memory, visual_tokens], dim=1))
        k = self._split_heads(self.to_k(kv))
        v = self._split_heads(self.to_v(kv))
        update = F.scaled_dot_product_attention(q, k, v)
        update = update.transpose(1, 2).reshape_as(memory)
        memory = memory + torch.sigmoid(self.attn_gate) * self.to_out(update)
        return memory + torch.sigmoid(self.mlp_gate) * self.mlp(self.norm_mlp(memory))


class PersistentWorldMemory(nn.Module):
    """Fixed-size recurrent memory updated from clean history-video features.

    A training window is not assumed to follow the previous shuffled batch.
    Instead, the K clean latent frames are consumed chronologically from a reset
    state.  At rollout time an explicit prior state may be supplied, providing
    persistence across action chunks without hidden mutable module state.
    """

    def __init__(
        self,
        *,
        video_dim: int,
        dim: int = 1024,
        num_tokens: int = 64,
        num_heads: int = 8,
        num_blocks: int = 2,
        ffn_mult: int = 4,
    ):
        super().__init__()
        if num_tokens <= 0 or num_blocks <= 0:
            raise ValueError("`memory.num_tokens` and `memory.num_blocks` must be positive")
        self.video_dim = int(video_dim)
        self.dim = int(dim)
        self.num_tokens = int(num_tokens)
        self.slot_embedding = nn.Parameter(
            torch.randn(1, self.num_tokens, self.dim) * self.dim ** -0.5
        )
        self.history_down = nn.Linear(self.video_dim, self.dim)
        self.blocks = nn.ModuleList(
            [_MemoryUpdateBlock(self.dim, int(num_heads), int(ffn_mult)) for _ in range(int(num_blocks))]
        )
        self.output_norm = nn.LayerNorm(self.dim)
        # The cached action path reuses the frozen/pretrained video expert's
        # layer-specific K/V projections.  One shared adapter is enough and
        # avoids roughly 190M per-layer memory K/V adapter parameters.
        self.to_video = nn.Linear(self.dim, self.video_dim)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            (int(batch_size), self.num_tokens, self.dim),
            device=device,
            dtype=dtype,
        )

    def _prepare_state(
        self,
        state: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        detach: bool,
    ) -> torch.Tensor:
        if state is None:
            return self.initial_state(batch_size, device=device, dtype=dtype)
        if state.ndim == 2:
            state = state.unsqueeze(0)
        expected = (batch_size, self.num_tokens, self.dim)
        if tuple(state.shape) != expected:
            raise ValueError(f"`memory_state` must have shape {expected}, got {tuple(state.shape)}")
        state = state.to(device=device, dtype=dtype)
        return state.detach() if detach else state

    def update(
        self,
        history: torch.Tensor,
        *,
        tokens_per_frame: int,
        state: Optional[torch.Tensor] = None,
        detach_previous: bool = True,
    ) -> torch.Tensor:
        """Consume only history tokens, frame by frame, and return the new state."""
        if history.ndim != 3 or history.shape[-1] != self.video_dim:
            raise ValueError(
                f"`history` must be [B,S,{self.video_dim}], got {tuple(history.shape)}"
            )
        tokens_per_frame = int(tokens_per_frame)
        if tokens_per_frame <= 0 or history.shape[1] % tokens_per_frame:
            raise ValueError(
                "history token count must be divisible by positive `tokens_per_frame`, "
                f"got S={history.shape[1]} and tokens_per_frame={tokens_per_frame}"
            )
        memory = self._prepare_state(
            state,
            batch_size=history.shape[0],
            device=history.device,
            dtype=history.dtype,
            detach=detach_previous,
        )
        slots = self.slot_embedding.to(device=history.device, dtype=history.dtype)
        visual = self.history_down(history)
        for frame_tokens in visual.split(tokens_per_frame, dim=1):
            for block in self.blocks:
                memory = block(memory, frame_tokens, slots)
        return self.output_norm(memory)

    def as_video_tokens(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or state.shape[1:] != (self.num_tokens, self.dim):
            raise ValueError(
                f"memory state must be [B,{self.num_tokens},{self.dim}], got {tuple(state.shape)}"
            )
        return self.to_video(state)


def build_persistent_memory(config: dict, *, video_dim: int, device=None, dtype=None) -> PersistentWorldMemory:
    memory = PersistentWorldMemory(
        video_dim=video_dim,
        dim=int(config.get("dim", 1024)),
        num_tokens=int(config.get("num_tokens", 64)),
        num_heads=int(config.get("num_heads", 8)),
        num_blocks=int(config.get("num_blocks", 2)),
        ffn_mult=int(config.get("ffn_mult", 4)),
    )
    if device is not None or dtype is not None:
        memory = memory.to(device=device, dtype=dtype)
    return memory
