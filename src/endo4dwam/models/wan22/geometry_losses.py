"""Losses for the geometry / motion readout."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def clip_shared_affine_depth_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    *,
    beta: float = 1.0,
    min_scale: float = 1e-3,
    sample_weight: Optional[torch.Tensor] = None,
    gradient_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SmoothL1 after aligning the prediction to the target with ONE (s, b) per clip.

    DA3 depth is affine-ambiguous, so the prediction may only be compared up to a
    scale and shift. Solving (s, b) *per frame* would let the model cheat: each
    frame could be rescaled independently and the cross-frame geometry -- the part
    a world model actually needs -- would carry no gradient. Sharing one (s, b)
    across the whole future clip keeps that structure.

    Two details that are easy to get wrong:

    * `s` is clamped positive. Without it a prediction with near/far inverted can
      be aligned back through a negative scale, which is precisely the failure the
      sign QC exists to catch -- it would be silently absorbed here instead.
    * (s, b) are detached. They come from a least-squares fit of the prediction
      itself, so letting gradient flow through them lets the network lower the
      loss by moving the optimal alignment rather than by improving the depth.

    Shapes: pred/target/weight are [B, S, H, W]. Returns (loss, s, b).
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    b_size = pred.shape[0]
    p = pred.float().reshape(b_size, -1)
    t = target.float().reshape(b_size, -1)
    w = torch.ones_like(p) if weight is None else weight.float().reshape(b_size, -1)

    sw = w.sum(dim=1).clamp(min=1.0)
    sp, st = (w * p).sum(dim=1), (w * t).sum(dim=1)
    spp, spt = (w * p * p).sum(dim=1), (w * p * t).sum(dim=1)

    denom = sw * spp - sp * sp
    safe = denom.abs() > 1e-8
    scale = torch.where(safe, (sw * spt - sp * st) / denom.masked_fill(~safe, 1.0),
                        torch.ones_like(denom))
    scale = scale.clamp(min=min_scale)
    shift = (st - scale * sp) / sw

    scale = scale.detach().unsqueeze(1)
    shift = shift.detach().unsqueeze(1)

    aligned_flat = scale * p + shift
    per_pixel = F.smooth_l1_loss(aligned_flat, t, reduction="none", beta=beta)
    loss = (per_pixel * w).sum(dim=1) / sw
    if gradient_weight:
        aligned = aligned_flat.reshape_as(pred).float()
        target_f = target.float()
        valid = torch.ones_like(target_f) if weight is None else weight.float()

        def directional(a: torch.Tensor, t_: torch.Tensor, m: torch.Tensor, dim: int):
            if dim == -1:
                da, dt = a[..., 1:] - a[..., :-1], t_[..., 1:] - t_[..., :-1]
                pair = m[..., 1:] * m[..., :-1]
            else:
                da, dt = a[..., 1:, :] - a[..., :-1, :], t_[..., 1:, :] - t_[..., :-1, :]
                pair = m[..., 1:, :] * m[..., :-1, :]
            err = F.smooth_l1_loss(da, dt, reduction="none", beta=beta)
            numer = (err * pair).reshape(b_size, -1).sum(dim=1)
            denom_grad = pair.reshape(b_size, -1).sum(dim=1).clamp(min=1.0)
            return numer / denom_grad

        grad_loss = 0.5 * (
            directional(aligned, target_f, valid, -1)
            + directional(aligned, target_f, valid, -2)
        )
        loss = loss + float(gradient_weight) * grad_loss
    if sample_weight is not None:
        loss = loss * sample_weight.float().reshape(b_size)
    return loss.mean(), scale.squeeze(1), shift.squeeze(1)


def flow_loss(pred: torch.Tensor, target: torch.Tensor,
              mask: Optional[torch.Tensor] = None, *, loss_type: str = "charbonnier",
              beta: float = 0.05, epsilon: float = 1e-3,
              alpha: float = 0.5) -> torch.Tensor:
    """Masked robust loss on a 2-channel flow field.

    Flow has no scale ambiguity, so it is supervised directly. `beta` is small
    because the targets are small: the endoscope video is 69% near-static, with a
    median normalised magnitude around 8e-4, so a beta of 1.0 would put every
    real motion inside the quadratic region and flatten the signal.

    Shapes: pred/target [B, S, 2, H, W]; mask [B, S, H, W].
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    diff = pred.float() - target.float()
    if loss_type == "charbonnier":
        if epsilon <= 0 or not 0 < alpha <= 1:
            raise ValueError("Charbonnier `epsilon` must be >0 and `alpha` in (0,1]")
        # Subtract the zero-error floor so a perfect/masked-out prediction is
        # exactly zero, which keeps metrics and distributed loss logs intuitive.
        per_component = (diff.square() + epsilon ** 2).pow(alpha) - epsilon ** (2 * alpha)
    elif loss_type == "smooth_l1":
        per_component = F.smooth_l1_loss(
            pred.float(), target.float(), reduction="none", beta=beta
        )
    else:
        raise ValueError(f"Unsupported flow loss type: {loss_type}")
    per_pixel = per_component.sum(dim=2)
    if mask is None:
        return per_pixel.mean()
    m = mask.float()
    return (per_pixel * m).sum() / m.sum().clamp(min=1.0)
