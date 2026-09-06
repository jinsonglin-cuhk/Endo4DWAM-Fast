import logging
import os
import inspect
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf

from .trainer import Wan22Trainer
from .utils.logging_config import get_logger, setup_logging
from .utils.video_io import save_mp4
from .utils import misc

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _validate_pipeline_config(cfg: DictConfig) -> None:
    """Fail cheap configuration errors before loading data or the 5B model."""
    model_cfg = cfg.model
    data_cfg = cfg.data.train
    video_cfg = model_cfg.video_dit_config
    geometry_cfg = model_cfg.get("geometry") or {}
    memory_cfg = model_cfg.get("memory") or {}

    history = int(video_cfg.num_history_latent_frames)
    data_history = int(data_cfg.num_history_latent_frames)
    if history < 1 or data_history != history:
        raise ValueError(
            "Model and dataset num_history_latent_frames must match and be >= 1, "
            f"got model={history}, data={data_history}"
        )

    training_attention_path = str(model_cfg.get("training_attention_path", "mixed"))
    if training_attention_path not in {"mixed", "cached"}:
        raise ValueError("model.training_attention_path must be one of: mixed, cached")

    num_layers = int(video_cfg.num_layers)
    if geometry_cfg.get("enable", False):
        supported = {
            "endo4dwam.runtime.create_endo4dwam",
            "endo4dwam.runtime.create_endo4dwam_joint",
        }
        if str(model_cfg._target_) not in supported:
            raise ValueError("Geometry pipeline supports Endo4DWAM base/joint, not IDM")
        for branch, loss_name in (("depth", "lambda_depth"), ("motion", "lambda_flow")):
            branch_cfg = geometry_cfg.get(branch) or {}
            if branch_cfg.get("enable", False) and float(model_cfg.loss[loss_name]) <= 0:
                raise ValueError(f"Enabled geometry branch requires positive {loss_name}")
        if not (data_cfg.get("geometry") or {}).get("enable", False):
            raise ValueError("Geometry model requires an enabled geometry dataloader")

        capture_layers = [int(value) for value in geometry_cfg.get("capture_layers", ())]
        if not capture_layers:
            raise ValueError("geometry.capture_layers must contain at least one layer")
        if len(capture_layers) != 4:
            raise ValueError("geometry DPT readout requires exactly four capture_layers")
        if len(set(capture_layers)) != len(capture_layers):
            raise ValueError("geometry.capture_layers must be unique")
        invalid_layers = [value for value in capture_layers if not 0 <= value < num_layers]
        if invalid_layers:
            raise ValueError(
                f"geometry.capture_layers must be in [0,{num_layers - 1}], got {invalid_layers}"
            )

        geo_dim = int(geometry_cfg.get("geo_dim", 1024))
        geo_heads = int(geometry_cfg.get("num_heads", 8))
        if geo_heads <= 0 or geo_dim % geo_heads:
            raise ValueError("geometry.geo_dim must be divisible by geometry.num_heads")

        grid = list(geometry_cfg.get("register_grid") or ())
        if len(grid) != 2 or any(int(value) <= 0 for value in grid):
            raise ValueError("geometry.register_grid must contain two positive integers")
        head_cfg = geometry_cfg.get("head") or {}
        head_type = str(head_cfg.get("type", "da3")).lower()
        if head_type not in {"edge", "da3"}:
            raise ValueError("geometry.head.type must be one of: edge, da3")
        if head_type == "edge" and not Path(str(head_cfg.get("weights_path", ""))).is_file():
            raise FileNotFoundError(
                f"EdGE head weights not found: {head_cfg.get('weights_path', '')}"
            )
        if head_type == "edge":
            edge_source = Path(str(head_cfg.get("source_path", "")))
            if not (edge_source / "edge/models/components/heads/dpt_head.py").is_file():
                raise FileNotFoundError(f"EdGE source tree not found: {edge_source}")
        patch_size = int(head_cfg.get("patch_size", 32))
        video_size = [int(value) for value in data_cfg.video_size]
        output_size = [int(grid[0]) * patch_size, int(grid[1]) * patch_size]
        if output_size != video_size:
            raise ValueError(
                "geometry.register_grid * geometry.head.patch_size must equal data video_size, "
                f"got {output_size} and {video_size}"
            )

        ratio = int(data_cfg.action_video_freq_ratio)
        num_frames = int(data_cfg.num_frames)
        sampled_video_frames = (num_frames - 1) // ratio + 1
        temporal_factor = int(data_cfg.vae_temporal_downsample_factor)
        latent_steps = (sampled_video_frames - 1) // temporal_factor + 1
        expected_steps = latent_steps - history
        configured_steps = int(geometry_cfg.get("num_supervised_steps", 0))
        if expected_steps <= 0 or configured_steps != expected_steps:
            raise ValueError(
                "geometry.num_supervised_steps must equal T_latent - K > 0, "
                f"got configured={configured_steps}, expected={expected_steps}"
            )

    if memory_cfg.get("enable", False):
        if str(model_cfg._target_) != "endo4dwam.runtime.create_endo4dwam":
            raise ValueError("Persistent memory is implemented only for the base Endo4DWAM reference policy")
        if str(video_cfg.video_attention_mask_mode) not in {
            "first_frame_causal", "first_k_frames_causal"
        }:
            raise ValueError("Persistent memory requires causal history-video attention")
        if training_attention_path != "cached":
            raise ValueError("Persistent memory requires model.training_attention_path=cached")
        action_reads = bool(memory_cfg.get("action_read", True))
        geometry_reads = bool(memory_cfg.get("geometry_read", True)) and bool(
            geometry_cfg.get("enable", False)
        )
        if not action_reads and not geometry_reads:
            raise ValueError(
                "Persistent memory has no supervised consumer: enable memory.action_read "
                "or enable geometry together with memory.geometry_read"
            )
        capture_layer = int(memory_cfg.get("capture_layer", 18))
        if not 0 <= capture_layer < num_layers:
            raise ValueError(
                f"memory.capture_layer must be in [0,{num_layers - 1}], got {capture_layer}"
            )
        memory_dim = int(memory_cfg.get("dim", 1024))
        memory_heads = int(memory_cfg.get("num_heads", 8))
        if memory_heads <= 0 or memory_dim % memory_heads:
            raise ValueError("memory.dim must be divisible by memory.num_heads")


def _apply_lora_to_video_expert(model, lora_cfg) -> None:
    """Inject PEFT LoRA into the video expert (WanVideoDiT) in-place.

    Analogous to EndoWAM's Cosmos25._maybe_apply_peft_lora_to_transformer.
    The PeftModel wrapper is discarded after injection; model.video_expert and
    model.mot.mixtures["video"] share the same object so both see the LoRA layers.

    Default target modules (Wan DiT naming, equivalent to EndoWAM Cosmos targets):
      self_attn.{q,k,v,o}    ↔  to_q, to_k, to_v, to_out.0
      cross_attn.{q,k,v,o}   (no Cosmos equivalent; cross-attn not present there)
      ffn.0, ffn.2            ↔  ff.net.0.proj, ff.net.2

    LoRA params matching EndoWAM defaults:  rank=16, alpha=32, dropout=0.05, train_base=False.
    """
    if isinstance(lora_cfg, DictConfig):
        lora_cfg = OmegaConf.to_container(lora_cfg, resolve=True)
    if not isinstance(lora_cfg, dict):
        raise ValueError(f"`lora_cfg` must be dict-like, got {type(lora_cfg)}")

    enabled = lora_cfg.get("enable", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in ("1", "true", "yes", "on")
    if not bool(enabled):
        return

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError(
            "LoRA requires `peft` to be installed. "
            "Install it with: pip install peft"
        ) from e

    rank = int(lora_cfg.get("rank", 16))
    alpha_val = lora_cfg.get("alpha", None)
    if alpha_val is None:
        alpha_val = rank * 2
    alpha_val = int(alpha_val)
    dropout = float(lora_cfg.get("dropout", 0.05))
    bias = str(lora_cfg.get("bias", "none"))
    train_base = bool(lora_cfg.get("train_base", False))

    target_modules = lora_cfg.get("target_modules", None)
    if target_modules is None:
        target_modules = [
            "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
            "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
            "ffn.0", "ffn.2",
        ]
    elif isinstance(target_modules, str):
        target_modules = [s.strip() for s in target_modules.split(",") if s.strip()]
    else:
        target_modules = list(target_modules)

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha_val,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias=bias,
    )

    # Inject LoRA into video expert (modifies nn.Linear layers in-place via setattr).
    # Discard the returned PeftModel wrapper — we use model.video_expert directly.
    # model.mot.mixtures["video"] is the same Python object as model.video_expert,
    # so both reflect the in-place LoRA modifications automatically.
    get_peft_model(model.video_expert, peft_config)

    if not train_base:
        for p in model.video_expert.parameters():
            p.requires_grad = False
        for n, p in model.video_expert.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

    model._lora_enabled = True
    model._lora_train_base = train_base

    n_train = sum(p.numel() for p in model.video_expert.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.video_expert.parameters())
    logger.info(
        "Video expert LoRA: r=%d, alpha=%d, dropout=%.2f, targets=%s, "
        "trainable=%.3fM / %.3fM params",
        rank, alpha_val, dropout, target_modules, n_train / 1e6, n_tot / 1e6,
    )


def create_wan22_model(
    model_id: str,
    tokenizer_model_id: str,
    dit_config,
    tokenizer_max_len: int = 512,
    train_shift: float = 5.0,
    infer_shift: float = 5.0,
    num_train_timesteps: int = 1000,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.wan22 import Wan22Core

    if isinstance(dit_config, DictConfig):
        dit_config = OmegaConf.to_container(dit_config, resolve=True)
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must resolve to a dict, got {type(dit_config)}")

    return Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        redirect_common_files=bool(redirect_common_files),
        dit_config=dit_config,
        train_shift=float(train_shift),
        infer_shift=float(infer_shift),
        num_train_timesteps=int(num_train_timesteps),
    )


def create_endo4dwam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    geometry=None,
    memory=None,
    action_to_video_grad_scale: float = 1.0,
    action_to_memory_grad_scale: float = 1.0,
    training_attention_path: str = "mixed",
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    lora=None,
):
    from .models.wan22.endo4dwam import Endo4DWAM

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for Endo4DWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(geometry, DictConfig):
        geometry = OmegaConf.to_container(geometry, resolve=True)
    if geometry is None:
        geometry = {}
    if not isinstance(geometry, dict):
        raise ValueError(f"`geometry` must resolve to a dict, got {type(geometry)}")
    if isinstance(memory, DictConfig):
        memory = OmegaConf.to_container(memory, resolve=True)
    if memory is None:
        memory = {}
    if not isinstance(memory, dict):
        raise ValueError(f"`memory` must resolve to a dict, got {type(memory)}")
    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    model = Endo4DWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        loss_lambda_depth=float(loss.get("lambda_depth", 0.0)),
        loss_lambda_flow=float(loss.get("lambda_flow", 0.0)),
        action_to_video_grad_scale=float(action_to_video_grad_scale),
        action_to_memory_grad_scale=float(action_to_memory_grad_scale),
        training_attention_path=str(training_attention_path),
        geometry_config=geometry,
        memory_config=memory,
    )
    if lora is not None:
        _apply_lora_to_video_expert(model, lora)
    return model


def create_endo4dwam_joint(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    geometry=None,
    action_to_video_grad_scale: float = 1.0,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    lora=None,
):
    from .models.wan22.endo4dwam_joint import Endo4DWAMJoint

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for Endo4DWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(geometry, DictConfig):
        geometry = OmegaConf.to_container(geometry, resolve=True)
    if geometry is None:
        geometry = {}
    if not isinstance(geometry, dict):
        raise ValueError(f"`geometry` must resolve to a dict, got {type(geometry)}")
    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    model = Endo4DWAMJoint.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        loss_lambda_depth=float(loss.get("lambda_depth", 0.0)),
        loss_lambda_flow=float(loss.get("lambda_flow", 0.0)),
        action_to_video_grad_scale=float(action_to_video_grad_scale),
        geometry_config=geometry,
    )
    if lora is not None:
        _apply_lora_to_video_expert(model, lora)
    return model


def create_endo4dwam_idm(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    geometry=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    lora=None,
):
    from .models.wan22.endo4dwam_idm import (
        Endo4DWAMIDM,
    )

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for Endo4DWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(geometry, DictConfig):
        geometry = OmegaConf.to_container(geometry, resolve=True)
    if geometry is None:
        geometry = {}
    if not isinstance(geometry, dict):
        raise ValueError(f"`geometry` must resolve to a dict, got {type(geometry)}")
    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    model = Endo4DWAMIDM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        loss_lambda_depth=float(loss.get("lambda_depth", 0.0)),
        loss_lambda_flow=float(loss.get("lambda_flow", 0.0)),
        geometry_config=geometry,
    )
    if lora is not None:
        _apply_lora_to_video_expert(model, lora)
    return model


def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def run_training(cfg: DictConfig):
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    if cfg.resume and Path(str(cfg.resume)).is_dir():
        old_path = Path(str(cfg.resume)).resolve().parents[2] / "config.yaml"
        if old_path.is_file():
            old_cfg = OmegaConf.load(old_path)
            for key in ("model", "data"):
                if OmegaConf.to_container(old_cfg[key], resolve=True) != config_payload[key]:
                    raise ValueError(f"Full resume requires unchanged {key} config. Use a weights .pt warm start and a new output_dir.")
    _validate_pipeline_config(cfg)
    config_path = Path(cfg.output_dir) / "config.yaml"
    if int(os.environ.get("RANK", "0")) == 0 and not cfg.resume and config_path.exists():
        raise FileExistsError(f"Run already exists: {config_path.parent}; use a new output_dir")

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    # Validate label metadata and dataset setup before allocating the 5B model.
    train_ds, val_ds = build_datasets(cfg.data)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    if int(os.environ.get("RANK", "0")) == 0:
        temporary = config_path.with_suffix(".yaml.tmp")
        OmegaConf.save(config_payload, temporary)
        temporary.replace(config_path)

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()

def run_inference(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    inference_cfg = cfg.inference
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(inference_cfg.device))
    checkpoint_path = inference_cfg.get("checkpoint_path")
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            logger.info("Loading finetuned checkpoint: %s", checkpoint_path)
            model.load_checkpoint(checkpoint_path)
        else:
            logger.warning("Checkpoint not found, skipping load: %s", checkpoint_path)
    model.eval()
    
    def center_crop_resize(img: Image, width: int, height: int) -> Image.Image:
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        resized = img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        return resized.crop((left, top, left + width, top + height))

    input_image = Image.open(str(inference_cfg.input_image_path)).convert("RGB")
    input_image = center_crop_resize(input_image, width=inference_cfg.width, height=inference_cfg.height)
    arr = np.array(input_image, dtype=np.float32)
    x = torch.from_numpy(arr)
    x = x.to(device=model.device, dtype=model.torch_dtype)
    x = x * (2.0 / 255.0) - 1.0
    x = repeat(x, "H W C -> B C H W", B=1)
    output_mp4 = str(inference_cfg.output_mp4)

    infer_kwargs = {
        "prompt": str(inference_cfg.prompt),
        "negative_prompt": str(inference_cfg.negative_prompt),
        "text_cfg_scale": float(inference_cfg.text_cfg_scale),
        "action_cfg_scale": float(inference_cfg.action_cfg_scale),
        "input_image": x,
        "num_frames": int(inference_cfg.num_frames),
        "num_inference_steps": int(inference_cfg.num_inference_steps),
        "sigma_shift": None if inference_cfg.get("sigma_shift") is None else float(inference_cfg.sigma_shift),
        "seed": int(inference_cfg.seed),
        "rand_device": str(inference_cfg.rand_device),
        "tiled": bool(inference_cfg.tiled),
    }

    infer_out = model.infer(**infer_kwargs)
    video = infer_out["video"]
    save_mp4(video, output_mp4, fps=15)
    logger.info("Saved inference video to %s", output_mp4)
    return output_mp4
