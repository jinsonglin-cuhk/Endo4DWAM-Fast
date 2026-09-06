#!/usr/bin/env bash
# Endo4DWAM-Joint + LoRA — EndoWAM endoscope dataset
#
# Model  : Endo4DWAMJoint (action attends full video sequence)
# LoRA   : injected on WanVideoDiT video expert (analogous to EndoWAM Cosmos LoRA)
# Data   : endowam_pseudo_z60  (3 procedures = 3 LeRobot roots)
#
# Usage:
#   Fresh start:   bash scripts/train_endowam_lora_joint.sh
#   Resume:        bash scripts/train_endowam_lora_joint.sh --resume
#
# Pre-requisites (one-time, before first run):
#   1. python scripts/build_endowam_episodes_stats.py \
#          --data_root /mnt/data2/ljs/Endo4DWAM/Endo4DWAM/dataset/endowam_pseudo_z60
#   2. python scripts/precompute_text_embeds.py task=endowam_joint_1cam_1e-4

set -euo pipefail
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_PATH")")"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ============================================================================
# GPU / process configuration
# ============================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# ============================================================================
# Run identity  (fixed so that --resume can locate the checkpoint dir)
# NOTE: renamed from the historical fastwam_*_endowam_rot45 ids. The dataset
# changed to endowam_pseudo_z60 (3 roots, no rot augmentation), so the old
# checkpoints are not resumable against this data anyway.
# ============================================================================
RUN_ROOT="${RUN_ROOT:-./runs/endowam_joint_lora}"
RUN_ID="${RUN_ID:-endo4dwam_joint_lora_z60_crop_split}"

# ============================================================================
# LoRA configuration  (matches EndoWAM defaults)
# ============================================================================
# EndoWAM Cosmos targets:    to_q, to_k, to_v, to_out.0, ff.net.0.proj, ff.net.2
# Endo4DWAM Wan DiT equivalents:
#   self-attn    → self_attn.{q,k,v,o}
#   cross-attn   → cross_attn.{q,k,v,o}
#   FFN up/down  → ffn.0, ffn.2
LORA_ENABLE=true
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TRAIN_BASE=false
LORA_TARGET_MODULES="self_attn.q,self_attn.k,self_attn.v,self_attn.o,cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o,ffn.0,ffn.2"

# ============================================================================
# Training hyperparameters  (aligned with EndoWAM train_endowam_causal_gaze_latent.sh)
# ============================================================================
MAX_STEPS=40000          # = EndoWAM trainer.max_train_steps
BATCH_SIZE=4             # per-GPU
LEARNING_RATE=1e-4
LR_SCHEDULER_TYPE=cosine
WEIGHT_DECAY=1e-2
GRADIENT_ACCUMULATION=8
MAX_GRAD_NORM=1.0
LOG_EVERY=10
SAVE_EVERY=1000          # = EndoWAM trainer.save_interval
SAVE_TOTAL_LIMIT=1       # keep only latest checkpoint  (= EndoWAM save_total_limit=1)
EVAL_EVERY=500           # = EndoWAM trainer.eval_interval
EVAL_INFERENCE_STEPS=10

# ============================================================================
# Resume logic  (auto-detect latest state checkpoint when --resume is passed)
# ============================================================================
OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}"
RESUME=null

if [[ "${1:-}" == "--resume" ]]; then
    shift
    state_root="${OUTPUT_DIR}/checkpoints/state"
    latest_step=-1
    for candidate in "${state_root}"/step_*; do
        [[ -d "$candidate" && -f "$candidate/trainer_state.json" ]] || continue
        step="${candidate##*/step_}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        if (( 10#$step > latest_step )); then
            latest_step=$((10#$step))
            RESUME="$candidate"
        fi
    done
    if [[ "$RESUME" == null ]]; then
        echo "[ERROR] No complete training state in ${state_root}; refusing to start fresh with --resume." >&2
        exit 1
    fi
fi
if [[ "$RESUME" == null && -f "${OUTPUT_DIR}/config.yaml" ]]; then
    echo "[ERROR] Run already exists: ${OUTPUT_DIR}. Use --resume or a new RUN_ID." >&2
    exit 1
fi

# ============================================================================
# Setup
# ============================================================================
mkdir -p "${OUTPUT_DIR}"
cp "$SCRIPT_PATH" "${OUTPUT_DIR}/"   # archive the launch script alongside the run

echo "[launch] task=endowam_joint_1cam_1e-4 nproc=${NPROC_PER_NODE} output_dir=${OUTPUT_DIR} resume=${RESUME}"

# ============================================================================
# Launch
# ============================================================================
accelerate launch \
    --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
    --num_processes "${NPROC_PER_NODE}" \
    scripts/train.py \
    task=endowam_joint_1cam_1e-4 \
    "output_dir=${OUTPUT_DIR}" \
    \
    "model.lora.enable=${LORA_ENABLE}" \
    "model.lora.rank=${LORA_RANK}" \
    "model.lora.alpha=${LORA_ALPHA}" \
    "model.lora.dropout=${LORA_DROPOUT}" \
    "model.lora.train_base=${LORA_TRAIN_BASE}" \
    "model.lora.target_modules='${LORA_TARGET_MODULES}'" \
    \
    "max_steps=${MAX_STEPS}" \
    "batch_size=${BATCH_SIZE}" \
    "learning_rate=${LEARNING_RATE}" \
    "lr_scheduler_type=${LR_SCHEDULER_TYPE}" \
    "weight_decay=${WEIGHT_DECAY}" \
    "gradient_accumulation_steps=${GRADIENT_ACCUMULATION}" \
    "max_grad_norm=${MAX_GRAD_NORM}" \
    "log_every=${LOG_EVERY}" \
    "save_every=${SAVE_EVERY}" \
    "save_total_limit=${SAVE_TOTAL_LIMIT}" \
    "eval_every=${EVAL_EVERY}" \
    "eval_num_inference_steps=${EVAL_INFERENCE_STEPS}" \
    "resume=${RESUME}" \
    "$@"
