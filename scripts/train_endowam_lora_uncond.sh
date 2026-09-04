#!/usr/bin/env bash
# FastWAM (uncond) + LoRA — EndoWAM endoscope dataset
#
# Model  : FastWAM base (action attends only first-frame video tokens)
# LoRA   : injected on WanVideoDiT video expert (analogous to EndoWAM Cosmos LoRA)
# Data   : endowam_pseudo_z60_rot45  (3 procedures × 8 rot angles = 24 LeRobot roots)
#
# Usage:
#   Fresh start:   bash scripts/train_endowam_lora_uncond.sh
#   Resume:        bash scripts/train_endowam_lora_uncond.sh --resume
#
# Pre-requisites (one-time, before first run):
#   1. python scripts/build_endowam_episodes_stats.py \
#          --data_root /mnt/data2/ljs/EndoWAM/dataset/endowam_pseudo_z60_rot45
#   2. python scripts/precompute_text_embeds.py task=endowam_uncond_1cam_1e-4

set -euo pipefail

# ============================================================================
# GPU / process configuration
# ============================================================================
export CUDA_VISIBLE_DEVICES=6,7
NPROC_PER_NODE=2

# ============================================================================
# Run identity  (fixed so that --resume can locate the checkpoint dir)
# ============================================================================
RUN_ROOT=./runs/endowam_uncond_lora
RUN_ID=fastwam_uncond_lora_endowam_rot45

# ============================================================================
# LoRA configuration  (matches EndoWAM defaults)
# ============================================================================
LORA_ENABLE=true
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TRAIN_BASE=false
LORA_TARGET_MODULES="self_attn.q,self_attn.k,self_attn.v,self_attn.o,cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o,ffn.0,ffn.2"

# ============================================================================
# Training hyperparameters  (aligned with EndoWAM train_endowam_causal_gaze_latent.sh)
# ============================================================================
MAX_STEPS=80000
BATCH_SIZE=6
LEARNING_RATE=1e-4
LR_SCHEDULER_TYPE=cosine
WEIGHT_DECAY=1e-2
GRADIENT_ACCUMULATION=4
MAX_GRAD_NORM=1.0
LOG_EVERY=10
SAVE_EVERY=1000
SAVE_TOTAL_LIMIT=1
EVAL_EVERY=500
EVAL_INFERENCE_STEPS=10

# ============================================================================
# Resume logic
# ============================================================================
OUTPUT_DIR="${RUN_ROOT}/${RUN_ID}"
RESUME=null

if [[ "${1:-}" == "--resume" ]]; then
    state_root="${OUTPUT_DIR}/checkpoints/state"
    if [[ -d "${state_root}" ]]; then
        latest_state=$(ls "${state_root}" 2>/dev/null \
            | grep -E '^step_[0-9]+$' \
            | sort -t_ -k2 -n \
            | tail -1)
        if [[ -n "${latest_state}" ]]; then
            RESUME="${state_root}/${latest_state}"
            echo "[INFO] Resuming from: ${RESUME}"
        else
            echo "[WARN] --resume passed but no state checkpoint found in ${state_root}; starting fresh."
        fi
    else
        echo "[WARN] --resume passed but checkpoint state dir ${state_root} does not exist; starting fresh."
    fi
fi

# ============================================================================
# Setup
# ============================================================================
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

echo "[launch] task=endowam_uncond_1cam_1e-4 nproc=${NPROC_PER_NODE} output_dir=${OUTPUT_DIR} resume=${RESUME}"

# ============================================================================
# Launch
# ============================================================================
accelerate launch \
    --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
    --num_processes "${NPROC_PER_NODE}" \
    scripts/train.py \
    task=endowam_uncond_1cam_1e-4 \
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
    "resume=${RESUME}"
