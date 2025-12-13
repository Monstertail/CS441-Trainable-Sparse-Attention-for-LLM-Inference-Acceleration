#!/usr/bin/env bash

set -euo pipefail

# Compare efficiency of full attention and multiple sparse attention variants
# using checkpoints produced by pretrain/train.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT_DIR="${PROJECT_ROOT}/pretrain/ckpt"
EFF_SCRIPT="${PROJECT_ROOT}/evaluation/efficiency.py"

# Optionally pass the training step as the first argument (default: 5000)
STEP="${1:-5000}"

echo "Using checkpoints from step ${STEP}"
echo "Checkpoint directory: ${CKPT_DIR}"



python "${EFF_SCRIPT}" \
  --checkpoint "${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_step_${STEP}.pt" \
  --model_type sparse_conv

python "${EFF_SCRIPT}" \
  --checkpoint "${CKPT_DIR}/sparse_attn_mlp_c16_f16_n4_step_${STEP}.pt" \
  --model_type sparse_mlp

python "${EFF_SCRIPT}" \
  --checkpoint "${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_step_${STEP}.pt" \
  --model_type sparse_attn

python "${EFF_SCRIPT}" \
  --checkpoint "${CKPT_DIR}/sparse_attn_mean_c16_f16_n4_step_${STEP}.pt" \
  --model_type sparse_mean

python "${EFF_SCRIPT}" \
  --checkpoint "${CKPT_DIR}/full_attn_step_${STEP}.pt" \
  --model_type full


