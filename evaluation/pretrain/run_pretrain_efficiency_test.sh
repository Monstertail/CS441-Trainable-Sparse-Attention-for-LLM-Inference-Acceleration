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

SEQ_LENS="${2:-"512 4096"}"

echo "Using checkpoints from step ${STEP}"
echo "Sequence lengths: ${SEQ_LENS}"
echo "Checkpoint directory: ${CKPT_DIR}"


for SEQ_LEN in ${SEQ_LENS}; do
  echo ""
  echo "=============================="
  echo "==> Efficiency @ seq_len=${SEQ_LEN}"
  echo "=============================="

  # Sparse conv
  if [[ -f "${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
    python "${EFF_SCRIPT}" \
      --checkpoint "${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
      --model_type sparse_conv
  else
    echo "Skip sparse_conv: ${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt not found"
  fi

  # Sparse mlp
  if [[ -f "${CKPT_DIR}/sparse_attn_mlp_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
    python "${EFF_SCRIPT}" \
      --checkpoint "${CKPT_DIR}/sparse_attn_mlp_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
      --model_type sparse_mlp
  else
    echo "Skip sparse_mlp: ${CKPT_DIR}/sparse_attn_mlp_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt not found"
  fi

  # Sparse attn
  if [[ -f "${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
    python "${EFF_SCRIPT}" \
      --checkpoint "${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
      --model_type sparse_attn
  else
    echo "Skip sparse_attn: ${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt not found"
  fi

  # Sparse mean
  if [[ -f "${CKPT_DIR}/sparse_attn_mean_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
    python "${EFF_SCRIPT}" \
      --checkpoint "${CKPT_DIR}/sparse_attn_mean_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
      --model_type sparse_mean
  else
    echo "Skip sparse_mean: ${CKPT_DIR}/sparse_attn_mean_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt not found"
  fi

  # Full attention
  if [[ -f "${CKPT_DIR}/full_attn_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
    python "${EFF_SCRIPT}" \
      --checkpoint "${CKPT_DIR}/full_attn_seq${SEQ_LEN}_step_${STEP}.pt" \
      --model_type full
  else
    echo "Skip full: ${CKPT_DIR}/full_attn_seq${SEQ_LEN}_step_${STEP}.pt not found"
  fi
done
