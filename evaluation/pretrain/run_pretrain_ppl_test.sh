#!/usr/bin/env bash

set -euo pipefail

# In-distribution test:  pretrain/data/enwik8.gz
# Out-of-distribution:   data_collection/cs441_synthetic_test.json
# This script is a thin wrapper that passes parameters to evaluation/perplexity.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT_DIR="${PROJECT_ROOT}/pretrain/ckpt"
PPL_SCRIPT="${PROJECT_ROOT}/evaluation/perplexity.py"

# Usage:
#   bash run_pretrain_ppl_test.sh [STEP] [SEQ_LENS]
# Default: STEP=5000, SEQ_LENS="512 4096"

STEP="${1:-5000}"
SEQ_LENS="${2:-"512 4096"}"

echo "Evaluating perplexity at step ${STEP}"
echo "Sequence lengths: ${SEQ_LENS}"
echo "Checkpoint directory: ${CKPT_DIR}"

for SEQ_LEN in ${SEQ_LENS}; do
  echo ""
  echo "=============================="
  echo "==> Perplexity @ seq_len=${SEQ_LEN}"
  echo "=============================="

  # Full attention
  if [[ -f "${CKPT_DIR}/full_attn_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
    echo ""
    echo "==> full attention checkpoint"
    python "${PPL_SCRIPT}" \
      --checkpoint "${CKPT_DIR}/full_attn_seq${SEQ_LEN}_step_${STEP}.pt" \
      --model_type full
  else
    echo "Skip full attention: ${CKPT_DIR}/full_attn_seq${SEQ_LEN}_step_${STEP}.pt not found"
  fi

  # Sparse conv
  if [[ -f "${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
    echo ""
    echo "==> sparse conv checkpoint"
    python "${PPL_SCRIPT}" \
      --checkpoint "${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
      --model_type sparse_conv
  fi

  # Sparse mlp
  if [[ -f "${CKPT_DIR}/sparse_attn_mlp_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
    echo ""
    echo "==> sparse mlp checkpoint"
    python "${PPL_SCRIPT}" \
      --checkpoint "${CKPT_DIR}/sparse_attn_mlp_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
      --model_type sparse_mlp
  fi

  # Sparse attn
  if [[ -f "${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
    echo ""
    echo "==> sparse attn checkpoint"
    python "${PPL_SCRIPT}" \
      --checkpoint "${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
      --model_type sparse_attn
  fi

  # Sparse mean
  if [[ -f "${CKPT_DIR}/sparse_attn_mean_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
    echo ""
    echo "==> sparse mean checkpoint"
    python "${PPL_SCRIPT}" \
      --checkpoint "${CKPT_DIR}/sparse_attn_mean_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
      --model_type sparse_mean
  fi
done
