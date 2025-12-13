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
#   bash run_pretrain_ppl_test.sh <STEP> [SEQ_LEN]
# Default: STEP=5000, SEQ_LEN=512

STEP="${1:-5000}"
SEQ_LEN="${2:-512}"

echo "Evaluating perplexity at step ${STEP}, seq_len=${SEQ_LEN}"
echo "Checkpoint directory: ${CKPT_DIR}"

# Full attention
if [[ -f "${CKPT_DIR}/full_attn_step_${STEP}.pt" ]]; then
  echo ""
  echo "==> full attention checkpoint"
  python "${PPL_SCRIPT}" \
    --checkpoint "${CKPT_DIR}/full_attn_step_${STEP}.pt" \
    --model_type full \
    --seq_len "${SEQ_LEN}"
else
  echo "Skip full attention: ${CKPT_DIR}/full_attn_step_${STEP}.pt not found"
fi

# Sparse conv
if [[ -f "${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_step_${STEP}.pt" ]]; then
  echo ""
  echo "==> sparse conv checkpoint"
  python "${PPL_SCRIPT}" \
    --checkpoint "${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_step_${STEP}.pt" \
    --model_type sparse_conv \
    --seq_len "${SEQ_LEN}"
fi

# Sparse mlp
if [[ -f "${CKPT_DIR}/sparse_attn_mlp_c16_f16_n4_step_${STEP}.pt" ]]; then
  echo ""
  echo "==> sparse mlp checkpoint"
  python "${PPL_SCRIPT}" \
    --checkpoint "${CKPT_DIR}/sparse_attn_mlp_c16_f16_n4_step_${STEP}.pt" \
    --model_type sparse_mlp \
    --seq_len "${SEQ_LEN}"
fi

# Sparse attn
if [[ -f "${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_step_${STEP}.pt" ]]; then
  echo ""
  echo "==> sparse attn checkpoint"
  python "${PPL_SCRIPT}" \
    --checkpoint "${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_step_${STEP}.pt" \
    --model_type sparse_attn \
    --seq_len "${SEQ_LEN}"
fi

# Sparse mean
if [[ -f "${CKPT_DIR}/sparse_attn_mean_c16_f16_n4_step_${STEP}.pt" ]]; then
  echo ""
  echo "==> sparse mean checkpoint"
  python "${PPL_SCRIPT}" \
    --checkpoint "${CKPT_DIR}/sparse_attn_mean_c16_f16_n4_step_${STEP}.pt" \
    --model_type sparse_mean \
    --seq_len "${SEQ_LEN}"
fi


