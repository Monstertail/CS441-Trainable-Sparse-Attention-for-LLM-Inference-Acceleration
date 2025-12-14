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

# Optional:
# - arg3: csv path
# Any remaining args are passed through to evaluation/perplexity.py
ARG3="${3:-}"
DEFAULT_CSV_PATH="${SCRIPT_DIR}/ppl_step${STEP}.csv"

if [[ -n "${ARG3}" && "${ARG3}" != -* ]]; then
  CSV_PATH="${ARG3}"
  EXTRA_ARGS=("${@:4}")
else
  CSV_PATH="${DEFAULT_CSV_PATH}"
  EXTRA_ARGS=("${@:3}")
fi

echo "Evaluating perplexity at step ${STEP}"
echo "Sequence lengths: ${SEQ_LENS}"
echo "Checkpoint directory: ${CKPT_DIR}"
echo "CSV output: ${CSV_PATH}"

rm -f "${CSV_PATH}"

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
      --model_type full \
      --csv_path "${CSV_PATH}" \
      "${EXTRA_ARGS[@]}"
  else
    echo "Skip full attention: ${CKPT_DIR}/full_attn_seq${SEQ_LEN}_step_${STEP}.pt not found"
  fi

  # Sparse conv
  if [[ -f "${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
    echo ""
    echo "==> sparse conv checkpoint"
    python "${PPL_SCRIPT}" \
      --checkpoint "${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
      --model_type sparse_conv \
      --csv_path "${CSV_PATH}" \
      "${EXTRA_ARGS[@]}"
  fi

  # Sparse mlp
  if [[ -f "${CKPT_DIR}/sparse_attn_mlp_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
    echo ""
    echo "==> sparse mlp checkpoint"
    python "${PPL_SCRIPT}" \
      --checkpoint "${CKPT_DIR}/sparse_attn_mlp_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
      --model_type sparse_mlp \
      --csv_path "${CSV_PATH}" \
      "${EXTRA_ARGS[@]}"
  fi

  # Sparse attn
  if [[ -f "${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
    echo ""
    echo "==> sparse attn checkpoint"
    python "${PPL_SCRIPT}" \
      --checkpoint "${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
      --model_type sparse_attn \
      --csv_path "${CSV_PATH}" \
      "${EXTRA_ARGS[@]}"
  fi

  # Sparse mean
  if [[ -f "${CKPT_DIR}/sparse_attn_mean_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
    echo ""
    echo "==> sparse mean checkpoint"
    python "${PPL_SCRIPT}" \
      --checkpoint "${CKPT_DIR}/sparse_attn_mean_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
      --model_type sparse_mean \
      --csv_path "${CSV_PATH}" \
      "${EXTRA_ARGS[@]}"
  fi
done
