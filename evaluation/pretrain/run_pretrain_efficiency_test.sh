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

# Optional:
# - arg3: batch sizes as a quoted list, e.g. "1 8 16 32 64"
# - arg4: csv path
# Any remaining args are passed through to evaluation/efficiency.py
ARG3="${3:-}"
ARG4="${4:-}"

DEFAULT_BATCH_SIZES="1 8 16 32 64"
DEFAULT_CSV_PATH="${SCRIPT_DIR}/efficiency_step${STEP}.csv"

if [[ -n "${ARG3}" && "${ARG3}" =~ ^[0-9[:space:]]+$ ]]; then
  BATCH_SIZES="${ARG3}"
  if [[ -n "${ARG4}" && "${ARG4}" != -* ]]; then
    CSV_PATH="${ARG4}"
    EXTRA_ARGS=("${@:5}")
  else
    CSV_PATH="${DEFAULT_CSV_PATH}"
    EXTRA_ARGS=("${@:4}")
  fi
else
  BATCH_SIZES="${DEFAULT_BATCH_SIZES}"
  CSV_PATH="${DEFAULT_CSV_PATH}"
  EXTRA_ARGS=("${@:3}")
fi

echo "Using checkpoints from step ${STEP}"
echo "Sequence lengths: ${SEQ_LENS}"
echo "Batch sizes: ${BATCH_SIZES}"
echo "Checkpoint directory: ${CKPT_DIR}"
echo "CSV output: ${CSV_PATH}"

rm -f "${CSV_PATH}"


for SEQ_LEN in ${SEQ_LENS}; do
  echo ""
  echo "=============================="
  echo "==> Efficiency @ seq_len=${SEQ_LEN}"
  echo "=============================="

  for BS in ${BATCH_SIZES}; do
    echo ""
    echo "-- batch_size=${BS} --"

    # Sparse conv
    if [[ -f "${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
      python "${EFF_SCRIPT}" \
        --checkpoint "${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
        --model_type sparse_conv \
        --batch_size "${BS}" \
        --csv_path "${CSV_PATH}" \
        "${EXTRA_ARGS[@]}"
    else
      echo "Skip sparse_conv: ${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt not found"
    fi

    # Sparse mlp
    if [[ -f "${CKPT_DIR}/sparse_attn_mlp_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
      python "${EFF_SCRIPT}" \
        --checkpoint "${CKPT_DIR}/sparse_attn_mlp_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
        --model_type sparse_mlp \
        --batch_size "${BS}" \
        --csv_path "${CSV_PATH}" \
        "${EXTRA_ARGS[@]}"
    else
      echo "Skip sparse_mlp: ${CKPT_DIR}/sparse_attn_mlp_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt not found"
    fi

    # Sparse attn
    if [[ -f "${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
      python "${EFF_SCRIPT}" \
        --checkpoint "${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
        --model_type sparse_attn \
        --batch_size "${BS}" \
        --csv_path "${CSV_PATH}" \
        "${EXTRA_ARGS[@]}"
    else
      echo "Skip sparse_attn: ${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt not found"
    fi

    # Sparse mean
    if [[ -f "${CKPT_DIR}/sparse_attn_mean_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
      python "${EFF_SCRIPT}" \
        --checkpoint "${CKPT_DIR}/sparse_attn_mean_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
        --model_type sparse_mean \
        --batch_size "${BS}" \
        --csv_path "${CSV_PATH}" \
        "${EXTRA_ARGS[@]}"
    else
      echo "Skip sparse_mean: ${CKPT_DIR}/sparse_attn_mean_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt not found"
    fi

    # Full attention
    if [[ -f "${CKPT_DIR}/full_attn_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
      python "${EFF_SCRIPT}" \
        --checkpoint "${CKPT_DIR}/full_attn_seq${SEQ_LEN}_step_${STEP}.pt" \
        --model_type full \
        --batch_size "${BS}" \
        --csv_path "${CSV_PATH}" \
        "${EXTRA_ARGS[@]}"
    else
      echo "Skip full: ${CKPT_DIR}/full_attn_seq${SEQ_LEN}_step_${STEP}.pt not found"
    fi
  done
done
