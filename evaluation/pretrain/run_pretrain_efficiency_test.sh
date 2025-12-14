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

SEQ_LEN="4096"

# Args:
#   bash run_pretrain_efficiency_test.sh [STEP] [BATCH_SIZES] [PROMPT_LENS] [CSV_PATH] [-- extra args for efficiency.py]
#
# Examples:
#   bash run_pretrain_efficiency_test.sh 5000
#   bash run_pretrain_efficiency_test.sh 5000 "1 8 16" "250 500 750" /tmp/eff.csv --no_kv_cache
#
# Defaults:
# - batch sizes: 1 8 16 32 64
# - prompt lens: 250 500 750 1000 1500 1800
# - gen_len: fixed to 100 (can be overridden by extra args)
#
BATCH_SIZES="${2:-"1 8 16 32 64"}"
PROMPT_LENS="${3:-"500 1000 1500 2000 2500 3000 3500 3900"}"
CSV_PATH="${4:-"${SCRIPT_DIR}/efficiency_step${STEP}_seq${SEQ_LEN}.csv"}"
EXTRA_ARGS=("${@:5}")

echo "Using checkpoints from step ${STEP}"
echo "Sequence length: ${SEQ_LEN}"
echo "Batch sizes: ${BATCH_SIZES}"
echo "Prompt lengths: ${PROMPT_LENS}"
echo "Checkpoint directory: ${CKPT_DIR}"
echo "CSV output: ${CSV_PATH}"

rm -f "${CSV_PATH}"


echo ""
echo "=============================="
echo "==> Efficiency sweep @ seq_len=${SEQ_LEN} (gen_len=100 fixed)"
echo "=============================="

for BS in ${BATCH_SIZES}; do
  for PL in ${PROMPT_LENS}; do
    echo ""
    echo "-- batch_size=${BS}, prompt_len=${PL}, gen_len=100 --"

    # Sparse conv
    if [[ -f "${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
      python "${EFF_SCRIPT}" \
        --checkpoint "${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
        --model_type sparse_conv \
        --batch_size "${BS}" \
        --prompt_len "${PL}" \
        --gen_len 100 \
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
        --prompt_len "${PL}" \
        --gen_len 100 \
        --csv_path "${CSV_PATH}" \
        "${EXTRA_ARGS[@]}"
    else
      echo "Skip sparse_mlp: ${CKPT_DIR}/sparse_attn_mlp_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt not found"
    fi

    # Sparse attn (AttentionPool)
    if [[ -f "${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" ]]; then
      python "${EFF_SCRIPT}" \
        --checkpoint "${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt" \
        --model_type sparse_attn \
        --batch_size "${BS}" \
        --prompt_len "${PL}" \
        --gen_len 100 \
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
        --prompt_len "${PL}" \
        --gen_len 100 \
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
        --prompt_len "${PL}" \
        --gen_len 100 \
        --csv_path "${CSV_PATH}" \
        "${EXTRA_ARGS[@]}"
    else
      echo "Skip full: ${CKPT_DIR}/full_attn_seq${SEQ_LEN}_step_${STEP}.pt not found"
    fi
  done
done
