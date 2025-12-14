#!/usr/bin/env bash

set -euo pipefail

# Load different seq_len=4096 checkpoints and generate one cs441 example.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CKPT_DIR="${PROJECT_ROOT}/pretrain/ckpt"
GEN_SCRIPT="${PROJECT_ROOT}/evaluation/lightweight_output_example.py"

# Usage:
#   bash run_pretrain_example.sh [STEP] [OUT_CSV] [EXAMPLE_IDX] [GEN_LEN] [-- extra args]
# Defaults:
#   STEP=5000
#   OUT_CSV=${SCRIPT_DIR}/exp_result/generation_step${STEP}_seq4096.csv
#   EXAMPLE_IDX=0
#   GEN_LEN=256

STEP="${1:-5000}"
OUT_CSV="${2:-"${SCRIPT_DIR}/exp_result/generation_step${STEP}_seq4096.csv"}"
EXAMPLE_IDX="${3:-0}"
GEN_LEN="${4:-256}"
EXTRA_ARGS=("${@:5}")

SEQ_LEN="4096"

echo "Step: ${STEP}"
echo "Seq len: ${SEQ_LEN}"
echo "Example idx: ${EXAMPLE_IDX}"
echo "Gen len: ${GEN_LEN}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Output CSV: ${OUT_CSV}"

rm -f "${OUT_CSV}"

run_one() {
  local model_type="$1"
  local ckpt_path="$2"

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "Skip ${model_type}: ${ckpt_path} not found"
    return 0
  fi

  python "${GEN_SCRIPT}" \
    --checkpoint "${ckpt_path}" \
    --model_type "${model_type}" \
    --example_idx "${EXAMPLE_IDX}" \
    --gen_len "${GEN_LEN}" \
    --out_csv "${OUT_CSV}" \
    --use_kv_cache \
    --datasets both \
    "${EXTRA_ARGS[@]}"
}

run_one full "${CKPT_DIR}/full_attn_seq${SEQ_LEN}_step_${STEP}.pt"
run_one sparse_conv "${CKPT_DIR}/sparse_attn_conv_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt"
run_one sparse_mlp "${CKPT_DIR}/sparse_attn_mlp_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt"
run_one sparse_attn "${CKPT_DIR}/sparse_attn_attn_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt"
run_one sparse_mean "${CKPT_DIR}/sparse_attn_mean_c16_f16_n4_seq${SEQ_LEN}_step_${STEP}.pt"


echo "Done. CSV at: ${OUT_CSV}"