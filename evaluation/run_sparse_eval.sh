#!/bin/bash

# Evaluation script for Sparse Attention model
# Usage: bash run_sparse_eval.sh [adapter_path] [task_name] [split] [compare_base]
#
# Examples:
#   bash run_sparse_eval.sh                              # Evaluate on full test set
#   MAX_SAMPLES=100 bash run_sparse_eval.sh              # Evaluate on first 100 samples
#   MAX_SAMPLES=50 bash run_sparse_eval.sh path gsm8k   # Evaluate 50 samples on GSM8K

set -e

# Arguments (use relative paths from evaluation/ directory)
# Default: use the most recent checkpoint from train/ckpt or train/results
ADAPTER_PATH=${1:-"../train/ckpt/sparse_attn_exp1-gsm8k-3.0epoch-Llama-3.2-1B-Instruct-sparse"}
TASK_NAME=${2:-"gsm8k"}
SPLIT=${3:-"test"}
COMPARE_BASE=${4:-"false"}  # Set to "true" to also evaluate base model

MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
DATA_PATH=${DATA_PATH:-"../train/data/gsm8k"}  # Relative path from evaluation/

# Generation hyperparameters
MAX_NEW_TOKENS=512
TEMPERATURE=0.0  # Greedy decoding

# Sample limit (optional, set via environment variable)
# Example: MAX_SAMPLES=100 bash run_sparse_eval.sh
MAX_SAMPLES=${MAX_SAMPLES:-"100"}  # Empty = all samples

# Output
if [ "$COMPARE_BASE" = "true" ]; then
    OUTPUT_FILE="./eval_results/${TASK_NAME}_${SPLIT}_comparison.json"
else
    OUTPUT_FILE="./eval_results/${TASK_NAME}_${SPLIT}_sparse_only.json"
fi

echo "============================================"
echo "Evaluating Sparse Attention Model"
echo "============================================"
echo "Adapter Path: $ADAPTER_PATH"
echo "Task: $TASK_NAME"
echo "Split: $SPLIT"
echo "Compare with Base: $COMPARE_BASE"
echo "============================================"

# Build command
CMD="python evaluate_sparse_attention.py \
    --model_id $MODEL_ID \
    --adapter_path $ADAPTER_PATH \
    --task_name $TASK_NAME \
    --data_path $DATA_PATH \
    --split $SPLIT \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --output_file $OUTPUT_FILE"

# Add --compare_base flag if requested
if [ "$COMPARE_BASE" = "true" ]; then
    CMD="$CMD --compare_base"
fi

# Add --max_samples if specified
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
    echo "Max Samples: $MAX_SAMPLES"
fi

# Run evaluation
eval $CMD

echo ""
echo "============================================"
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_FILE"
echo "============================================"

