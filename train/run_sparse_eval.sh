#!/bin/bash

# Evaluation script for Sparse Attention model
# Usage: bash run_sparse_eval.sh [adapter_path] [task_name]

set -e

# Arguments
ADAPTER_PATH=${1:-"./ckpt/sparse_attn_exp1-gsm8k-3epoch-Llama-3.2-1B-sparse"}
TASK_NAME=${2:-"gsm8k"}
SPLIT=${3:-"test"}

MODEL_ID="meta-llama/Llama-3.2-1B"
DATA_PATH="/path/to/data"  # TODO: Update this path

# Generation hyperparameters
MAX_NEW_TOKENS=512
TEMPERATURE=0.0  # Greedy decoding

# Output
OUTPUT_FILE="./eval_results/${TASK_NAME}_${SPLIT}_results.json"

echo "============================================"
echo "Evaluating Sparse Attention Model"
echo "============================================"
echo "Adapter Path: $ADAPTER_PATH"
echo "Task: $TASK_NAME"
echo "Split: $SPLIT"
echo "============================================"

python evaluate_sparse_attention.py \
    --model_id $MODEL_ID \
    --adapter_path $ADAPTER_PATH \
    --task_name $TASK_NAME \
    --data_path $DATA_PATH \
    --split $SPLIT \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --output_file $OUTPUT_FILE

echo ""
echo "============================================"
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_FILE"
echo "============================================"

