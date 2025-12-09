#!/bin/bash

# Training script for Sparse Attention on Llama 3.2 1B
# Usage: bash run_sparse_train.sh [experiment_name] [task_name] [cuda_device]
# Example: bash run_sparse_train.sh exp1 gsm8k 0
#          bash run_sparse_train.sh exp2 gsm8k 1,2  # Multi-GPU

set -e

# Default values
EXPERIMENT_NAME=${1:-"sparse_attn_exp1"}
TASK_NAME=${2:-"gsm8k"}
CUDA_DEVICE=${3:-"8"}  # Default to GPU 8
MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"  # Use Instruct version (has chat_template)
DATA_PATH="/home/xinyuya2/jinwei/nsa/CS441-Trainable-Sparse-Attention-for-LLM-Inference-Acceleration/train/data/gsm8k"

# Training hyperparameters
BATCH_SIZE=4
GRAD_ACCUM=4
N_EPOCHS=3
LR=2e-4
WARMUP_RATIO=0.1

# Wandb configuration
USE_WANDB=true                # Set to true to enable wandb logging
WANDB_PROJECT="nsa"           # W&B project name
WANDB_RUN_NAME=""             # Leave empty for auto-generated name

# Sparse attention hyperparameters
COMPRESS_BLOCK_SIZE=16
COMPRESS_STRIDE=8
SELECTION_BLOCK_SIZE=16
NUM_SELECTED_BLOCKS=4
SLIDING_WINDOW_SIZE=64
K_COMPRESS_METHOD="max_pool"  # or "mlp" (default: max_pool for efficiency)
V_COMPRESS_METHOD="max_pool"  # or "mlp" (default: max_pool for efficiency)

echo "============================================"
echo "Training Sparse Attention Model"
echo "============================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Task: $TASK_NAME"
echo "Model: $MODEL_ID"
echo "CUDA Device: $CUDA_DEVICE"
echo "Wandb: $USE_WANDB (Project: $WANDB_PROJECT)"
echo "============================================"

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Build wandb arguments
WANDB_ARGS=""
if [ "$USE_WANDB" = true ]; then
    WANDB_ARGS="--use_wandb --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_RUN_NAME" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb_run_name $WANDB_RUN_NAME"
    fi
fi

python train_sparse_attention.py \
    --model_id $MODEL_ID \
    --output_name $EXPERIMENT_NAME \
    --task_name $TASK_NAME \
    --data_path $DATA_PATH \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --n_epochs $N_EPOCHS \
    --learning_rate $LR \
    --warmup_ratio $WARMUP_RATIO \
    --compress_block_size $COMPRESS_BLOCK_SIZE \
    --compress_stride $COMPRESS_STRIDE \
    --selection_block_size $SELECTION_BLOCK_SIZE \
    --num_selected_blocks $NUM_SELECTED_BLOCKS \
    --sliding_window_size $SLIDING_WINDOW_SIZE \
    --k_compress_method $K_COMPRESS_METHOD \
    --v_compress_method $V_COMPRESS_METHOD \
    $WANDB_ARGS

echo ""
echo "============================================"
echo "Training completed!"
echo "Model saved to: ./ckpt/${EXPERIMENT_NAME}-${TASK_NAME}-${N_EPOCHS}epoch-Llama-3.2-1B-sparse"
echo "============================================"

