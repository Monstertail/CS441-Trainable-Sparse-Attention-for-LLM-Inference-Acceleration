#!/bin/bash

# Training script for Sparse Attention on Llama 3.2 1B with Distillation
# Usage: bash run_sparse_train.sh [experiment_name] [task_name] [teacher_gpu] [student_gpu]
# Example: bash run_sparse_train.sh exp1 gsm8k 0 1  # Teacher on GPU 0, Student on GPU 1
#          bash run_sparse_train.sh exp1 gsm8k 0 0  # Both on GPU 0 (single GPU)

set -e

# Default values
EXPERIMENT_NAME=${1:-"sparse_attn_exp1"}
TASK_NAME=${2:-"gsm8k"}
TEACHER_GPU=${3:-"8"}  # Default: Teacher on GPU 0
STUDENT_GPU=${4:-"8"}  # Default: Student on GPU 1
MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"  # Use Instruct version (has chat_template)
DATA_PATH="/home/xinyuya2/jinwei/nsa/CS441-Trainable-Sparse-Attention-for-LLM-Inference-Acceleration/train/data/gsm8k"

# Training hyperparameters
BATCH_SIZE=2  # Reduced from 4 to avoid OOM with dual models
GRAD_ACCUM=16  # Increased to maintain effective batch size of 16
N_EPOCHS=3
LR=1e-5  # Further reduced for stability (large grad_norm observed)
WARMUP_RATIO=0.2  # Increased from 0.1 for more stable training

# Wandb configuration
USE_WANDB=true                # Set to true to enable wandb logging
WANDB_PROJECT="nsa"           # W&B project name
WANDB_RUN_NAME=""             # Leave empty for auto-generated name

# Distillation settings
USE_DISTILLATION=true            # Use dual-model distillation (default: true)

# Sparse attention hyperparameters
COMPRESS_BLOCK_SIZE=16
COMPRESS_STRIDE=8
SELECTION_BLOCK_SIZE=16
NUM_SELECTED_BLOCKS=4
SLIDING_WINDOW_SIZE=64
K_COMPRESS_METHOD="max_pool"  # or "mlp" (default: max_pool for efficiency)
V_COMPRESS_METHOD="mlp"       # or "max_pool" (default: mlp for better learning)

echo "============================================"
echo "Training Sparse Attention Model with Distillation"
echo "============================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "Task: $TASK_NAME"
echo "Model: $MODEL_ID"
echo "Distillation: $USE_DISTILLATION"
echo "Teacher GPU: $TEACHER_GPU"
echo "Student GPU: $STUDENT_GPU"
echo "Wandb: $USE_WANDB (Project: $WANDB_PROJECT)"
echo "============================================"

# Set CUDA visible devices and enable auto-assignment
# User provides physical GPU IDs, we set them visible and let Python auto-assign
if [ "$TEACHER_GPU" = "$STUDENT_GPU" ]; then
    # Single GPU mode
    export CUDA_VISIBLE_DEVICES=$TEACHER_GPU
    echo "🖥️  Single GPU mode: physical GPU $TEACHER_GPU → cuda:0"
else
    # Dual GPU mode
    export CUDA_VISIBLE_DEVICES=$TEACHER_GPU,$STUDENT_GPU
    echo "🖥️  Dual GPU mode: physical GPUs $TEACHER_GPU,$STUDENT_GPU → cuda:0,cuda:1"
fi

# Use 'auto' to let Python script auto-assign based on CUDA_VISIBLE_DEVICES
TEACHER_DEVICE="auto"
STUDENT_DEVICE="auto"

echo "  → Device assignment: auto (will be determined by Python script)"

# Build wandb arguments
WANDB_ARGS=""
if [ "$USE_WANDB" = true ]; then
    WANDB_ARGS="--use_wandb --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_RUN_NAME" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb_run_name $WANDB_RUN_NAME"
    fi
fi

# Build distillation arguments (use remapped device IDs!)
DISTILL_ARGS=""
if [ "$USE_DISTILLATION" = true ]; then
    DISTILL_ARGS="--use_distillation --teacher_device $TEACHER_DEVICE --student_device $STUDENT_DEVICE"
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
    $DISTILL_ARGS \
    $WANDB_ARGS

echo ""
echo "============================================"
echo "Training completed!"
echo "Model saved to: ./ckpt/${EXPERIMENT_NAME}-${TASK_NAME}-${N_EPOCHS}epoch-Llama-3.2-1B-Instruct-sparse"
echo ""
echo "Training Configuration:"
echo "  - Distillation: $USE_DISTILLATION"
echo "  - Teacher GPU: $TEACHER_GPU"
echo "  - Student GPU: $STUDENT_GPU"
echo "  - Compression: K=$K_COMPRESS_METHOD, V=$V_COMPRESS_METHOD"
echo "============================================"

