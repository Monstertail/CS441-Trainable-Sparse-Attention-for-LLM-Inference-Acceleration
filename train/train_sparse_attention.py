"""
Training script for Sparse Attention on Llama 3.2 1B
Following softCoT's pipeline for GSM8K and other reasoning tasks
"""

import argparse
from tqdm import tqdm
import os

import torch
import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from fastNLP import logger

from data_loader import GSM8KLoader, StrategyQALoader, AugASDivLoader, AQuALoader
from sparse_attention_model import LlamaWithSparseAttention
from sparse_distillation_model import SparseDistillationModel
from utils import pre_process_gsm8k, pre_process_strategy_qa, pre_process_aqua, CustomDataCollator


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sparse Attention on Llama 3.2 1B")
    
    # Model args
    parser.add_argument(
        '--model_id', 
        type=str, 
        default='meta-llama/Llama-3.2-1B',
        help='Base model ID'
    )
    parser.add_argument(
        '--output_name', 
        type=str, 
        required=True,
        help='Output experiment name'
    )
    
    # Data args
    parser.add_argument(
        '--task_name', 
        type=str, 
        choices=['gsm8k', 'strategyqa', 'asdiv-aug', 'aqua'],
        default='gsm8k',
        help='Task to train on'
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='/path/to/data/dir',
        help='Root path to dataset files'
    )
    parser.add_argument(
        '--k_shot', 
        type=int, 
        default=0,
        help='Number of training examples (0 = full dataset)'
    )
    
    # Training args
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=4,
        help='Batch size per device'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=4,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--n_epochs', 
        type=float, 
        default=3.0,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=2e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.1,
        help='Warmup ratio'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )
    
    # Sparse attention args
    parser.add_argument(
        '--compress_block_size',
        type=int,
        default=16,
        help='Block size for coarse-grained compression'
    )
    parser.add_argument(
        '--compress_stride',
        type=int,
        default=8,
        help='Stride for compression (for sliding blocks)'
    )
    parser.add_argument(
        '--selection_block_size',
        type=int,
        default=16,
        help='Block size for fine-grained selection'
    )
    parser.add_argument(
        '--num_selected_blocks',
        type=int,
        default=4,
        help='Number of blocks to select for fine attention'
    )
    parser.add_argument(
        '--sliding_window_size',
        type=int,
        default=64,
        help='Sliding window size for local attention'
    )
    parser.add_argument(
        '--k_compress_method',
        type=str,
        choices=['max_pool', 'mlp'],
        default='max_pool',
        help='Compression method for K (default: max_pool for efficiency)'
    )
    parser.add_argument(
        '--v_compress_method',
        type=str,
        choices=['max_pool', 'mlp'],
        default='max_pool',
        help='Compression method for V (default: max_pool for efficiency)'
    )
    
    # Distillation and device args
    parser.add_argument(
        '--use_distillation',
        action='store_true',
        default=True,
        help='Use dual-model distillation (default: True)'
    )
    parser.add_argument(
        '--teacher_device',
        type=str,
        default='auto',
        help='Device for teacher model (default: auto, or specify cuda:0, cuda:1, etc.)'
    )
    parser.add_argument(
        '--student_device',
        type=str,
        default='auto',
        help='Device for student model (default: auto, or specify cuda:0, cuda:1, etc.)'
    )
    
    # Logging and monitoring
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Use Weights & Biases for logging'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='sparse-attention',
        help='W&B project name'
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='W&B run name (default: auto-generated)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info(f'Arguments: {args.__dict__}')
    
    # Setup directories
    model_name = args.model_id.split('/')[-1]
    post_fix = f'{args.task_name}-{args.n_epochs}epoch-{model_name}-sparse'
    output_dir = f'./results/{args.output_name}-{post_fix}'
    log_dir = f'./logs/{args.output_name}-{post_fix}'
    save_model_dir = f'./ckpt/{args.output_name}-{post_fix}'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)
    
    logger.info(f'Output Dir: {output_dir}')
    logger.info(f'Log Dir: {log_dir}')
    logger.info(f'Save Model Dir: {save_model_dir}')
    
    # Initialize wandb if requested
    if args.use_wandb:
        try:
            import wandb
            
            # Auto-generate run name if not provided
            wandb_run_name = args.wandb_run_name or f"{args.output_name}-{post_fix}"
            
            wandb.init(
                project=args.wandb_project,
                name=wandb_run_name,
                config={
                    'model_id': args.model_id,
                    'task_name': args.task_name,
                    'n_epochs': args.n_epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'use_distillation': args.use_distillation,
                    'teacher_device': args.teacher_device if args.use_distillation else None,
                    'student_device': args.student_device if args.use_distillation else None,
                    'compress_block_size': args.compress_block_size,
                    'num_selected_blocks': args.num_selected_blocks,
                    'k_compress_method': args.k_compress_method,
                    'v_compress_method': args.v_compress_method,
                }
            )
            logger.info(f"✅ W&B initialized: {wandb.run.url}")
        except ImportError:
            logger.warning("⚠️  wandb not installed. Install with: pip install wandb")
            args.use_wandb = False
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine backbone type
    if 'Llama' in args.model_id:
        special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
        backbone = 'llama'
    elif 'Qwen' in args.model_id:
        special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
        backbone = 'qwen'
    else:
        raise NotImplementedError(f"Model {args.model_id} not supported yet")
    
    # Initialize model with sparse attention
    sparse_attn_config = {
        'compress_block_size': args.compress_block_size,
        'compress_stride': args.compress_stride,
        'selection_block_size': args.selection_block_size,
        'num_selected_blocks': args.num_selected_blocks,
        'sliding_window_size': args.sliding_window_size,
        'k_compress_method': args.k_compress_method,
        'v_compress_method': args.v_compress_method,
    }
    
    if args.use_distillation:
        logger.info("🔧 Initializing dual-model distillation (Teacher + Student)...")
        
        # Auto-assign devices if not specified
        teacher_device = args.teacher_device
        student_device = args.student_device
        
        if teacher_device == 'auto' or student_device == 'auto':
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_visible:
                # Parse visible GPUs
                visible_gpus = [int(x.strip()) for x in cuda_visible.split(',') if x.strip()]
                if len(visible_gpus) >= 2:
                    teacher_device = 'cuda:0'  # First visible GPU (remapped)
                    student_device = 'cuda:1'  # Second visible GPU (remapped)
                    logger.info(f"🤖 Auto-assigned: Teacher→cuda:0 (physical GPU {visible_gpus[0]}), Student→cuda:1 (physical GPU {visible_gpus[1]})")
                elif len(visible_gpus) == 1:
                    teacher_device = 'cuda:0'
                    student_device = 'cuda:0'
                    logger.warning(f"⚠️  Only 1 GPU available, using same device for both models")
                else:
                    teacher_device = 'cuda:0'
                    student_device = 'cuda:0'
                    logger.warning(f"⚠️  No CUDA_VISIBLE_DEVICES set, using cuda:0 for both")
            else:
                # Check available GPUs directly
                num_gpus = torch.cuda.device_count()
                if num_gpus >= 2:
                    teacher_device = 'cuda:0'
                    student_device = 'cuda:1'
                    logger.info(f"🤖 Auto-assigned: Teacher→cuda:0, Student→cuda:1")
                else:
                    teacher_device = 'cuda:0'
                    student_device = 'cuda:0'
                    logger.warning(f"⚠️  Only {num_gpus} GPU(s) available, using same device")
        
        logger.info(f"  Teacher device: {teacher_device}")
        logger.info(f"  Student device: {student_device}")
        
        model = SparseDistillationModel(
            model_id=args.model_id,
            sparse_attn_config=sparse_attn_config,
            teacher_device=teacher_device,
            student_device=student_device,
        )
    else:
        logger.info("Initializing single model with sparse attention adapters...")
        
        model = LlamaWithSparseAttention(
            model_id=args.model_id,
            sparse_attn_config=sparse_attn_config,
            device_map='auto',
        )
        
        # Print trainable parameter statistics
        model.get_trainable_parameters()
    
    # Load and preprocess dataset
    logger.info(f"Loading {args.task_name} dataset...")
    
    if args.task_name == 'gsm8k':
        db = GSM8KLoader().load(args.data_path)
        preprocess_method = pre_process_gsm8k
    elif args.task_name == 'strategyqa':
        db = StrategyQALoader().load(args.data_path)
        preprocess_method = pre_process_strategy_qa
    elif args.task_name == 'asdiv-aug':
        db = AugASDivLoader().load(args.data_path)
        preprocess_method = pre_process_gsm8k
    elif args.task_name == 'aqua':
        db = AQuALoader().load(args.data_path)
        preprocess_method = pre_process_aqua
    else:
        raise NotImplementedError(f"Task {args.task_name} not implemented")
    
    train_dataset = db.get_dataset('train')
    eval_dataset = db.get_dataset('dev')
    
    if args.k_shot > 0:
        train_dataset = train_dataset[:args.k_shot]
        logger.info(f"Using k-shot learning with k={args.k_shot}")
    
    # Preprocess data
    logger.info("Preprocessing training data...")
    train_rows = []
    for ins in tqdm(train_dataset, desc='Preprocess Training Set'):
        # For sparse attention, we don't need assistant model or thought tokens
        # We'll adapt the preprocessing to work with single model
        row = preprocess_method(
            ins,
            tokenizer,
            tokenizer,  # Use same tokenizer (no assistant model)
            num_thought_tokens=0,  # No thought tokens
            add_bot_eot=True,
            split='train',
            base_special_token=special_token,
            assistant_special_token=special_token,
            base_backbone=backbone,
            assistant_backbone=backbone,
        )
        train_rows.append(row)
    
    logger.info("Preprocessing evaluation data...")
    eval_rows = []
    for ins in tqdm(eval_dataset, desc='Preprocess Eval Set'):
        row = preprocess_method(
            ins,
            tokenizer,
            tokenizer,
            num_thought_tokens=0,
            add_bot_eot=True,
            split='dev',
            base_special_token=special_token,
            assistant_special_token=special_token,
            base_backbone=backbone,
            assistant_backbone=backbone,
        )
        eval_rows.append(row)
    
    train_data = Dataset.from_pandas(pd.DataFrame(train_rows))
    eval_data = Dataset.from_pandas(pd.DataFrame(eval_rows))
    
    logger.info(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Evaluation (updated for transformers 4.51.0+)
        eval_strategy='epoch',  # Changed from 'evaluation_strategy' in newer transformers
        save_strategy='epoch',
        save_total_limit=5,
        
        # Training
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Optimization
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=0.1,  # Very strict clipping (saw grad_norm > 1000)
        
        # Precision
        bf16=True,
        bf16_full_eval=True,
        
        # Logging
        logging_dir=log_dir,
        logging_steps=10,
        logging_first_step=True,
        
        # Misc
        save_safetensors=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,  # Important: keep all columns
        
        # Disable automatic multi-GPU (we manage devices manually)
        ddp_find_unused_parameters=False,
        
        # Report (add wandb if enabled)
        report_to=['tensorboard'] + (['wandb'] if args.use_wandb else []),
    )
    
    # Use CustomDataCollator from utils.py (already has padding logic)
    # This is the same collator used in SoftCoT, well-tested and handles:
    # - Dynamic padding to max length in batch
    # - Proper handling of labels (-100 for padding)
    # - Multiple input fields (we only use input_ids, attention_mask, labels)
    data_collator = CustomDataCollator()
    
    # Prevent Trainer from using DataParallel (we manually manage dual-GPU)
    # Override the detected GPU count to prevent automatic multi-GPU wrapping
    training_args._n_gpu = 1  # Tell Trainer to treat this as single-GPU
    
    # Initialize trainer (Note: we manually manage multi-GPU in the model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("🚀 Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"💾 Saving sparse attention adapters to {save_model_dir}")
    if args.use_distillation:
        model.save_student(save_model_dir)  # Dual-model: save student adapters
    else:
        model.save_adapters(save_model_dir)  # Single-model: save adapters
    
    # Also save the full training config
    import json
    config_path = os.path.join(save_model_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'model_id': args.model_id,
            'task_name': args.task_name,
            'sparse_attn_config': sparse_attn_config,
            'training_args': {
                'n_epochs': args.n_epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
            }
        }, f, indent=2)
    
    logger.info(f"✅ Training completed! Model saved to {save_model_dir}")


if __name__ == "__main__":
    main()

