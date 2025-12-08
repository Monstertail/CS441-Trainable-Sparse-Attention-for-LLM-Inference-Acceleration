"""
Evaluation script for Sparse Attention model
"""

import argparse
import json
import os
from tqdm import tqdm
import re

import torch
from transformers import AutoTokenizer
from fastNLP import logger

from data_loader import GSM8KLoader, StrategyQALoader, AugASDivLoader, AQuALoader
from sparse_attention_model import LlamaWithSparseAttention


def extract_answer_gsm8k(text):
    """Extract numerical answer from GSM8K format"""
    # Look for boxed answer
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        answer = match.group(1).replace(',', '')
        try:
            return float(answer) if '.' in answer else int(answer)
        except:
            return None
    
    # Look for "final answer is: X"
    match = re.search(r'final answer is[:\s]+([0-9.,]+)', text.lower())
    if match:
        answer = match.group(1).replace(',', '')
        try:
            return float(answer) if '.' in answer else int(answer)
        except:
            return None
    
    return None


def extract_answer_strategyqa(text):
    """Extract Yes/No answer"""
    text_lower = text.lower()
    
    # Look for explicit yes/no
    if 'answer is yes' in text_lower or 'answer: yes' in text_lower:
        return 'Yes'
    if 'answer is no' in text_lower or 'answer: no' in text_lower:
        return 'No'
    
    # Fallback: check last sentence
    sentences = text.split('.')
    for sent in reversed(sentences):
        if 'yes' in sent.lower():
            return 'Yes'
        if 'no' in sent.lower():
            return 'No'
    
    return None


def evaluate_model(
    model,
    tokenizer,
    dataset,
    task_name,
    max_new_tokens=512,
    temperature=0.0,
    batch_size=1,
):
    """
    Evaluate model on dataset
    """
    model.eval()
    
    results = []
    correct = 0
    total = 0
    
    # Choose answer extraction function
    if task_name in ['gsm8k', 'aqua', 'asdiv-aug']:
        extract_answer = extract_answer_gsm8k
    elif task_name == 'strategyqa':
        extract_answer = extract_answer_strategyqa
    else:
        raise NotImplementedError(f"Task {task_name} not supported")
    
    with torch.no_grad():
        for instance in tqdm(dataset, desc='Evaluating'):
            # Prepare input
            if task_name in ['gsm8k', 'aqua', 'asdiv-aug']:
                question = instance['question']
                input_text = f"Solve the following math problem:\n\nProblem: {question}\n\nSolution:"
                
                # Get ground truth
                answer_text = instance['answer']
                if isinstance(answer_text, str) and '####' in answer_text:
                    gt_answer = answer_text.split('####')[-1].strip().replace(',', '')
                    try:
                        gt_answer = float(gt_answer) if '.' in gt_answer else int(gt_answer)
                    except:
                        gt_answer = None
                else:
                    gt_answer = answer_text
                    
            elif task_name == 'strategyqa':
                question = instance['question']
                input_text = f"Answer the following question with Yes or No:\n\nQuestion: {question}\n\nAnswer:"
                gt_answer = 'Yes' if instance['answer'] else 'No'
            else:
                continue
            
            # Tokenize
            inputs = tokenizer(
                input_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024
            ).to(model.base_model.device)
            
            # Generate
            outputs = model.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode
            generated_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Extract answer
            pred_answer = extract_answer(generated_text)
            
            # Check correctness
            is_correct = False
            if pred_answer is not None and gt_answer is not None:
                if isinstance(gt_answer, (int, float)):
                    try:
                        is_correct = abs(float(pred_answer) - float(gt_answer)) < 1e-3
                    except:
                        is_correct = False
                else:
                    is_correct = str(pred_answer).lower() == str(gt_answer).lower()
            
            if is_correct:
                correct += 1
            total += 1
            
            # Store result
            results.append({
                'question': question,
                'ground_truth': gt_answer,
                'generated_text': generated_text,
                'predicted_answer': pred_answer,
                'correct': is_correct,
            })
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sparse Attention model")
    
    parser.add_argument(
        '--model_id',
        type=str,
        default='meta-llama/Llama-3.2-1B',
        help='Base model ID'
    )
    parser.add_argument(
        '--adapter_path',
        type=str,
        required=True,
        help='Path to trained sparse attention adapters'
    )
    parser.add_argument(
        '--task_name',
        type=str,
        choices=['gsm8k', 'strategyqa', 'asdiv-aug', 'aqua'],
        required=True,
        help='Task to evaluate on'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='/path/to/data/dir',
        help='Path to dataset'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'dev', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=512,
        help='Maximum new tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (0 = greedy)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output file for detailed results'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Arguments: {args.__dict__}")
    
    # Load model
    logger.info(f"Loading model from {args.model_id}...")
    
    # Load training config if available
    config_path = os.path.join(args.adapter_path, 'training_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            training_config = json.load(f)
            sparse_attn_config = training_config.get('sparse_attn_config', {})
            logger.info(f"Loaded sparse attention config: {sparse_attn_config}")
    else:
        sparse_attn_config = None
        logger.warning("No training config found, using default sparse attention config")
    
    model = LlamaWithSparseAttention(
        model_id=args.model_id,
        sparse_attn_config=sparse_attn_config,
        device_map='auto',
    )
    
    # Load trained adapters
    logger.info(f"Loading adapters from {args.adapter_path}...")
    model.load_adapters(args.adapter_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    logger.info(f"Loading {args.task_name} dataset...")
    
    if args.task_name == 'gsm8k':
        db = GSM8KLoader().load(args.data_path)
    elif args.task_name == 'strategyqa':
        db = StrategyQALoader().load(args.data_path)
    elif args.task_name == 'asdiv-aug':
        db = AugASDivLoader().load(args.data_path)
    elif args.task_name == 'aqua':
        db = AQuALoader().load(args.data_path)
    else:
        raise NotImplementedError(f"Task {args.task_name} not implemented")
    
    eval_dataset = db.get_dataset(args.split)
    logger.info(f"Loaded {len(eval_dataset)} examples from {args.split} split")
    
    # Evaluate
    logger.info("Starting evaluation...")
    eval_results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=eval_dataset,
        task_name=args.task_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    # Print results
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation Results on {args.task_name} ({args.split} split)")
    logger.info(f"{'='*50}")
    logger.info(f"Accuracy: {eval_results['accuracy']:.2%}")
    logger.info(f"Correct: {eval_results['correct']} / {eval_results['total']}")
    logger.info(f"{'='*50}\n")
    
    # Save detailed results
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output_file, 'w') as f:
            json.dump({
                'accuracy': eval_results['accuracy'],
                'correct': eval_results['correct'],
                'total': eval_results['total'],
                'task': args.task_name,
                'split': args.split,
                'model_id': args.model_id,
                'adapter_path': args.adapter_path,
                'results': eval_results['results'],
            }, f, indent=2)
        
        logger.info(f"Detailed results saved to {args.output_file}")


if __name__ == "__main__":
    main()

