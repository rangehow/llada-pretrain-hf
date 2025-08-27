import argparse
import json
from typing import Dict, List, Any
from itertools import cycle
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModel, 
    PreTrainedModel, 
    PreTrainedTokenizer
)

import ray

from task import dname2func, TaskConfig


class LogProbCollator:
    """A collator for batch processing log-probability evaluation data. It is generic and does not need changes."""
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids_list, continuation_masks, attention_masks = [], [], []
        group_ids, is_correct_list = [], []
        max_length = 0
        
        for item in batch:
            input_ids = item['input_ids']
            continuation_ids = item['continuation_ids']
            
            full_sequence = input_ids + continuation_ids
            input_ids_list.append(full_sequence)
            
            continuation_mask = [0] * len(input_ids) + [1] * len(continuation_ids)
            continuation_masks.append(continuation_mask)
            
            attention_mask = [1] * len(full_sequence)
            attention_masks.append(attention_mask)
            
            group_ids.append(item['group_id'])
            is_correct_list.append(item['is_correct'])
            max_length = max(max_length, len(full_sequence))
        
        padded_input_ids, padded_continuation_masks, padded_attention_masks = [], [], []
        for seq, cont_mask, att_mask in zip(input_ids_list, continuation_masks, attention_masks):
            pad_length = max_length - len(seq)
            padded_input_ids.append(seq + [self.pad_token_id] * pad_length)
            padded_continuation_masks.append(cont_mask + [0] * pad_length)
            padded_attention_masks.append(att_mask + [0] * pad_length)

        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_masks, dtype=torch.long),
            'continuation_mask': torch.tensor(padded_continuation_masks, dtype=torch.bool),
            'group_ids': group_ids,
            'is_correct': is_correct_list
        }


def compute_logprobs_causal(model: PreTrainedModel, batch: Dict[str, torch.Tensor], **kwargs) -> List[float]:
    input_ids = batch['input_ids'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)
    continuation_mask = batch['continuation_mask'].to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits
        logits_for_scoring = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = F.log_softmax(logits_for_scoring, dim=-1)
        token_logprobs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        continuation_mask_for_scoring = continuation_mask[:, 1:]
        masked_logprobs = token_logprobs * continuation_mask_for_scoring
        total_logprobs = masked_logprobs.sum(dim=1)
        
    return total_logprobs.cpu().tolist()


def compute_logprobs_masked(
    model: PreTrainedModel, 
    batch: Dict[str, torch.Tensor], 
    tokenizer: PreTrainedTokenizer
) -> List[float]:
    """
    Computes the pseudo-log-likelihood of continuation tokens using a Masked LM.
    This is the efficient, batched version.
    """
    input_ids = batch['input_ids']
    continuation_mask = batch['continuation_mask']
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    device = model.device

    if mask_token_id is None:
        raise ValueError("Tokenizer for Masked LM must have a `mask_token_id`.")

    batched_input_ids = []
    batched_attention_masks = []
    extraction_info = []
    
    for i in range(input_ids.shape[0]):
        sample_input_ids = input_ids[i]
        sample_cont_mask = continuation_mask[i]
        
        continuation_indices = torch.where(sample_cont_mask)[0]
        seq_len = torch.where(sample_input_ids != pad_token_id)[0].max() + 1
        
        for token_idx in continuation_indices:
            token_idx = token_idx.item()
            original_token_id = sample_input_ids[token_idx].item()
            
            masked_input_ids = sample_input_ids.clone()
            masked_input_ids[token_idx] = mask_token_id
            
            batched_input_ids.append(masked_input_ids[:seq_len])
            batched_attention_masks.append(torch.ones(seq_len, dtype=torch.long))
            
            extraction_info.append((i, token_idx, original_token_id))

    if not batched_input_ids:
        return [0.0] * input_ids.shape[0]

    padded_inputs = tokenizer.pad(
        {"input_ids": batched_input_ids},
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=padded_inputs['input_ids'].to(device),
            attention_mask=padded_inputs['attention_mask'].to(device)
        )
        logits = outputs.logits

    total_log_probs = [0.0] * input_ids.shape[0]
    log_preds = F.log_softmax(logits, dim=-1)

    for idx, (original_batch_idx, token_idx, original_token_id) in enumerate(extraction_info):
        token_log_prob = log_preds[idx, token_idx, original_token_id].item()
        total_log_probs[original_batch_idx] += token_log_prob
        
    return total_log_probs
    

def calculate_accuracy(group_results: Dict[str, List[Dict]]) -> tuple:
    correct_predictions = 0
    total_groups = len(group_results)
    if total_groups == 0:
        return 0.0, 0, 0
    for group_id, group_data in group_results.items():
        best_option = max(group_data, key=lambda x: x['logprob'])
        if best_option['is_correct'] == 1:
            correct_predictions += 1
    accuracy = correct_predictions / total_groups
    return accuracy, correct_predictions, total_groups


@ray.remote(num_gpus=1)
class Evaluator:
    def __init__(self, model_name: str, model_type: str, trust_remote_code: bool):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=trust_remote_code,
            use_fast=True
        )
        if self.tokenizer.pad_token_id is None and model_type == 'causal':
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code
            )
        except:
            self.model = AutoModel.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code
            )
        self.model.to('cuda').eval() 

        self.logprob_computer_fn = compute_logprobs_causal if model_type == 'causal' else compute_logprobs_masked

    def evaluate_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        batch_logprobs = self.logprob_computer_fn(
            model=self.model,
            batch=batch,
            tokenizer=self.tokenizer
        )
        
        results = []
        for logprob, group_id, is_correct in zip(batch_logprobs, batch['group_ids'], batch['is_correct']):
            result = {'group_id': group_id, 'logprob': logprob, 'is_correct': is_correct}
            results.append(result)
        return results


def evaluate_dataset_ray(
    actors: List[Evaluator],
    tokenizer: PreTrainedTokenizer,
    dataset,
    batch_size: int,
    num_workers: int
) -> tuple:
    print(f"Starting Ray evaluation on {len(dataset)} samples with {len(actors)} GPUs...")
    
    collator = LogProbCollator(tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collator,
        num_workers=num_workers
    )

    all_results = []
    group_results = {}
    
    result_futures = []
    actor_pool = cycle(actors)
    for batch in tqdm(dataloader, desc="Distributing batches to GPUs"):
        actor = next(actor_pool)
        result_futures.append(actor.evaluate_batch.remote(batch))

    for future in tqdm(result_futures, desc="Collecting results from GPUs"):
        batch_results = ray.get(future)
        all_results.extend(batch_results)
    
    for result in all_results:
        group_id = result['group_id']
        if group_id not in group_results:
            group_results[group_id] = []
        group_results[group_id].append(result)

    return all_results, group_results


def get_model_short_name(model_path: str) -> str:
    """
    从模型路径中提取简短的模型名称
    """
    path = Path(model_path)
    
    # 如果路径以 checkpoint-xxx 结尾，则取上一级目录名
    if path.name.startswith('checkpoint-'):
        model_name = path.parent.name
    else:
        model_name = path.name
    
    # 进一步清理模型名称
    # 移除可能的前缀路径信息
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    
    # 如果名称仍然很长，可以进行进一步简化
    if len(model_name) > 80:  # 可以根据需要调整长度限制
        # 尝试提取关键信息，但保持完整性
        parts = model_name.split('_')
        if len(parts) > 5:
            # 保留前几个关键部分和最后一个部分
            model_name = '_'.join(parts[:3] + parts[-1:])
    
    return model_name



def create_output_structure(output_dir: str, model_path: str) -> tuple:
    """
    创建更好的输出目录结构
    返回: (model_dir, model_short_name)
    """
    model_short_name = get_model_short_name(model_path)
    
    # 创建以模型名称命名的子目录
    model_dir = Path(output_dir) / model_short_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    return str(model_dir), model_short_name


def save_results(results_data: Dict, model_dir: str, task_name: str, model_short_name: str):
    """
    保存结果到文件，支持多种格式
    """
    # 1. 保存详细结果到JSON文件
    detailed_file = Path(model_dir) / f"{task_name}.json"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # 2. 保存简要结果到summary文件
    summary_file = Path(model_dir) / "summary.jsonl"
    summary_data = {
        'model': model_short_name,
        'task': task_name,
        'accuracy': results_data['accuracy'],
        'correct': results_data['correct_predictions'],
        'total': results_data['total_questions'],
        'timestamp': datetime.now().isoformat()
    }
    
    # 追加到summary文件
    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(summary_data, ensure_ascii=False) + '\n')
    
    print(f"✓ Detailed results saved to: {detailed_file}")
    print(f"✓ Summary appended to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate language models on log-probability tasks.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or name of the model.")
    parser.add_argument("--model_type", type=str, required=True, choices=['causal', 'masked'], help="Type of model.")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated list of tasks to evaluate (e.g., hellaswag,arc_challenge).")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size per GPU.")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples for testing. 0 for no limit.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save evaluation results.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--trust_remote_code", action='store_true', help="Trust remote code for models.")

    args = parser.parse_args()
    print("Arguments:", args)

    # 创建更好的输出目录结构
    model_dir, model_short_name = create_output_structure(args.output_dir, args.model_name_or_path)
    print(f"Results will be saved to: {model_dir}")
    print(f"Model short name: {model_short_name}")

    ray.init()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise ValueError("No GPUs found. This script requires at least one GPU.")
    print(f"Found {num_gpus} GPUs. Creating one actor per GPU.")

    actors = [
        Evaluator.remote(
            model_name=args.model_name_or_path,
            model_type=args.model_type,
            trust_remote_code=args.trust_remote_code
        ) for _ in range(num_gpus)
    ]
    
    tokenizer_for_collator = AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        trust_remote_code=args.trust_remote_code, 
        use_fast=True
    )
    if tokenizer_for_collator.pad_token_id is None and args.model_type == 'causal':
        tokenizer_for_collator.pad_token_id = tokenizer_for_collator.eos_token_id

    if args.model_type == 'masked' and args.batch_size > 2:
        print(f"Warning: Batch size {args.batch_size} might be very slow for masked LMs. Consider reducing it.")

    # 存储所有任务的结果，用于最终汇总
    all_task_results = []

    tasks = args.tasks.split(',')
    for task_name in tasks:
        task_name = task_name.strip()
        if not task_name: continue

        print(f"\n----- Starting evaluation for task: {task_name} -----")
        
        config = TaskConfig(tokenizer=tokenizer_for_collator)
        if task_name not in dname2func:
            print(f"Error: Task '{task_name}' not found. Skipping.")
            continue
        dataset = dname2func[task_name](config)
        if args.limit > 0:
            dataset = dataset.select(range(min(args.limit, len(dataset))))


        results, group_results = evaluate_dataset_ray(
            actors=actors,
            tokenizer=tokenizer_for_collator,
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        accuracy, correct, total = calculate_accuracy(group_results)

        print(f"\n=== {task_name.upper()} Evaluation Results ===")
        print(f"Model: {model_short_name}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) ({correct}/{total})")

        # 准备保存的数据
        output_data = {
            'model_name': args.model_name_or_path,
            'model_short_name': model_short_name,
            'model_type': args.model_type,
            'task': task_name,
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_questions': total,
            'timestamp': datetime.now().isoformat(),
            'args': vars(args),
            'detailed_results': results
        }
        
        # 保存结果
        save_results(output_data, model_dir, task_name, model_short_name)
        
        # 添加到汇总列表
        all_task_results.append({
            'task': task_name,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        })

    # 保存所有任务的汇总结果
    if all_task_results:
        overall_summary = {
            'model_name': args.model_name_or_path,
            'model_short_name': model_short_name,
            'model_type': args.model_type,
            'timestamp': datetime.now().isoformat(),
            'tasks': all_task_results,
            'average_accuracy': sum(r['accuracy'] for r in all_task_results) / len(all_task_results),
            'args': vars(args)
        }
        
        summary_path = Path(model_dir) / "overall_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(overall_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Overall summary saved to: {summary_path}")
        print(f"✓ Average accuracy across all tasks: {overall_summary['average_accuracy']:.4f}")

    ray.shutdown()


if __name__ == "__main__":
    main()
