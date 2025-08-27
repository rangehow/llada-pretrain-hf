"""
task.py

这个文件定义了将各种 Hugging Face 数据集转换为统一格式以进行对数概率（log-probability）评测的逻辑。
核心是 `_finalize_dataset_for_logprob` 函数，它将每个多项选择题扩展为 N 行（N是选项数量），
每一行代表一个可能的答案，并包含计算 logprob 所需的所有信息。

每个任务函数（例如 `mmlu`, `hellaswag`）都使用 @register2dict 装饰器进行注册，
以便在 main.py 中可以通过任务名称字符串动态调用。
"""

import datasets
from dataclasses import dataclass
from typing import Any, Optional, Dict

# --- 1. Configuration Dataclass ---
@dataclass
class TaskConfig:
    """
    Configuration for data processing tasks.
    """
    tokenizer: Any
    local_dir: Optional[str] = None


# --- 2. Registration Decorator and Dictionary ---
dname2func: Dict[str, Any] = {}
def register2dict():
    def decorator(func):
        if func.__name__ not in dname2func:
            dname2func[func.__name__] = func
        else:
            print(f"Error: Function name '{func.__name__}' is already registered.")
            exit(1)
        return func
    return decorator


# --- 3. Core Finalizing Function (REFACTORED & UNIFIED) ---

def _finalize_dataset_for_logprob(
    dataset_to_finalize: datasets.Dataset,
    task_name_str: str,
    config: TaskConfig,
    question_col: str,
    options_col: str,
    answer_col: str,
    is_sentence_completion: bool = False
) -> datasets.Dataset:
    """
    [重构后] 将多项选择题数据集转换为 Logprob 评测格式。
    此函数是统一的处理入口，直接处理原始数据，避免了多阶段处理带来的状态不一致问题。

    Args:
        dataset_to_finalize: 原始数据集。
        task_name_str: 任务名称 (用于日志)。
        config: 配置对象，主要用于获取 tokenizer。
        question_col: 数据集中表示“问题”或“上下文”的列名。
        options_col: 数据集中表示“选项”的列名 (应为 list of strings)。
        answer_col: 数据集中表示“答案”的列名。
        is_sentence_completion: 任务类型标志。True 表示句子补全 (如 HellaSwag)，
                                 False 表示标准多选题 (如 MMLU)。
    """
    tokenizer = config.tokenizer

    def expand_and_tokenize(batch):
        new_batch = {
            "input_ids": [], "continuation_ids": [], "is_correct": [],
            "group_id": [], "task_name": []
        }

        for i in range(len(batch[question_col])):
            group_id = batch["id"][i]
            question_text = batch[question_col][i]
            options = batch[options_col][i]
            labels = batch[answer_col][i]

            # --- 核心修复：在单一函数内统一处理各种答案格式 ---
            original_answer_index = -1
            if isinstance(labels, str):
                if labels.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    original_answer_index = ord(labels.upper()) - ord('A')
                elif labels.isdigit():
                    original_answer_index = int(labels)
            elif isinstance(labels, list): # For [0, 1, 0, 0] format
                original_answer_index = labels.index(1) if 1 in labels else -1
            elif isinstance(labels, int): # For 0, 1, 2, 3 format
                original_answer_index = labels

            if original_answer_index == -1 or original_answer_index >= len(options):
                continue # Skip invalid samples
            # --- 答案处理结束 ---

            # 根据任务类型构建 prompt 和 continuation
            if is_sentence_completion:
                prompt_str = question_text.strip()
                input_ids = tokenizer(prompt_str, add_special_tokens=False)['input_ids']
                if tokenizer.bos_token_id is not None:
                    input_ids = [tokenizer.bos_token_id] + input_ids

                for j, opt_text in enumerate(options):
                    continuation_str = " " + opt_text
                    continuation_ids = tokenizer(continuation_str, add_special_tokens=False)['input_ids']
                    
                    new_batch["input_ids"].append(input_ids)
                    new_batch["continuation_ids"].append(continuation_ids)
                    new_batch["is_correct"].append(1 if j == original_answer_index else 0)
                    new_batch["group_id"].append(group_id)
                    new_batch["task_name"].append(task_name_str)
            else: # Standard Multiple-Choice Question format
                prompt_str = question_text + "\n\n"
                for j, opt in enumerate(options):
                    label = chr(ord('A') + j)
                    prompt_str += f"{label}. {opt}\n"
                
                prompt_str = prompt_str.strip() + "\n\nAnswer:"
                input_ids = tokenizer(prompt_str, add_special_tokens=False)['input_ids']
                if tokenizer.bos_token_id is not None:
                    input_ids = [tokenizer.bos_token_id] + input_ids

                for j in range(len(options)):
                    choice_label = chr(ord('A') + j)
                    continuation_str = " " + choice_label
                    continuation_ids = tokenizer(continuation_str, add_special_tokens=False)['input_ids']

                    new_batch["input_ids"].append(input_ids)
                    new_batch["continuation_ids"].append(continuation_ids)
                    new_batch["is_correct"].append(1 if j == original_answer_index else 0)
                    new_batch["group_id"].append(group_id)
                    new_batch["task_name"].append(task_name_str)

        return new_batch

    if "id" not in dataset_to_finalize.column_names:
        dataset_to_finalize = dataset_to_finalize.add_column("id", range(len(dataset_to_finalize)))

    final_dataset = dataset_to_finalize.map(
        expand_and_tokenize,
        batched=True,
        remove_columns=dataset_to_finalize.column_names,
        desc=f"[{task_name_str}] Expanding for logprob evaluation"
    )
    return final_dataset


# --- 4. Task-Specific Functions (SIMPLIFIED) ---

@register2dict()
def mmlu_pro(config: TaskConfig) -> datasets.Dataset:
    task_name = "mmlu_pro"
    print(f"Processing task: {task_name} (Logprob Mode)")
    dataset = datasets.load_dataset('TIGER-Lab/MMLU-Pro', split='test', cache_dir=config.local_dir)
    return _finalize_dataset_for_logprob(
        dataset, task_name, config,
        question_col="question",
        options_col="options",
        answer_col="answer" # 'A', 'B', etc.
    )

@register2dict()
def mmlu(config: TaskConfig) -> datasets.Dataset:
    task_name = "mmlu"
    print(f"Processing task: {task_name} (Logprob Mode)")
    dataset = datasets.load_dataset('cais/mmlu', 'all', split='test', cache_dir=config.local_dir)
    return _finalize_dataset_for_logprob(
        dataset, task_name, config,
        question_col="question",
        options_col="choices",
        answer_col="answer" # integer index
    )

@register2dict()
def truthfulqa(config: TaskConfig) -> datasets.Dataset:
    task_name = "truthfulqa"
    print(f"Processing task: {task_name} (Logprob Mode)")
    dataset = datasets.load_dataset('truthful_qa', 'multiple_choice', split='validation', cache_dir=config.local_dir)
    
    # Pre-process to flatten the nested columns
    def map_prepare_cols(example):
        return {
            "question": example['question'],
            "options": example['mc1_targets']['choices'],
            "answer": example['mc1_targets']['labels'] # list of 0/1
        }
    dataset = dataset.map(map_prepare_cols, remove_columns=dataset.column_names)
    
    return _finalize_dataset_for_logprob(
        dataset, task_name, config,
        question_col="question",
        options_col="options",
        answer_col="answer"
    )

@register2dict()
def gpqa_diamond(config: TaskConfig) -> datasets.Dataset:
    task_name = "gpqa_diamond"
    print(f"Processing task: {task_name} (Logprob Mode)")
    dataset = datasets.load_dataset('Idavidrein/gpqa', 'gpqa_diamond', split='train', cache_dir=config.local_dir)

    # Pre-process to combine options into a single list
    def map_prepare_cols(example):
        options = [
            example['Correct Answer'], 
            example['Incorrect Answer 1'],
            example['Incorrect Answer 2'],
            example['Incorrect Answer 3']
        ]
        return {
            "question": example['Question'],
            "options": options,
            "answer": 0 # Correct answer is always the first one in the original list
        }
    dataset = dataset.map(map_prepare_cols, remove_columns=dataset.column_names)
    
    return _finalize_dataset_for_logprob(
        dataset, task_name, config,
        question_col="question",
        options_col="options",
        answer_col="answer"
    )

@register2dict()
def hellaswag(config: TaskConfig) -> datasets.Dataset:
    task_name = "hellaswag"
    print(f"Processing task: {task_name} (Logprob Mode)")
    dataset = datasets.load_dataset('Rowan/hellaswag', split='validation', cache_dir=config.local_dir)
    return _finalize_dataset_for_logprob(
        dataset, task_name, config,
        question_col="ctx",
        options_col="endings",
        answer_col="label", # string '0', '1', etc.
        is_sentence_completion=True
    )

@register2dict()
def arc_easy(config: TaskConfig) -> datasets.Dataset:
    task_name = "arc_easy"
    print(f"Processing task: {task_name} (Logprob Mode)")
    dataset = datasets.load_dataset('ai2_arc', 'ARC-Easy', split='validation', cache_dir=config.local_dir)

    # Pre-process to convert answerKey ('A', '1', etc.) to an integer index
    def map_prepare_cols(example):
        try:
            # Find the index of the answerKey in the label list
            correct_index = example['choices']['label'].index(example['answerKey'])
        except (ValueError, KeyError):
            correct_index = -1 # Mark as invalid if not found
        return {
            "question": example['question'],
            "options": example['choices']['text'],
            "answer": correct_index
        }
    dataset = dataset.map(map_prepare_cols, remove_columns=dataset.column_names)
    
    # Filter out any samples that had invalid answers
    dataset = dataset.filter(lambda x: x['answer'] != -1)
    
    return _finalize_dataset_for_logprob(
        dataset, task_name, config,
        question_col="question",
        options_col="options",
        answer_col="answer"
    )