import argparse
import json
import logging
import os
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    LlamaForCausalLM,
)
from .collator import NTPCollator,LLaDACollator,CausalLMCollator
from .mlm_schedule import LazyScheduledMLMProbProvider,LazyMLMProbSchedulerCallback
from .configuration_niu import NiuConfig
from .llada.modeling_llada import LLaDAModelLM
from .llada.configuration_llada import LLaDAConfig
from .trainer import MultipleLossTrainer
from .utils.debug_func import analyze_weights,debug_data

# 在 main 函数的开始部分
import torch.multiprocessing as mp
from .utils.load_dataset import get_dataset
# --- 设置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
import ast

def is_main_process():
    """检查是否为主进程"""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True
    except:
        return True



def check_for_checkpoints(output_dir):
    """
    检查指定的输出目录下是否存在类似 checkpoint- 的文件夹（更简练的版本）。
    """
    import re
    return os.path.exists(output_dir) and any(
        os.path.isdir(os.path.join(output_dir, item)) and re.match(r"^checkpoint-", item)
        for item in os.listdir(output_dir)
    )



def main():
    # --- 1. 设置 ArgumentParser ---
    parser = argparse.ArgumentParser(description="使用可配置参数训练一个MLM模型")

    # 路径参数
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="预训练模型或本地模型/分词器的路径。")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="训练数据集的名称（如：finefineweb）。")

    parser.add_argument("--validation_dataset_name", type=str, default="finefineweb_validation",
                        help="验证数据集的名称（如：finefineweb_validation）。")
    parser.add_argument("--config_path", type=str, required=True,
                        help="模型配置文件的路径。")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="模型 checkpoints 和输出的保存路径。")
    parser.add_argument("--mode",default="llada")
    
    # MLM Schedule 参数
    parser.add_argument("--mlm_start_prob", type=float, default=0.25)
    parser.add_argument("--mlm_end_prob", type=float, default=0.15)
    parser.add_argument("--mlm_schedule_type", type=str, default='cosine')
    parser.add_argument("--tail_bias_factor", type=float, default=1.5)

    # 数据处理参数
    parser.add_argument("--max_length", type=int, default=512, help="输入序列的最大长度。")

    # TrainingArguments 参数
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)

    parser.add_argument("--per_device_eval_batch_size", type=int, default=2,
                        help="每个设备的评估批次大小。")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)

    parser.add_argument("--evaluation_strategy", type=str, default="steps",
                        help="评估策略 ('no', 'steps', 'epoch')。")
    parser.add_argument("--eval_steps", type=int, default=5000,
                        help="每隔多少步进行一次评估。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action='store_true')

    args = parser.parse_args()

    # --- 2. 打印和保存参数配置 ---
    # 只在主进程打印和保存参数配置
    if is_main_process():
        logging.info("=" * 80)
        logging.info("训练参数配置:")
        logging.info("=" * 80)
        args_dict = vars(args)
        for key, value in args_dict.items():
            logging.info(f"{key:30}: {value}")
        logging.info("=" * 80)
        
        os.makedirs(args.output_dir, exist_ok=True)
        args_json_path = os.path.join(args.output_dir, "training_args.json")
        with open(args_json_path, "w", encoding="utf-8") as f:
            json.dump(args_dict, f, ensure_ascii=False, indent=4)
        logging.info(f"所有参数已保存至: {args_json_path}")



    # --- 4. 加载数据集和分词器 ---
    if is_main_process():
        logging.info(f"加载训练数据集 '{args.dataset_name}'...")
    train_dataset = get_dataset(args.dataset_name)

    eval_dataset = None
    if args.validation_dataset_name:
        if is_main_process():
            logging.info(f"加载验证数据集 '{args.validation_dataset_name}'...")
        eval_dataset = get_dataset(args.validation_dataset_name)


    model_path =  args.model_name_or_path
    if is_main_process():
        logging.info(f"从路径 '{model_path}' 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.eos_token is None:
        tokenizer.eos_token_id = 50279
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = 50285

    shared_step = mp.Value('i', 0)


    if args.mode == 'llada':
        config = LLaDAConfig.from_pretrained(args.config_path)
        model = LLaDAModelLM(config,init_params=True)
        config.register_for_auto_class()
        model.register_for_auto_class("AutoModel")
    elif args.mode == 'llama':
        config = AutoConfig.from_pretrained(args.config_path)
        model = LlamaForCausalLM(config)
    else:
        assert False
    # analyze_weights(model)


    if args.mode == 'llama':
        collator = NTPCollator(tokenizer, max_length=args.max_length)
    elif args.mode == 'llada':
        collator = LLaDACollator(tokenizer,max_length=args.max_length)
        

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={'min_lr_rate': 0.01},
        warmup_ratio=args.warmup_ratio,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        data_seed=args.seed,
        seed=args.seed,
        bf16=True,
        adam_beta2 = 0.95,
        weight_decay = 0.1,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to='none',
        include_num_input_tokens_seen = True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size, 
        eval_strategy=args.evaluation_strategy, 
        eval_steps=args.eval_steps,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        # eval_on_start = True,
    )


    # --- 7. 初始化并开始训练 ---
    trainer = MultipleLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # <-- 变量名更新
        eval_dataset=eval_dataset,   # <-- 新增，传递验证集
        data_collator=collator,
        # callbacks=None if args.mode == 'llama' or args.mode == 'llada' else [lazy_prob_scheduler_callback],
        keys_you_want_to_log = ['lm_loss','current_mlm_prob','masked_lm_loss','non_masked_lm_loss']
    )


    # if is_main_process() and args.mode!='llama':
    #     debug_data(trainer, tokenizer, collator)

    if check_for_checkpoints(args.output_dir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()



    

if __name__ == "__main__":
    main()