
from torch import nn
def analyze_weights(model: nn.Module):
    """
    递归检测并打印PyTorch模型中每个权重矩阵的均值和标准差。

    Args:
        model (nn.Module): 需要分析的PyTorch模型。
    """
    print("--- 开始分析模型权重 ---")
    
    # model.named_parameters() 会递归地返回模型所有参数的 (名称, 张量) 对
    for name, param in model.named_parameters():
        # 我们通过检查张量的维度来区分权重和偏置
        # 权重矩阵的维度通常 > 1, 而偏置的维度等于 1
        if param.dim() > 1:
            # 使用 torch.no_grad() 来确保我们不会计算梯度，这能节省内存和计算资源
            with torch.no_grad():
                mean = param.data.mean()
                std = param.data.std()
                print(f"层/参数名称: {name:<25} | 均值 (Mean): {mean:+.6f} | 标准差 (Std): {std:.6f}")
                
    print("--- 模型权重分析完成 ---")


def debug_data(trainer,tokenizer,collator):
    dataloader = trainer.get_train_dataloader()

    # 检查collator的结果
    logging.info("开始检查collator的结果...")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 1:  # 只检查前1个batch
            break
            
        logging.info(f"\n=== Batch {batch_idx + 1} ===")
        logging.info(f"Batch大小: {batch['input_ids'].shape[0]}")
        logging.info(f"序列长度: {batch['input_ids'].shape[1]}")
        
        # 检查是否存在token_change_labels
        has_token_change_labels = 'token_change_labels' in batch
        
        # 检查前1个样本
        for sample_idx in range(min(1, batch['input_ids'].shape[0])):
            logging.info(f"\n--- 样本 {sample_idx + 1} ---")
            
            input_ids = batch['input_ids'][sample_idx]
            labels = batch['labels'][sample_idx]
            token_change_labels = batch['token_change_labels'][sample_idx] if has_token_change_labels else None
            attention_mask = batch['attention_mask'][sample_idx]
            
            # 转换为tokens
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # 找到有效长度（排除padding）
            valid_length = attention_mask.sum().item()
            total_length = len(input_ids)
            
            logging.info(f"总token数量: {total_length}")
            logging.info(f"非pad token数量: {valid_length}")
            logging.info(f"padding token数量: {total_length - valid_length}")
            
            current_mlm_prob = collator._get_mlm_probability()
            logging.info(f"当前MLM概率: {current_mlm_prob:.3f}")
            
            if has_token_change_labels:
                logging.info("\n详细Token信息:")
                logging.info("格式: [位置] Token名称 (ID) | 输入状态 | MLM标签 | 变化标签 | 注意力")
            else:
                logging.info("\n详细Token信息 (无变化标签):")
                logging.info("格式: [位置] Token名称 (ID) | 输入状态 | MLM标签 | 注意力")
            logging.info("-" * 100)
            
            # 统计计数器
            masked_count = 0
            random_count = 0
            keep_count = 0
            original_count = 0
            pad_count = 0
            
            # 显示所有token，包括pad
            for i in range(total_length):
                token_id = input_ids[i].item()
                token = tokens[i]
                label = labels[i].item()
                change_label = token_change_labels[i].item() if has_token_change_labels else None
                attention = attention_mask[i].item()
                
                # 判断是否为padding
                is_padding = (attention == 0)
                
                if is_padding:
                    # padding token
                    status = "PAD"
                    mlm_label_info = f"标签:{label}"
                    change_info = f"变化:{change_label}" if has_token_change_labels else ""
                    pad_count += 1
                else:
               
                    if token == tokenizer.mask_token:
                        status = f"MASK"
                        mlm_label_info = f"{tokenizer.decode(label)}"
                        masked_count += 1
                        change_info = f"变化:{change_label}" if has_token_change_labels else ""
                    elif token_id == label:
                        status = f"KEEP"
                        mlm_label_info = f"{tokenizer.decode(label)}"
                        keep_count +=1 
                        change_info = f"变化:{change_label}" if has_token_change_labels else ""
                    elif label!=-100:
                        status = f"随机替换"
                        mlm_label_info = f"{tokenizer.decode(label)}"
                        random_count += 1
                        change_info = f"变化:{change_label}" if has_token_change_labels else ""
                    else:
                        status = "无需计算loss"
                        mlm_label_info = "标签:无(-100)"
                        change_info = f"变化:{change_label}" if has_token_change_labels else ""
                        original_count += 1

                
                # 显示token信息
                attention_info = f"注意力:{attention}"
                if has_token_change_labels:
                    logging.info(f"[{i:2d}] {token:6}  | {status:10} | {mlm_label_info:20} | {change_info:6} | {attention_info}")
                else:
                    logging.info(f"[{i:2d}] {token:6}  | {status:10} | {mlm_label_info:20} | {attention_info}")
            
            # 详细统计信息
            logging.info(f"\n=== 统计信息 ===")
            logging.info(f"总token数量: {total_length}")
            logging.info(f"有效token数量: {valid_length}")
            logging.info(f"padding token数量: {pad_count}")
            logging.info(f"")
            logging.info(f"MLM处理统计:")
            logging.info(f"  - 原始未处理: {original_count}")
            logging.info(f"  - MASK替换: {masked_count}")
            if has_token_change_labels:
                logging.info(f"  - 随机替换: {random_count}")
                logging.info(f"  - 保持原样: {keep_count}")
            logging.info(f"  - 总MLM处理: {masked_count + random_count + keep_count}")
            logging.info(f"")
            logging.info(f"比例统计:")
            logging.info(f"  - 当前MLM概率设置: {current_mlm_prob:.3f}")
            if valid_length > 0:
                actual_mlm_ratio = (masked_count + random_count + keep_count) / valid_length
                mask_ratio = masked_count / valid_length
                
                logging.info(f"  - 实际MLM处理比例: {actual_mlm_ratio:.3f}")
                logging.info(f"  - MASK比例: {mask_ratio:.3f}")
                
                if has_token_change_labels:
                    random_ratio = random_count / valid_length
                    keep_ratio = keep_count / valid_length
                    logging.info(f"  - 随机替换比例: {random_ratio:.3f}")
                    logging.info(f"  - 保持原样比例: {keep_ratio:.3f}")
            
            # 显示标签分布
            label_distribution = {}
            change_label_distribution = {} if has_token_change_labels else None
            
            for i in range(total_length):
                label = labels[i].item()
                label_distribution[label] = label_distribution.get(label, 0) + 1
                
                if has_token_change_labels:
                    change_label = token_change_labels[i].item()
                    change_label_distribution[change_label] = change_label_distribution.get(change_label, 0) + 1
            
            logging.info(f"\n=== 标签分布 ===")
            logging.info(f"MLM标签分布: {label_distribution}")
            if has_token_change_labels:
                logging.info(f"变化标签分布: {change_label_distribution}")
            
            # 显示原始文本和处理后文本的对比
            try:
                # 重建原始文本
                original_ids = input_ids.clone()
                for i in range(total_length):
                    if labels[i].item() != -100:
                        original_ids[i] = labels[i]
                
                # 只显示有效部分的文本
                original_text = tokenizer.decode(original_ids[:valid_length], skip_special_tokens=False)
                current_text = tokenizer.decode(input_ids[:valid_length], skip_special_tokens=False)
                
                logging.info(f"\n=== 文本对比 ===")
                logging.info(f"原始文本: {original_text}")
                logging.info(f"处理后文本: {current_text}")
                
                # 显示差异
                original_tokens = tokenizer.convert_ids_to_tokens(original_ids[:valid_length])
                current_tokens = tokenizer.convert_ids_to_tokens(input_ids[:valid_length])
                
                differences = []
                for i, (orig, curr) in enumerate(zip(original_tokens, current_tokens)):
                    if orig != curr:
                        differences.append(f"位置{i}: {orig} → {curr}")
                
                if differences:
                    logging.info(f"Token差异: {'; '.join(differences)}")
                else:
                    logging.info("没有Token差异")
                    
            except Exception as e:
                logging.warning(f"无法重建文本对比: {e}")



def count_token_of_dataset(dataset,tokenizer):
    logging.info("=" * 80)
    logging.info("开始统计数据集的总 token 数量（这可能需要一些时间）...")

    # 定义一个用于 map 的函数，它对一批文本进行分词并返回每个文本的 token 数量
    def count_tokens_in_batch(batch):
        # 对 'text' 列中的所有文本进行分词，不进行填充或截断，以获得真实 token 数量
        # add_special_tokens=False 确保我们只计算内容本身的 token
        tokenized_texts = tokenizer(batch['text'], truncation=False, add_special_tokens=False)
        # 返回一个新列，其中包含每个文本的 token 数量列表
        return {"num_tokens": [len(ids) for ids in tokenized_texts['input_ids']]}

    # 使用 map 方法并行处理数据集
    # - batched=True: 允许 count_tokens_in_batch 一次处理多行数据，效率更高
    # - num_proc: 指定并行处理的进程数，可以显著加速
    # - remove_columns: 处理后只保留新创建的 'num_tokens' 列，以节省内存
    token_counts_dataset = dataset.map(
        count_tokens_in_batch,
        batched=True,
        num_proc=16,  # 复用 dataloader 的 worker 数量作为并行数
        remove_columns=dataset.column_names
    )

    # 对新数据集中 'num_tokens' 列的所有值求和
    total_tokens = sum(token_counts_dataset['num_tokens'])

    logging.info(f"数据集总 token 数量统计完成。")
    # 使用逗号格式化数字，方便阅读
    logging.info(f">>> 数据集 '{args.dataset_name}' 的总 token 数量为: {total_tokens:,}")
    logging.info("=" * 80)