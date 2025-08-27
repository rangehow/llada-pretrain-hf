import os
import datasets
from transformers import AutoTokenizer
from tqdm import tqdm

# --- 1. 配置参数 ---
# 定义输入输出路径和模型参数，方便修改
DATASET_PATH = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/dataset/m-a-p/FineFineWeb-sample/main'
TOKENIZER_PATH = '/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/answerdotai/ModernBERT-base/main'
OUTPUT_PATH = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/datasets/filtered_finefineweb'
MAX_LENGTH = 4096
NUM_PROC = 64  # 并行处理的进程数
BATCH_SIZE = 1000 # map操作的批处理大小

# --- 2. 加载数据集和Tokenizer ---
print("Step 1: Loading dataset and tokenizer...")

# 加载数据集，利用多进程加速
# 注意：datasets.load_dataset已经为您处理好了数据加载的并行化
dataset = datasets.load_dataset(DATASET_PATH, num_proc=NUM_PROC,split='train')
print(f"  - Original dataset loaded. Size: {len(dataset):,} examples.")

# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
print("  - Tokenizer loaded successfully.")

# --- 3. 计算每个样本的Token长度（高效、并行、批处理） ---
print(f"\nStep 2: Calculating token length for each example (batched=True, num_proc={NUM_PROC})...")

def get_token_length(batch):
    """
    一个批处理函数，用于计算batch中每条文本的token长度。
    tokenizer处理一个batch的文本列表比单条处理快得多。
    """
    # 我们只关心长度，所以不需要padding或truncation
    tokenized_output = tokenizer(batch['text'], truncation=False, padding=False)
    # 将计算出的长度列表作为一个新列'token_length'添加到batch中
    batch['token_length'] = [len(ids) for ids in tokenized_output['input_ids']]
    return batch

# 使用 .map() 并行和批处理地添加'token_length'列
# 这是整个流程中最高效的部分
dataset_with_lengths = dataset.map(
    get_token_length,
    batched=True,
    batch_size=BATCH_SIZE,
    num_proc=NUM_PROC,
    desc="Calculating token lengths" # 添加tqdm进度条描述
)

print(f"  - Finished calculating token lengths.")

# --- 4. 过滤数据集 ---
print(f"\nStep 3: Filtering out examples longer than {MAX_LENGTH} tokens (num_proc={NUM_PROC})...")

# .filter() 同样支持多进程并行，速度很快
filtered_dataset = dataset_with_lengths.filter(
    lambda example: example['token_length'] <= MAX_LENGTH,
    num_proc=NUM_PROC,
    desc="Filtering dataset" # 添加tqdm进度条描述
)

print(f"  - Filtering complete.")
print(f"  - Original dataset size: {len(dataset):,}")
print(f"  - Filtered dataset size:  {len(filtered_dataset):,}")
print(f"  - Number of examples removed: {len(dataset) - len(filtered_dataset):,}")


# --- 5. 统计剩余数据集的总Token数 ---
print("\nStep 4: Calculating total tokens in the filtered dataset...")

# 从已经计算好的 'token_length' 列求和
# 这种方式比重新tokenize高效得多，并且内存友好，因为它会流式处理数据
# total_tokens = sum(filtered_dataset['token_length'])

# # 如果数据集非常大，担心内存，也可以使用迭代器的方式，效果类似
# total_tokens = 0
# for length in tqdm(filtered_dataset.select_columns(['token_length']), desc="Summing tokens"):
#     total_tokens += length['token_length']


# print(f"  - Total number of tokens in the filtered dataset: {total_tokens:,}")

# --- 6. 清理并保存最终数据集 ---
print(f"\nStep 5: Saving the final dataset to disk...")

# 在保存前，移除临时的'token_length'列，保持数据集干净
final_dataset = filtered_dataset.remove_columns(['token_length'])

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 保存到磁盘
final_dataset.save_to_disk(OUTPUT_PATH,max_shard_size='50MB')

print(f"  - Dataset successfully saved to: {OUTPUT_PATH}")
print("\nDone!")