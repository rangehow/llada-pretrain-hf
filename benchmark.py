import torch
import time
from transformers import LlamaForCausalLM, ModernBertForMaskedLM, AutoConfig

# --- 1. 模型和设备初始化 ---
print("="*50)
print("Initializing models and device...")

# 检查是否有可用的GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    print("Warning: Running on CPU. Performance will be significantly slower and not representative of typical use cases.")

# 加载模型配置 (这是您提供的代码)
# 假设配置文件路径正确
try:
    llama_config = AutoConfig.from_pretrained("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/llama_400M.json")
    llama = LlamaForCausalLM(llama_config)

    modernbert_config = AutoConfig.from_pretrained("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/modernbert_large.json")
    modernbert = ModernBertForMaskedLM(modernbert_config)
except Exception as e:
    print(f"Error loading model configs: {e}")
    print("Please ensure the config paths are correct. Exiting.")
    exit()


# 将模型移动到指定设备，并设置为评估模式 .eval()
# .eval() 会禁用 dropout 等训练特有的层
llama.to(device).eval()
modernbert.to(device).eval()

print("Models initialized and moved to device.")
print("="*50)


# --- 2. 性能测试函数 ---

def measure_forward_speed(model, model_name, batch_size, seq_len, device, num_runs=20, warmup_runs=5):
    """
    测量模型一次前向传播的平均耗时。

    Args:
        model: 要测试的 PyTorch 模型。
        model_name (str): 模型的名称，用于打印。
        batch_size (int): 批处理大小。
        seq_len (int): 序列长度。
        device (str): "cuda" 或 "cpu"。
        num_runs (int): 用于计算平均值的运行次数。
        warmup_runs (int): 预热运行的次数。

    Returns:
        float: 平均耗时（毫秒）。
    """
    # 构造随机输入数据
    # token ID 的范围是 [0, vocab_size-1]
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

    # 禁用梯度计算，节省显存和计算
    with torch.no_grad():
        # 预热阶段
        for _ in range(warmup_runs):
            _ = model(input_ids)

        # 确保预热操作已在GPU上完成
        if device == "cuda":
            torch.cuda.synchronize()

        # 精确计时阶段
        # 使用 torch.cuda.Event 来精确测量GPU执行时间
        if device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            timings = []
            for _ in range(num_runs):
                start_event.record()
                _ = model(input_ids)
                end_event.record()
                
                # 等待GPU操作完成
                torch.cuda.synchronize()
                
                # 记录耗时（毫秒）
                timings.append(start_event.elapsed_time(end_event))
            
            avg_time_ms = sum(timings) / len(timings)

        else: # CPU 计时
            start_time = time.time()
            for _ in range(num_runs):
                _ = model(input_ids)
            end_time = time.time()
            avg_time_ms = ((end_time - start_time) / num_runs) * 1000

    return avg_time_ms

# --- 3. 定义测试配置并运行测试 ---

# 定义一系列 (batch_size, sequence_length) 的组合进行测试
# 注意：非常长的序列（如 > 8192）可能会导致显存溢出（Out of Memory）
# 我们从较小的尺寸开始，逐步增大
test_configs = [
    # Batch Size = 1, Sequence Length 变化
    (1, 512),
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (1, 8192),   # 在大多数消费级/中端GPU上可能开始遇到问题
    (1, 16384),  # 极有可能在标准GPU上失败
    
    # Sequence Length = 1024, Batch Size 变化
    (2, 1024),
    (4, 1024),
    (8, 1024),
]

models_to_test = {
    "Llama-400M": llama,
    "ModernBERT-Large": modernbert,
}

print("\nStarting performance benchmark...")
print("-" * 70)
print(f"{'Model Name':<20} | {'Batch Size':<12} | {'Seq Length':<12} | {'Avg Time (ms/pass)':<20}")
print("-" * 70)

for model_name, model in models_to_test.items():
    for bs, sl in test_configs:
        try:
            # 运行测试
            avg_time = measure_forward_speed(model, model_name, bs, sl, device)
            print(f"{model_name:<20} | {bs:<12} | {sl:<12} | {avg_time:<20.4f}")
        
        except torch.cuda.OutOfMemoryError:
            # 捕获显存溢出错误
            print(f"{model_name:<20} | {bs:<12} | {sl:<12} | {'Out of Memory (OOM)':<20}")
        except Exception as e:
            # 捕获其他可能的错误
            print(f"{model_name:<20} | {bs:<12} | {sl:<12} | Error: {str(e)[:30]}")
    print("-" * 70)