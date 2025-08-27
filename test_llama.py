import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def load_model_and_tokenizer(model_path):
    """加载模型和tokenizer"""
    print(f"正在加载模型: {model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=128, temperature=1, top_p=0.9):
    """生成文本续写"""
    # 编码输入，不自动添加special tokens
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    
    # 获取模型设备
    device = next(model.parameters()).device
    
    # 将输入移到模型设备上
    inputs = inputs.to(device)
    
    # 手动在开头添加BOS token
    if tokenizer.bos_token_id is not None:
        bos_token = torch.tensor([[tokenizer.bos_token_id]], dtype=inputs.dtype, device=device)
        inputs = torch.cat([bos_token, inputs], dim=1)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    return generated_text

def main():
    model_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/llama_400M_finefineweb/checkpoint-75500"
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        return
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # 测试prompts
    test_prompts = [
        "What's your hobby?",
        "I love reading books.",
        "The weather is",
        "My favorite food is",
        "How do you",
        "Yesterday I went to",
        "The most important thing in life is",
        "Technology has changed",
        "When I was young",
        "In the future"
    ]
    
    print("=" * 60)
    print("开始测试Llama模型续写功能")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[测试 {i}/10]")
        print(f"Prompt: {prompt}")
        print("-" * 40)
        
        try:
            # --- 【调用方式调整】 ---
            # 现在 generate_text 函数直接返回了续写部分
            continuation = generate_text(model, tokenizer, prompt)
            print(f"续写: {continuation.strip()}") # strip() 仍然有用，可以去除首尾多余的空格
            
        except Exception as e:
            print(f"生成失败: {str(e)}")
        
        print("-" * 40)

if __name__ == "__main__":
    main()