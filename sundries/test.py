import torch

from transformers import AutoTokenizer,AutoModel


torch.set_float32_matmul_precision('high')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 检查GPU是否可用
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


print(f"Using device: {device}")

# model_path ="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/niu_1B_100b_finefineweb_lowvariance/checkpoint-584500"
# model_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/niu_400M_finefineweb_lowvariance/checkpoint-309339"
model_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/llada_1B_100b_finefineweb/checkpoint-256000"
print(model_path)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
print("Loading model...")
model = AutoModel.from_pretrained(model_path,trust_remote_code=True,torch_dtype=torch.bfloat16)
model.to(device)  # 移动模型到GPU
model.eval()  # 设置为评估模式




# 测试输入 - 添加多个不同类型的测试用例
test_inputs = [
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

# 遍历每个测试输入
for i, test_input in enumerate(test_inputs):
 
    print(f"\n{'='*50}")
    print(f"Test {i+1}: {test_input}")
    print('='*50)

    # 手动encode输入，不添加特殊tokens
    input_ids = tokenizer.encode(test_input, add_special_tokens=False, return_tensors="pt").to(device)

    # 添加BOS token到input_ids前面
    bos_token_id = tokenizer.bos_token_id
    bos_tensor = torch.tensor([[bos_token_id]], device=device)
    input_ids = torch.cat([bos_tensor, input_ids], dim=1)

    attention_mask = torch.ones_like(input_ids).to(device)  # 手动创建attention mask


    print(f"Input shape: {input_ids.shape}")
    print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

    # 获取mask token id
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        # 如果tokenizer没有mask token，使用[MASK]或者vocab中的特殊token
        mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
        if mask_token_id == tokenizer.unk_token_id:
            # 如果还是没有，使用pad token作为替代
            mask_token_id = tokenizer.pad_token_id



    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_token_id=mask_token_id,
            max_new_tokens=128,  
            num_diffusion_steps=1000, 
            temperature=1.0,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            debug=False,  
            tokenizer=tokenizer,
            use_token_change_classifier=False,
            decode_top_k_positions=1,
            force_ar_progression=True,
            block_size=32,
        )
    
    # 显示新生成的部分
    original_length = input_ids.shape[1]
    new_tokens = generated_ids[0][original_length:]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    print(f"Generated: {test_input} -> {new_text}")
    # import pdb
    # pdb.set_trace()
