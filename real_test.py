import datasets
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig,AutoModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import traceback
from pathlib import Path
from llada.modeling_llada import LLaDAModelLM
# 动态导入 NLTK
try:
    from nltk.util import ngrams
except ImportError:
    print("NLTK not found. Please install it for diversity metrics: pip install nltk")
    def ngrams(sequence, n): return []

from diffusion.modeling_niu import ModernBertForDiffusionLM
from collator import MLMCollator

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    一个健壮的模型评估器，用于比较不同模型的性能。
    """

    def __init__(self, tokenizer_path: str, dataset_path: str, device: str = 'cuda'):
        self.device = device
        self.model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        logger.info(f"模型将使用数据类型: {self.model_dtype}")

        # ... (其他 __init__ 代码保持不变) ...
        logger.info(f"加载分词器: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        if self.tokenizer.eos_token is None: self.tokenizer.eos_token_id = 50279

        logger.info(f"加载数据集: {dataset_path}")
        full_dataset = datasets.load_dataset(dataset_path, split='train', streaming=True)
        self.test_dataset = list(full_dataset.shuffle(seed=42).take(1000))



        self.prompts = [
            "The future of artificial intelligence is", "To make a delicious pizza from scratch, you need to",
            "Once upon a time, in a land far, far away,", "The most important thing in life is",
            "Technology has changed our world by", "The capital of France is",
            "Watch out,"
        ]

        self.generation_params = {
            "max_new_tokens": 50, "num_diffusion_steps": 10, "temperature_mlm": 1.0,
            "do_sample": False, "top_k": 50,
        }

        logger.info("加载外部评估模型 (Qwen3-0.6B-Base)...")
        self.eval_model = AutoModelForCausalLM.from_pretrained(
            '/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen3-0.6B-Base/main',
            torch_dtype=self.model_dtype
        ).to(device)
        self.eval_tokenizer = AutoTokenizer.from_pretrained('/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen3-0.6B-Base/main')
        if self.eval_tokenizer.pad_token is None:
            self.eval_tokenizer.pad_token = self.eval_tokenizer.eos_token

        self.result_columns = [
            "Model", "Perplexity", "Avg MLM Loss", "TC Accuracy", "TC F1-Score",
            "TC Precision", "TC Recall", "Gen Diversity (dist-1)",
            "Gen Diversity (dist-2)", "Gen PPL (by Qwen3)"  # 更新列名
        ]

    def _load_model_safe(self, model_path: str):
        """安全地加载模型，并强制使用正确的Dtype。"""
        logger.info(f"尝试从 '{model_path}' 加载模型...")

        if 'llada' in model_path:
            model = LLaDAModelLM.from_pretrained(
                model_path,
                torch_dtype=self.model_dtype,
                trust_remote_code=True
            )
            logger.info("成功使用加载LLADA模型。")
        else:
            # 方案A: 使用 from_pretrained，并直接指定 torch_dtype
            model = ModernBertForDiffusionLM.from_pretrained(
                model_path,
                torch_dtype=self.model_dtype  # <--- FIX: 核心修复点
            )
            logger.info("成功使用加载MODERNBERT模型。")
        return model
    



    def _calculate_diversity(self, texts):
        tokens = [text.split() for text in texts]
        unigrams = [item for sublist in tokens for item in sublist]
        bigrams = [item for sublist in tokens for item in ngrams(sublist, 2)]
        dist_1 = len(set(unigrams)) / len(unigrams) if len(unigrams) > 0 else 0
        dist_2 = len(set(bigrams)) / len(bigrams) if len(bigrams) > 0 else 0
        return dist_1, dist_2

    @torch.inference_mode()
    def _calculate_generation_ppl(self, texts, batch_size=1):
        self.eval_model.eval()
        total_loss, total_tokens = 0, 0
        for i in tqdm(range(0, len(texts), batch_size), desc="评估生成文本PPL (by Qwen3)"):  # 更新描述
            batch = texts[i:i+batch_size]
            if not any(batch): continue
            inputs = self.eval_tokenizer(batch, return_tensors='pt', truncation=True, max_length=512).to(self.device)
            labels = inputs.input_ids.clone()

            with torch.autocast(device_type=self.device, dtype=self.model_dtype):
                outputs = self.eval_model(**inputs, labels=labels)

            token_counts = (labels != -100).sum()
            if token_counts > 0:
                total_loss += outputs.loss.item() * token_counts
                total_tokens += token_counts
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        return np.exp(avg_loss.cpu()) if hasattr(avg_loss, 'cpu') else np.exp(avg_loss)

    @torch.inference_mode()
    def _evaluate_generation(self, model):
        model.eval()
        generated_texts = []
        for prompt in tqdm(self.prompts, desc="生成文本续写"):
            input_ids = self.tokenizer.encode(prompt, return_tensors=None,add_special_tokens=False)
            input_ids = torch.tensor([[self.tokenizer.bos_token_id] + input_ids],device=self.device)

            with torch.autocast(device_type=self.device, dtype=self.model_dtype):
                output_ids = model.generate(
                    input_ids=input_ids,
                    mask_token_id=self.tokenizer.mask_token_id,
                    debug = True,
                    use_token_change_classifier=False,
                    tokenizer=self.tokenizer,
                    **self.generation_params
                )

            full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_part = full_text[len(prompt):].strip()
            generated_texts.append(generated_part)

        dist_1, dist_2 = self._calculate_diversity(generated_texts)
        gen_ppl = self._calculate_generation_ppl(generated_texts)
        return {"dist-1": dist_1, "dist-2": dist_2, "gen_ppl_by_gpt2": gen_ppl, "generated_samples": generated_texts}

    @torch.inference_mode()
    def _evaluate_mlm_and_token_change(self, model, batch_size=8):
        """合并计算困惑度和Token变化任务的评估"""
        model.eval()
        collator = MLMCollator(tokenizer=self.tokenizer, mlm_probability=0.8)
        
        # MLM相关指标
        mlm_losses = []
        
        # Token变化任务相关指标
        all_preds, all_labels = [], []
        
        for i in tqdm(range(0, len(self.test_dataset), batch_size), desc="评估MLM和Token变化任务"):
            batch_samples = self.test_dataset[i:min(i + batch_size, len(self.test_dataset))]
            batch = collator(batch_samples)
            
            # 只对tensor类型的数据进行设备移动
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 使用 autocast 来匹配模型精度
            with torch.autocast(device_type=self.device, dtype=self.model_dtype):
                outputs = model(**batch)  # 直接使用**batch传递参数

            # 收集MLM损失
            if hasattr(outputs, 'mlm_loss'):
                mlm_losses.append(outputs.mlm_loss.item())
            else:
                mlm_losses.append(outputs.loss.item())
            # 收集Token变化任务预测结果
            if hasattr(outputs, 'token_change_logits') and outputs.token_change_logits is not None:
                logits = outputs.token_change_logits
                preds = torch.argmax(logits, dim=-1)
                active_mask = batch['attention_mask'] == 1
                all_preds.extend(preds[active_mask].cpu().numpy())
                all_labels.extend(batch['token_change_labels'][active_mask].cpu().numpy())

        # 计算MLM相关指标
        mean_mlm_loss = np.mean(mlm_losses) if mlm_losses else float('inf')
        perplexity = np.exp(mean_mlm_loss)
        
        # 计算Token变化任务指标
        tc_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1-score": 0.0}
        if all_preds and all_labels:
            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='binary', zero_division=0
            )
            tc_metrics = {
                "accuracy": accuracy, 
                "precision": precision, 
                "recall": recall, 
                "f1-score": f1
            }
        
        return {
            "perplexity": perplexity,
            "avg_mlm_loss": mean_mlm_loss,
            "tc_metrics": tc_metrics
        }

    def evaluate_single_model(self, model_path: str, model_name: str):
        results = {col: float('nan') for col in self.result_columns}
        results["Model"] = model_name
        model = None

        try:
            model = self._load_model_safe(model_path)
            if model is None:
                raise RuntimeError(f"模型加载失败: {model_name}")
            model.to(self.device)

            # 合并的MLM和Token变化任务评估
            combined_results = self._evaluate_mlm_and_token_change(model)
            
            # 更新结果
            results["Perplexity"] = combined_results["perplexity"]
            results["Avg MLM Loss"] = combined_results["avg_mlm_loss"]
            results.update({
                "TC Accuracy": combined_results["tc_metrics"]['accuracy'],
                "TC F1-Score": combined_results["tc_metrics"]['f1-score'],
                "TC Precision": combined_results["tc_metrics"]['precision'], 
                "TC Recall": combined_results["tc_metrics"]['recall']
            })

            # 生成任务评估保持不变
            gen_metrics = self._evaluate_generation(model)
            results.update({
                "Gen Diversity (dist-1)": gen_metrics['dist-1'],
                "Gen Diversity (dist-2)": gen_metrics['dist-2'],
                "Gen PPL (by Qwen3)": gen_metrics['gen_ppl_by_gpt2']
            })

            logger.info(f"\n--- {model_name} 生成样本 ---")
            for i, prompt in enumerate(self.prompts):
                logger.info(f"提示: {prompt}\n生成: {gen_metrics['generated_samples'][i]}\n")
                
        except Exception as e:
            logger.error(f"评估 '{model_name}' 时发生严重错误: {e}")
            traceback.print_exc()
        finally:
            if model is not None: del model
            torch.cuda.empty_cache()
            logger.info(f"'{model_name}' 评估完成，资源已释放。")
        return results

def main():
    """主评估函数"""


    torch.set_float32_matmul_precision('high')


    model_configs = {
        "Cosine_Schedule63": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/cosine_m0.6_r0.3/checkpoint-18572",
        "Linear_Schedule45": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/linear_m0.4_r0.5/checkpoint-18572",
        "Linear_Schedule54":"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/linear_m0.5_r0.4/checkpoint-18572",
        "Linear_Schedule63": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/linear_m0.6_r0.3/checkpoint-18572",
        "Random_Schedule63": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/random_m0.6_r0.3/checkpoint-18572",
        "Linear_Schedule72":"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/linear_m0.7_r0.2/checkpoint-18572",
        "Linear_Schedule81":"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/linear_m0.8_r0.1/checkpoint-18572",
        "Linear_Schedule1":"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/linear_m1_r0/checkpoint-18572",
        # "llada":"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/llada_500m/checkpoint-18572"
    }
    tokenizer_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/cosine_m0.6_r0.3/checkpoint-18572"
    dataset_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/dataset/EleutherAI/fineweb-edu-dedup-10b/main"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")

    evaluator = ModelEvaluator(tokenizer_path, dataset_path, device)

    all_results = []
    for model_name, model_path in model_configs.items():
        logger.info(f"\n{'='*25} 开始评估模型: {model_name} {'='*25}")
        result = evaluator.evaluate_single_model(model_path, model_name)
        all_results.append(result)

    df_results = pd.DataFrame(all_results)


    logger.info("\n\n" + "="*80)
    logger.info("                           最终评估结果对比")
    logger.info("="*80)
    pd.set_option('display.max_columns', None); pd.set_option('display.width', 1000); pd.set_option('display.float_format', '{:.4f}'.format)
    print(df_results.to_string(index=False))

    output_file = "model_evaluation_results.csv"
    df_results.to_csv(output_file, index=False, float_format='%.4f')
    logger.info(f"\n结果已保存到: {output_file}")

    logger.info("\n" + "="*50); logger.info("                   最佳模型分析"); logger.info("="*50)
    metrics_to_analyze = {
        "最低困惑度 (PPL)": ("Perplexity", "min"),
        "最低MLM损失": ("Avg MLM Loss", "min"),
        "最高TC准确率": ("TC Accuracy", "max"),
        "最高TC F1": ("TC F1-Score", "max"),
        "最高生成多样性 (dist-2)": ("Gen Diversity (dist-2)", "max"),
        "最低生成困惑度 (by Qwen3)": ("Gen PPL (by Qwen3)", "min")  # 更新指标名称
    }
    for metric_name, (column, direction) in metrics_to_analyze.items():
        if column in df_results.columns and df_results[column].notna().any():
            best_idx = df_results[column].idxmin() if direction == "min" else df_results[column].idxmax()
            best_model, best_value = df_results.loc[best_idx, "Model"], df_results.loc[best_idx, column]
            logger.info(f"{metric_name:<30}: {best_model:<20} ({best_value:.4f})")


if __name__ == '__main__':
    main()