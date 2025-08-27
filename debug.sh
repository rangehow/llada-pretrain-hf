#!/bin/bash
export NUMEXPR_MAX_THREADS=1000

source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/.bashrc

mamba activate sglang
echo $HF_HOME

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04
export WANDB_DISABLED=true
set -e

# --- 配置你的路径和参数 ---

# 1. 路径设置
# 将这里的路径替换为你的实际路径
MODEL_PATH="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/answerdotai/ModernBERT-base/main"
# MODEL_PATH="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen3-4B-Instruct-2507/main"
# DATASET_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/dataset/EleutherAI/fineweb-edu-dedup-10b/main"
# DATASET_NAME="debug"
# DATASET_NAME="fineweb_10b"
# DATASET_NAME="ultra_fineweb"
# DATASET_NAME="finefineweb"
DATASET_NAME="filtered_finefineweb"

# 2. 训练超参数
EPOCHS=2
LEARNING_RATE=5e-4
BATCH_SIZE=8 # per_device_train_batch_size
GRAD_ACCUM_STEPS=32


MLM_SCHEDULE_TYPE=random
MLM_PROB_START=1
MLM_PROB_END=0

# CONFIG_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/modernbert_large.json
# CONFIG_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/llama_400M.json
# 模型配置文件基础路径
BASE_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion"

# # 模型配置文件名
# OUTPUT_DIR="diffusion/model_output/debug_llada"
# CONFIG_FILE="llada_1b.json"
# MODE=llada


# OUTPUT_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/debug_llama
# CONFIG_FILE="llama_1b.json"
# MODE=llama

CONFIG_PATH="${BASE_PATH}/model_config/${CONFIG_FILE}"

CUDA_VISIBLE_DEVICES=0 python -m diffusion.main \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --config_path "${CONFIG_PATH}" \
  --num_train_epochs ${EPOCHS} \
  --learning_rate ${LEARNING_RATE} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size 16 \
  --eval_steps 1000 \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  --dataloader_num_workers 0 \
  --warmup_ratio 0.01 \
  --mlm_start_prob "${MLM_PROB_START}" \
  --mlm_end_prob "${MLM_PROB_END}" \
  --logging_steps 1 \
  --save_total_limit 2 \
  --seed 42 \
  --validation_dataset_name finefineweb_validation \
  --max_length 3000 \
  --mlm_schedule_type "${MLM_SCHEDULE_TYPE}" \
  --mode "${MODE}" \
  --save_steps 1000 \
  --bf16




