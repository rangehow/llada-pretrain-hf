#!/bin/bash
# 这个脚本现在接收10个参数来运行训练任务



# --- 从命令行参数中读取配置，当然你也可以考虑直接hard code ---
OUTPUT_DIR=$1
MLM_SCHEDULE_TYPE=$2
MLM_PROB_START=$3
MLM_PROB_END=$4
BATCH_SIZE=$5
GRAD_ACCUM_STEPS=$6
CONFIG_PATH=$7
MODE=${8}
DATASET_NAME=${9}
MAX_LENGTH=${10}
EPOCHS=${11}
tail_bias_factor=${12}


echo "--- 任务配置 ---"
echo "输出目录: ${OUTPUT_DIR}"
echo "MLM 调度器: ${MLM_SCHEDULE_TYPE}"
echo "MLM 概率起始值: ${MLM_PROB_START}"
echo "MLM 概率结束值: ${MLM_PROB_END}"
echo "Batch Size: ${BATCH_SIZE}"
echo "梯度累积步数: ${GRAD_ACCUM_STEPS}"
echo "模型配置文件: ${CONFIG_PATH}"
echo "运行模式: ${MODE}"
echo "数据集名称: ${DATASET_NAME}"
echo "最大长度: ${MAX_LENGTH}"
echo "----------------"


# --- 固定配置 ---
# 确保脚本在遇到错误时退出
set -e

# 激活环境和设置环境变量
export NUMEXPR_MAX_THREADS=1000
export SWANLAB_SAVE_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/swanlab
export SWANLAB_API_KEY=EogSfKa8RaKM7NHRF8GSf
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/.bashrc
mamba activate sglang
echo "HF_HOME is set to: $HF_HOME"

# 进入工作目录
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04
export WANDB_DISABLED=true

# 固定的路径
MODEL_PATH="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/answerdotai/ModernBERT-base/main"

# 固定的训练超参数

LEARNING_RATE=2e-4



# --- 运行训练命令 ---
accelerate launch -m diffusion.main \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --config_path "${CONFIG_PATH}" \
  --num_train_epochs ${EPOCHS} \
  --learning_rate ${LEARNING_RATE} \
  --tail_bias_factor ${tail_bias_factor} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  --dataloader_num_workers 8 \
  --warmup_ratio 0.01 \
  --mlm_start_prob "${MLM_PROB_START}" \
  --mlm_end_prob "${MLM_PROB_END}" \
  --logging_steps 10 \
  --save_total_limit 2 \
  --seed 42 \
  --max_length "${MAX_LENGTH}" \
  --mlm_schedule_type "${MLM_SCHEDULE_TYPE}" \
  --mode "${MODE}" \
  --bf16

echo "训练任务完成。"