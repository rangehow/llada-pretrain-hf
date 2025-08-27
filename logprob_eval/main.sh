#!/bin/bash

# ==============================================================================
# Batch Evaluation Script for Language Models
#
# This script runs the `evaluate_logprobs.py` script for a predefined list
# of models. It's designed to be easily configurable.
#
# How to Use:
# 1. Make sure `evaluate_logprobs.py` and `task.py` are in the same directory.
# 2. Configure the `MODELS`, `MODEL_TYPES`, and `TASKS_TO_RUN` variables below.
# 3. Make this script executable: `chmod +x run_batch_eval.sh`
# 4. Run the script: `./run_batch_eval.sh`
# ==============================================================================
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/logprob_eval
export NUMEXPR_MAX_THREADS=1000

source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/.bashrc

mamba activate sglang
echo $HF_HOME
export WANDB_DISABLED=true
set -e
# --- Configuration ---

# Add the models you want to evaluate here.
# Make sure the `MODELS` and `MODEL_TYPES` arrays have the same number of elements
# and correspond to each other.
MODELS=(
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/llama_1B_100b_finefineweb/checkpoint-203000"
    # "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/niu_400M_finefineweb_lowvariance/checkpoint-309339"
    # "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output_old/llada_400m_50B_token/checkpoint-113209"
    # "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/niu_1B_100b_finefineweb_lowvariance/checkpoint-571500"
    # "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/niu_1B_100b_finefineweb/checkpoint-530000"
    # "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/llada_1B_fineweb100B/checkpoint-226417"
)

# Specify the type for each corresponding model in the `MODELS` array.
MODEL_TYPES=(
    "causal"
    # "causal"
    # "masked"
    # "causal"
)

# Comma-separated list of tasks to run for EACH model.
# These task names must exist in your `task.py`'s `dname2func` dictionary.
TASKS_TO_RUN="hellaswag"

# --- Evaluation Parameters ---

# Directory to store all result files.
OUTPUT_DIR="evaluation_results"

# Set batch size. Causal models can handle larger batches.
# This will be overridden for masked models in the loop below.
BASE_BATCH_SIZE=32

# Set to a positive number for quick testing, or 0 to use the full dataset.
LIMIT=0 

# Set to `true` if your models require trusting remote code (e.g., some Llama/Mistral versions).
TRUST_REMOTE_CODE=true

# --- Script Logic ---

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if the model and type arrays have the same length
if [ ${#MODELS[@]} -ne ${#MODEL_TYPES[@]} ]; then
    echo "Error: The number of models and model types do not match."
    echo "Please check the 'MODELS' and 'MODEL_TYPES' arrays."
    exit 1
fi

# Prepare the --trust_remote_code flag
TRUST_FLAG=""
if [ "$TRUST_REMOTE_CODE" = true ]; then
    TRUST_FLAG="--trust_remote_code"
fi

# Loop through all the models
for i in "${!MODELS[@]}"; do
    model_name="${MODELS[$i]}"
    model_type="${MODEL_TYPES[$i]}"
    
    echo "======================================================================"
    echo "Starting Evaluation for: ${model_name} (Type: ${model_type})"
    echo "======================================================================"

    # Adjust batch size for masked models as they are much slower
    batch_size=$BASE_BATCH_SIZE

    # Construct and run the command
    # The `tee` command will print the output to the console AND save it to a log file.
    log_file="${OUTPUT_DIR}/log_${model_name//\//_}.txt"

    python main.py \
        --model_name_or_path "$model_name" \
        --model_type "$model_type" \
        --tasks "$TASKS_TO_RUN" \
        --batch_size "$batch_size" \
        --limit "$LIMIT" \
        --output_dir "$OUTPUT_DIR" \
        $TRUST_FLAG 2>&1 | tee "$log_file"

    # Check if the last command was successful
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: Evaluation failed for model ${model_name}."
        echo "Check the log file for details: ${log_file}"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    else
        echo "----------------------------------------------------------------------"
        echo "Successfully completed evaluation for: ${model_name}"
        echo "Log saved to: ${log_file}"
        echo "----------------------------------------------------------------------"
    fi
done

echo "======================================================================"
echo "All batch evaluations complete."
echo "Results and logs are saved in the '${OUTPUT_DIR}' directory."
echo "======================================================================"