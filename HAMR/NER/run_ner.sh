#!/bin/bash
#SBATCH --job-name=HAMR_ner    # Job name
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=8               # Number of CPU cores per task
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --mem=50G                       # Memory allocation
#SBATCH --time=12:00:00                 # Maximum runtime
#SBATCH --partition=bigTiger            # Submission partition
#SBATCH --nodelist=itiger01             # Specify node (can be removed if not needed)

# Exit on error
set -e

# Set environment variables to disable flash and memory-efficient SDP-Attention
export PYTORCH_SDP_DISABLE_FLASH=1
export PYTORCH_SDP_DISABLE_MEM_EFFICIENT=1

# Activate conda environment
source /project/hrao/miniconda3/bin/activate
conda activate imbalance_env

# Change the dataset here!
dataset="mit_restaurant"
# dataset="bionlp2004"
# dataset="tweetner7_2020"

# Change the model here!
model_name="microsoft/deberta-v3-base"
# model_name="answerdotai/ModernBERT-base"

train_file="HAMR/Datasets/${dataset}/train.json"
validation_file="HAMR/Datasets/${dataset}/dev.json"
test_file="HAMR/Datasets/${dataset}/test.json"



log_base_dir="HAMR/NER/logs"
mkdir -p "$log_base_dir"
log_file="${log_base_dir}/${dataset}_${SLURM_JOB_ID}.log"
exec > >(tee -a "$log_file") 2>&1

output_dir="HAMR/NER/models/${dataset}_${SLURM_JOB_ID}"
mkdir -p "$output_dir"

# Hardness-aware sampling parameters
HARDNESS_ALPHA=0.1       
KNN_K=10
KNN_LAMBDA=0.2
KNN_FREQ=1
KNN_HARD_RATIO=0.2
WNET_LR=3e-4                   
META_UPDATE_LR=2e-4            
LEARNING_RATE=2e-5


python HAMR/NER/train_ner.py \
    --model_name_or_path ${model_name} \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --test_file ${test_file} \
    --output_dir ${output_dir} \
    --do_train --do_eval --do_predict \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --fp16 \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs 8 \
    --max_seq_length 512 \
    --overwrite_output_dir \
    --seed 42 \
    --log_level debug \
    --logging_strategy steps \
    --logging_steps 500 \
    --eval_strategy epoch \
    --report_to tensorboard \
    --disable_tqdm True \
    --eval_accumulation_steps 8 \
    --save_strategy epoch \
    --embedding_dir HAMR/embedding_data/${dataset} \
    --hardness_aware_sampling True \
    --hardness_alpha ${HARDNESS_ALPHA} \
    --knn_k ${KNN_K} \
    --knn_lambda ${KNN_LAMBDA} \
    --knn_build_freq ${KNN_FREQ} \
    --knn_hard_sample_ratio ${KNN_HARD_RATIO} \
    --wnet_lr ${WNET_LR} \
    --meta_update_lr ${META_UPDATE_LR} \
    --meta_update_scale_factor 2.0 \
    --remove_unused_columns False

# sbatch HAMR/NER/run_ner.sh
