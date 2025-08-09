#!/bin/bash
#SBATCH --job-name=HAMR_cls      # Job name
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=8       # Number of CPU cores per task
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --mem=32G               # Memory allocation
#SBATCH --time=12:00:00         # Maximum runtime
#SBATCH --partition=bigTiger    # Partition to submit to
#SBATCH --nodelist=itiger01
# Exit on error
set -e

# Activate conda environment
source /project/hrao/miniconda3/bin/activate
conda activate imbalance_env

# Change the dataset here!
# dataset="hurricane_irma_2017"
dataset="cyclone_idai_2019"
# dataset="sst5"

# Change the model here!
model_name="microsoft/deberta-v3-base"
# model_name="answerdotai/ModernBERT-base"


HARDNESS_ALPHA=0.7       
KNN_K=10
KNN_LAMBDA=0.1
KNN_HARD_RATIO=0.3
LEARNING_RATE=4e-5
WNET_LR=1e-4                   
META_UPDATE_LR=2e-4            


train_file="HAMR/Datasets/${dataset}/train.json"
validation_file="HAMR/Datasets/${dataset}/dev.json"
test_file="HAMR/Datasets/${dataset}/test.json"



log_base_dir="HAMR/CLS/logs"
mkdir -p "$log_base_dir"
log_file="${log_base_dir}/${dataset}_${SLURM_JOB_ID}.log"
exec > >(tee -a "$log_file") 2>&1

output_dir="HAMR/CLS/models/${dataset}_${SLURM_JOB_ID}"
mkdir -p "$output_dir"


python HAMR/CLS/train_cls.py \
    --model_name_or_path ${model_name} \
    --log_level info \
    --logging_strategy epoch \
    --logging_steps 500 \
    --seed 42 \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --test_file ${test_file} \
    --text_column_names text \
    --label_column_name label \
    --shuffle_train_dataset \
    --metric_name f1 \
    --do_train --do_eval --do_predict \
    --max_seq_length 512 \
    --per_device_train_batch_size 8 \
    --fp16 \
    --output_dir ${output_dir} \
    --save_strategy=epoch \
    --hardness_aware_sampling true \
    --num_train_epochs 8 \
    --learning_rate ${LEARNING_RATE} \
    --wnet_lr ${WNET_LR} \
    --meta_update_lr ${META_UPDATE_LR} \
    --hardness_alpha ${HARDNESS_ALPHA} \
    --knn_k ${KNN_K} \
    --knn_lambda ${KNN_LAMBDA} \
    --knn_hard_sample_ratio ${KNN_HARD_RATIO} \
    --knn_build_freq 1 \
    --meta_update_scale_factor 2.0 \
    --embedding_dir HAMR/embedding_data/${dataset} \
    --weighted_loss true \
    --weighted_sampling true



# sbatch HAMR/CLS/run_cls.sh



