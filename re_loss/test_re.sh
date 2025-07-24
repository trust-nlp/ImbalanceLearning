#!/bin/bash
#SBATCH --job-name=re_loss      # 作业名称
#SBATCH --ntasks=1              # 任务数
#SBATCH --cpus-per-task=8       # 每个任务的CPU核心数
#SBATCH --gres=gpu:1            # 请求1个GPU
#SBATCH --mem=32G               # 内存分配
#SBATCH --time=12:00:00         # 最大运行时间
#SBATCH --partition=bigTiger    # 提交的分区
#SBATCH --nodelist=itiger01
# 脚本遇到错误时退出
set -e

# 激活conda环境
source /project/hrao/miniconda3/bin/activate
conda activate imbalance_env


dataset="conll04_json"
# dataset="crossre_filtered_json"
# dataset="nyt_multi_filtered"
# dataset="chemprot_converted"
# dataset="ddi_corpus_converted"

data_path="/project/hrao/data"
train_file="${data_path}/${dataset}/train.json"
validation_file="${data_path}/${dataset}/validation.json"
test_file="${data_path}/${dataset}/test.json"

# /project/hrao/data/nyt_multi_filtered


# 定义日志目录和日志文件名
log_base_dir="Meta-Resample/re_loss/log"
mkdir -p "$log_base_dir"
log_file="${log_base_dir}/${dataset}_re_diceloss_run_${SLURM_JOB_ID}.log"
# 将 stdout 和 stderr 全部重定向到 $log_file
exec > >(tee -a "$log_file") 2>&1


# microsoft/deberta-v3-base
# bert-base-uncased
# microsoft/deberta-v3-large
# allenai/scibert_scivocab_uncased

python Meta-Resample/re_loss/run_re.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --train_file ${train_file} \
  --validation_file ${validation_file} \
  --test_file ${test_file} \
  --max_seq_length 512 \
  --num_train_epochs 8 \
  --per_device_train_batch_size 32 \
  --do_train --do_eval --do_predict \
  --logging_steps 50 \
  --save_steps 500 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --add_special_markers true \
  --negative_ratio 1.0 \
  --overwrite_cache true \
  --output_dir ./re_model \
  --loss_name focal \
  --seed 42

echo "脚本测试完成！" 


# sbatch Meta-Resample/re_loss/test_re.sh