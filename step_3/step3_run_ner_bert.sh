#!/bin/bash
#SBATCH --job-name=ner_meta_resample    # 作业名称
#SBATCH --ntasks=1                      # 任务数
#SBATCH --cpus-per-task=8               # 每个任务的 CPU 核心数
#SBATCH --gres=gpu:1                    # 请求 1 个 GPU
#SBATCH --mem=50G                       # 内存分配
#SBATCH --time=12:00:00                 # 最大运行时间
#SBATCH --partition=bigTiger            # 提交分区
#SBATCH --nodelist=itiger03             # 指定节点（可按需移除）

# 脚本遇到错误时退出
set -e

# 设置环境变量以禁用 flash 和 memory-efficient SDP-Attention
export PYTORCH_SDP_DISABLE_FLASH=1
export PYTORCH_SDP_DISABLE_MEM_EFFICIENT=1

dataset="mit_movie_trivia"   # ← 这里修改为你要跑的"单个"数据集名称
# dataset="bionlp2004"
# dataset="mit_restaurant"
# dataset="ontonotes5"
# dataset="tweetner7_2020"

######################
# （2）创建必要的输出、日志目录（自动根据 dataset 生成子目录）
######################
output_base_dir="/project/hrao/Meta-Resample/output"
log_base_dir="/project/hrao/Meta-Resample/log/step_3"
mkdir -p "${output_base_dir}/${dataset}_bert"
mkdir -p "${log_base_dir}/${dataset}_bert"

######################
# （3）激活 conda 环境
######################
source /project/hrao/miniconda3/bin/activate
conda activate imbalance_env

######################
# （4）设置模型、文件路径，以及硬度感知采样参数
######################
model="google-bert/bert-base-uncased"
data_path="/project/hrao/data"
script_path="/project/hrao/Meta-Resample/step_3/step3_run_ner.py"

# 硬度感知采样参数（如不需要可以留空或删掉相应行）
HARDNESS_SAMPLING_ENABLED=True
HARDNESS_ALPHA=0.5             # 尝试 1.0
HARDNESS_KNN_K=15
HARDNESS_KNN_LAMBDA=0
HARDNESS_KNN_FREQ=1
VNET_LR=1e-3                   # 尝试降低 VNet 学习率
META_UPDATE_LR=1e-4         # 保持或确认是否使用
META_UPDATE_SCALE_FACTOR=2.0   # 尝试减小 scale factor

######################
# （5）组装训练脚本的参数
######################
train_file="${data_path}/${dataset}/train.conll"
validation_file="${data_path}/${dataset}/dev.conll"
test_file="${data_path}/${dataset}/test.conll"

output_dir="${output_base_dir}/${dataset}_bert_run"
log_file="${log_base_dir}/${dataset}_bert_run_${SLURM_JOB_ID}.log"

ARGS="--model_name_or_path ${model} \
      --train_file ${train_file} \
      --validation_file ${validation_file} \
      --test_file ${test_file} \
      --output_dir ${output_dir} \
      --do_train --do_eval --do_predict \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 2 \
      --learning_rate 2e-5 \
      --num_train_epochs 8 \
      --max_seq_length 256 \
      --overwrite_output_dir \
      --seed 42 \
      --log_level debug \
      --logging_strategy steps \
      --eval_strategy epoch \
      --report_to tensorboard \
      --disable_tqdm True"

if [ "${HARDNESS_SAMPLING_ENABLED}" = "True" ]; then
    ARGS+=" --hardness_aware_sampling ${HARDNESS_SAMPLING_ENABLED} \
            --hardness_alpha ${HARDNESS_ALPHA} \
            --knn_k ${HARDNESS_KNN_K} \
            --knn_lambda ${HARDNESS_KNN_LAMBDA} \
            --knn_build_freq ${HARDNESS_KNN_FREQ} \
            --vnet_lr ${VNET_LR} \
            --meta_update_lr ${META_UPDATE_LR} \
            --meta_update_scale_factor ${META_UPDATE_SCALE_FACTOR} \
            --remove_unused_columns False"
fi

######################
# （6）执行训练脚本
######################
echo "===== 开始运行：数据集 = ${dataset} ====="
echo "训练文件：${train_file}"
python "${script_path}" ${ARGS} > "${log_file}" 2>&1

status=$?
if [ $status -eq 0 ]; then
    echo "===== 运行完成：数据集 = ${dataset} ====="
    echo "日志存放在：${log_file}"
else
    echo "===== 运行失败：数据集 = ${dataset}, 错误码：$status ====="
    echo "错误日志末尾："
    tail -n 30 "${log_file}"
    exit $status
fi
