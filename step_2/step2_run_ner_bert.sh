#!/bin/bash
#SBATCH --job-name=ner_meta_resample    # 作业名称
#SBATCH --ntasks=1                  # 任务数
#SBATCH --cpus-per-task=8           # 每个任务的 CPU 核心数
#SBATCH --gres=gpu:1                # 只请求1个GPU
#SBATCH --mem=50G                   # 内存分配
#SBATCH --time=12:00:00             # 最大运行时间
#SBATCH --partition=bigTiger        # 提交的分区
#SBATCH --nodelist=itiger03

# 脚本遇到错误时退出
set -e

# 创建所有需要的目录（合并重复的目录创建）
mkdir -p /project/hrao/Meta-Resample/log/step_2
mkdir -p /project/hrao/Meta-Resample/output


# 激活 miniconda3 环境
source /project/hrao/miniconda3/bin/activate
conda activate imbalance_env

# 设置模型和数据路径
model="google-bert/bert-base-uncased"
data_path="/project/hrao/data"
script_path="/project/hrao/Meta-Resample/step_2/step2_run_ner.py"
output_base_dir="/project/hrao/Meta-Resample/output"
log_dir="/project/hrao/Meta-Resample/log/step_2"

# 硬度感知采样参数
HARDNESS_SAMPLING_ENABLED=True
HARDNESS_BETA=0.9
HARDNESS_ALPHA=0.5
HARDNESS_EPSILON=1e-6
HARDNESS_INITIAL_LOSS=1
HARDNESS_P_MAX_VALUE=0.01

HARDNESS_KNN_K=15
HARDNESS_KNN_LAMBDA=0.5
HARDNESS_KNN_FREQ=1

# bionlp
# baseline: F1=0.7208
# 1）10 0.5 f1=0.7131
# 2) 10 0.75 f1=0.7131
# 3）15 0.5 f1=0.7158

# step_v2 HARDNESS_ALPHA=0.45 F1= 0.7167

# 获取当前作业序号
job_id=${SLURM_JOB_ID}

# 定义数据集数组
declare -a datasets=(
    # "bionlp2004"
    "mit_movie_trivia"
    # "mit_restaurant"
    # "ontonotes5"
    # "tweetner7_2020"
)

# 定义运行次数
num_runs=1

# 创建记录并行任务的数组
declare -a pids=()
declare -a task_descs=()

# 运行任务函数
run_task() {
    local dataset=$1
    local run_id=$2
    local output_dir_base="${output_base_dir}/${dataset}_bert"
    local log_dir_base="${log_dir}/${dataset}_bert"
    
    # 为每个数据集任务创建日志目录
    mkdir -p "${log_dir}/${dataset}_bert"
    
    # 切换到脚本所在目录
    cd "$(dirname ${script_path})"
    
    # 训练数据路径
    local train_file="${data_path}/${dataset}/train.conll"
    local validation_file="${data_path}/${dataset}/dev.conll"
    local test_file="${data_path}/${dataset}/test.conll"
    
    local output_dir="${output_dir_base}_run${run_id}"
    local log_file="${log_dir_base}_run${run_id}_${job_id}.log"
    
    echo "开始运行: 数据集=${dataset}, 运行次数=${run_id}"
    echo "使用训练文件: $train_file"
    
    # 运行训练脚本（移除多余的CUDA设置，让SLURM自动分配）
    ARGS="--model_name_or_path ${model} \
        --train_file ${train_file} \
        --validation_file ${validation_file} \
        --test_file ${test_file} \
        --output_dir ${output_dir} \
        --do_train --do_eval --do_predict \
        --per_device_train_batch_size 32 \
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
        --hardness_ema_beta ${HARDNESS_BETA} \
        --hardness_alpha ${HARDNESS_ALPHA} \
        --hardness_epsilon ${HARDNESS_EPSILON} \
        --initial_ema_loss_value ${HARDNESS_INITIAL_LOSS} \
        --hardness_p_max ${HARDNESS_P_MAX_VALUE} \
        --knn_k ${HARDNESS_KNN_K} \
        --knn_lambda ${HARDNESS_KNN_LAMBDA} \
        --knn_build_freq ${HARDNESS_KNN_FREQ} \
        --remove_unused_columns False"
    fi
    
    # Add per_device_eval_batch_size
    ARGS+=" --per_device_eval_batch_size 2"

    python ${script_path} $ARGS > ${log_file} 2>&1
    
    local status=$?
    if [ $status -eq 0 ]; then
        echo "任务完成: 数据集=${dataset}, 运行次数=${run_id}"
    else
        echo "任务失败: 数据集=${dataset}, 运行次数=${run_id}, 错误码: $status"
        # 显示错误日志末尾
        echo "错误日志末尾:"
        tail -n 30 ${log_file}
    fi
}

# 修改资源监控函数，添加超时机制
monitor_resources() {
    local timeout=$1
    local start_ts=$(date +%s)
    
    while true; do
        local now_ts=$(date +%s)
        # 如果超时，就退出
        if [ $((now_ts - start_ts)) -ge $timeout ]; then
            echo "监控超时，退出监控"
            break
        fi
        
        # 检查是否还有运行中的任务
        local running_tasks=0
        for i in "${!pids[@]}"; do
            if kill -0 ${pids[$i]} 2>/dev/null; then
                running_tasks=$((running_tasks + 1))
            fi
        done
        
        # 如果没有运行中的任务，退出监控
        if [ $running_tasks -eq 0 ] && [ ${#pids[@]} -gt 0 ]; then
            echo "所有任务已完成，退出监控"
            break
        fi
        
        echo "=== $(date) ==="
        echo "=== GPU使用情况 ==="
        nvidia-smi
        echo "=== 内存使用情况 ==="
        free -h
        echo "=== 运行中的任务 ==="
        for i in "${!pids[@]}"; do
            if kill -0 ${pids[$i]} 2>/dev/null; then
                echo "  ${task_descs[$i]} (PID: ${pids[$i]}) 正在运行"
            else
                echo "  ${task_descs[$i]} (PID: ${pids[$i]}) 已结束"
            fi
        done
        echo "=================="
        sleep 300  # 每5分钟记录一次
    done
}

# 任务计数器
task_count=0

# 串行运行指定次数
for run_id in $(seq 1 $num_runs); do
    echo "开始第 ${run_id} 次运行..."
    
    # 并行运行不同数据集的任务
    for dataset in "${datasets[@]}"; do
        run_task "$dataset" "$run_id" &
        pid=$!
        pids+=($pid)
        task_descs+=("${dataset}_run${run_id}")
        task_count=$((task_count + 1))
        echo "启动数据集任务: ${dataset}, 运行次数=${run_id}, PID=${pid}"
        
        # 添加短暂延迟，避免任务同时启动造成资源竞争
        sleep 5
    done
    
    # 等待当前运行的所有任务完成
    while [ ${#pids[@]} -gt 0 ]; do
        for i in "${!pids[@]}"; do
            if ! kill -0 ${pids[$i]} 2>/dev/null; then
                wait ${pids[$i]}
                unset pids[$i]
                pids=("${pids[@]}")
                break
            fi
        done
        sleep 1
    done
    
    echo "第 ${run_id} 次运行完成"
done

echo "已启动 $task_count 个数据集任务"

# 启动监控（1小时后自动退出）
monitor_resources 3600 > "${log_dir}/resource_monitor_${job_id}.log" &
monitor_pid=$!

# 使用方法: sbatch /project/hrao/Meta-Resample/step_2/step2_run_ner_bert.sh