#!/bin/bash
#SBATCH --job-name=embed_conll          # 任务名
#SBATCH --ntasks=1                      # MPI 任务数
#SBATCH --cpus-per-task=8               # 每个任务 CPU 核心数
#SBATCH --gres=gpu:1                    # 请求 1 张 GPU
#SBATCH --mem=50G                       # 总内存
#SBATCH --time=04:00:00                 # 最长运行时间
#SBATCH --partition=bigTiger            # 分区
#SBATCH --nodelist=itiger01             # 可选：指定节点

##### 配置输出目录 & 日志 #####


dataset="conll04_json"
# dataset="crossre_filtered_json"
# dataset="nyt_multi_filtered"
# dataset="chemprot_converted"
# dataset="ddi_corpus_converted"

EMB_DIR="Meta-Resample/embedding_data/${dataset}"
LOG_DIR="Meta-Resample/embedding_data"
SCRIPT_PATH="Meta-Resample/embedding_data/ner_embed_json.py"
JSON_DIR="Meta-Resample/data/${dataset}"


MODEL_DIR="Meta-Resample/embedding_models/bge-large-en-v1.5"

mkdir -p "${EMB_DIR}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/embed_${SLURM_JOB_ID}.log"

echo "===== $(date) Starting embedding job ${SLURM_JOB_ID} on ${SLURM_NODELIST} =====" | tee -a "${LOG_FILE}"

##### 1) 激活 conda 环境 #####
source /project/hrao/miniconda3/bin/activate
conda activate imbalance_env
echo ">>> Activated imbalance_env" | tee -a "${LOG_FILE}"

##### 2) 检查 GPU 驱动 & CUDA env #####
echo ">>> Checking GPU / CUDA / PyTorch ..." | tee -a "${LOG_FILE}"
nvidia-smi 2>&1 | tee -a "${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_FILE}"

python3 - << 'EOF' 2>&1 | tee -a "${LOG_FILE}"
import torch, os
print("torch.cuda.is_available() :", torch.cuda.is_available())
print("torch.version.cuda       :", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES     :", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("GPU name                 :", torch.cuda.get_device_name(0))
EOF

##### 3) 强制在未检测到 GPU 时退出 #####
python3 - << 'EOF'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("ERROR: GPU requested but not detected. Aborting.")
EOF
echo ">>> GPU detected, proceeding with embedding." | tee -a "${LOG_FILE}"

##### 4) 执行 embedding 脚本 #####
echo ">>> Running ner_embed_json.py ..." | tee -a "${LOG_FILE}"
python3 "${SCRIPT_PATH}" \
    --json_dir "${JSON_DIR}" \
    --model_dir "${MODEL_DIR}" \
    --output_dir "${EMB_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "===== $(date) Job ${SLURM_JOB_ID} completed =====" | tee -a "${LOG_FILE}"

# sbatch Meta-Resample/embedding_data/embed_json_job.sh