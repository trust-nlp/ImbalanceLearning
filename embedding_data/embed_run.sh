#!/bin/bash
#SBATCH --job-name=embed_conll
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger05

##### Configure output directory & log #####

# CLS datasets
# dataset="sst5"
# dataset="hurricane_irma_2017"
dataset="cyclone_idai_2019"

SCRIPT_PATH="HAMR/embedding_data/embed_cls.py"


# # NER datasets
# # dataset="bionlp2004"
# # dataset="mit_restaurant"
# dataset="tweetner7_2020"

# SCRIPT_PATH="HAMR/embedding_data/embed_ner.py"


EMB_DIR="HAMR/embedding_data/${dataset}"
LOG_DIR="HAMR/embedding_data/logs"

JSON_DIR="HAMR/Datasets/${dataset}"


MODEL_DIR="/project/hrao/imbalance/models/bge-large-en-v1.5"

mkdir -p "${EMB_DIR}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/embed_${SLURM_JOB_ID}.log"

echo "===== $(date) Starting embedding job ${SLURM_JOB_ID} on ${SLURM_NODELIST} =====" | tee -a "${LOG_FILE}"

##### 1) Activate conda environment #####
source /project/hrao/miniconda3/bin/activate
conda activate imbalance_env
echo ">>> Activated imbalance_env" | tee -a "${LOG_FILE}"

##### 2) Check GPU driver & CUDA env #####
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

##### 3) Exit if GPU is not detected #####
python3 - << 'EOF'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("ERROR: GPU requested but not detected. Aborting.")
EOF
echo ">>> GPU detected, proceeding with embedding." | tee -a "${LOG_FILE}"

##### 4) Run embedding script #####
echo ">>> Running ner_embed_json.py ..." | tee -a "${LOG_FILE}"
python3 "${SCRIPT_PATH}" \
    --json_dir "${JSON_DIR}" \
    --model_dir "${MODEL_DIR}" \
    --output_dir "${EMB_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "===== $(date) Job ${SLURM_JOB_ID} completed =====" | tee -a "${LOG_FILE}"

# sbatch HAMR/embedding_data/embed_run.sh