# Model-Agnostic Meta Learning for Class Imbalance Adaptation

HAMR (**H**ardness-**A**ware **M**eta-**R**esampling) is a training framework that enhances model performance on imbalanced text datasets. It dynamically identifies and prioritizes difficult examples during training, leading to better generalization on class imbalance scenario for both **Token Classification (NER)** and **Sequence Classification** tasks.

## How It Works

HAMR extends the standard Hugging Face `Trainer` with a simple but powerful meta-learning loop:

1.  **Learn Sample Hardness:** A small meta-network (`WNet`) learns to assign a weight to each training sample based on its loss. Higher loss (harder example) means a higher weight.

2.  **Resample with Weights:** A custom sampler (`HardnessAwareSampler`) uses these weights to over-sample hard examples and under-sample easy ones, creating more effective training batches.

3.  **Boost Neighbors (Optional):** Using pre-computed sentence embeddings and a [FAISS](https://github.com/facebookresearch/faiss) index, HAMR identifies the semantic neighbors of the hardest samples and boosts their sampling priority, encouraging the model to learn from entire challenging regions of the data space.

## Quick Start

### 1. Setup Environment

```bash
# Create and activate conda environment
conda create -n hamr_env python=3.9
conda activate hamr_env

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install FAISS for GPU
conda install -c pytorch faiss-gpu

# Install other dependencies
pip install transformers datasets evaluate sentence-transformers seqeval
```

### 2. Prepare Data & Embeddings

Place your raw dataset files (JSON/JSONL) in `HAMR/Datasets/<dataset_name>/`. Then, generate the sentence embeddings required for the k-NN booster.

```bash
# 1. Edit HAMR/embedding_data/embed_run.sh to set your `dataset` name
# 2. Run the script via SLURM
sbatch HAMR/embedding_data/embed_run.sh
```

This will save sentence embeddings as `.npy` files in `HAMR/embedding_data/<dataset_name>/`.

### 3. Train a Model

Configure and run the training script for your task (e.g., NER).

```bash
# 1. Edit the run script (e.g., HAMR/NER/run_ner_bionlp.sh):
#    - Set the `dataset` name.
#    - Ensure `--embedding_dir` points to the output from the previous step.
#    - Adjust HAMR hyperparameters (see table below).

# 2. Launch the training job
sbatch HAMR/NER/run_ner_bionlp.sh
```
Logs will be saved to `HAMR/NER/logs/` and the final model to `HAMR/NER/models/`.

## Project Structure

```
/HAMR/
├── CLS/                     # Sequence Classification implementation
│   ├── custom_trainer.py
│   └── run_cls.sh
├── NER/                     # Token Classification (NER) implementation
│   ├── custom_trainer.py
│   └── run_ner_bionlp.sh
└── embedding_data/          # Scripts for generating sentence embeddings
    ├── embed_cls.py
    ├── embed_ner.py
    └── embed_run.sh
```