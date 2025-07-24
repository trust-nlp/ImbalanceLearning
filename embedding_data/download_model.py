#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download and save the BAAI/bge-large-en-v1.5 model and tokenizer.
"""

from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download
import os

def download_and_save(model_name: str, save_dir: str):
    """
    Download the specified model and save it to a local directory.

    Args:
        model_name (str): The model identifier on Hugging Face, e.g., "BAAI/bge-large-en-v1.5"
        save_dir (str): The local directory path to save the model.
    """
    # 1. Download the full model files (including config, pytorch_model.bin, tokenizer, etc.) using snapshot_download
    print(f"Downloading model files: {model_name} ...")
    repo_dir = snapshot_download(repo_id=model_name, cache_dir=save_dir)
    print(f"Model files have been downloaded to: {repo_dir}")

    # 2. Load the tokenizer and model (optional)
    print("Loading tokenizer and model ...")
    tokenizer = AutoTokenizer.from_pretrained(repo_dir)
    model = AutoModel.from_pretrained(repo_dir)

    # 3. Save the tokenizer and model to the same directory
    out_dir = os.path.join(save_dir, model_name.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving to: {out_dir} ...")
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    print("Save completed!")

if __name__ == "__main__":
    # Model identifier on Hugging Face
    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    # Local directory to save the model (customizable)
    SAVE_DIRECTORY = "./embedding_models"

    download_and_save(MODEL_NAME, SAVE_DIRECTORY)
