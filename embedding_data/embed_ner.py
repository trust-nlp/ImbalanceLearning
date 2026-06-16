#!/usr/bin/env python
# embed_json_tokens.py

import os
import json
import argparse
import numpy as np
import torch
import hashlib
from sentence_transformers import SentenceTransformer
import logging

# Configure logging (keep Code2's logging style)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_json_tokens(file_path: str) -> list[str]:
    """
    读取 JSON/JSONL，逐条记录从 'tokens'（字符串列表）恢复句子：" ".join(tokens)，
    返回按文件顺序排列的文本列表。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # JSONL（逐行 JSON）或 JSON（数组 / 单对象）
    if "\n" in raw and raw.lstrip()[:1] != "[":
        objs = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                objs.append(obj)
            except json.JSONDecodeError:
                logging.warning(f"跳过无法解析的 JSON 行：{line[:120]}...")
    else:
        try:
            objs = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"无法解析 {file_path} 为 JSON：{e}") from e
        if isinstance(objs, dict):
            objs = [objs]

    texts: list[str] = []
    for obj in objs:
        toks = obj.get("tokens")
        if isinstance(toks, list) and all(isinstance(t, str) for t in toks):
            texts.append(" ".join(toks))
        else:
            logging.warning("跳过记录：缺少有效的 'tokens'（字符串列表）字段")
    return texts


def uid_of(text: str) -> int:
    """用 SHA1 生成短 UID，训练端用同一规则校验顺序一致性。"""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def embed_and_save(train_json: str, model_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    logging.info(f"Loading SentenceTransformer model from: {model_dir}")
    model = SentenceTransformer(model_dir, device=device)

    texts = read_json_tokens(train_json)
    if not texts:
        raise RuntimeError("输入文件中没有可用的 'tokens' 字段以恢复文本")

    logging.info(f"Loaded {len(texts)} sentences from {train_json}")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    emb_path = os.path.join(output_dir, "train_embeddings.npy")
    np.save(emb_path, embeddings)
    logging.info(f"Saved embeddings to {emb_path} (shape: {embeddings.shape})")

    uids = np.array([uid_of(s) for s in texts], dtype=np.uint64)
    uid_path = os.path.join(output_dir, "train_uids.npy")
    np.save(uid_path, uids)
    logging.info(f"Saved UIDs to {uid_path} (count: {uids.shape[0]})")


def main():
    parser = argparse.ArgumentParser(
        description="Embed train.json/train.jsonl (records with 'tokens') to train_embeddings.npy + train_uids.npy"
    )
    parser.add_argument("--train_json", required=True, help="Path to train.json or train.jsonl")
    parser.add_argument("--model_dir", required=True, help="SentenceTransformer model path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    embed_and_save(args.train_json, args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
