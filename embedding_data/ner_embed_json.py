# /project/hrao/imbalance/embedding_data/ner_embed_json.py

import os
import json
import argparse
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def read_json_sentences(file_path):
    """
    支持两种 JSON 格式:
      1) JSONL  —— 每行是一个对象, 行内含 "tokens".
      2) 单 JSON —— 可以是对象或对象列表, 每个对象含 "tokens".

    返回: List[str] — 句子文本, 每条为 "token1 token2 ..."
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # JSONL 判定: 含换行且首字符不是 "[" (避免把 JSON 数组误认为 JSONL)
    if "\n" in raw and raw.lstrip()[0] != "[":
        objs = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        objs = json.loads(raw)
        if isinstance(objs, dict):          # 单对象包装多条
            objs = [objs]

    sentences = []
    for obj in objs:
        tokens = obj.get("tokens")
        if tokens:                          # 忽略缺失 tokens 的记录
            sentences.append(" ".join(tokens))
    return sentences


def embed_and_save(json_dir: str, model_dir: str, output_dir: str):
    """
    遍历 json/jsonl 文件 → 句子级嵌入 → 保存 *.npy
    """
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model = SentenceTransformer(model_dir, device=device)

    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith((".json", ".jsonl")):
            continue

        file_path = os.path.join(json_dir, fname)
        sentences = read_json_sentences(file_path)
        print(f"[INFO] {fname}: {len(sentences)} sentences")

        if not sentences:                   # 空文件直接跳过
            continue

        embeddings = model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        base = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{base}_embeddings.npy")
        np.save(out_path, embeddings)
        print(f"[SAVE] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Embed .json / .jsonl files via SentenceTransformer")
    parser.add_argument("--json_dir", required=True, help="Directory with .json/.jsonl files")
    parser.add_argument("--model_dir", required=True, help="SentenceTransformer model path")
    parser.add_argument("--output_dir", required=True, help="Where to write *.npy embeddings")
    args = parser.parse_args()

    embed_and_save(args.json_dir, args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
