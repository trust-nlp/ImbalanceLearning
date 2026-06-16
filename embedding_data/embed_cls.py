import os
import json
import argparse
import numpy as np
import torch
import hashlib
from sentence_transformers import SentenceTransformer


def read_json_texts(file_path: str) -> list[str]:
    """读取 JSON/JSONL，提取每条记录的 'text' 字段，返回按文件顺序排列的文本列表。"""
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # JSONL（逐行 JSON）或 JSON（数组 / 单对象）
    if "\n" in raw and raw.lstrip()[:1] != "[":
        objs = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        objs = json.loads(raw)
        if isinstance(objs, dict):
            objs = [objs]

    texts: list[str] = []
    for obj in objs:
        t = obj.get("text")
        if isinstance(t, str):
            texts.append(t)
    return texts


def uid_of(text: str) -> int:
    """用 SHA1 生成短 UID，训练端用同一规则校验顺序一致性。"""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def embed_and_save(train_json: str, model_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_dir, device=device)

    texts = read_json_texts(train_json)
    if not texts:
        raise RuntimeError("train.json 中没有可用的 'text' 字段")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    np.save(os.path.join(output_dir, "train_embeddings.npy"), embeddings)
    uids = np.array([uid_of(s) for s in texts], dtype=np.uint64)
    np.save(os.path.join(output_dir, "train_uids.npy"), uids)


def main():
    parser = argparse.ArgumentParser(description="Embed train.json to train_embeddings.npy + train_uids.npy")
    parser.add_argument("--train_json", required=True, help="Path to train.json or train.jsonl")
    parser.add_argument("--model_dir", required=True, help="SentenceTransformer model path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    embed_and_save(args.train_json, args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
