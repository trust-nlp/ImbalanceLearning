#!/usr/bin/env python3
"""Create balanced train/validation splits for HAMR experiments.

Input:
  - original train.json
  - original dev.json / validation.json

Output:
  - balanced train file: original train minus examples moved to validation
  - balanced validation file: original validation plus selected training examples

By default, use --output_dir to write the balanced files into a new folder while
preserving the input filenames, for example:

  python build_balanced_splits.py \
    --task cls \
    --train Datasets/sst5/train.json \
    --validation Datasets/sst5/dev.json \
    --output_dir Datasets_balanced/sst5

This writes:
  - Datasets_balanced/sst5/train.json
  - Datasets_balanced/sst5/dev.json

The output schema is unchanged:
  - CLS: {"text": str, "label": int, "label_text": str}
  - NER: {"tokens": list[str], "ner_tags": list[str]}
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


Record = dict[str, Any]


def read_json_records(path: Path) -> list[Record]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if raw.lstrip().startswith("["):
        data = json.loads(raw)
    else:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array, JSON object, or JSONL records.")
    return data


def write_json_records(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        return

    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def median_int(values: list[int]) -> int:
    if not values:
        return 0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return int(values[mid])
    return int(math.ceil((values[mid - 1] + values[mid]) / 2))


def infer_task(train: list[Record], validation: list[Record]) -> str:
    sample = next((x for x in train + validation if x), None)
    if sample is None:
        raise ValueError("Cannot infer task from empty train and validation sets.")
    if "tokens" in sample and "ner_tags" in sample:
        return "ner"
    if "text" in sample and "label" in sample:
        return "cls"
    raise ValueError(
        "Cannot infer task. Expected CLS columns text/label or NER columns tokens/ner_tags."
    )


def build_cls_splits(
    train: list[Record],
    validation: list[Record],
    seed: int,
) -> tuple[list[Record], list[Record], dict[str, Any]]:
    rng = random.Random(seed)
    validation_counts = Counter(ex["label"] for ex in validation)
    train_counts = Counter(ex["label"] for ex in train)
    labels = sorted(set(validation_counts) | set(train_counts))
    target = median_int([validation_counts.get(label, 0) for label in labels])

    by_label: dict[Any, list[tuple[int, Record]]] = defaultdict(list)
    for idx, ex in enumerate(train):
        by_label[ex["label"]].append((idx, ex))
    for items in by_label.values():
        rng.shuffle(items)

    moved_indices: set[int] = set()
    balanced_validation = list(validation)

    for label in labels:
        need = max(0, target - validation_counts.get(label, 0))
        for idx, ex in by_label.get(label, []):
            if need == 0:
                break
            moved_indices.add(idx)
            balanced_validation.append(ex)
            need -= 1

    balanced_train = [ex for idx, ex in enumerate(train) if idx not in moved_indices]
    stats = {
        "task": "cls",
        "target_per_label": target,
        "moved_from_train": len(moved_indices),
        "validation_counts_before": dict(validation_counts),
        "validation_counts_after": dict(Counter(ex["label"] for ex in balanced_validation)),
    }
    return balanced_train, balanced_validation, stats


def tag_entity_type(tag: str | None) -> str | None:
    if not tag or tag == "O":
        return None
    if "-" in tag:
        return tag.split("-", 1)[1]
    return tag


def ner_entity_counts(ex: Record) -> Counter[str]:
    counts: Counter[str] = Counter()
    prev_type: str | None = None

    for tag in ex.get("ner_tags", []):
        entity_type = tag_entity_type(tag)
        if entity_type is None:
            prev_type = None
            continue

        starts_span = str(tag).startswith("B-") or entity_type != prev_type
        if starts_span:
            counts[entity_type] += 1
        prev_type = entity_type

    return counts


def build_ner_splits(
    train: list[Record],
    validation: list[Record],
    seed: int,
) -> tuple[list[Record], list[Record], dict[str, Any]]:
    rng = random.Random(seed)

    validation_counts: Counter[str] = Counter()
    train_counts: Counter[str] = Counter()
    train_example_counts: list[Counter[str]] = []

    for ex in validation:
        validation_counts.update(ner_entity_counts(ex))
    for ex in train:
        counts = ner_entity_counts(ex)
        train_example_counts.append(counts)
        train_counts.update(counts)

    entity_types = sorted(set(validation_counts) | set(train_counts))
    target = median_int([validation_counts.get(entity_type, 0) for entity_type in entity_types])
    deficit: Counter[str] = Counter(
        {
            entity_type: max(0, target - validation_counts.get(entity_type, 0))
            for entity_type in entity_types
        }
    )

    remaining_indices = list(range(len(train)))
    rng.shuffle(remaining_indices)
    moved_indices: set[int] = set()
    balanced_validation = list(validation)

    while any(need > 0 for need in deficit.values()):
        best_idx = None
        best_gain = 0

        for idx in remaining_indices:
            counts = train_example_counts[idx]
            gain = sum(min(counts[entity_type], deficit[entity_type]) for entity_type in deficit)
            if gain > best_gain:
                best_idx = idx
                best_gain = gain

        if best_idx is None or best_gain == 0:
            break

        moved_indices.add(best_idx)
        balanced_validation.append(train[best_idx])
        remaining_indices.remove(best_idx)
        for entity_type, count in train_example_counts[best_idx].items():
            deficit[entity_type] = max(0, deficit[entity_type] - count)

    after_counts: Counter[str] = Counter()
    for ex in balanced_validation:
        after_counts.update(ner_entity_counts(ex))

    balanced_train = [ex for idx, ex in enumerate(train) if idx not in moved_indices]
    stats = {
        "task": "ner",
        "target_per_entity_type": target,
        "moved_from_train": len(moved_indices),
        "remaining_deficit": dict(deficit),
        "validation_counts_before": dict(validation_counts),
        "validation_counts_after": dict(after_counts),
    }
    return balanced_train, balanced_validation, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build balanced train/validation JSON files for CLS or NER datasets."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for balanced outputs. Preserves input filenames.",
    )
    parser.add_argument("--out_train", type=Path, default=None)
    parser.add_argument("--out_validation", type=Path, default=None)
    parser.add_argument("--task", choices=["auto", "cls", "ner"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats", type=Path, default=None)
    args = parser.parse_args()

    has_explicit_outputs = args.out_train is not None or args.out_validation is not None
    if args.output_dir is None and not has_explicit_outputs:
        parser.error("Provide either --output_dir or both --out_train and --out_validation.")
    if has_explicit_outputs and (args.out_train is None or args.out_validation is None):
        parser.error("--out_train and --out_validation must be provided together.")
    if args.output_dir is not None and has_explicit_outputs:
        parser.error("Use either --output_dir or explicit output files, not both.")

    if args.output_dir is not None:
        args.out_train = args.output_dir / args.train.name
        args.out_validation = args.output_dir / args.validation.name

    return args


def main() -> None:
    args = parse_args()
    train = read_json_records(args.train)
    validation = read_json_records(args.validation)

    task = infer_task(train, validation) if args.task == "auto" else args.task
    if task == "cls":
        balanced_train, balanced_validation, stats = build_cls_splits(
            train=train,
            validation=validation,
            seed=args.seed,
        )
    else:
        balanced_train, balanced_validation, stats = build_ner_splits(
            train=train,
            validation=validation,
            seed=args.seed,
        )

    write_json_records(args.out_train, balanced_train)
    write_json_records(args.out_validation, balanced_validation)

    stats.update(
        {
            "train_size_before": len(train),
            "train_size_after": len(balanced_train),
            "validation_size_before": len(validation),
            "validation_size_after": len(balanced_validation),
        }
    )

    if args.stats is not None:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
