# """Finetuning the library models for relation extraction."""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import Value, load_dataset
from itertools import combinations

import os
from datetime import datetime

import transformers
from sklearn.metrics import classification_report
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import torch.nn as nn
from loss_functions import FocalLoss, DiceLoss
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

NO_RELATION = "no_relation"
ENTITY_MARKERS = ("<e1>", "</e1>", "<e2>", "</e2>")

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    add_special_markers: bool = field(
        default=True,
        metadata={"help": "Whether to wrap head/tail entity spans with special marker tokens."},
    )
    negative_ratio: float = field(
        default=1.0,
        metadata={"help": "Expected #negative pairs / #positive pairs. 0 表示不加负样本"},
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classification task'
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    evaluation_strategy: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional hook for older Transformers versions that lack "
                "`evaluation_strategy` in TrainingArguments. "
                "If provided, we copy it into `training_args` later."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None:
            if self.train_file is None or self.validation_file is None:
                raise ValueError(" training/validation file or a dataset name.")

            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert validation_extension == train_extension, (
                "`validation_file` should have the same extension (csv or json) as `train_file`."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    loss_name: str = field(
        default="ce",
        metadata={"help": "Loss function: ce | focal | dice"}
    )


def get_label_list(raw_dataset, split: str = "train") -> list[str]:
    rel_types = set()
    for rels in raw_dataset[split]["relations"]:
        for rel in rels:
            rel_types.add(rel["type"])
    rel_types.add(NO_RELATION)
    return sorted(list(rel_types))


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 如果在 Slurm 环境下，取 SLURM_JOB_ID，否则用时间戳
    job_id = os.environ.get("SLURM_JOB_ID")
    suffix = job_id if job_id is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
    training_args.output_dir = f"{training_args.output_dir}_{suffix}"

    # --- legacy-compat for evaluation_strategy ---------------------------------
    if data_args.evaluation_strategy is not None:
        if hasattr(training_args, "evaluation_strategy"):
            # 旧版 TrainingArguments 可能没有该字段；有就覆盖
            training_args.evaluation_strategy = data_args.evaluation_strategy
            logger.info(f"Set training_args.evaluation_strategy = {data_args.evaluation_strategy}")
        else:
            logger.warning(
                "Current Transformers version lacks `evaluation_strategy` in TrainingArguments; "
                "argument will be ignored."
            )

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files, or specify a dataset name
    # to load from huggingface/datasets. In ether case, you can specify a the key of the column(s) containing the text and
    # the key of the column containing the label. If multiple columns are specified for the text, they will be joined together
    # for the actual text value.
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        # Try print some info about the dataset
        logger.info(f"Dataset loaded: {raw_datasets}")
        logger.info(raw_datasets)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert test_extension == train_extension, (
                    "`test_file` should have the same extension (csv or json) as `train_file`."
                )
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a dataset name or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    if data_args.remove_splits is not None:
        for split in data_args.remove_splits.split(","):
            logger.info(f"removing split {split}")
            raw_datasets.pop(split)

    if data_args.train_split_name is not None:
        logger.info(f"using {data_args.train_split_name} as train set")
        raw_datasets["train"] = raw_datasets[data_args.train_split_name]
        raw_datasets.pop(data_args.train_split_name)

    if data_args.validation_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["validation"] = raw_datasets[data_args.validation_split_name]
        raw_datasets.pop(data_args.validation_split_name)

    if data_args.test_split_name is not None:
        logger.info(f"using {data_args.test_split_name} as test set")
        raw_datasets["test"] = raw_datasets[data_args.test_split_name]
        raw_datasets.pop(data_args.test_split_name)

    if data_args.remove_columns is not None:
        for split in raw_datasets.keys():
            for column in data_args.remove_columns.split(","):
                logger.info(f"removing column {column} from split {split}")
                raw_datasets[split] = raw_datasets[split].remove_columns(column)

    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    # Simplified for relation extraction task
    label_list = get_label_list(raw_datasets, split="train")
    num_labels = len(label_list)
    label_to_id = {v: i for i, v in enumerate(label_list)}
    id2label = {id: label for label, id in label_to_id.items()}

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    logger.info("setting problem type to single label classification")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        
        # 在这里直接提供所有配置信息
        num_labels=num_labels,
        label2id=label_to_id,
        id2label=id2label,
        problem_type="single_label_classification",
        
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        
        # 强烈建议设置为 True，这样即使缓存被污染，也能强制替换分类头
        ignore_mismatched_sizes=True,
    )

    if data_args.add_special_markers:
        tokenizer.add_tokens(list(ENTITY_MARKERS), special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False



    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def _span(entities, ent_raw):
        """
        给定实体容器 (list 或 dict) 与实体id，返回对应 span dict。
        - list : id 可以是 int / str(含前缀 e1、E02 等)；
        - dict : 允许键为 'e1' / '1' / 1 等任意形式。
        若找不到则抛出 KeyError，并把可用键写进消息里，方便排查。
        """
        # ---------- list 结构 ----------
        if isinstance(entities, list):
            try:
                # 假设 ent_raw 是可以被 int() 直接转换的
                ent_idx = int(ent_raw)
                return entities[ent_idx]
            except (ValueError, TypeError, IndexError) as e:
                # 失败时附上更详细的上下文
                raise KeyError(
                    f"Invalid entity index '{ent_raw}' for list of size {len(entities)}"
                ) from e

        # ---------- dict 结构 ----------
        # 直接尝试 ent_raw 作为 key，失败就抛 KeyError
        if ent_raw in entities:
            return entities[ent_raw]
        
        # 兼容性尝试：如果 ent_raw 是 "1" 而 key 是 1 (int)
        if isinstance(ent_raw, str) and ent_raw.isdigit():
            if int(ent_raw) in entities:
                return entities[int(ent_raw)]

        raise KeyError(
            f"Cannot find entity id '{ent_raw}' in entities dict; "
            f"available keys={list(entities.keys())[:10]} ..."
        )

    def _ensure_span_dict(span):
        """
        强制将实体跨度转换成含 'start'/'end' 键的字典。
        - 支持 dict / int / (start, end) 三种形式；
        - 其它形式直接抛错，方便早点发现数据问题。
        """
        if isinstance(span, dict):
            if "start" in span and "end" in span:
                return span
            raise KeyError("Entity dict 缺少 'start' / 'end' 键")

        if isinstance(span, int):
            return {"start": span, "end": span + 1}

        if isinstance(span, (list, tuple)) and len(span) == 2:
            return {"start": int(span[0]), "end": int(span[1])}

        raise KeyError(f"Unsupported entity span format: {type(span)} → {span}")


    def build_instance(tokens, entities, h_id, t_id, rel_type):
        h_span = _ensure_span_dict(_span(entities, h_id))
        t_span = _ensure_span_dict(_span(entities, t_id))

        out_toks = []
        for idx, tok in enumerate(tokens):
            if idx == h_span["start"]:
                out_toks.append(ENTITY_MARKERS[0])
            if idx == t_span["start"]:
                out_toks.append(ENTITY_MARKERS[2])

            out_toks.append(tok)

            if idx == h_span["end"] - 1:
                out_toks.append(ENTITY_MARKERS[1])
            if idx == t_span["end"] - 1:
                out_toks.append(ENTITY_MARKERS[3])

        return " ".join(out_toks), rel_type

    def preprocess_function(examples):
        """
        批处理版本（batched=True）——把一个 batch 内 *所有* 正/负关系
        扁平展开为独立样本。
        每条关系对应一条 sentence + label。
        """

        flat_sentences, flat_labels = [], []

        for tokens, entities, relations in zip(
                examples["tokens"], examples["entities"], examples["relations"]):

            pos_pairs = set()

            # ---------- 正样本 ----------
            for r in relations:
                if isinstance(r, dict):
                    h_id, t_id, rel_label = r["head"], r["tail"], r["type"]
                else:  # tuple / list
                    if isinstance(r[0], int) and isinstance(r[1], int):
                        h_id, t_id, rel_label = r[0], r[1], r[2]
                    else:
                        rel_label, h_id, t_id = r[0], r[1], r[2]

                try:
                    sent, _ = build_instance(tokens, entities, h_id, t_id, rel_label)
                except Exception as e:                      # 任何异常直接跳过
                    logger.warning(f"[SKIP-POS] {e}")
                    continue

                flat_sentences.append(sent)
                flat_labels.append(label_to_id[rel_label])
                pos_pairs.add(tuple(sorted((h_id, t_id))))

            # ---------- 负样本 (已修复) ----------
            if data_args.negative_ratio > 0:
                if isinstance(entities, list):
                    all_ids = list(range(len(entities)))
                else:
                    all_ids = [k for k, v in entities.items() if isinstance(v, dict) and "start" in v]

                # 1. 找出所有潜在的负样本对
                potential_neg_pairs = []
                for h_id, t_id in combinations(all_ids, 2):
                    # 使用 sorted 匹配 pos_pairs 中的格式
                    if tuple(sorted((h_id, t_id))) not in pos_pairs:
                        potential_neg_pairs.append((h_id, t_id))

                # 2. 计算需要采样的负样本数量
                num_pos = len(pos_pairs)
                # 如果没有正样本，就不添加负样本
                if num_pos == 0:
                    continue
                
                num_neg_to_sample = int(num_pos * data_args.negative_ratio)
                num_neg_to_sample = min(num_neg_to_sample, len(potential_neg_pairs))

                # 3. 从负样本池中随机采样
                neg_samples = random.sample(potential_neg_pairs, num_neg_to_sample)

                for h_id, t_id in neg_samples:
                    try:
                        sent, _ = build_instance(tokens, entities, h_id, t_id, NO_RELATION)
                    except Exception as e:
                        logger.warning(f"[SKIP-NEG] {e}")
                        continue
                    flat_sentences.append(sent)
                    flat_labels.append(label_to_id[NO_RELATION])

        # 若整个 batch 都被过滤掉，返回空 dict → datasets 会自动丢弃
        if not flat_sentences:
            return {}

        tokenized = tokenizer(
            flat_sentences, padding=padding, truncation=True, max_length=max_seq_length
        )
        tokenized["label"] = flat_labels
        return tokenized

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        column_names = raw_datasets["train"].column_names     # ['entities', 'tokens', ...]
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,                       # 关键：批处理并扁平化
            remove_columns=column_names,        # 移除原始字段，只保留 tokenizer 输出
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
            else:
                logger.warning("Validation dataset not found. Falling back to test dataset for validation.")
                eval_dataset = raw_datasets["test"]
        else:
            eval_dataset = raw_datasets["validation"]

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        # remove label column if it exists
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # --- metrics ---
    metric_f1  = evaluate.load("f1",       cache_dir=model_args.cache_dir)
    metric_acc = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
    logger.info("Metrics loaded: macro-F1, micro-F1, accuracy")

    def compute_metrics(p: EvalPrediction):
        """Return macro-F1, micro-F1, accuracy, and a detailed report."""
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds  = np.argmax(logits, axis=1)

        # 打印详细的分类报告，这是诊断问题的关键
        # target_names=label_list 会显示类别名称而不是ID
        report = classification_report(
            y_true=p.label_ids,
            y_pred=preds,
            target_names=label_list,
            digits=4
        )
        print("\n" + report)
        
        # 原有的指标计算保持不变
        macro_f1 = metric_f1.compute(predictions=preds, references=p.label_ids,
                                     average="macro")["f1"]
        micro_f1 = metric_f1.compute(predictions=preds, references=p.label_ids,
                                     average="micro")["f1"]
        acc      = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
        return {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "accuracy": acc,
        }

    class SequenceTrainer(Trainer):
        def __init__(self, *args, loss_name: str = "ce", **kwargs):
            super().__init__(*args, **kwargs)
            if loss_name == "focal":
                self.criterion = FocalLoss(gamma=2.0)
            elif loss_name == "dice":
                # α 设 0.8 ⇒ 80% Dice + 20% CE，可显著提升稳定性；如需纯 Dice 把 α 改 1.0
                self.criterion = DiceLoss(alpha=1, square_denominator=False)
            else:  # 默认交叉熵
                self.criterion = nn.CrossEntropyLoss()

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits            # [B, C]
            loss = self.criterion(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = SequenceTrainer(
        loss_name=model_args.loss_name,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        # 运行预测并同时计算测试集指标
        predict_output = trainer.predict(predict_dataset, metric_key_prefix="test")

        # 记录并保存指标
        metrics = predict_output.metrics
        trainer.log_metrics("test", metrics)      # 打到控制台和日志文件
        trainer.save_metrics("test", metrics)     # 保存到 output_dir/test_results.json

        # 转 logits → 标签文本
        predictions = np.argmax(predict_output.predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        with open(output_predict_file, "w") as writer:
            writer.write("index\tprediction\n")
            for idx, item in enumerate(predictions):
                writer.write(f"{idx}\t{label_list[item]}\n")
        logger.info(f"Test predictions saved at {output_predict_file}")
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "relation-extraction"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()