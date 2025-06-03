# ===== 彻底禁用 flash / mem-efficient SDP-Attention，强制使用 math 实现 =====
import os
os.environ["PYTORCH_SDP_DISABLE_FLASH"] = "1"
os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT"] = "1"   # 仅当 PyTorch>=2.1 时生效
import torch
# 如果是 PyTorch ≥2.1，显式关闭 flash/mem-efficient、启用 math path；否则 sdpa_kernel("math") 兜底
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except AttributeError:
    from torch.nn.attention import sdpa_kernel
    sdpa_kernel("math")
# ========================================================================

"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

# ===== 彻底禁用 flash / mem-efficient SDP-Attention，强制使用 math 实现 =====
import os
os.environ["PYTORCH_SDP_DISABLE_FLASH"] = "1"
os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT"] = "1"   # 仅当 PyTorch>=2.1 时生效
# 早导入 torch 并显式切 backend
import torch
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Union, Any, List, Tuple

import datasets
import evaluate
import numpy as np
from datasets import ClassLabel, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import random

# 条件导入将在后面根据use_hardness_sampling进行

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.50.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

logger = logging.getLogger(__name__)


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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
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
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1].lower()
                if extension not in ["csv", "json", "conll", "txt"]:
                    raise ValueError("`train_file` should be a csv, json, conll or txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1].lower()
                if extension not in ["csv", "json", "conll", "txt"]:
                    raise ValueError("`validation_file` should be a csv, json, conll or txt file.")
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1].lower()
                if extension not in ["csv", "json", "conll", "txt"]:
                    raise ValueError("`test_file` should be a csv, json, conll or txt file.")
        self.task_name = self.task_name.lower()


@dataclass
class HardnessArgs:
    hardness_aware_sampling: bool = field(default=False, metadata={"help": "Enable hardness-aware sampling."})
    hardness_alpha: float = field(default=1.0, metadata={"help": "Exponent for scaling meta_probs in sampler."})
    knn_k: int = field(default=5, metadata={"help": "Top-k neighbors for each hard entity"})
    knn_lambda: float = field(default=0.5, metadata={"help": "Weight boost coefficient for neighbor sentences"})
    knn_build_freq: int = field(default=1, metadata={"help": "Rebuild FAISS index every N epochs"})
    vnet_lr: float = field(default=1e-4, metadata={"help": "Learning rate for the VNet optimizer."})
    meta_update_lr: float = field(default=1e-5, metadata={"help": "Learning rate for meta-model simulated update."})
    meta_update_scale_factor: float = field(default=0.1, metadata={"help": "Scaling factor for VNet output in meta_probs update."})


# 定义CustomDataCollatorForTokenClassification类
class CustomDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 统一字段命名为orig_idx
        original_indices = None
        # Keep track of keys that are not meant for the parent collator but we want to preserve or handle manually
        keys_to_remove_before_super_call = [] 

        if features and "orig_idx" in features[0]: # Check if features is not empty
            original_indices = [feature.pop("orig_idx") for feature in features]
        elif features:
            pass
        else:
            pass

        # Identify other non-tensorizable/raw columns to remove before calling super
        if features:
            if "tokens" in features[0]:
                keys_to_remove_before_super_call.append("tokens")
            if "ner_tags" in features[0]:
                keys_to_remove_before_super_call.append("ner_tags")
            # Add any other raw columns here if they cause issues with the parent collator

            if keys_to_remove_before_super_call:
                for feature in features:
                    for key_to_remove in keys_to_remove_before_super_call:
                        feature.pop(key_to_remove, None) # Remove safely
        
        batch = super().__call__(features) # Call the parent's collate logic with cleaned features

        if original_indices is not None:
            batch["orig_idx"] = torch.tensor(original_indices, dtype=torch.long)
        else:
            pass
        
        return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, HardnessArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, hardness_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, hardness_args = parser.parse_args_into_dataclasses()

    # Early check for hardness-aware sampling to minimize side effects
    use_hardness_sampling = hardness_args.hardness_aware_sampling and training_args.do_train

    # 条件导入
    if use_hardness_sampling:
        from custom_trainer import HardnessAwareTrainer
        from custom_trainer import KnnEpochEndCallback

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_ner", model_args, data_args)

    # Setup logging
    log_level_for_setup = training_args.get_process_log_level() # Get the desired log level from arguments

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level_for_setup, # Set this for the root logger; affects all loggers that propagate to root
    )

    # Set specific log levels for the main script's logger and library loggers
    # `logger` here refers to logging.getLogger(__name__) in this file (step1_run_ner.py)
    # Its level will now be correctly inherited from the root logger set by basicConfig,
    # or explicitly set here.
    logger.setLevel(log_level_for_setup)
    datasets.utils.logging.set_verbosity(log_level_for_setup)
    transformers.utils.logging.set_verbosity(log_level_for_setup)

    # Ensure transformers logs have a handler and are formatted, as per library's recommendation
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

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
            
        # 处理conll格式文件
        if extension == "conll":
            def load_conll_dataset(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                tokens = []
                labels = []
                current_tokens = []
                current_labels = []
                
                for line in lines:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            current_tokens.append(parts[0])
                            current_labels.append(parts[-1])
                    elif current_tokens:
                        tokens.append(current_tokens)
                        labels.append(current_labels)
                        current_tokens = []
                        current_labels = []
                
                if current_tokens:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
                
                return {"tokens": tokens, "ner_tags": labels}
            
            # 加载数据集
            raw_datasets = datasets.DatasetDict()
            for split, file_path in data_files.items():
                dataset_dict = load_conll_dataset(file_path)
                raw_datasets[split] = datasets.Dataset.from_dict(dataset_dict)
        else:
            raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"bloom", "gpt2", "roberta", "deberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    model.config.output_hidden_states = True
    
    # 确保模型使用 math attention
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "math"
    else:
        setattr(model.config, "use_flash_attention", False)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )

    # Model has labels -> use them.
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if sorted(model.config.label2id.keys()) == sorted(label_list):
            # Reorganize `label_list` to match the ordering of the model.
            if labels_are_int:
                label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
                label_list = [model.config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [model.config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: "
                f"model labels: {sorted(model.config.label2id.keys())}, dataset labels:"
                f" {sorted(label_list)}.\nIgnoring the model labels as a result.",
            )

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = dict(enumerate(label_list))

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if data_args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        
        train_dataset = raw_datasets["train"]
        
        # 先截断再添加索引
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        # ---- create meta-validation split for meta-learning (random 5% each epoch) ----
        META_RATIO = 0.1  # 5%
        raw_train = train_dataset  # 先保存整体训练集
        def rebuild_meta_dataset():
            total_len = len(raw_train)
            meta_size = max(1, int(total_len * META_RATIO))
            idxs = list(range(total_len))
            random.shuffle(idxs)
            chosen = idxs[:meta_size]
            subset = raw_train.select(chosen)
            return subset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on meta dataset",
            )
        meta_dataset = rebuild_meta_dataset()
        # Remove the selected examples from train_dataset
        meta_indices = set(range(len(meta_dataset)))  # 先不用，实际后面 Trainer 内部会按采样，但此处保留 train_dataset 原样
        logger.info(f"[MetaSplit] meta_dataset={len(meta_dataset)}, train_dataset={len(train_dataset)}")

        # Only initialize variables needed for hardness sampling when it's enabled
        if use_hardness_sampling:
            def add_original_indices(example, idx):
                # 统一字段命名为orig_idx
                example["orig_idx"] = idx
                return example
            
            current_train_dataset_len = len(train_dataset)
            
            train_dataset = train_dataset.map(
                add_original_indices, 
                with_indices=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Adding original indices to train dataset"
            )
            logger.info("Adding orig_idx for hardness-aware sampling.")
            logger.info(f"[DEBUG_ORIG_IDX] After add_original_indices, train_dataset columns: {train_dataset.column_names}")
            logger.info(f"[DEBUG_ORIG_IDX] Sample 0 after add_original_indices: {train_dataset[0]}")
        else:
            current_train_dataset_len = None
        
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            
        if use_hardness_sampling:
            logger.info(f"Train dataset columns after tokenization: {train_dataset.column_names}")
            logger.info(f"[DEBUG_ORIG_IDX] After tokenize_and_align_labels, train_dataset columns: {train_dataset.column_names}")
            logger.info(f"[DEBUG_ORIG_IDX] Sample 0 after tokenize_and_align_labels: {train_dataset[0]}")
            if 'orig_idx' not in train_dataset.column_names:
                logger.warning("'orig_idx' is missing from tokenized train_dataset columns when hardness_aware_sampling is enabled. "
                               "This might cause issues if not handled by the collator.")
    else:
        train_dataset = None

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                tokenize_and_align_labels,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    if use_hardness_sampling:
        logger.info("Using CustomDataCollatorForTokenClassification for hardness-aware sampling.")
        data_collator = CustomDataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    else:
        logger.info("Using standard DataCollatorForTokenClassification.")
        data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Metrics
    metric = evaluate.load("seqeval", cache_dir=model_args.cache_dir)

    def compute_metrics(p):
        # predictions, labels = p
        # predictions = np.argmax(predictions, axis=2)

        predictions_from_eval_pred, labels = p
        
        actual_predictions_np = predictions_from_eval_pred
        if isinstance(predictions_from_eval_pred, tuple):
            logger.info(
                f"compute_metrics received a tuple for predictions. Assuming the first element contains the actual logits. "
                f"Number of elements in tuple: {len(predictions_from_eval_pred)}"
            )
            actual_predictions_np = predictions_from_eval_pred[0]
            for i, item in enumerate(predictions_from_eval_pred):
                shape_info = item.shape if hasattr(item, 'shape') else 'N/A (not an array)'
                logger.info(f"  Tuple element {i}: type={type(item)}, shape={shape_info}")

        # Ensure actual_predictions_np is a numpy array before np.argmax
        if not isinstance(actual_predictions_np, np.ndarray):
            try:
                actual_predictions_np = np.array(actual_predictions_np)
                logger.info(f"Converted actual_predictions_np to numpy array. New shape: {actual_predictions_np.shape}")
            except Exception as e:
                logger.error(f"Could not convert actual_predictions_np to numpy array. Error: {e}")
                # Fallback or re-raise, depending on desired behavior. For now, let it proceed to argmax to see error there.
                pass

        if not hasattr(actual_predictions_np, 'shape') or len(actual_predictions_np.shape) < 2:
             logger.error(f"actual_predictions_np does not have expected dimensions. Shape: {actual_predictions_np.shape if hasattr(actual_predictions_np, 'shape') else 'N/A'}")
             # Depending on how critical this is, you might want to return empty metrics or raise an error.
             # For now, let it attempt argmax, which will likely fail and provide more context if shape is wrong.

        predictions = np.argmax(actual_predictions_np, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "overall_precision": results["overall_precision"],
                "overall_recall": results["overall_recall"],
                "overall_f1": results["overall_f1"],
                "overall_accuracy": results["overall_accuracy"],
            }

    # Initialize our Trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset if training_args.do_train else None,
        "eval_dataset": eval_dataset if training_args.do_eval else None,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "tokenizer": tokenizer,
        "meta_dataset": meta_dataset if training_args.do_train and use_hardness_sampling else None,
    }

    # 根据use_hardness_sampling选择trainer类
    if use_hardness_sampling:
        logger.info("Enabling HardnessAwareTrainer.")
        trainer = HardnessAwareTrainer(
            **trainer_kwargs,
            train_dataset_len=current_train_dataset_len, 
            hardness_aware_sampling=hardness_args.hardness_aware_sampling,
            hardness_alpha=hardness_args.hardness_alpha,
            knn_k=hardness_args.knn_k,
            knn_lambda=hardness_args.knn_lambda,
            knn_build_freq=hardness_args.knn_build_freq,
            vnet_lr=hardness_args.vnet_lr,
            meta_update_lr=hardness_args.meta_update_lr,
            meta_update_scale_factor=hardness_args.meta_update_scale_factor,
        )
        # 创建并添加新的回调实例
        knn_callback = KnnEpochEndCallback(trainer) # Pass the trainer instance
        trainer.add_callback(knn_callback)
        logger.info("Added KnnEpochEndCallback to HardnessAwareTrainer.")
    else:
        logger.info("Using standard Trainer.")
        trainer = Trainer(**trainer_kwargs)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        original_output_hidden_states = None
        if hasattr(model.config, "output_hidden_states"):
            original_output_hidden_states = model.config.output_hidden_states
            model.config.output_hidden_states = False
            logger.info("Temporarily disabled model.config.output_hidden_states for evaluation.")

        metrics = trainer.evaluate()

        if original_output_hidden_states is not None and hasattr(model.config, "output_hidden_states"):
            model.config.output_hidden_states = original_output_hidden_states
            logger.info("Restored model.config.output_hidden_states after evaluation.")

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        original_output_hidden_states_predict = None
        if hasattr(model.config, "output_hidden_states"):
            original_output_hidden_states_predict = model.config.output_hidden_states
            model.config.output_hidden_states = False
            logger.info("Temporarily disabled model.config.output_hidden_states for prediction.")

        # trainer.predict() might return a tuple (logits, hidden_states, ...) 
        # if model.config.output_hidden_states is True.
        predict_output = trainer.predict(predict_dataset, metric_key_prefix="predict")
        
        if original_output_hidden_states_predict is not None and hasattr(model.config, "output_hidden_states"):
            model.config.output_hidden_states = original_output_hidden_states_predict
            logger.info("Restored model.config.output_hidden_states after prediction.")

        raw_predictions_from_predict = predict_output.predictions
        labels = predict_output.label_ids # or predict_output.labels depending on version/usage
        metrics = predict_output.metrics

        actual_predictions_np = raw_predictions_from_predict
        if isinstance(raw_predictions_from_predict, tuple):
            logger.info(
                f"[Predict Block] Received a tuple for predictions. Assuming the first element contains the actual logits. "
                f"Number of elements in tuple: {len(raw_predictions_from_predict)}"
            )
            actual_predictions_np = raw_predictions_from_predict[0]
            for i, item in enumerate(raw_predictions_from_predict):
                shape_info = item.shape if hasattr(item, 'shape') else 'N/A (not an array)'
                logger.info(f"  [Predict Block] Tuple element {i}: type={type(item)}, shape={shape_info}")
        
        # Ensure actual_predictions_np is a numpy array before np.argmax
        if not isinstance(actual_predictions_np, np.ndarray):
            try:
                actual_predictions_np = np.array(actual_predictions_np)
                logger.info(f"[Predict Block] Converted actual_predictions_np to numpy array. New shape: {actual_predictions_np.shape}")
            except Exception as e:
                logger.error(f"[Predict Block] Could not convert actual_predictions_np to numpy array. Error: {e}")
                # Depending on the desired behavior, you might re-raise or handle it.
                # For now, let it proceed to argmax to see the error there if conversion failed.
                pass

        if not hasattr(actual_predictions_np, 'shape') or len(actual_predictions_np.shape) < 2:
             logger.error(f"[Predict Block] actual_predictions_np does not have expected dimensions. Shape: {actual_predictions_np.shape if hasattr(actual_predictions_np, 'shape') else 'N/A'}")
             # Handle error appropriately

        predictions = np.argmax(actual_predictions_np, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "token-classification"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    # 修复推送与建卡逻辑 - 确保无论是否push，都生成model card
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
        trainer.create_model_card(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()