"""
Fine-tuning the library models for token classification.
"""


# ===== Force-disable flash / mem-efficient SDP-Attention, use math implementation =====
import os
os.environ["PYTORCH_SDP_DISABLE_FLASH"] = "1"
os.environ["PYTORCH_SDP_DISABLE_MEM_EFFICIENT"] = "1"   # Only effective for PyTorch>=2.1

# ------------------------------------------------------------
# Disable torch.compile (minimal one-liner, no compatibility branches)
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.compile = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda x: x))
# ------------------------------------------------------------

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
from collections import defaultdict

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

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a training/validation/test file.")
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
    knn_hard_sample_ratio: float = field(
        default=0.1, metadata={"help": "The ratio of top-weighted samples to be considered 'hard' for KNN neighbor boosting."}
    )
    wnet_lr: float = field(default=3e-4, metadata={"help": "Learning rate for the WNet optimizer."})
    meta_update_lr: float = field(default=1e-3, metadata={"help": "Learning rate for meta-model simulated update."})
    meta_update_scale_factor: float = field(default=0.1, metadata={"help": "Scaling factor for VNet output in meta_probs update."})

    # === NEW ===
    embedding_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory that stores pre-computed *.npy sentence embeddings produced by embed_conll.py."},
    )


# Define CustomDataCollatorForTokenClassification class
class CustomDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Standardize field name to orig_idx
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

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, HardnessArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, hardness_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, hardness_args = parser.parse_args_into_dataclasses()

    # **Force-disable torch.compile (to avoid conflicts with second-order gradients)**
    training_args.torch_compile = False

    # Early check for hardness-aware sampling to minimize side effects
    use_hardness_sampling = hardness_args.hardness_aware_sampling and training_args.do_train

    # Conditional imports
    if use_hardness_sampling:
        from custom_trainer import HardnessAwareTrainer
        from custom_trainer import KnnEpochEndCallback

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_ner", model_args, data_args)

    # Setup logging
    log_level_for_setup = training_args.get_process_log_level()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    # Suppress redundant INFO from datasets and transformers libraries
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    logger.setLevel(log_level_for_setup)
    datasets.utils.logging.set_verbosity(log_level_for_setup)
    transformers.utils.logging.set_verbosity(log_level_for_setup)

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

    # ---------- extra deterministic seeding ----------
    import random, numpy as np, torch
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
            
        # Handle conll format files
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
            
            # Load datasets
            raw_datasets = datasets.DatasetDict()
            for split, file_path in data_files.items():
                dataset_dict = load_conll_dataset(file_path)
                raw_datasets[split] = datasets.Dataset.from_dict(dataset_dict)
        else:
            raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Get the column names for input/target.
    # Determine which dataset to get metadata from, with priority: train -> validation -> test
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
        features = raw_datasets["test"].features
    else:
        # If nothing to do, provide an error message
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict` must be True.")

    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = (
        f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1]
    )

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    # 1. Determine if labels are of ClassLabel (integer type)
    #    This variable will be used later, so it must be defined in all branches.
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)

    # 2. Load model config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # 3. If in training mode, we need to determine the label list from the data and update the config
    if training_args.do_train:
        if labels_are_int:
            label_list = features[label_column_name].feature.names
        else:
            label_list = get_label_list(raw_datasets["train"][label_column_name])

        # Update label information in the config
        config.num_labels = len(label_list)
        config.label2id = {l: i for i, l in enumerate(label_list)}
        config.id2label = {i: l for i, l in enumerate(label_list)}

    # 4. If in prediction/evaluation mode, label info is already in config, use it directly
    #    Also, we need the label_list variable to be available later in the code
    if hasattr(config, "id2label") and config.id2label:
        label_list = [config.id2label[i] for i in range(config.num_labels)]
    else:
        if not training_args.do_train:
            raise ValueError(
                "The model config doesn't contain label information. "
                "You must provide a --train_file to build the label list."
            )

    num_labels = len(label_list)
    label_to_id = {l: i for i, l in enumerate(label_list)}  # Regenerate from label_list to ensure consistency

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
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
    model.config.use_cache = False

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This script requires a fast tokenizer. To convert a slow tokenizer into a fast one, "
            "please see https://huggingface.co/docs/transformers/fast_tokenizers"
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
                f" {sorted(label_list)}.\nIgnoring the model labels as a result."
            )

    # Set the correspondences label/ID inside the model config
    # Ensure final label_to_id and model.config.label2id are consistent
    model.config.label2id = label_to_id
    model.config.id2label = {v: k for k, v in label_to_id.items()}

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
        raw_train = train_dataset            # Save the original training set for meta-split
        
        # Truncate first, then add index
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        # Initialize meta_dataset placeholder
        meta_dataset = None

        # Only initialize variables needed for hardness sampling when it's enabled
        if use_hardness_sampling:
            def add_original_indices(example, idx):
                # Standardize field name to orig_idx
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

        # ---------- NEW: Load pre-computed sentence embeddings ----------
        if use_hardness_sampling:
            if hardness_args.embedding_dir is None:
                raise ValueError("--embedding_dir must be provided when hardness-aware sampling is enabled.")

            # Infer the embedding filename corresponding to the current training set (train_embeddings.npy)
            train_split_name = os.path.splitext(os.path.basename(data_args.train_file))[0]
            emb_path = os.path.join(hardness_args.embedding_dir, f"{train_split_name}_embeddings.npy")
            if not os.path.isfile(emb_path):
                raise FileNotFoundError(f"Pre-computed embedding file not found: {emb_path}")

            _emb_np = np.load(emb_path)
            if _emb_np.shape[0] != len(train_dataset):
                raise ValueError(f"Embedding count ({_emb_np.shape[0]}) != train_dataset size ({len(train_dataset)}).")
            train_sentence_embeddings = torch.from_numpy(_emb_np).float()   # Keep on CPU, will be moved to device by Trainer later
            logger.info(f"[Embeddings] Loaded pre-computed sentence embeddings from {emb_path}  "
                        f"shape={_emb_np.shape}")
        
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

        # ---------- use validation split as meta-dataset ----------
        if use_hardness_sampling and training_args.do_train:
            meta_dataset = eval_dataset

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

        if not isinstance(actual_predictions_np, np.ndarray):
            try:
                actual_predictions_np = np.array(actual_predictions_np)
                logger.info(f"Converted actual_predictions_np to numpy array. New shape: {actual_predictions_np.shape}")
            except Exception as e:
                logger.error(f"Could not convert actual_predictions_np to numpy array. Error: {e}")
                pass

        if not hasattr(actual_predictions_np, 'shape') or len(actual_predictions_np.shape) < 2:
             logger.error(f"actual_predictions_np does not have expected dimensions. Shape: {actual_predictions_np.shape if hasattr(actual_predictions_np, 'shape') else 'N/A'}")

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

        # seqeval returns a dictionary with overall and per-class metrics
        results = metric.compute(predictions=true_predictions, references=true_labels)
        
        # Create a new dictionary to store all the desired metrics
        final_metrics = {}

        # 1. Add overall metrics (Micro F1 and Accuracy)
        final_metrics["micro_f1"] = results["overall_f1"]
        final_metrics["accuracy"] = results["overall_accuracy"]

        # 2. Extract F1, Precision, and Recall for each entity type, and calculate Macro F1
        entity_f1_scores = []
        for key, value in results.items():
            if isinstance(value, dict) and "f1" in value:
                # Extract F1, Precision, Recall for each class
                entity_name = key
                final_metrics[f"f1_{entity_name}"] = value["f1"]
                final_metrics[f"precision_{entity_name}"] = value["precision"]
                final_metrics[f"recall_{entity_name}"] = value["recall"]
                entity_f1_scores.append(value["f1"])
        
        # Calculate Macro F1
        final_metrics["macro_f1"] = np.mean(entity_f1_scores) if entity_f1_scores else 0.0

        # Return a dictionary containing all detailed metrics
        return final_metrics

    # Initialize our Trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset if training_args.do_train else None,
        "eval_dataset": eval_dataset if training_args.do_eval else None,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "tokenizer": tokenizer,
        "meta_dataset": meta_dataset,   # Always the validation set or None
    }

    # Select trainer class based on use_hardness_sampling
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
            knn_hard_sample_ratio=hardness_args.knn_hard_sample_ratio,
            wnet_lr=hardness_args.wnet_lr,
            meta_update_lr=hardness_args.meta_update_lr,
            meta_update_scale_factor=hardness_args.meta_update_scale_factor,
            precomputed_embeddings=train_sentence_embeddings,
        )
        # Create and add the new callback instance
        knn_callback = KnnEpochEndCallback(trainer) # Pass the trainer instance
        trainer.add_callback(knn_callback)
        logger.info("Added KnnEpochEndCallback to HardnessAwareTrainer.")
    else:
        logger.info("Using standard Trainer.")
        # Remove parameters not accepted by the standard Trainer
        standard_trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k != 'meta_dataset'}
        trainer = Trainer(**standard_trainer_kwargs)

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

    # Fix push and card creation logic - ensure model card is generated regardless of push
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