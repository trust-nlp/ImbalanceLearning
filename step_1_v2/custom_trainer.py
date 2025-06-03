# custom_trainer.py
import torch
import torch.nn as nn
from transformers import Trainer
from typing import Optional, Dict, Union, Any, List, Tuple
import logging
from custom_sampler import HardnessAwareSampler # Assuming custom_sampler.py is in the same directory

logger = logging.getLogger(__name__)

class HardnessAwareTrainer(Trainer):
    def __init__(self, *args, 
                 train_dataset_len: Optional[int] = None, # Needed for sampler
                 hardness_aware_sampling: bool = False,
                 hardness_ema_beta: float = 0.9,
                 hardness_alpha: float = 1.0,
                 hardness_epsilon: float = 1e-6,
                 hardness_p_max: Optional[float] = None,
                 initial_ema_loss_value: float = 1.0, # Initial loss for EMA
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.hardness_aware_sampling_enabled = hardness_aware_sampling
        self.ema_beta = hardness_ema_beta
        self.hardness_alpha = hardness_alpha
        self.hardness_epsilon = hardness_epsilon
        self.hardness_p_max = hardness_p_max
        self.initial_ema_loss_value = initial_ema_loss_value
        
        self.train_dataset_len = train_dataset_len # This comes from meta_aug_run_ner.py

        if self.hardness_aware_sampling_enabled:
            if self.train_dataset is None: # self.train_dataset is set by super().__init__
                raise ValueError("HardnessAwareTrainer: train_dataset must be provided for hardness-aware sampling.")
            
            # If train_dataset_len was not provided or is 0, try to get it from self.train_dataset
            if self.train_dataset_len is None or self.train_dataset_len == 0:
                logger.warning("train_dataset_len not provided or is 0 for HardnessAwareTrainer. "
                               "This indicates an issue with how it was passed from the main script if sampling is ON.")

                if self.hardness_aware_sampling_enabled and (self.train_dataset_len is None or self.train_dataset_len == 0):
                    raise ValueError("HardnessAwareTrainer: train_dataset_len is 0 or None, which is invalid for hardness-aware sampling. "
                                     "It should be the length of the pre-tokenized training set passed from the main script.")

            # Initialize EMA losses: 将EMA张量放到训练设备
            # This tensor will be shared with the sampler.
            device = kwargs.get("device", self.args.device)
            self.ema_losses = torch.full((self.train_dataset_len,), 
                                         self.initial_ema_loss_value, 
                                         dtype=torch.float32, device=device) # 放到训练设备
            
            # 初始化 neighbor_boost：所有句子初始增益为 0
            self._neighbor_boost = torch.zeros(self.train_dataset_len, dtype=torch.float32, device=device)

            logger.info("HardnessAwareTrainer initialized for hardness-aware sampling.")
            logger.info(f"  EMA Beta: {self.ema_beta}, Alpha: {self.hardness_alpha}, Epsilon: {self.hardness_epsilon}")
            logger.info(f"  Initial EMA Loss: {self.initial_ema_loss_value}, P_max: {self.hardness_p_max}")
            logger.info(f"  EMA losses tensor shape: {self.ema_losses.shape}, device: {self.ema_losses.device}")


    def _get_train_sampler(self, train_dataset: Optional[Any] = None) -> Optional[torch.utils.data.Sampler]:
        if not self.hardness_aware_sampling_enabled:
            return super()._get_train_sampler()

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        logger.info("Creating HardnessAwareSampler for training.")
        # The sampler will use the self.ema_losses tensor which gets updated by the Trainer
        return HardnessAwareSampler(
            dataset_len=self.train_dataset_len,
            ema_losses=self.ema_losses, # Pass the shared tensor
            alpha=self.hardness_alpha,
            epsilon=self.hardness_epsilon,
            p_max=self.hardness_p_max,
            num_samples=None # Draw len(dataset) samples per epoch
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: Optional[int] = -1):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        # 统一字段命名为orig_idx
        original_indices = inputs.pop("orig_idx", None) # orig_idx: (batch_size,)

        # 简化标签取用逻辑 - 提前取labels
        labels = inputs.pop("labels", None)
        
        # 保证model(**inputs, labels=...)拿到初始loss
        outputs = model(**inputs, labels=labels) # outputs.loss is (batch_size, ) if reduction='none'
                                # or scalar if reduction='mean'/'sum' in model's loss_fct

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "compute_loss"): # For models that compute loss internally
                 loss = unwrapped_model.compute_loss(outputs, labels)
            else: # Default Hugging Face behavior
                loss = self.label_smoother(outputs, labels, shift_labels=True) if self.label_smoother else outputs.loss
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]


        # --- Custom logic for per-sentence loss and EMA update ---
        # Diagnostic logging for the conditions
        logger.info(f"[EMA_DEBUG] Checking conditions for hardness-aware block at step {self.state.global_step}:")
        logger.info(f"[EMA_DEBUG]   self.hardness_aware_sampling_enabled: {self.hardness_aware_sampling_enabled}")
        logger.info(f"[EMA_DEBUG]   original_indices is not None: {original_indices is not None}")
        if original_indices is not None:
            logger.info(f"[EMA_DEBUG]   len(original_indices): {len(original_indices)}")
        logger.info(f"[EMA_DEBUG]   model.training: {model.training}")

        if self.hardness_aware_sampling_enabled and original_indices is not None and model.training:
            logger.info(f"[Max Entity Loss] 成功进入硬度感知采样模块，batch_size={len(original_indices)}, step={self.state.global_step}")
            
            logits = outputs.logits # (batch_size, seq_len, num_classes)
            # 直接使用之前取到的labels
            true_labels = labels # (batch_size, seq_len)
            
            logger.debug(f"[Max Entity Loss] logits shape: {logits.shape}, true_labels shape: {true_labels.shape}")

            if true_labels is not None:
                loss_fct = nn.CrossEntropyLoss(reduction='none') # Get per-token losses
                
                # --- 修改为 Max Entity Loss 按实体 span 计算句子损失 ---
                # 获取 id->标签 名称映射
                id2label = model.config.id2label  # dict: {id: "B-PER", ...}

                per_token_loss = loss_fct(
                    logits.view(-1, model.config.num_labels), 
                    true_labels.view(-1)
                ).view(true_labels.shape[0], true_labels.shape[1])
                mask = (true_labels != -100)
                batch_size, seq_len = true_labels.shape

                per_sentence_loss_list = []
                for s in range(batch_size):
                    # 扫描出所有实体 span（B- 开头 + 后续 I-）
                    spans = []
                    for i, lbl_id in enumerate(true_labels[s].tolist()):
                        lbl = id2label.get(lbl_id, "")
                        if lbl.startswith("B-"):
                            start = i
                            end = i
                            # 连续的 I- 同类标签归入同一实体
                            while end + 1 < seq_len and id2label.get(true_labels[s, end+1].item(), "").startswith("I-"):
                                end += 1
                            spans.append((start, end))
                    # 计算每个实体的平均 token 损失
                    entity_losses = []
                    for (st, ed) in spans:
                        # per_token_loss 已经剔除 padding 标签
                        entity_losses.append(per_token_loss[s, st:ed+1].mean())
                    # 若有实体，取最大损失；否则退回到句子平均
                    if entity_losses:
                        sent_loss = torch.stack(entity_losses).max()
                    else:
                        sent_loss = (per_token_loss[s] * mask[s]).sum() / mask[s].sum().clamp(min=1)
                    per_sentence_loss_list.append(sent_loss)
                per_sentence_loss = torch.stack(per_sentence_loss_list)  # (batch_size,)
                per_sentence_loss = per_sentence_loss.detach() # Detach from graph

                # Update EMA losses (EMA张量已经在训练设备上)
                # Ensure original_indices are valid and on the correct device for indexing
                current_device_indices = original_indices.to(self.ema_losses.device)

                # Move per_sentence_loss to the device of ema_losses before update
                per_sentence_loss_device_sync = per_sentence_loss.to(self.ema_losses.device)
                
                old_ema = self.ema_losses[current_device_indices]
                new_ema = self.ema_beta * old_ema + (1 - self.ema_beta) * per_sentence_loss_device_sync
                self.ema_losses[current_device_indices] = new_ema
                
                # For debugging, log a few EMA updates
                if self.args.logging_steps and self.state.global_step % self.args.logging_steps == 0 : # Log more frequently
                     logger.debug(f"Step {self.state.global_step}: Updated EMA for indices "
                                 f"{current_device_indices.tolist()[:3]}... "
                                 f"from {old_ema.tolist()[:3]}... + new_losses {per_sentence_loss_device_sync.tolist()[:3]}... "
                                 f"to {new_ema.tolist()[:3]}...")
                     logger.debug(f"EMA losses sample (first 10 after update on this step): {self.ema_losses[:10].cpu().tolist()}")

        return (loss, outputs) if return_outputs else loss 