# custom_trainer.py
import torch
import torch.nn as nn
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Optional, Dict, Union, Any, List, Tuple
import logging
from custom_sampler import HardnessAwareSampler # Assuming custom_sampler.py is in the same directory
import faiss
from collections import defaultdict

logger = logging.getLogger(__name__)

# 新的回调类
class KnnEpochEndCallback(TrainerCallback):
    def __init__(self, trainer_instance: 'HardnessAwareTrainer'):
        super().__init__()
        self.trainer = trainer_instance # Hold a reference to the trainer
        self._epoch_counter = 0 # Local epoch counter for this callback
        logger.info("KnnEpochEndCallback initialized.")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info(">>> KnnEpochEndCallback.on_epoch_end called.")
        
        # Access trainer's attributes directly through self.trainer
        if not self.trainer.hardness_aware_sampling_enabled:
            logger.info("KnnEpochEndCallback: Hardness sampling is disabled in trainer, skipping KNN logic.")
            return

        self._epoch_counter += 1
        logger.info(f"KnnEpochEndCallback: Epoch counter: {self._epoch_counter}, KNN build freq: {self.trainer.knn_build_freq}.")
        if self._epoch_counter % self.trainer.knn_build_freq != 0:
            logger.info(f"KnnEpochEndCallback: Skipping KNN index build for epoch {self._epoch_counter} (counter % freq != 0).")
            return

        logger.info(f"KnnEpochEndCallback: Number of collected entity_embeds before check: {len(self.trainer._entity_embeds)}")
        if not self.trainer._entity_embeds:
            logger.info("KnnEpochEndCallback: No entity embeddings collected. Skipping KNN index build.")
            return
        
        logger.info(f"KnnEpochEndCallback: Starting KNN index build. Collected {len(self.trainer._entity_embeds)} embeddings.")

        unique_embeds_dict = {}
        for gid, embed in self.trainer._entity_embeds:
            if gid not in unique_embeds_dict:
                unique_embeds_dict[gid] = embed
        
        if not unique_embeds_dict:
            logger.info("KnnEpochEndCallback: No unique entity embeddings. Skipping KNN index build.")
            self.trainer._entity_embeds.clear()
            return

        indices = list(unique_embeds_dict.keys())
        embed_list = list(unique_embeds_dict.values())
        
        embed_mat = torch.stack(embed_list).numpy().astype("float32")
        dim = embed_mat.shape[1]
        num_entities_to_index = embed_mat.shape[0]
        logger.info(f"KnnEpochEndCallback: Building FAISS index with {num_entities_to_index} unique embeddings of dim {dim}.")

        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embed_mat)
        index.add(embed_mat)
        logger.info("KnnEpochEndCallback: FAISS index built.")

        k_to_search = min(self.trainer.knn_k + 1, num_entities_to_index)
        if k_to_search <=1 and self.trainer.knn_k > 0:
             logger.warning(f"KnnEpochEndCallback: Not enough entities ({num_entities_to_index}) for knn_k={self.trainer.knn_k}. Searching for {max(0, k_to_search-1)} neighbors.")
        
        if num_entities_to_index == 0 or k_to_search == 0:
            logger.info("KnnEpochEndCallback: No entities or k=0. Skipping search.")
            if hasattr(self.trainer, '_neighbor_boost') and self.trainer._neighbor_boost is not None:
                self.trainer._neighbor_boost.fill_(0.0)
        else:
            _, knn_idx = index.search(embed_mat, k_to_search)
            logger.info(f"KnnEpochEndCallback: FAISS search done.")

            neighbor_cnt = defaultdict(float)
            for i in range(num_entities_to_index):
                for neighbor_rank in range(1, k_to_search):
                    neighbor_original_index_in_embed_mat = knn_idx[i, neighbor_rank]
                    gid_j = indices[neighbor_original_index_in_embed_mat]
                    neighbor_cnt[gid_j] += 1.0
            logger.info(f"KnnEpochEndCallback: Neighbor counts calculated: {len(neighbor_cnt)} sentences boosted.")

            # Log id of trainer's _neighbor_boost before update
            if hasattr(self.trainer, '_neighbor_boost') and self.trainer._neighbor_boost is not None:
                logger.info(f"KnnEpochEndCallback: self.trainer._neighbor_boost id BEFORE update: {id(self.trainer._neighbor_boost)}, device: {self.trainer._neighbor_boost.device}")
            else:
                logger.warning("KnnEpochEndCallback: self.trainer._neighbor_boost is None or does not exist BEFORE update attempt.")

            # Create boost_vec on the same device as self.trainer._neighbor_boost or ema_losses
            target_device = self.trainer.ema_losses.device # Assuming ema_losses is on the correct device (GPU)
            if hasattr(self.trainer, '_neighbor_boost') and self.trainer._neighbor_boost is not None:
                target_device = self.trainer._neighbor_boost.device

            boost_vec = torch.zeros_like(self.trainer._neighbor_boost, device=target_device) # Ensure correct device
            if neighbor_cnt:
                max_cnt = max(neighbor_cnt.values())
                if max_cnt > 0:
                    for gid, c in neighbor_cnt.items():
                        if 0 <= gid < len(boost_vec): # gid is an original_idx
                             boost_vec[gid] = c / max_cnt
            
            # Perform in-place update
            if hasattr(self.trainer, '_neighbor_boost') and self.trainer._neighbor_boost is not None:
                self.trainer._neighbor_boost.copy_(boost_vec)
                logger.info(f"KnnEpochEndCallback: self.trainer._neighbor_boost (id: {id(self.trainer._neighbor_boost)}) updated IN-PLACE. Max boost: {self.trainer._neighbor_boost.max().item() if len(neighbor_cnt)>0 else 0.0}")
            else:
                logger.error("KnnEpochEndCallback: self.trainer._neighbor_boost is None, cannot perform in-place update!")

        self.trainer._entity_embeds.clear()
        logger.info("KnnEpochEndCallback: Entity embeddings cache cleared.")

class HardnessAwareTrainer(Trainer):
    def __init__(self, *args, 
                 train_dataset_len: Optional[int] = None, # Needed for sampler
                 hardness_aware_sampling: bool = False,
                 hardness_ema_beta: float = 0.9,
                 hardness_alpha: float = 1.0,
                 hardness_epsilon: float = 1e-6,
                 hardness_p_max: Optional[float] = None,
                 initial_ema_loss_value: float = 1.0, # Initial loss for EMA
                 knn_k: int = 5,
                 knn_lambda: float = 0.5,
                 knn_build_freq: int = 1,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.hardness_aware_sampling_enabled = hardness_aware_sampling
        self.ema_beta = hardness_ema_beta
        self.hardness_alpha = hardness_alpha
        self.hardness_epsilon = hardness_epsilon
        self.hardness_p_max = hardness_p_max
        self.initial_ema_loss_value = initial_ema_loss_value
        self.knn_k = knn_k
        self.knn_lambda = knn_lambda
        self.knn_build_freq = knn_build_freq # This will be used by the callback
        self._entity_embeds: List[Tuple[int, torch.Tensor]] = [] # Remains here, populated by compute_loss
        
        self.train_dataset_len = train_dataset_len

        logger.info(f"HardnessAwareTrainer initialized. Hardness Sampling Enabled: {self.hardness_aware_sampling_enabled}")
        if self.hardness_aware_sampling_enabled:
            logger.info(f"  KNN K: {self.knn_k}, KNN Lambda: {self.knn_lambda}, KNN Build Freq: {self.knn_build_freq}")
            if self.train_dataset_len is None or self.train_dataset_len == 0:
                 logger.warning("  train_dataset_len is None or 0. This might be an issue if not set before sampler creation.")
            else:
                 logger.info(f"  train_dataset_len: {self.train_dataset_len}")
                 # Initialize _neighbor_boost on the trainer's main device (e.g., GPU)
                 device_for_boost = self.args.device 
                 self._neighbor_boost = torch.zeros(self.train_dataset_len, dtype=torch.float32, device=device_for_boost)
                 logger.info(f"  Initialized _neighbor_boost tensor on device {device_for_boost} with shape ({self.train_dataset_len},) and id {id(self._neighbor_boost)}")

        if self.hardness_aware_sampling_enabled:
            if self.train_dataset is None:
                raise ValueError("HardnessAwareTrainer: train_dataset must be provided for hardness-aware sampling.")
            
            if self.train_dataset_len is None or self.train_dataset_len == 0:
                # This check is now earlier, which is good.
                raise ValueError("HardnessAwareTrainer: train_dataset_len is 0 or None, invalid for hardness-aware sampling.")

            # Initialize ema_losses on the trainer's main device
            device_for_ema = self.args.device
            self.ema_losses = torch.full((self.train_dataset_len,), 
                                         self.initial_ema_loss_value, 
                                         dtype=torch.float32, device=device_for_ema) # Use trainer's device
            
            logger.info(f"  EMA Beta: {self.ema_beta}, Alpha: {self.hardness_alpha}, Epsilon: {self.hardness_epsilon}")
            logger.info(f"  Initial EMA Loss: {self.initial_ema_loss_value}, P_max: {self.hardness_p_max}")
            logger.info(f"  EMA losses tensor shape: {self.ema_losses.shape}, device: {self.ema_losses.device}, id: {id(self.ema_losses)}")

    def _get_train_sampler(self, train_dataset: Optional[Any] = None) -> Optional[torch.utils.data.Sampler]:
        if not self.hardness_aware_sampling_enabled:
            return super()._get_train_sampler()

        if self.train_dataset is None: # This is redundant if checked in __init__, but safe
            raise ValueError("Trainer: training requires a train_dataset.")
        
        logger.info("Creating HardnessAwareSampler for training.")
        
        passed_boost_tensor = self._neighbor_boost if hasattr(self, '_neighbor_boost') and self._neighbor_boost is not None else None
        if passed_boost_tensor is not None:
            logger.info(f"  Trainer._get_train_sampler passing _neighbor_boost with id {id(passed_boost_tensor)} (device: {passed_boost_tensor.device}) to sampler.")
        else:
            logger.info("  Trainer._get_train_sampler passing None as _neighbor_boost to sampler.")

        return HardnessAwareSampler(
            dataset_len=self.train_dataset_len,
            ema_losses=self.ema_losses, # This is a reference to the tensor in Trainer
            alpha=self.hardness_alpha,
            epsilon=self.hardness_epsilon,
            p_max=self.hardness_p_max,
            neighbor_boost=passed_boost_tensor, # Pass direct tensor reference (should be on GPU)
            lambda_boost=self.knn_lambda,
            num_samples=None 
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: Optional[int] = -1):
        original_indices = inputs.pop("orig_idx", None)
        labels = inputs.pop("labels", None)
        outputs = model(**inputs, labels=labels)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "compute_loss"): 
                 loss = unwrapped_model.compute_loss(outputs, labels)
            else: 
                loss = self.label_smoother(outputs, labels, shift_labels=True) if self.label_smoother else outputs.loss
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError("The model did not return a loss...")
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # --- Hardness logic which populates self._entity_embeds ---
        if self.hardness_aware_sampling_enabled and original_indices is not None and model.training:
            logits = outputs.logits 
            true_labels = labels
            
            if true_labels is not None:
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                id2label = model.config.id2label
                per_token_loss = loss_fct(logits.view(-1, model.config.num_labels), true_labels.view(-1)).view(true_labels.shape[0], true_labels.shape[1])
                mask = (true_labels != -100)
                batch_size, seq_len = true_labels.shape
                per_sentence_loss_list = []
                spans_per_sentence: List[List[Tuple[int, int]]] = [[] for _ in range(batch_size)]
                entity_losses_per_sentence: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

                for s in range(batch_size):
                    current_spans = []
                    for i, lbl_id in enumerate(true_labels[s].tolist()):
                        lbl = id2label.get(lbl_id, "")
                        if lbl.startswith("B-"):
                            start = i; end = i
                            current_entity_type = lbl.split('-')[1] if len(lbl.split('-')) > 1 else None
                            while end + 1 < seq_len:
                                next_lbl_id = true_labels[s, end+1].item()
                                next_lbl = id2label.get(next_lbl_id, "")
                                if next_lbl.startswith("I-"):
                                    next_entity_type = next_lbl.split('-')[1] if len(next_lbl.split('-')) > 1 else None
                                    if next_entity_type == current_entity_type: end += 1
                                    else: break
                                else: break
                            current_spans.append((start, end))
                    spans_per_sentence[s] = current_spans
                    current_entity_losses = []
                    for (st, ed) in current_spans:
                        span_mask = mask[s, st:ed+1]
                        if span_mask.sum() > 0:
                             current_entity_losses.append( (per_token_loss[s, st:ed+1] * span_mask).sum() / span_mask.sum())
                        else: 
                             current_entity_losses.append(torch.tensor(0.0, device=per_token_loss.device))
                    entity_losses_per_sentence[s] = current_entity_losses
                    if current_entity_losses: sent_loss = torch.stack(current_entity_losses).max()
                    elif mask[s].sum() > 0: sent_loss = (per_token_loss[s] * mask[s]).sum() / mask[s].sum().clamp(min=1)
                    else: sent_loss = torch.tensor(0.0, device=per_token_loss.device)
                    per_sentence_loss_list.append(sent_loss)
                per_sentence_loss = torch.stack(per_sentence_loss_list).detach()

                logger.debug(f"[Embeddings] In compute_loss. model.training={model.training}, num_sentences_in_batch={len(per_sentence_loss)}, global_step={self.state.global_step}")
                if per_sentence_loss.numel() > 0: logger.debug(f"[Embeddings] Per-sentence losses (min/max/mean): {per_sentence_loss.min().item():.4f}/{per_sentence_loss.max().item():.4f}/{per_sentence_loss.mean().item():.4f}")
                else: logger.warning("[Embeddings] per_sentence_loss is empty in current batch.")

                if model.config.output_hidden_states and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    if per_sentence_loss.numel() == 0:
                        logger.warning("[Embeddings] per_sentence_loss empty, cannot compute hard_threshold.")
                    else:
                        hard_threshold = torch.quantile(per_sentence_loss.cpu(), 0.8).item()
                        logger.debug(f"[Embeddings] Hard threshold: {hard_threshold:.4f}")
                        num_embeddings_added_this_batch = 0
                        with torch.no_grad():
                            last_hidden = outputs.hidden_states[-1]
                            for b_idx, sent_loss_val in enumerate(per_sentence_loss):
                                if sent_loss_val.item() < hard_threshold: continue
                                max_loss_val = sent_loss_val.item()
                                for span_info, ent_loss_val in zip(spans_per_sentence[b_idx], entity_losses_per_sentence[b_idx]):
                                    if abs(ent_loss_val.item() - max_loss_val) < 1e-6:
                                        st, ed = span_info
                                        if mask[b_idx, st:ed+1].sum() > 0:
                                            ent_repr = last_hidden[b_idx, st:ed+1][mask[b_idx, st:ed+1]].mean(0)
                                            gid = original_indices[b_idx].item()
                                            self._entity_embeds.append((gid, ent_repr.detach().cpu()))
                                            num_embeddings_added_this_batch += 1
                                        else: logger.debug(f"[Embeddings] Skipped GID {original_indices[b_idx].item()}, span [{st}-{ed}] masked.")
                                        break 
                        if num_embeddings_added_this_batch > 0: logger.debug(f"[Embeddings] Added {num_embeddings_added_this_batch} embeddings. Total: {len(self._entity_embeds)}.")
                else: logger.warning("[Embeddings] Hidden states not available for embedding saving.")

                current_device_indices = original_indices.to(self.ema_losses.device)
                per_sentence_loss_device_sync = per_sentence_loss.to(self.ema_losses.device)
                old_ema = self.ema_losses[current_device_indices]
                new_ema = self.ema_beta * old_ema + (1 - self.ema_beta) * per_sentence_loss_device_sync
                self.ema_losses[current_device_indices] = new_ema
        return (loss, outputs) if return_outputs else loss 