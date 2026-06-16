# custom_trainer.py
# ===== Insurance: Disable torch.compile when this file is run standalone =====
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.compile = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda x: x))
torch.use_deterministic_algorithms(True, warn_only=True)

torch._functorch.config.donated_buffer = False
import torch.nn as nn
from torch.func import functional_call
from torch.utils.data import DataLoader
from collections import OrderedDict
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Optional, Dict, Union, Any, List, Tuple
import logging
from custom_sampler import HardnessAwareSampler
import faiss
import numpy as np
from collections import defaultdict
import copy
import random
from weight_network import WNet
from train_ner import CustomDataCollatorForTokenClassification

logger = logging.getLogger(__name__)
# Log INFO and above
logger.setLevel(logging.INFO)

class KnnEpochEndCallback(TrainerCallback):
    """
    At the end of each epoch, find high-weight hard samples and boost the sampling
    weights of their neighbors. This implements the "hard sample neighbor boosting" logic.
    """
    def __init__(self, trainer_instance: "HardnessAwareTrainer"):
        super().__init__()
        self.trainer = trainer_instance

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        tr = self.trainer
        if (
            not tr.hardness_aware_sampling_enabled
            or tr.knn_lambda == 0
            or tr.faiss_index is None
            or state.epoch is None
            or int(state.epoch) == 0
            or (int(state.epoch) % tr.knn_build_freq) != 0
        ):
            return

        logger.info(f"[KNN] Boosting neighbors of hard samples at epoch {state.epoch:.0f}")

        # 1. Get current sample weights (meta_probs) as a proxy for "hardness"
        current_probs = tr.meta_probs.clone().detach().cpu()

        # 2. Identify "hard samples": select samples with top N% weights
        num_hard_samples = int(len(current_probs) * tr.knn_hard_sample_ratio)
        if num_hard_samples == 0:
            logger.warning("[KNN] No hard samples to boost based on ratio. Skipping.")
            return

        # Find indices of samples with the highest weights
        hard_indices = torch.topk(current_probs, k=num_hard_samples).indices
        logger.info(f"[KNN] Identified {len(hard_indices)} hard samples to find neighbors for.")

        # 3. Extract and L2-normalize embeddings of hard samples
        hard_embeddings = tr._emb_matrix[hard_indices].cpu().numpy()
        faiss.normalize_L2(hard_embeddings)

        # 4. Batch-find neighbors for these hard samples
        _, neighbor_indices = tr.faiss_index.search(hard_embeddings, tr.knn_k)

        # 5. Count how many times each neighbor was "hit"
        unique_neighbors, counts = np.unique(neighbor_indices.ravel(), return_counts=True)
        
        # Create a new boost tensor and populate with counts
        boost = torch.zeros_like(tr._neighbor_boost) # Create on CPU
        boost[unique_neighbors] = torch.from_numpy(counts).float()

        # 6. Normalize and update the trainer's boost tensor
        if boost.max() > 0:
            boost = boost / boost.max()
        
        # .copy_() updates the tensor content in-place without changing memory address
        tr._neighbor_boost.copy_(boost)
        logger.info(f"[KNN] Neighbor boost updated for {len(unique_neighbors)} unique neighbors. Max boost value: {boost.max().item():.4f}")


class HardnessAwareTrainer(Trainer):
    def __init__(self, *args, 
                 meta_dataset=None,
                 train_dataset_len: Optional[int] = None,
                 hardness_aware_sampling: bool = False,
                 hardness_alpha: float = 1.0,
                 knn_k: int = 5,
                 knn_lambda: float = 0.5,
                 knn_build_freq: int = 1,
                 knn_hard_sample_ratio: float = 0.2,
                 wnet_lr: float = 1e-4,
                 meta_update_lr: float = 2e-4,
                 meta_update_scale_factor: float = 2.0,
                 precomputed_embeddings=None,
                 sampler_oversample_ratio: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # ** Ensure Trainer does not call torch.compile **
        self.args.torch_compile = False
        logger.debug(f"[MetaInit] meta_dataset_len = {len(meta_dataset) if meta_dataset is not None else 0}")

        # ----- meta-validation loader -----
        self.meta_dataset = meta_dataset
        if self.meta_dataset is not None:
            self.meta_dataloader = DataLoader(
                self.meta_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
                generator=torch.Generator().manual_seed(self.args.seed)
            )
            self._meta_iter = iter(self.meta_dataloader)
        else:
            self.meta_dataloader = None
            self._meta_iter = None

        # Learning rate for a "virtual" inner step
        self.inner_lr = meta_update_lr

        self.hardness_aware_sampling_enabled = hardness_aware_sampling
        self.hardness_alpha = hardness_alpha
        self.knn_k = knn_k
        self.knn_lambda = knn_lambda
        self.knn_build_freq = knn_build_freq
        self.knn_hard_sample_ratio = knn_hard_sample_ratio

        self.train_dataset_len = train_dataset_len
        self.wnet_lr = wnet_lr
        self.meta_update_lr = meta_update_lr
        self.meta_update_scale_factor = meta_update_scale_factor
        self.sampler_oversample_ratio = float(sampler_oversample_ratio)

        logger.info(f"HardnessAwareTrainer initialized. Hardness Sampling Enabled: {self.hardness_aware_sampling_enabled}")
        if self.hardness_aware_sampling_enabled:
            logger.info(f"  H.Alpha: {self.hardness_alpha}, WNet LR: {self.wnet_lr}, MetaUpdateScale: {self.meta_update_scale_factor}")
            if self.train_dataset_len is None or self.train_dataset_len == 0:
                raise ValueError("HardnessAwareTrainer: train_dataset_len must be > 0")
            
            buffer_device = torch.device("cpu")      # For large matrices only
            self._neighbor_boost = torch.zeros(self.train_dataset_len, dtype=torch.float32, device=buffer_device)
            self.meta_probs = torch.full(
                (self.train_dataset_len,),
                1.0 / self.train_dataset_len,
                dtype=torch.float32,
                device=buffer_device
            )
            self._emb_matrix = precomputed_embeddings.float().cpu()      # Resides on CPU
            self.wnet = WNet(1, 100, 1).to(self.args.device)             # On the same GPU as the main model
            self.wopt = torch.optim.Adam(self.wnet.parameters(), lr=self.wnet_lr, weight_decay=1e-4)

            # ---------- Store sentence embeddings & build FAISS index ----------
            if precomputed_embeddings is None:
                raise ValueError("precomputed_embeddings must be provided when hardness-aware sampling is enabled.")

            dim = self._emb_matrix.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            emb_l2 = self._emb_matrix.detach().cpu().numpy()
            faiss.normalize_L2(emb_l2)          # Cosine similarity = dot product of normalized vectors
            self.faiss_index.add(emb_l2)
            logger.info(f"[FAISS] Added {emb_l2.shape[0]} vectors, dim={dim}")

    # Clear GPU cache before each evaluation to avoid OOM from storing predictions
    def evaluate(self, *args, **kwargs):
        torch.cuda.empty_cache()
        return super().evaluate(*args, **kwargs)

    def _get_train_sampler(self, train_dataset: Optional[Any] = None) -> Optional[torch.utils.data.Sampler]:
        if not self.hardness_aware_sampling_enabled:
            logger.info("[SamplerSetup] Hardness sampling disabled, using super()._get_train_sampler().")
            return super()._get_train_sampler()

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        logger.debug("Creating HardnessAwareSampler for training.")
        
        passed_boost_tensor = self._neighbor_boost
        logger.debug(f"[SamplerSetup] Initializing HardnessAwareSampler with: "
                    f"dataset_len={self.train_dataset_len}, "
                    f"alpha={self.hardness_alpha}, "
                    f"lambda_boost (knn_lambda)={self.knn_lambda}, "
                    f"meta_probs_sum_on_init={self.meta_probs.sum().item():.4f}, "
                    f"meta_probs_device={self.meta_probs.device}, "
                    f"neighbor_boost_is_not_none={passed_boost_tensor is not None}, "
                    f"neighbor_boost_device={passed_boost_tensor.device if passed_boost_tensor is not None else 'N/A'}")
        
        if passed_boost_tensor is not None:
            logger.info(f"  Trainer._get_train_sampler passing _neighbor_boost id={id(passed_boost_tensor)}, device={passed_boost_tensor.device}")

        n = int(self.train_dataset_len * self.sampler_oversample_ratio)
        return HardnessAwareSampler(
            dataset_len=self.train_dataset_len,
            meta_probs=self.meta_probs,
            alpha=self.hardness_alpha,
            epsilon=1e-6,
            neighbor_boost=passed_boost_tensor,
            lambda_boost=self.knn_lambda,
            num_samples=n,
        )

    def _get_per_sentence_loss(self, outputs: Union[Dict, Tuple], labels: torch.Tensor, model_config: Any) -> torch.Tensor:
        """
        Calculates per-sentence loss.
        A simple version: average loss of actual tokens in the sentence.
        Here, we use the average loss of actual tokens as a starting point.
        """
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        logits = outputs.logits if isinstance(outputs, dict) else outputs[1]
        
        batch_size, seq_len, num_labels = logits.shape
        
        per_token_loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        per_token_loss = per_token_loss.view(batch_size, seq_len)
        
        mask = (labels != -100)  # Ignore padding tokens
        
        # Sum loss over actual tokens and divide by number of actual tokens
        sentence_loss = (per_token_loss * mask).sum(dim=1)
        num_actual_tokens = mask.sum(dim=1).float()
        
        # Avoid division by zero for sentences with no actual tokens
        per_sentence_loss_avg = (per_token_loss * mask).max(dim=1).values
        
        return per_sentence_loss_avg

    def _normalize_loss(self, loss_tensor: torch.Tensor) -> torch.Tensor:
        """
        Batch-level z-score: z = (loss - mean) / std.
        No longer using EMA.
        """
        loss_tensor = loss_tensor.to(self.args.device)
        if loss_tensor.numel() == 0:
            raise ValueError("loss_tensor is empty; check dataloader/batch construction")
        mean = loss_tensor.mean()
        std = loss_tensor.std(unbiased=False).clamp(min=1e-6)
        return (loss_tensor - mean) / std

    def _compute_sample_weights(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample weights from normalized loss z using WNet.
        Returns weights with mean ~1 and clipped range.
        """
        if z.dim() != 2 or z.size(-1) != 1:
            raise ValueError(f"Expected z shape [B,1], got {tuple(z.shape)}")

        w_logit = self.wnet(z).squeeze(-1)  # [B]
        w = torch.sigmoid(w_logit) + 1e-6
        w = w / (w.mean().detach() + 1e-6)
        w = w.clamp(min=0.1, max=8.0)
        return w

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes loss for evaluation/prediction.
        Not used by the custom training_step for loss calculation during training.
        """
        inputs.pop("orig_idx", None)
        labels = inputs.pop("labels", None)

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """
        1. Calc current batch's sentence-level loss -> z-score -> VNet outputs w in (0,1)
        2. Train main model with w * loss
        3. Update VNet with a simulated gradient step + meta_loss on a meta-batch
        4. Update self.meta_probs with w for the Sampler
        """
        logger.debug(f"[TrainStep-Start] Got inputs keys = {list(inputs.keys())}")
        model.train()
        logger.debug(f"[TrainStep-Start] Device = {self.args.device}, labels_present = {'labels' in inputs}")
        inputs = self._prepare_inputs(inputs)
        original_indices = inputs.pop("orig_idx", None)
        labels = inputs.pop("labels", None)
        if labels is None:
            raise ValueError("Labels missing")

        # ---------- Main model forward pass ----------
        outputs = model(**inputs, labels=labels)
        per_sentence_loss = self._get_per_sentence_loss(outputs, labels, model.config)   # [B]
        logger.debug(f"[Loss-PerSentence] min={per_sentence_loss.min().item():.6f}, max={per_sentence_loss.max().item():.6f}, mean={per_sentence_loss.mean().item():.6f}")

        if self.hardness_aware_sampling_enabled:
            # ---------- WNet forward (pre-update) ----------
            z = self._normalize_loss(per_sentence_loss.detach()).unsqueeze(-1)  # [B,1]
            w_pre = self._compute_sample_weights(z)

            if self.state.global_step % 50 == 0:
                logger.info(
                    f"[W(pre)] step={self.state.global_step} mean={w_pre.mean():.2f}  "
                    f"p10={w_pre.quantile(0.1).item():.2f}  p90={w_pre.quantile(0.9).item():.2f}"
                )

            # ---------- Inner loss for virtual step (build meta-grad graph) ----------
            inner_loss = (w_pre * per_sentence_loss).mean()

            wnet_updated = False
            if self.meta_dataloader is not None:
                try:
                    logger.debug("[MetaBatch-Pull] Fetching one meta_batch")
                    meta_batch = next(self._meta_iter)
                except StopIteration:
                    self._meta_iter = iter(self.meta_dataloader)
                    logger.debug("[MetaBatch-Pull] Iterator reset, fetching again")
                    meta_batch = next(self._meta_iter)

                meta_inputs = self._prepare_inputs(meta_batch)
                logger.debug(f"[MetaBatch-Inputs] keys = {list(meta_inputs.keys())}")
                meta_labels = meta_inputs.pop("labels")
                if meta_labels is None:
                    raise ValueError("Meta labels missing")

                logger.debug(f"[MetaStep] About to functional_call with inner_lr = {self.inner_lr}")
                grads = torch.autograd.grad(
                    inner_loss,
                    tuple(model.parameters()),
                    create_graph=True,
                )

                updated_param_dict = OrderedDict()
                for (name, param), grad in zip(model.named_parameters(), grads):
                    updated_param_dict[name] = param - self.inner_lr * grad
                for name, buf in model.named_buffers():
                    updated_param_dict[name] = buf

                meta_inputs["labels"] = meta_labels
                meta_outputs = functional_call(model, updated_param_dict, (), meta_inputs, strict=False)
                meta_loss = meta_outputs.loss
                logger.debug(f"[MetaLoss] value = {meta_loss.item():.6f}")

                # Update WNet
                self.wopt.zero_grad()
                meta_loss.backward(retain_graph=True)
                self.wopt.step()
                wnet_updated = True

            # ---------- WNet forward (post-update) ----------
            # Main-model final update must use post-update weights.
            with torch.no_grad():
                w_post = self._compute_sample_weights(z) if wnet_updated else w_pre

            if self.state.global_step % 50 == 0:
                logger.info(
                    f"[W(post)] step={self.state.global_step} mean={w_post.mean():.2f}  "
                    f"p10={w_post.quantile(0.1).item():.2f}  p90={w_post.quantile(0.9).item():.2f}"
                )

            # ---------- Update sampler probabilities using post-update weights ----------
            if original_indices is None:
                raise ValueError("orig_idx missing while hardness_aware_sampling_enabled is True")

            gamma = 0.99
            self.meta_probs.mul_(gamma)
            self.meta_probs.index_add_(0, original_indices.cpu(), (1 - gamma) * w_post.cpu())
            self.meta_probs.add_(1e-4)
            self.meta_probs.div_(self.meta_probs.sum())

            loss_main = (w_post.detach() * per_sentence_loss).mean()
        else:
            loss_main = outputs.loss

        # ---------- Main model backward ----------
        self.optimizer.zero_grad(set_to_none=True)
        self.accelerator.backward(loss_main)
        return loss_main.detach() / self.args.gradient_accumulation_steps