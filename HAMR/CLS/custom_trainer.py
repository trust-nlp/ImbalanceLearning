# cls/Meta_Resample/custom_trainer_cls.py
# Precaution: Disable torch.compile when this file is run standalone.
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.compile = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda x: x))

torch._functorch.config.donated_buffer = False
import torch.nn as nn
from torch.func import functional_call
from torch.utils.data import DataLoader
from collections import OrderedDict
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Optional, Dict, Union, Any, List, Tuple
import logging
# Enable global INFO level logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
from custom_sampler import HardnessAwareSampler
from weight_network import WNet
import faiss
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class KnnEpochEndCallback(TrainerCallback):
    """
    At the end of each epoch, finds hard samples with high weights and boosts
    the sampling weights of their neighbors.
    """
    def __init__(self, trainer_instance: "HardnessAwareTrainerForCls"):
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
            not tr.weighted_sampling
            or not tr.hardness_aware_sampling_enabled
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

        # 2. Identify "hard samples" by selecting top N% based on weights
        num_hard_samples = int(len(current_probs) * tr.knn_hard_sample_ratio)
        if num_hard_samples == 0:
            logger.warning("[KNN] No hard samples to boost based on ratio. Skipping.")
            return

        # Get indices of samples with the highest weights
        hard_indices = torch.topk(current_probs, k=num_hard_samples).indices
        logger.info(f"[KNN] Identified {len(hard_indices)} hard samples to find neighbors for.")

        # 3. Extract and L2-normalize embeddings of hard samples
        hard_embeddings = tr._emb_matrix[hard_indices].cpu().numpy()
        faiss.normalize_L2(hard_embeddings)

        # 4. Batch search for neighbors of these hard samples
        _, neighbor_indices = tr.faiss_index.search(hard_embeddings, tr.knn_k)

        # 5. Count "hits" for each neighbor
        unique_neighbors, counts = np.unique(neighbor_indices.ravel(), return_counts=True)
        
        # Create a new boost tensor and populate with counts
        boost = torch.zeros_like(tr._neighbor_boost) # Created on CPU
        boost[unique_neighbors] = torch.from_numpy(counts).float()

        # 6. Normalize and update the boost tensor in the trainer
        if boost.max() > 0:
            boost = boost / boost.max()
        
        tr._neighbor_boost.copy_(boost)
        logger.info(f"[KNN] Neighbor boost updated for {len(unique_neighbors)} unique neighbors. Max boost value: {boost.max().item():.4f}")


class HardnessAwareTrainerForCls(Trainer):
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
                 weighted_loss: bool = True,
                 weighted_sampling: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure Trainer does not call torch.compile
        self.args.torch_compile = False
        logger.info(f"[MetaInit] meta_dataset_len = {len(meta_dataset) if meta_dataset is not None else 0}")

        self.meta_dataset = meta_dataset
        if self.meta_dataset is not None:
            generator = torch.Generator().manual_seed(self.args.seed)
            self.meta_dataloader = DataLoader(
                self.meta_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
                generator=generator,
            )
            self._meta_iter = iter(self.meta_dataloader)
        else:
            self.meta_dataloader = None
            self._meta_iter = None

        self.inner_lr = meta_update_lr

        self.hardness_aware_sampling_enabled = hardness_aware_sampling
        self.hardness_alpha = hardness_alpha
        self.knn_k = knn_k
        self.knn_lambda = knn_lambda
        self.knn_build_freq = knn_build_freq
        self.knn_hard_sample_ratio = knn_hard_sample_ratio

        self.train_dataset_len = train_dataset_len
        self.wnet_lr = wnet_lr
        self.meta_update_scale_factor = meta_update_scale_factor
        self.weighted_loss = weighted_loss
        self.weighted_sampling = weighted_sampling

        logger.info(f"HardnessAwareTrainerForCls initialized. Hardness Sampling Enabled: {self.hardness_aware_sampling_enabled}")
        if self.hardness_aware_sampling_enabled:
            logger.info(f"  H.Alpha: {self.hardness_alpha}, WNet LR: {self.wnet_lr}, MetaUpdateScale: {self.meta_update_scale_factor}")
            if self.train_dataset_len is None or self.train_dataset_len == 0:
                raise ValueError("HardnessAwareTrainer: train_dataset_len must be greater than 0")

            device_for_tensors = torch.device("cpu")  # Store large tensors on CPU to save GPU memory
            self._neighbor_boost = torch.zeros(self.train_dataset_len, dtype=torch.float32, device=device_for_tensors)
            self.meta_probs = torch.full(
                (self.train_dataset_len,),
                1.0 / self.train_dataset_len,
                dtype=torch.float32,
                device=device_for_tensors
            )
            self.wnet = WNet(1, 100, 1).to(device_for_tensors)
            self.wopt = torch.optim.Adam(self.wnet.parameters(), lr=self.wnet_lr, weight_decay=1e-4)

            if precomputed_embeddings is None:
                raise ValueError("precomputed_embeddings must be provided when hardness-aware sampling is enabled.")

            self._emb_matrix = precomputed_embeddings.cpu()
            dim = self._emb_matrix.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            emb_l2 = self._emb_matrix.detach().cpu().numpy()
            faiss.normalize_L2(emb_l2)
            self.faiss_index.add(emb_l2)
            logger.info(f"[FAISS] Added {emb_l2.shape[0]} vectors, dim={dim}")

            if getattr(self.args, "gradient_checkpointing", False):
                self.model.gradient_checkpointing_enable()

    def _get_train_sampler(
        self,
        dataset,  # The current train_dataset passed by transformers.Trainer
    ) -> Optional[torch.utils.data.Sampler]:
        """
        Returns a HardnessAwareSampler if hard sampling is enabled,
        otherwise falls back to the parent's default implementation.
        """
        if not self.hardness_aware_sampling_enabled or not self.weighted_sampling:
            return super()._get_train_sampler(dataset)

        if dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        logger.info("Creating HardnessAwareSampler for training.")
        return HardnessAwareSampler(
            dataset_len=self.train_dataset_len,
            meta_probs=self.meta_probs,
            alpha=self.hardness_alpha,
            epsilon=1e-6,
            neighbor_boost=self._neighbor_boost,
            lambda_boost=self.knn_lambda,
            num_samples=None,        # One epoch iterates through all samples once
            replacement=False,
        )

    def _get_per_sample_loss(self, outputs, labels):
        """
        Calculates per-sample loss for classification tasks.
        """
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        logits = outputs.logits
        return loss_fct(logits, labels)

    def _normalize_loss(self, loss_tensor: torch.Tensor) -> torch.Tensor:
        """Batch-level z-score normalization: z = (loss - mean) / std."""
        if loss_tensor.numel() <= 1:
            return torch.zeros_like(loss_tensor)
        mean = loss_tensor.mean()
        std = loss_tensor.std(unbiased=False).clamp(min=1e-6)
        return (loss_tensor - mean) / std

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Standard compute_loss for evaluation and prediction.
        Training loss is handled in training_step.
        """
        inputs.pop("orig_idx", None)
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_total_examples: int | None = None,
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        original_indices = inputs.pop("orig_idx", None)

        # Forward pass to get per-sample loss
        with self.compute_loss_context_manager():
            outputs = model(**inputs)
        per_sample_loss = self._get_per_sample_loss(outputs, inputs["labels"])

        # --- WNet forward pass and weight calculation ---
        z = self._normalize_loss(per_sample_loss.detach()).unsqueeze(-1)
        z = z.to(self.wnet.linear1.weight.device)
        w_logit = self.wnet(z).squeeze(-1)
        w = torch.sigmoid(w_logit)
        w = w / (w.mean().detach() + 1e-6)  # Normalize mean to 1
        w = w.clamp(min=0.1, max=8.0)
        w = w.to(per_sample_loss.device)

        # --- Inner (meta) gradient step ---
        inner_loss = (w * per_sample_loss).mean()

        if self.meta_dataloader is not None:
            try:
                meta_batch = next(self._meta_iter)
            except StopIteration:
                self._meta_iter = iter(self.meta_dataloader)
                meta_batch = next(self._meta_iter)

            meta_inputs = self._prepare_inputs(meta_batch)
            meta_inputs.pop("orig_idx", None)
            
            grads = torch.autograd.grad(inner_loss, tuple(p for p in model.parameters() if p.requires_grad), create_graph=True)
            
            param_dict = OrderedDict(p for p in model.named_parameters() if p[1].requires_grad)
            updated_param_dict = OrderedDict()
            
            grad_iter = iter(grads)
            for name, param in param_dict.items():
                grad_val = next(grad_iter)
                updated_param_dict[name] = param - self.inner_lr * grad_val

            for name, buf in model.named_buffers():
                updated_param_dict[name] = buf
            
            # Compute meta_loss on meta-batch with virtually updated parameters
            meta_outputs = functional_call(model, updated_param_dict, (), meta_inputs)
            meta_loss = self._get_per_sample_loss(meta_outputs, meta_inputs["labels"]).mean()
            
            # Update WNet
            self.wopt.zero_grad()
            torch.autograd.backward(meta_loss,
                                     retain_graph=True,
                                     inputs=list(self.wnet.parameters()))
            self.wopt.step()

        # --- Update sampling probabilities (EMA) ---
        if original_indices is not None:
            gamma = 0.99
            with torch.no_grad():
                self.meta_probs.mul_(gamma)
                self.meta_probs.index_add_(0,
                                           original_indices.to(self.meta_probs.device),
                                           (1 - gamma) * w.detach().to(self.meta_probs.device))
                self.meta_probs.add_(1e-6)
                self.meta_probs.div_(self.meta_probs.sum())

        # --- Compute main model gradients ---
        if self.weighted_loss:
            main_loss = (w.detach() * per_sample_loss).mean()
        else:
            main_loss = per_sample_loss.mean()
        
        if self.args.gradient_accumulation_steps > 1:
            main_loss = main_loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(main_loss)

        return main_loss.detach() 