# custom_sampler.py
import torch
from torch.utils.data import Sampler
import logging
from typing import Optional

logger = logging.getLogger(__name__)
# Log INFO and above
logger.setLevel(logging.INFO)

class HardnessAwareSampler(Sampler):
    """
    Samples elements based on their meta_probs (produced by VNet), multiplied by neighbor_boost.
    meta_probs are updated externally by the Trainer, reflecting the probability
    that a sample should be drawn next.
    """
    def __init__(
        self,
        dataset_len: int,
        meta_probs: torch.Tensor,  # Shared tensor, updated by Trainer
        alpha: float = 0.5,
        epsilon: float = 1e-6,
        num_samples: Optional[int] = None,
        neighbor_boost: Optional[torch.Tensor] = None,
        lambda_boost: float = 0.0,
    ):
        self.dataset_len = dataset_len
        self.meta_probs = meta_probs # Direct reference to trainer's meta_probs
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_samples = int(num_samples) if num_samples is not None else self.dataset_len
        self.lambda_boost = lambda_boost
        self.neighbor_boost = neighbor_boost  # Should be a direct reference to trainer's tensor (on GPU)
        self.replacement = True
        
        if self.neighbor_boost is not None:
            logger.info(f"HardnessAwareSampler initialized with neighbor_boost tensor (id: {id(self.neighbor_boost)}, device: {self.neighbor_boost.device}).")
        else:
            logger.info("HardnessAwareSampler initialized with neighbor_boost as None.")

        if not isinstance(self.meta_probs, torch.Tensor):
            raise TypeError(f"meta_probs must be a torch.Tensor, got {type(self.meta_probs)}")
        if self.meta_probs.numel() != self.dataset_len:
            raise ValueError(f"meta_probs length ({self.meta_probs.numel()}) must match dataset_len ({self.dataset_len})")

    def _calculate_probabilities(self) -> torch.Tensor:
        base = (self.meta_probs.cpu().float() + self.epsilon) ** self.alpha
        if self.neighbor_boost is not None:
            base = base * (1.0 + self.lambda_boost * self.neighbor_boost.cpu().float())
        probs = base / base.sum()
        return probs.clamp(min=1e-12)

    def __iter__(self):
        p = self._calculate_probabilities()
        idx = torch.multinomial(p, self.num_samples, replacement=True)
        u, c = idx.unique(return_counts=True)
        logger.info(f"[Sampler] draws={self.num_samples}, unique={u.numel()}, max_repeats={int(c.max().item())}")
        return iter(idx.tolist())

    def __len__(self):
        return self.num_samples

    # removed deprecated methods