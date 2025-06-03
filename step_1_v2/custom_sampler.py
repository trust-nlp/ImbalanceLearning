# custom_sampler.py
import torch
from torch.utils.data import Sampler
import logging
from typing import Optional # Added Optional for p_max type hint

logger = logging.getLogger(__name__)

class HardnessAwareSampler(Sampler):
    """
    Samples elements based on their EMA loss, giving higher probability to harder examples.
    The EMA losses are expected to be updated externally by the Trainer.
    """
    def __init__(self, 
                 dataset_len: int,
                 ema_losses: torch.Tensor, # Shared tensor with Trainer
                 alpha: float = 1.0,
                 epsilon: float = 1e-6,
                 p_max: Optional[float] = None, # Optional: max probability cap
                 num_samples: Optional[int] = None):
        self.dataset_len = dataset_len
        self.ema_losses = ema_losses # This is a reference to the tensor in Trainer
        self.alpha = alpha
        self.epsilon = epsilon
        self.p_max = p_max
        self.num_samples = num_samples if num_samples is not None else self.dataset_len
        
        if not isinstance(self.ema_losses, torch.Tensor):
            raise TypeError(f"ema_losses must be a torch.Tensor, got {type(self.ema_losses)}")
        if self.ema_losses.numel() != self.dataset_len:
            raise ValueError(f"ema_losses length ({self.ema_losses.numel()}) "
                             f"must match dataset_len ({self.dataset_len})")

        logger.info(f"HardnessAwareSampler initialized with dataset_len={self.dataset_len}, "
                    f"alpha={self.alpha}, epsilon={self.epsilon}, p_max={self.p_max}, "
                    f"num_samples={self.num_samples}")
        logger.info(f"Initial EMA losses (first 10): {self.ema_losses[:10].tolist()}")


    def _calculate_probabilities(self) -> torch.Tensor:
        # 精简概率计算
        current_ema_losses = self.ema_losses.cpu().float() # Work with a float copy
        
        # p_i \propto (L̄_i + ε)^α
        weights = (current_ema_losses + self.epsilon) ** self.alpha
        probs = weights / weights.sum()
        
        if self.p_max is not None:
            probs = probs.clamp(max=self.p_max)
            probs = probs / probs.sum()
            
        return probs

    def __iter__(self):
        probabilities = self._calculate_probabilities()
        
        # Ensure probabilities are valid for multinomial sampling
        if torch.any(probabilities < 0):
            logger.error("Negative probabilities found before multinomial sampling. This should not happen.")
            # Fallback to uniform if something went wrong
            probabilities = torch.ones(self.dataset_len, dtype=torch.float) / self.dataset_len
        
        if torch.all(probabilities == 0):
            logger.warning("All probabilities are zero. Falling back to uniform sampling.")
            probabilities = torch.ones(self.dataset_len, dtype=torch.float) / self.dataset_len

        indices = torch.multinomial(probabilities, self.num_samples, replacement=True)
        
        # Log some info about the sampled indices distribution for debugging
        unique_indices, counts = torch.unique(indices, return_counts=True)
        logger.debug(f"HardnessAwareSampler: Epoch sampling done. "
                    f"{len(unique_indices)} unique samples drawn. "
                    f"Max frequency: {counts.max().item()}. "
                    f"Sampled indices (first 10): {indices[:10].tolist()}")
        
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def update_sampler_config(self, ema_losses: torch.Tensor):
        """Allows external updates to the EMA losses if needed, though direct tensor update is preferred."""
        if ema_losses.numel() != self.dataset_len:
            raise ValueError("New EMA losses tensor length does not match dataset length.")
        self.ema_losses = ema_losses
        logger.info("HardnessAwareSampler: EMA losses reference updated.") 
