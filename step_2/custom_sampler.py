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
                 num_samples: Optional[int] = None,
                 neighbor_boost: Optional[torch.Tensor] = None,
                 lambda_boost: float = 0.0):
        self.dataset_len = dataset_len
        self.ema_losses = ema_losses # This is a reference to the tensor in Trainer
        self.alpha = alpha
        self.epsilon = epsilon
        self.p_max = p_max
        self.num_samples = num_samples if num_samples is not None else self.dataset_len
        self.lambda_boost = lambda_boost
        self.neighbor_boost = neighbor_boost  # Should be a direct reference to trainer's tensor (on GPU)
        
        if self.neighbor_boost is not None:
            logger.info(f"HardnessAwareSampler initialized with neighbor_boost tensor (id: {id(self.neighbor_boost)}, device: {self.neighbor_boost.device}).")
        else:
            logger.info("HardnessAwareSampler initialized with neighbor_boost as None.")
            
        if not isinstance(self.ema_losses, torch.Tensor):
            raise TypeError(f"ema_losses must be a torch.Tensor, got {type(self.ema_losses)}")
        if self.ema_losses.numel() != self.dataset_len:
            raise ValueError(f"ema_losses length ({self.ema_losses.numel()}) "
                             f"must match dataset_len ({self.dataset_len})")

    def _calculate_probabilities(self) -> torch.Tensor:
        # 精简概率计算
        current_ema_losses = self.ema_losses.cpu().float() # Work with a float copy for probability calculation
        
        # p_i \propto (L̄_i + ε)^α
        weights = (current_ema_losses + self.epsilon) ** self.alpha
        logger.debug(f"[Sampler] Weights before boost: sum={weights.sum().item():.4f}, min={weights.min().item():.4f}, max={weights.max().item():.4f}, non_zero={(weights > 0).sum().item()}/{weights.numel()}")

        if self.neighbor_boost is not None:
            current_neighbor_boost_for_calc = self.neighbor_boost.cpu().float()
            logger.debug(f"[Sampler] Neighbor_boost (id: {id(self.neighbor_boost)}, device: {self.neighbor_boost.device}): using .cpu().float() for calculation. Sum={current_neighbor_boost_for_calc.sum().item():.4f}, non_zero={(current_neighbor_boost_for_calc > 0).sum().item()}/{current_neighbor_boost_for_calc.numel()}, max={current_neighbor_boost_for_calc.max().item():.4f}")
            
            weights_after_boost = weights * (1 + self.lambda_boost * current_neighbor_boost_for_calc)
            logger.debug(f"[Sampler] Weights after boost (lambda={self.lambda_boost}): sum={weights_after_boost.sum().item():.4f}, min={weights_after_boost.min().item():.4f}, max={weights_after_boost.max().item():.4f}, non_zero={(weights_after_boost > 0).sum().item()}/{weights_after_boost.numel()}")
            
            # Check if boosting changed weights significantly
            if not torch.allclose(weights, weights_after_boost) and weights.sum() > 0:
                change_ratio = (weights_after_boost.sum() - weights.sum()) / weights.sum()
                logger.debug(f"[Sampler] Relative change in sum of weights due to boost: {change_ratio.item():.4%}")
            else:
                logger.debug("[Sampler] Neighbor boost did not significantly change weights or initial weights sum to zero.")
            weights = weights_after_boost # Apply the boost
        
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
        
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def update_sampler_config(self, ema_losses: torch.Tensor):
        """Allows external updates to the EMA losses if needed, though direct tensor update is preferred."""
        if ema_losses.numel() != self.dataset_len:
            raise ValueError("New EMA losses tensor length does not match dataset length.")
        self.ema_losses = ema_losses

    def set_neighbor_boost(self, boost_tensor):
        if boost_tensor.numel() != self.dataset_len:
            raise ValueError("neighbor_boost 长度不符")
        logger.warning("HardnessAwareSampler.set_neighbor_boost was called. This is unexpected with the current design of shared tensor reference.")