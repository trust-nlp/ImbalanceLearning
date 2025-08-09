# custom_sampler.py
import torch
from torch.utils.data import Sampler
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HardnessAwareSampler(Sampler):
    """
    Samples elements based on their meta_probs (from VNet), multiplied by neighbor_boost.
    meta_probs are updated by the Trainer, reflecting the probability of sampling each item.
    """
    def __init__(self,
                 dataset_len: int,
                 meta_probs: torch.Tensor,  # Shared tensor, updated by the Trainer
                 alpha: float = 0.5,
                 epsilon: float = 1e-6,
                 num_samples: Optional[int] = None,
                 neighbor_boost: Optional[torch.Tensor] = None,
                 lambda_boost: float = 0.0,
                 replacement: bool = False):
        self.dataset_len = dataset_len
        self.meta_probs = meta_probs # Direct reference to the meta_probs tensor in the Trainer
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_samples = num_samples if num_samples is not None else self.dataset_len
        self.lambda_boost = lambda_boost
        self.neighbor_boost = neighbor_boost  # Should be a direct reference to trainer's tensor (on GPU)
        self.replacement = replacement

        if self.neighbor_boost is not None:
            logger.info(f"HardnessAwareSampler initialized with neighbor_boost tensor (id: {id(self.neighbor_boost)}, device: {self.neighbor_boost.device}).")
        else:
            logger.info("HardnessAwareSampler initialized with neighbor_boost as None.")

        if not isinstance(self.meta_probs, torch.Tensor):
            raise TypeError(f"meta_probs must be a torch.Tensor, got {type(self.meta_probs)}")
        if self.meta_probs.numel() != self.dataset_len:
            raise ValueError(f"meta_probs length ({self.meta_probs.numel()}) must match dataset_len ({self.dataset_len})")

    def _calculate_probabilities(self) -> torch.Tensor:
        current_probs_from_trainer = self.meta_probs.cpu().float()
        logger.debug(f"[Sampler_CalcProbs] Input meta_probs_from_trainer (sum={current_probs_from_trainer.sum().item():.4f}, "
                     f"min={current_probs_from_trainer.min().item():.4f}, max={current_probs_from_trainer.max().item():.4f}, "
                     f"num_zeros={(current_probs_from_trainer == 0).sum().item()})")

        weights = (current_probs_from_trainer + self.epsilon) ** self.alpha
        logger.debug(f"[Sampler_CalcProbs] Weights after alpha (sum={weights.sum().item():.4f}, "
                     f"min={weights.min().item():.4f}, max={weights.max().item():.4f}, "
                     f"non_zero={(weights > 0).sum().item()}/{weights.numel()})")

        if self.neighbor_boost is not None:
            current_neighbor_boost_for_calc = self.neighbor_boost.cpu().float()
            logger.debug(f"[Sampler] Neighbor_boost (id: {id(self.neighbor_boost)}, device: {self.neighbor_boost.device}): using .cpu().float() for calculation. Sum={current_neighbor_boost_for_calc.sum().item():.4f}, non_zero={(current_neighbor_boost_for_calc > 0).sum().item()}/{current_neighbor_boost_for_calc.numel()}, max={current_neighbor_boost_for_calc.max().item():.4f}")

            weights_after_boost = weights * (1 + self.lambda_boost * current_neighbor_boost_for_calc)
            logger.debug(f"[Sampler] Weights after boost (lambda={self.lambda_boost}): sum={weights_after_boost.sum().item():.4f}, min={weights_after_boost.min().item():.4f}, max={weights_after_boost.max().item():.4f}, non_zero={(weights_after_boost > 0).sum().item()}/{weights_after_boost.numel()}")

            if not torch.allclose(weights, weights_after_boost) and weights.sum() > 0:
                change_ratio = (weights_after_boost.sum() - weights.sum()) / weights.sum()
                logger.debug(f"[Sampler] Relative change in sum of weights due to boost: {change_ratio.item():.4%}")
            else:
                logger.debug("[Sampler] Neighbor boost did not significantly change weights or initial weights sum to zero.")
            weights = weights_after_boost

        sum_weights = weights.sum()
        if sum_weights <= 1e-9:
            logger.warning(f"[Sampler_CalcProbs_WARN] Sum of weights is very small or zero ({sum_weights.item()}). "
                           f"This will lead to uniform sampling or errors. "
                           f"Input meta_probs sum: {current_probs_from_trainer.sum().item()}. Alpha: {self.alpha}")
            probs = torch.ones_like(weights) / weights.numel()
        else:
            probs = weights / sum_weights
            probs = probs.clamp(min=1e-12)  # Avoid negative/zero probabilities for multinomial

        logger.debug(f"[Sampler_CalcProbs] Final sampling probs (sum={probs.sum().item():.4f}, "
                     f"min={probs.min().item():.4f}, max={probs.max().item():.4f}, "
                     f"num_zeros={(probs == 0).sum().item()})")
        return probs

    def __iter__(self):
        probabilities = self._calculate_probabilities()

        if torch.any(probabilities < 0):
            logger.error("Negative probabilities found before multinomial sampling. This should not happen.")
            probabilities = torch.ones(self.dataset_len, dtype=torch.float) / self.dataset_len

        if torch.all(probabilities == 0):
            logger.warning("All probabilities are zero. Falling back to uniform sampling.")
            probabilities = torch.ones(self.dataset_len, dtype=torch.float) / self.dataset_len

        indices = torch.multinomial(probabilities, self.num_samples, replacement=self.replacement)
        logger.info(f"unique_in_batch={indices.unique().numel()}/{self.num_samples}")

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def update_sampler_config(self, meta_probs: torch.Tensor):
        """Allows external updates to the meta_probs if needed."""
        if meta_probs.numel() != self.dataset_len:
            raise ValueError("New meta_probs tensor length does not match dataset length.")
        self.meta_probs = meta_probs
        sample_idxs = list(range(min(5, self.dataset_len)))
        sample_vals = {idx: self.meta_probs[idx].item() for idx in sample_idxs}
        logger.debug(f"[meta_probs-Update] sample idx→prob {sample_vals}")

    def set_neighbor_boost(self, boost_tensor):
        if boost_tensor.numel() != self.dataset_len:
            raise ValueError("neighbor_boost length mismatch.")
        logger.warning("HardnessAwareSampler.set_neighbor_boost was called. This is unexpected with the current design of shared tensor reference.") 