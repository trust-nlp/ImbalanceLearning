# custom_trainer.py
import torch
import torch.nn as nn
from torch.func import functional_call
from torch.utils.data import DataLoader
from collections import OrderedDict
# from torch.nn.attention import sdpa_kernel
# from torch.nn.attention import SDPBackend, sdp_kernel
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Optional, Dict, Union, Any, List, Tuple
import logging
from custom_sampler import HardnessAwareSampler
import faiss
from collections import defaultdict
import copy
import random
from resnet import VNet
from step3_run_ner import CustomDataCollatorForTokenClassification  # 新增导入

logger = logging.getLogger(__name__)

# 新的回调类
class KnnEpochEndCallback(TrainerCallback):
    def __init__(self, trainer_instance: 'HardnessAwareTrainer'):
        super().__init__()
        self.trainer = trainer_instance

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.trainer.hardness_aware_sampling_enabled:
            return
        # 暂时禁用 KNN 功能
        return

class HardnessAwareTrainer(Trainer):
    def __init__(self, *args, 
                 meta_dataset=None,
                 train_dataset_len: Optional[int] = None,
                 hardness_aware_sampling: bool = False,
                 hardness_alpha: float = 1.0,
                 knn_k: int = 5,
                 knn_lambda: float = 0.5,
                 knn_build_freq: int = 1,
                 vnet_lr: float = 1e-5,
                 meta_update_lr: float = 1e-5,
                 meta_update_scale_factor: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"[MetaInit] meta_dataset_len = {len(meta_dataset) if meta_dataset is not None else 0}")

        # ----- meta-validation loader -----
        self.meta_dataset = meta_dataset
        if self.meta_dataset is not None:
            self.meta_dataloader = DataLoader(
                self.meta_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
            )
            self._meta_iter = iter(self.meta_dataloader)
        else:
            self.meta_dataloader = None
            self._meta_iter = None

        # 学习率用于一次"虚拟"内步
        self.inner_lr = meta_update_lr

        self.hardness_aware_sampling_enabled = hardness_aware_sampling
        self.hardness_alpha = hardness_alpha
        self.knn_k = knn_k
        self.knn_lambda = knn_lambda
        self.knn_build_freq = knn_build_freq
        self._entity_embeds: List[Tuple[int, torch.Tensor]] = []

        self.train_dataset_len = train_dataset_len
        self.vnet_lr = vnet_lr
        self.meta_update_lr = meta_update_lr
        self.meta_update_scale_factor = meta_update_scale_factor

        logger.info(f"HardnessAwareTrainer initialized. Hardness Sampling Enabled: {self.hardness_aware_sampling_enabled}")
        if self.hardness_aware_sampling_enabled:
            logger.info(f"  H.Alpha: {self.hardness_alpha}, VNet LR: {self.vnet_lr}, MetaUpdateScale: {self.meta_update_scale_factor}")
            if self.train_dataset_len is None or self.train_dataset_len == 0:
                raise ValueError("HardnessAwareTrainer: train_dataset_len 必须大于 0")
            
            device_for_tensors = self.args.device 
            self._neighbor_boost = torch.zeros(self.train_dataset_len, dtype=torch.float32, device=device_for_tensors)
            self.meta_probs = torch.full(
                (self.train_dataset_len,),
                1.0 / self.train_dataset_len,
                dtype=torch.float32,
                device=device_for_tensors
            )
            self.vnet = VNet(1, 100, 1).to(device_for_tensors)
            self.vopt = torch.optim.Adam(self.vnet.parameters(), lr=self.vnet_lr, weight_decay=1e-4)

            # 用于移动统计的成员变量 (EMA 方差)
            self.moving_mean = torch.tensor(0.0, device=device_for_tensors)
            self.moving_variance = torch.tensor(1.0, device=device_for_tensors) # 方差初始为1 (std为1)
            self.moving_variance = self.moving_variance.clamp(min=1e-6)  # 确保方差下限
            self.moving_avg_decay = 0.99 
            logger.info(f"  Normalization: Initial moving_mean=0.0, moving_variance=1.0, decay={self.moving_avg_decay}")

    def _get_train_sampler(self, train_dataset: Optional[Any] = None) -> Optional[torch.utils.data.Sampler]:
        if not self.hardness_aware_sampling_enabled:
            logger.info("[SamplerSetup] Hardness sampling disabled, using super()._get_train_sampler().")
            return super()._get_train_sampler()

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        logger.info("Creating HardnessAwareSampler for training.")
        
        passed_boost_tensor = self._neighbor_boost
        logger.info(f"[SamplerSetup] Initializing HardnessAwareSampler with: "
                    f"dataset_len={self.train_dataset_len}, "
                    f"alpha={self.hardness_alpha}, "
                    f"lambda_boost (knn_lambda)={self.knn_lambda}, "
                    f"meta_probs_sum_on_init={self.meta_probs.sum().item():.4f}, "
                    f"meta_probs_device={self.meta_probs.device}, "
                    f"neighbor_boost_is_not_none={passed_boost_tensor is not None}, "
                    f"neighbor_boost_device={passed_boost_tensor.device if passed_boost_tensor is not None else 'N/A'}")
        
        if passed_boost_tensor is not None:
            logger.info(f"  Trainer._get_train_sampler passing _neighbor_boost id={id(passed_boost_tensor)}, device={passed_boost_tensor.device}")

        return HardnessAwareSampler(
            dataset_len=self.train_dataset_len,
            meta_probs=self.meta_probs,
            alpha=self.hardness_alpha,
            epsilon=1e-6,
            neighbor_boost=passed_boost_tensor,
            lambda_boost=self.knn_lambda,
            num_samples=None
        )

    def _get_per_sentence_loss(self, outputs: Union[Dict, Tuple], labels: torch.Tensor, model_config: Any) -> torch.Tensor:
        """
        Calculates per-sentence loss.
        A simple version: average loss of actual tokens in the sentence.
        More sophisticated: max loss among entities, or other heuristics.
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
        per_sentence_loss_avg = sentence_loss / num_actual_tokens.clamp(min=1)
        
        return per_sentence_loss_avg

    def _update_loss_stats(self, current_batch_loss: torch.Tensor):
        # current_batch_loss 是 per_sentence_loss.detach()，在 self.args.device 上
        if current_batch_loss.numel() == 0:
            return

        batch_mean = current_batch_loss.mean()
        batch_variance = current_batch_loss.var(unbiased=False) # 使用有偏方差 (n) 或无偏方差 (n-1)

        # EMA 更新均值
        if not torch.isnan(batch_mean) and not torch.isinf(batch_mean):
            self.moving_mean = self.moving_avg_decay * self.moving_mean + \
                               (1 - self.moving_avg_decay) * batch_mean
        
        # EMA 更新方差
        if not torch.isnan(batch_variance) and not torch.isinf(batch_variance) and batch_variance >= 0: # 方差不能为负
            self.moving_variance = self.moving_avg_decay * self.moving_variance + \
                                   (1 - self.moving_avg_decay) * batch_variance
            self.moving_variance = self.moving_variance.clamp(min=1e-6) # 确保方差非负且有个下限

    def _normalize_loss(self, loss_tensor: torch.Tensor) -> torch.Tensor:
        loss_tensor_device = loss_tensor.to(self.args.device) # 确保在正确设备
        self._update_loss_stats(loss_tensor_device) 

        if loss_tensor_device.numel() == 0:
            return torch.tensor([], device=loss_tensor_device.device, dtype=loss_tensor_device.dtype)

        current_std = torch.sqrt(self.moving_variance) # 从方差计算标准差
        
        if current_std > 1e-6: # 确保 current_std 有效
            normalized_loss = (loss_tensor_device - self.moving_mean) / current_std
        else: 
            normalized_loss = loss_tensor_device - self.moving_mean # 仅中心化
        
        return normalized_loss

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
        1. 计算当前 batch 的句子级 loss → z-score → VNet 输出 w∈(0,1)
        2. 用 w * loss 训练主模型
        3. 用一次模拟梯度步 + meta-batch 上的 meta_loss 更新 VNet
        4. 用 w 更新 self.meta_probs 供 Sampler
        """
        logger.debug(f"[TrainStep-Start] Got inputs keys = {list(inputs.keys())}")
        model.train()
        logger.debug(f"[TrainStep-Start] Device = {self.args.device}, labels_present = {'labels' in inputs}")
        inputs = self._prepare_inputs(inputs)
        original_indices = inputs.pop("orig_idx", None)
        labels = inputs.pop("labels", None)
        if labels is None:
            raise ValueError("Labels missing")

        # ---------- 主模型正向 ----------
        outputs = model(**inputs, labels=labels)
        per_sentence_loss = self._get_per_sentence_loss(outputs, labels, model.config)   # [B]
        logger.debug(f"[Loss-PerSentence] min={per_sentence_loss.min().item():.6f}, max={per_sentence_loss.max().item():.6f}, mean={per_sentence_loss.mean().item():.6f}")

        if self.hardness_aware_sampling_enabled:
            # ---------- VNet 前向 ----------
            z = self._normalize_loss(per_sentence_loss.detach()).unsqueeze(-1)  # 均值+方差标准化
            logger.debug(f"[VNet-Input z] min={z.min().item():.6f}, max={z.max().item():.6f}, mean={z.mean().item():.6f}")

            w_logit = self.vnet(z).squeeze(-1)
            logger.debug(f"[VNet-w_logit] min={w_logit.min().item():.6f}, max={w_logit.max().item():.6f}, mean={w_logit.mean().item():.6f}")
            w = torch.sigmoid(w_logit) + 1e-6               # ← 只有一次 sigmoid
            w = w / (w.mean().detach() + 1e-6)              # 均值归一化到 1
            w = w.clamp(min=0.1, max=10.0)                  # 允许更大动态范围
            logger.debug(f"[VNet-w(after norm)] min={w.min().item():.6f}, max={w.max().item():.6f}, mean={w.mean().item():.6f}")

            # ---------- 内部 (meta) 梯度步骤 ----------
            inner_loss = (w * per_sentence_loss).mean()      # *带梯度*，供二阶

            if self.meta_dataloader is not None:
                # 1 . 取一个 meta-batch
                try:
                    logger.debug(f"[MetaBatch-Pull] Fetching one meta_batch")
                    meta_batch = next(self._meta_iter)
                except StopIteration:
                    self._meta_iter = iter(self.meta_dataloader)
                    logger.debug(f"[MetaBatch-Pull] Iterator 重置后再次 Fetch")
                    meta_batch = next(self._meta_iter)

                # 1.1 准备 meta inputs 并单独提取 labels
                meta_inputs = self._prepare_inputs(meta_batch)
                logger.debug(f"[MetaBatch-Inputs] keys = {list(meta_inputs.keys())}")
                meta_labels = meta_inputs.pop("labels")

                # 2 . 计算 current batch 对模型参数的梯度，并虚拟更新一次
                logger.debug(f"[MetaStep] About to functional_call with inner_lr = {self.inner_lr}")
                grads = torch.autograd.grad(
                    inner_loss,
                    tuple(model.parameters()),
                    create_graph=True,        # 允许二阶梯度
                )
                # 2.1 从 model.named_parameters() 构造更新后的参数字典
                updated_param_dict = OrderedDict()
                for (name, param), grad in zip(model.named_parameters(), grads):
                    updated_param_dict[name] = param - self.inner_lr * grad

                # 2.2 把所有 buffer 加入字典（保持原值不变）
                for name, buf in model.named_buffers():
                    updated_param_dict[name] = buf

                # 2.3 把 labels 放回去，确保 functional_call 时会计算 loss
                meta_inputs["labels"] = meta_labels

                # 3 . 用更新后的参数在 meta-batch 上前向
                # functional_call 的 args=()，kwargs=meta_inputs
                meta_outputs = functional_call(model, updated_param_dict, (), meta_inputs, strict=False)
                meta_loss = meta_outputs.loss
                logger.debug(f"[MetaLoss] value = {meta_loss.item():.6f}")

                # 4 . 更新 VNet
                self.vopt.zero_grad()
                meta_loss.backward()
                self.vopt.step()

            # ---------- 更新采样概率 (累加更新) ----------
            if self.hardness_aware_sampling_enabled and original_indices is not None:
                # 用 scatter_add_ 进行累加，而不是覆盖
                gamma = 0.9                                           # 指数衰减
                self.meta_probs.mul_(gamma)                          # 先整体衰减
                self.meta_probs.scatter_add_(
                    0,
                    original_indices.to(self.meta_probs.device),
                    w.detach().to(self.meta_probs.device)
                )
                sample_idxs = original_indices[:5].tolist() if original_indices is not None else []
                sample_vals = {idx: self.meta_probs[idx].item() for idx in sample_idxs}
                logger.debug(f"[meta_probs-Update] sample idx→prob {sample_vals}")
                # 防止权重无限膨胀，加上上限后归一化
                self.meta_probs.clamp_(min=1e-8, max=10.0)
                self.meta_probs.div_(self.meta_probs.sum())
        else:
            # 标准训练，不使用hardness aware sampling
            loss_main = outputs.loss

        # ---------- (重新 forward) 计算主模型梯度 ----------
        self.optimizer.zero_grad(set_to_none=True)
        loss_main = (w.detach() * per_sentence_loss).mean()  # 直接用第一次前向的 loss

        # Backprop 仅沿第二次 forward 的图
        self.accelerator.backward(loss_main)

        return loss_main.detach() / self.args.gradient_accumulation_steps 