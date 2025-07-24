# custom_trainer.py
# ===== 再保险：本文件单独运行时同样禁用 torch.compile =====
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.compile = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda x: x))

torch._functorch.config.donated_buffer = False
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader
from collections import OrderedDict
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Optional, Dict, Union, Any, List, Tuple
import logging
from .custom_sampler import HardnessAwareSampler
import faiss
from collections import defaultdict
import copy
import random
from .vnet import VNet

logger = logging.getLogger(__name__)

# 添加自定义损失函数类
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        intersection = torch.sum(probs * targets_one_hot, dim=1)
        cardinality = torch.sum(probs + targets_one_hot, dim=1)
        dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6)
        return (1. - dice_score).mean()

class KnnEpochEndCallback(TrainerCallback):
    """Recompute neighbor_boost every `knn_build_freq` epochs."""

    def __init__(self, trainer_instance: "HardnessAwareTrainer"):
        super().__init__()
        self.trainer = trainer_instance

    def on_epoch_end(self, args, state, control, **kwargs):
        tr = self.trainer
        # 【新增日志】确认回调函数被触发
        logger.info(f"[KNN Callback] on_epoch_end triggered at epoch {int(state.epoch)}.")

        if (
            not tr.hardness_aware_sampling_enabled
            or tr.knn_lambda == 0
            or tr.faiss_index is None
            or state.epoch is None
            or (int(state.epoch) % tr.knn_build_freq) != 0
        ):
            # 【新增日志】告知为何跳过更新
            logger.info(f"[KNN Callback] Skipping neighbor_boost update. "
                        f"Enabled={tr.hardness_aware_sampling_enabled}, "
                        f"lambda={tr.knn_lambda}, "
                        f"epoch={int(state.epoch)} % freq={tr.knn_build_freq} != 0 is { (int(state.epoch) % tr.knn_build_freq) != 0 }")
            return

        logger.info(f"[KNN Callback] Rebuilding neighbor_boost at epoch {int(state.epoch)}...")
        cpu_vecs = tr._emb_matrix.detach().cpu().numpy()
        sim, idx = tr.faiss_index.search(cpu_vecs, tr.knn_k + 1)   # 包含自身

        boost = torch.zeros_like(tr._neighbor_boost)
        for row in idx:
            neighbors = row[1:]           # 去掉自身
            boost[neighbors] += 1

        if boost.max() > 0:
            boost = boost / boost.max()
        
        tr._neighbor_boost.copy_(boost.to(tr.args.device))
        logger.info(f"[KNN Callback] neighbor_boost tensor updated. New max value: {tr._neighbor_boost.max().item():.4f}")

class HardnessAwareTrainer(Trainer):
    def __init__(self, *args, 
                 meta_dataset=None,
                 train_dataset_len: Optional[int] = None,
                 hardness_aware_sampling: bool = False,
                 hardness_alpha: float = 1.0,
                 knn_k: int = 5,
                 knn_lambda: float = 0.5,
                 knn_build_freq: int = 1,
                 vnet_lr: float = 1e-4,
                 meta_update_lr: float = 2e-4,
                 meta_update_scale_factor: float = 2.0,
                 precomputed_embeddings=None,
                 loss_name: str = "ce", # 新增 loss_name 参数
                 **kwargs):
        super().__init__(*args, **kwargs)
        # **确保 Trainer 不调用 torch.compile**
        self.args.torch_compile = False

        logger.info("[DEBUG] HardnessAwareTrainer.__init__: After super().__init__")
        self.hardness_aware_sampling_enabled = hardness_aware_sampling
        logger.info(f"[DEBUG] HardnessAwareTrainer.__init__: hardness_aware_sampling = {hardness_aware_sampling}")
        logger.info(f"[MetaInit] Hardness Sampling Enabled: {self.hardness_aware_sampling_enabled}")
        
        # ----- 合并 SequenceTrainer 的功能：初始化损失函数 -----
        self.loss_name = loss_name
        if self.loss_name == "focal":
            self.criterion = FocalLoss(gamma=2.0)
            logger.info("[Loss] Using FocalLoss")
        elif self.loss_name == "dice":
            self.criterion = DiceLoss(alpha=1.0)
            logger.info("[Loss] Using DiceLoss")
        else:  # 默认交叉熵
            self.criterion = nn.CrossEntropyLoss()
            logger.info("[Loss] Using CrossEntropyLoss")

        # 只有在启用 hardness sampling 时才初始化相关组件
        if self.hardness_aware_sampling_enabled:
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

            self.hardness_alpha = hardness_alpha
            self.knn_k = knn_k
            self.knn_lambda = knn_lambda
            self.knn_build_freq = knn_build_freq
            self._entity_embeds: List[Tuple[int, torch.Tensor]] = []

            self.train_dataset_len = train_dataset_len
            self.vnet_lr = vnet_lr
            self.meta_update_lr = meta_update_lr
            self.meta_update_scale_factor = meta_update_scale_factor

            logger.info(f"  H.Alpha: {self.hardness_alpha}, VNet LR: {self.vnet_lr}, MetaUpdateScale: {self.meta_update_scale_factor}")
            if self.train_dataset_len is None or self.train_dataset_len == 0:
                raise ValueError("HardnessAwareTrainer: train_dataset_len 必须大于 0")
            
            device_for_tensors = self.args.device 
            self._neighbor_boost = torch.zeros(self.train_dataset_len, dtype=torch.float32, device=device_for_tensors)
            # 重命名为 _sample_scores，并初始化为 1。Sampler会处理归一化。
            self._sample_scores = torch.ones(
                self.train_dataset_len,
                dtype=torch.float32,
                device=device_for_tensors
            )
            self.vnet = VNet(1, 100, 1).to(device_for_tensors)
            self.vopt = torch.optim.Adam(self.vnet.parameters(), lr=self.vnet_lr, weight_decay=1e-4)

            # ---------- NEW: build FAISS index ----------
            if precomputed_embeddings is None:
                raise ValueError("precomputed_embeddings is required for hardness-aware sampling.")
            self._emb_matrix = precomputed_embeddings.to(device_for_tensors)           # [N, D]
            dim = self._emb_matrix.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            _cpu = self._emb_matrix.detach().cpu().numpy()
            faiss.normalize_L2(_cpu)
            self.faiss_index.add(_cpu)
            logger.info(f"[FAISS] Indexed {_cpu.shape[0]} vectors (dim={dim})")
        else:
            # 当不启用 hardness sampling 时，设置默认值避免属性错误
            self.meta_dataset = None
            self.meta_dataloader = None
            self._meta_iter = None
            self.faiss_index = None
            self.train_dataset_len = 0

    def _get_train_sampler(self, train_dataset: Optional[Any] = None) -> Optional[torch.utils.data.Sampler]:
        if not self.hardness_aware_sampling_enabled:
            logger.info("[SamplerSetup] Hardness sampling disabled, using default sequential/random sampler.")
            return super()._get_train_sampler()

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        logger.info("[SamplerSetup] Creating HardnessAwareSampler for training.")
        
        passed_boost_tensor = self._neighbor_boost
        logger.info(f"[SamplerSetup] Initializing HardnessAwareSampler with: "
                    f"dataset_len={self.train_dataset_len}, "
                    f"alpha={self.hardness_alpha}, "
                    f"lambda_boost (knn_lambda)={self.knn_lambda}, "
                    f"sample_scores_sum_on_init={self._sample_scores.sum().item():.4f}, "
                    f"sample_scores_device={self._sample_scores.device}, "
                    f"neighbor_boost_is_not_none={passed_boost_tensor is not None}, "
                    f"neighbor_boost_device={passed_boost_tensor.device if passed_boost_tensor is not None else 'N/A'}")
        
        if passed_boost_tensor is not None:
            logger.info(f"  Trainer._get_train_sampler passing _neighbor_boost id={id(passed_boost_tensor)}, device={passed_boost_tensor.device}")

        return HardnessAwareSampler(
            dataset_len=self.train_dataset_len,
            meta_probs=self._sample_scores,  # <--- 使用修正后的张量
            alpha=self.hardness_alpha,
            epsilon=1e-6,
            neighbor_boost=passed_boost_tensor,
            lambda_boost=self.knn_lambda,
            num_samples=None,
            replacement=True,
        )

    def _get_per_sentence_loss(
        self,
        outputs: Union[Dict, Tuple],
        labels: torch.Tensor,
        model_config: Any
    ) -> torch.Tensor:
        """
        根据 self.loss_name 计算每个样本的损失 (reduction='none')。
        返回 [B] 长度的句级损失张量。
        """
        logits = outputs.logits if isinstance(outputs, dict) else outputs[1]

        # --- 序列分类任务 ---
        if logits.dim() == 2:
            if self.loss_name == "focal":
                ce_loss = F.cross_entropy(logits, labels, reduction='none')
                pt = torch.exp(-ce_loss)
                # self.criterion is an instance of FocalLoss
                focal_loss_per_item = self.criterion.alpha * (1 - pt)**self.criterion.gamma * ce_loss
                return focal_loss_per_item
            
            elif self.loss_name == "dice":
                probs = F.softmax(logits, dim=1)
                targets_one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()
                intersection = torch.sum(probs * targets_one_hot, dim=1)
                cardinality = torch.sum(probs + targets_one_hot, dim=1)
                dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6)
                return 1. - dice_score
            
            elif self.loss_name == "ce":
                return F.cross_entropy(logits, labels, reduction="none")
            
            else:
                raise ValueError(f"Unsupported loss_name for training step: {self.loss_name}")

        # --- 序列标注任务 (如果需要，也应按上面方式扩展) ---
        else: # logits.dim() == 3
             bsz, seq_len, num_labels = logits.shape
             # 保持原有逻辑，但明确指出其使用的是CE loss
             loss_fct = nn.CrossEntropyLoss(reduction="none")
             per_tok_loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
             per_tok_loss = per_tok_loss.view(bsz, seq_len)
             mask = (labels != -100)
             # 取一条句子中 **最大** token-loss 作为该句难度
             return (per_tok_loss * mask).max(dim=1).values



    def compute_loss(self, model, inputs, return_outputs=False):
        """
        集成了 SequenceTrainer 的功能，使用自定义损失函数。
        主要用于评估/预测，training_step 会自己处理损失计算。
        """
        inputs.pop("orig_idx", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits            # [B, C]
        loss = self.criterion(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """
        统一的训练步骤：根据 hardness_aware_sampling_enabled 决定使用哪种逻辑。
        """
        model.train()
        
        # 如果不启用 hardness sampling，执行标准训练步骤
        if not self.hardness_aware_sampling_enabled:
            # 使用 super() 调用父类方法，更简洁且不易出错
            if self.state.global_step % self.args.logging_steps == 0:
                logger.info("[TrainStep] Using standard training step (no hardness sampling).")
            return super().training_step(model, inputs, num_items_in_batch)

        # ----- 以下是启用 hardness sampling 时的核心逻辑 -----
        if self.state.global_step % self.args.logging_steps == 0:
            logger.info(f"--- [META-STEP START] global_step={self.state.global_step}: Activating meta-learning update. ---")

        inputs = self._prepare_inputs(inputs)
        original_indices = inputs.pop("orig_idx", None)
        labels = inputs.pop("labels", None)
        if labels is None:
            raise ValueError("Labels are missing in training inputs for meta-learning step.")

        # ===== 1. 主模型前向，计算原始损失 =====
        outputs = model(**inputs, labels=labels)
        per_sentence_loss = self._get_per_sentence_loss(outputs, labels, model.config)
        loss_main = per_sentence_loss.mean() # 主损失是无加权的

        # ===== 2. 元学习：更新VNet和采样分数 =====
        # 准备VNet的输入：使用裁剪后的损失，更稳定
        vnet_input = per_sentence_loss.detach().clamp(max=10.0).unsqueeze(-1)
        
        # VNet前向，得到权重
        w_logit = self.vnet(vnet_input).squeeze(-1)
        w = torch.sigmoid(w_logit)
        
        # 模拟内部更新步骤
        # 注意：这里的inner_loss仅用于计算对VNet的梯度，不用于更新主模型
        inner_loss = (w * per_sentence_loss).mean()

        # 获取meta batch
        if self.meta_dataloader is None:
             raise ValueError("Meta-learning step requires a meta_dataset, but it is None.")
        try:
            meta_batch = next(self._meta_iter)
        except StopIteration:
            self._meta_iter = iter(self.meta_dataloader)
            meta_batch = next(self._meta_iter)
        meta_inputs = self._prepare_inputs(meta_batch)
        meta_inputs.pop("orig_idx", None)
        meta_labels = meta_inputs.pop("labels")

        # 计算虚拟更新后的模型参数
        grads = torch.autograd.grad(
            inner_loss,
            (p for p in model.parameters() if p.requires_grad),
            create_graph=True
        )
        updated_param_dict = OrderedDict()
        grad_iter = iter(grads)
        for name, param in model.named_parameters():
            if param.requires_grad:
                updated_param_dict[name] = param - self.inner_lr * next(grad_iter)
        for name, buffer in model.named_buffers():
            updated_param_dict[name] = buffer
        
        # 在meta batch上计算元损失
        meta_inputs["labels"] = meta_labels
        meta_outputs = functional_call(model, updated_param_dict, (), meta_inputs)
        meta_loss = meta_outputs.loss

        # 更新 VNet
        self.vopt.zero_grad()
        meta_loss.backward()
        self.vopt.step()

        # ===== 3. 更新全局采样分数 =====
        # 直接用VNet的输出更新分数，更直接
        if original_indices is not None:
            # 使用EMA平滑更新
            gamma = 0.9
            old_scores = self._sample_scores[original_indices]
            new_scores = w.detach()
            self._sample_scores[original_indices] = gamma * old_scores + (1 - gamma) * new_scores
            
            if self.state.global_step % 50 == 0:
                logger.info(f"[Scores] step={self.state.global_step} updated scores mean={self._sample_scores.mean():.3f} "
                            f"min={self._sample_scores.min():.3f} max={self._sample_scores.max():.3f}")

        # ===== 4. 主模型反向传播 =====
        # 使用第1步计算的无加权主损失
        if self.args.n_gpu > 1:
            loss_main = loss_main.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss_main = loss_main / self.args.gradient_accumulation_steps
        
        self.accelerator.backward(loss_main)

        return loss_main.detach() 