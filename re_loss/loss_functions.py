import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    改进的 Dice 损失函数实现，专门处理 NER 任务中的 -100 标签，并支持类别权重
    """
    def __init__(self, smooth=1e-5, alpha=0.5, square_denominator=True, class_weights=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha  # LogDice = alpha * Dice + (1-alpha) * CE
        self.square_denominator = square_denominator
        self.class_weights = class_weights
        
    def forward(self, logits, labels, mask=None):
        """
        Args:
            logits: [batch_size, seq_len, num_labels] 或 [batch_size, num_labels] 预测 logits
            labels: [batch_size, seq_len] 或 [batch_size] 真实标签
            mask: [batch_size, seq_len] 或 [batch_size] 有效 token 的 mask
        """
        # --- 单句级多类分类 (Relation Extraction) ------------------------------
        if logits.dim() == 2:                       # [B, C]
            probs = F.softmax(logits, dim=-1)       # [B, C]
            num_labels = probs.size(1)
            one_hot = F.one_hot(labels, num_classes=num_labels).float()

            # 可选类别权重
            if self.class_weights is not None:
                w = self.class_weights.to(logits.device)
                probs = probs * w
                one_hot = one_hot * w

            inter = (probs * one_hot).sum(dim=0) * 2.0 + self.smooth
            if self.square_denominator:
                denom = (probs.pow(2) + one_hot.pow(2)).sum(dim=0) + self.smooth
            else:
                denom = (probs + one_hot).sum(dim=0) + self.smooth

            dice_loss = 1.0 - (inter / denom).mean()

            # Log-Dice：alpha∈[0,1)，<1 时再加一点 CE 稳定训练
            if self.alpha < 1.0:
                ce_loss = F.cross_entropy(logits, labels)
                return self.alpha * dice_loss + (1 - self.alpha) * ce_loss
            return dice_loss
        
        if logits.dim() != 3:
            raise ValueError(f"Expected logits 2D/3D, got {logits.shape}")
        
        _, _, num_labels = logits.size()
        
        # 创建标签 mask，排除 -100 的标签
        label_mask = (labels != -100)
        
        # 将 logits 转换为概率
        log_probs = F.log_softmax(logits, dim=-1)  # 使用log_softmax提高数值稳定性
        probs = torch.exp(log_probs)
        
        # 处理有效的标签（非-100）
        valid_labels = labels.clone()
        valid_labels[~label_mask] = 0  # 将 -100 替换为 0，便于创建 one-hot
        
        # 将标签转换为 one-hot 编码
        one_hot_labels = F.one_hot(valid_labels, num_classes=num_labels).float()
        
        # 应用 label_mask
        if mask is not None:
            label_mask = label_mask & mask.bool()
        
        # 扩展 label_mask 到最后一个维度
        expanded_label_mask = label_mask.unsqueeze(-1).expand(-1, -1, num_labels)
        
        # 只计算有效位置的 Dice loss
        masked_probs = probs * expanded_label_mask.float()
        masked_labels = one_hot_labels * expanded_label_mask.float()
        
        # 应用类别权重（如果有）
        if self.class_weights is not None:
            class_weights = self.class_weights.to(logits.device)
            weight_mask = class_weights.view(1, 1, -1).expand_as(masked_probs)
            masked_probs = masked_probs * weight_mask
            masked_labels = masked_labels * weight_mask
        
        # 计算 Dice 系数
        if self.square_denominator:
            numerator = 2.0 * (masked_probs * masked_labels).sum(dim=(1, 2)) + self.smooth
            denominator = (masked_probs * masked_probs).sum(dim=(1, 2)) + (masked_labels * masked_labels).sum(dim=(1, 2)) + self.smooth
        else:
            numerator = 2.0 * (masked_probs * masked_labels).sum(dim=(1, 2)) + self.smooth
            denominator = masked_probs.sum(dim=(1, 2)) + masked_labels.sum(dim=(1, 2)) + self.smooth
        
        # 计算 Dice 损失
        dice_loss = 1.0 - (numerator / denominator).mean()
        
        # 可选：结合交叉熵损失（Log-Dice）
        if self.alpha < 1.0:
            # 计算交叉熵损失（只在有效位置）
            ce_loss = -torch.sum(masked_labels * log_probs, dim=-1)  # [batch_size, seq_len]
            ce_loss = ce_loss * label_mask.float()
            ce_loss = ce_loss.sum() / (label_mask.sum() + 1e-12)
            
            # 组合 Dice 和交叉熵
            return self.alpha * dice_loss + (1 - self.alpha) * ce_loss
            
        return dice_loss


class FocalLoss(nn.Module):
    """
    改进的Focal损失函数，支持类别权重和-100标签处理
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # 类别权重，可以是None或者张量
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ce_loss_fct = nn.CrossEntropyLoss(weight=alpha, reduction='none', ignore_index=ignore_index)
        
    def forward(self, logits, labels, mask=None):
        """
        Args:
            logits: [batch_size, seq_len, num_labels] 或 [batch_size, num_labels] 预测logits
            labels: [batch_size, seq_len] 或 [batch_size] 真实标签
            mask: [batch_size, seq_len] 或 [batch_size] 有效token的mask
        """
        # 允许序列级分类任务 (logits: [B, C])
        if logits.dim() == 2:            # [B, C] → [B, 1, C]
            logits = logits.unsqueeze(1)
            labels = labels.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)
        if logits.dim() != 3:
            raise ValueError(f"Expected logits 2D/3D, got {logits.shape}")
        
        _, _, num_labels = logits.size()
        
        # 重塑logits和labels以适应CrossEntropyLoss
        logits_flat = logits.view(-1, num_labels)  # [batch_size*seq_len, num_labels]
        labels_flat = labels.view(-1)  # [batch_size*seq_len]
        
        # 计算常规交叉熵损失
        ce_loss = self.ce_loss_fct(logits_flat, labels_flat)  # [batch_size*seq_len]
        
        # 计算预测概率
        probs = F.softmax(logits_flat, dim=-1)  # [batch_size*seq_len, num_labels]
        
        # 获取每个位置对应标签的预测概率
        valid_indices = (labels_flat != self.ignore_index)
        valid_labels = labels_flat[valid_indices]
        valid_probs = probs[valid_indices]
        
        # 获取对应真实标签的概率
        valid_label_indices = torch.arange(valid_labels.size(0)).to(labels.device)
        pt = valid_probs[valid_label_indices, valid_labels]  # [num_valid]
        
        # 重新整形focal权重
        focal_weight = (1 - pt) ** self.gamma  # [num_valid]
        
        # 应用focal权重到交叉熵损失
        focal_ce = focal_weight * ce_loss[valid_indices]  # [num_valid]
        
        # 应用mask（如果提供）
        if mask is not None:
            mask_flat = mask.view(-1)[valid_indices].float()  # [num_valid]
            focal_ce = focal_ce * mask_flat
            if self.reduction == 'mean':
                return focal_ce.sum() / (mask_flat.sum() + 1e-12)
            elif self.reduction == 'sum':
                return focal_ce.sum()
        
        # 返回损失
        if self.reduction == 'mean':
            return focal_ce.mean()
        elif self.reduction == 'sum':
            return focal_ce.sum()
        return focal_ce


