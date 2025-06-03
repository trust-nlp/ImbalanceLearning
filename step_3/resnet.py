import torch
import torch.nn as nn
import torch.nn.functional as F

class VNet(nn.Module):
    """
    用于Meta-Weight-Net的两层MLP网络
    输入：损失值 [batch_size, 1]
    输出：线性值 [batch_size, 1] (直接输出,用于预测归一化的损失)
    """
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # 将 bias 设小一些，让 sigmoid 之后差异更明显
        nn.init.constant_(self.linear2.bias, 0.1)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        # **返回 raw logits，不再提前 sigmoid**
        return self.linear2(x)
    
    def params(self):
        """返回模型参数，用于optimizer"""
        return self.parameters() 