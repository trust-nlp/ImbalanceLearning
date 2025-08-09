import torch
import torch.nn as nn
import torch.nn.functional as F

class WNet(nn.Module):
    """
    A two-layer MLP for Meta-Weight-Net.
    Input: loss values [batch_size, 1]
    Output: linear values [batch_size, 1] (for predicting normalized loss)
    """
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super(WNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # Initialize bias to a small value for more distinct sigmoid outputs
        nn.init.constant_(self.linear2.bias, 0.1)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        # ** Return raw logits, no premature sigmoid **
        return self.linear2(x)
    
    def params(self):
        """Return model parameters for the optimizer"""
        return self.parameters()