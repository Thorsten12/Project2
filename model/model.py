import torch.nn as nn
import torch
from .vit import VIT
from .classifierHead import ClassifierHead

class FullModel(nn.Module):
    def __init__(self, H, C, P, D, num_heads, mlp_mult, depth):
        super().__init__()
        self.vit            = VIT(H=H, C=C, P=P, D=D, num_heads=num_heads, mlp_mult=mlp_mult, depth=depth)
        self.classifierHead = ClassifierHead(D=D, num_classes=10)
    
    def forward(self, x):
        x = self.vit(x)
        return self.classifierHead(x)
