import torch.nn as nn
import torch
from .vit import VIT
from .classifierHead import ClassifierHead

class FullModel(nn.Module):
    def __init__(self, H, C, P, D, num_heads, mlp_mult, classifier_dim):
        super().__init__()
        self.vit             = VIT(H=H, C=C, P=P, D=D, num_heads=num_heads, mlp_mult=mlp_mult)
        self.classifierHead  = ClassifierHead(D=classifier_dim, num_classes=10)
    
    def forward(self, x):
        x       = self.vit(x)
        logits  = self.classifierHead(x)
        probs   = torch.softmax(logits, dim=-1)
        return probs
