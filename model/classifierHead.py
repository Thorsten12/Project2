
from .layernorm import Layernorm
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, D, num_classes=10):
        super().__init__()
        self.D              = D
        self.num_classes    = 10
        self.norm           = Layernorm(D=D)
        self.linear         = nn.Linear(D, num_classes)

    def forward(self, x):
        x = x[:, 0, :]      # CLS Token
        x = self.norm(x)
        x = self.linear(x)
        return x
