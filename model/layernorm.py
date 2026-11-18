import torch
import torch.nn as nn

class Layernorm(nn.Module):
    def __init__(self, eps=1e-05, D=None):
        super().__init__()
        self.eps    = eps
        self.gamma  = nn.Parameter(torch.ones(D))
        self.beta   = nn.Parameter(torch.zeros(D))

    def forward(self, x):

        mean        = x.mean(dim=-1, keepdim=True)
        var         = x.var(dim=-1, keepdim=True, unbiased=False)

        x           = (x - mean) / torch.sqrt(var + self.eps)

        return self.gamma * x + self.beta