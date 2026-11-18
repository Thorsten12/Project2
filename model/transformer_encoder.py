"""
This is an Implementation of a simple Transformer Encoder.
"""

import torch.nn as nn
import torch

from .layernorm import Layernorm

class MSA(nn.Module):
    def __init__(self, num_heads, D):
        super().__init__()
        assert D % num_heads == 0

        self.num_heads  = num_heads
        self.D          = D
        self.head_dim   = D // num_heads

        self.Q          = nn.Linear(D, D) 
        self.V          = nn.Linear(D, D)
        self.K          = nn.Linear(D, D)

        self.proj       = nn.Linear(D, D)

    def forward(self, x):
        B, N, D         = x.shape

        q               = self.Q(x)
        v               = self.V(x)
        k               = self.K(x)

        q               = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k               = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v               = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attention       = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention       = attention.softmax(dim=-1)

        out             = attention @ v

        out             = out.transpose(1, 2).reshape(B, N, D)

        out             = self.proj(out)

        return out

class MLP(nn.Module):
    def __init__(self, D, mult=2):
        super().__init__()
        self.D = D
        self.mult = mult
        self.l_1 = nn.Linear(D, D * mult)
        self.act = nn.GELU()
        self.l_2 = nn.Linear(D * mult, D)

    def forward(self, x):
        x       = self.l_1(x)
        x       = self.act(x)
        x       = self.l_2(x)

        return x


class Transformer(nn.Module):
    def __init__(self, D=192, num_heads=3, mlp_mult=2):
        super().__init__()

        self.norm_1 = Layernorm(D=D)
        self.msa = MSA(num_heads=num_heads, D=D)

        self.norm_2 = Layernorm(D=D)
        self.mlp = MLP(D=D, mult=mlp_mult)
        
    def forward(self, x):
        x_pre_norm_1    = x
        x               = self.norm_1(x)
        x               = self.msa(x)
        x               = x + x_pre_norm_1

        x_pre_norm_2    = x
        x               = self.norm_2(x)
        x               = self.mlp(x)
        x               = x + x_pre_norm_2

        return x


if __name__ == "__main__":
    x = torch.randn(128, 64, 192)

    model = Transformer(D=192, num_heads=3, mlp_mult=2)

    out = model(x)

    print(out.shape)