"""
This Model serves as a backbone for a simple Classifier. It is an Adaptation of the "https://arxiv.org/pdf/2010.11929" "An Image is worth 16x16 Words: Transformers for Image Recognition At Scale".
"""

import torch
import torch.nn as nn
import torch.functional as F

from .embeddings import PatchEmbedding, PositionalEmbedding
from .transformer_encoder import Transformer

class VIT(nn.Module):
    def __init__(self, H, C, P, D, num_heads, mlp_mult, depth):
        super().__init__()
        N = (H // P) ** 2
        self.patch_embed = PatchEmbedding(H=H, C=C, P=P, D=D)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = PositionalEmbedding(N=N + 1, D=D)   # N+1
        self.blocks = nn.ModuleList(
            [Transformer(D=D, num_heads=num_heads, mlp_mult=mlp_mult) for _ in range(depth)]
        )

    def forward(self, x):
        x = self.patch_embed(x)                                  # (B, N, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)                           # (B, N+1, D)
        x = self.pos_embed(x)
        for blk in self.blocks:
            x = blk(x)
        return x
