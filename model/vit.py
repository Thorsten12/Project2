"""
This Model serves as a backbone for a simple Classifier. It is an Adaptation of the "https://arxiv.org/pdf/2010.11929" "An Image is worth 16x16 Words: Transformers for Image Recognition At Scale".
"""

import torch.nn as nn
import torch.functional as F

from .embeddings import PatchEmbedding, PositionalEmbedding
from .transformer_encoder import Transformer

class VIT(nn.Module):
    def __init__(self, H, C, P, D):
        super().__init__()

        N              = (H // P) ** 2  # Number of Patches

        self.patch_embed    = PatchEmbedding(H=H, C=C, P=P, D=D)
        self.pos_embed      = PositionalEmbedding(N=N, D=D)

        self.transformer    = Transformer(D=D, num_heads=3, mlp_mult=2)



    def forward(self, x):
        # embedding
        x = self.patch_embed(x)
        x = self.pos_embed(x) 

        x = self.transformer(x)

        return x

    
