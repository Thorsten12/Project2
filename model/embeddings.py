"""
In this code the Patch and Positional Embedding are present. Both are simple learnable Embeddings
"""
import torch
import torch.nn as nn
import torch.functional as F

class PatchEmbedding(nn.Module):
    # Patch Embedding -> (B, 64, 48) -> (B, 64, D)
    def __init__(self, 
                 H = 32,    # Height = Wight
                 C = 3,     # Color
                 P = 4,     # Patchsize
                 D = 192,   # Embedding Dim
                 ):
        super().__init__()
        self.proj           = nn.Conv2d(
            in_channels     = C,
            out_channels    = D,
            kernel_size     = P,
            stride          = P
        )

    def forward(self, x):
        # x (B, 3, 32, 32)

        x = self.proj(x)        # x (B, D, 8, 8)
        x = x.flatten(2)        # x (B, D, 64)
        x = x.transpose(1, 2)   # x (B, 64, D)

        return x

class PositionalEmbedding(nn.Module):
    # simple random-init Matrix with (B, N, D)
    def __init__(self, N, D):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, N, D))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed
        
    

if __name__ == "__main__":
    positional_embedding = PositionalEmbedding(64, 192)

    x = torch.randn(128, 64, 192)  # Patch Embeddings

    out = positional_embedding(x)

    print(out.shape)  # torch.Size([128, 64, 192])