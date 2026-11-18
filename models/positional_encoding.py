import math
import torch
from torch import nn


class SinusoidalPositionEncoding(nn.Module):
    """Only adds order information; no learned temporal processing."""
    def __init__(self, dim, length):
        super().__init__()
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        freq = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, length, dim)
        pe[0, :, 0::2] = torch.sin(position * freq)
        pe[0, :, 1::2] = torch.cos(position * freq[:dim // 2])
        self.register_buffer("encoding", pe, persistent=True)

    def forward(self, x):
        if x.shape[1:] != self.encoding.shape[1:]:
            raise ValueError("Unexpected temporal token dimensions")
        return x + self.encoding.to(dtype=x.dtype)
