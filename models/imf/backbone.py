"""PyTorch port of official imfDiT Transformer primitives (not original MF).

Manual attention deliberately supports the double backward used by JVP.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def scaled_linear(in_dim, out_dim, constant, bias=False):
    layer = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.normal_(layer.weight, std=constant / math.sqrt(in_dim))
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


def rotary(x, theta):
    # [B,L,heads,head_dim]; adjacent real/imaginary pairs as official JAX.
    length, dim = x.shape[1], x.shape[-1]
    freq = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=x.dtype) / dim)
    angle = torch.arange(length, device=x.device, dtype=x.dtype)[:, None] * freq[None]
    cos, sin = angle.cos()[None, :, None], angle.sin()[None, :, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class RoPEAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, self.heads = c["hidden_dim"], c["num_heads"]
        self.head_dim = d // self.heads
        self.theta = c["rope_theta"]
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, scaled_linear(d, d, c["weight_init_constant"]))
        self.q_norm = RMSNorm(self.head_dim, c["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, c["rms_norm_eps"])

    def forward(self, x):
        b, length, d = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, length, self.heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(b, length, self.heads, self.head_dim))
        v = self.v_proj(x).reshape(b, length, self.heads, self.head_dim).transpose(1, 2)
        q, k = rotary(q, self.theta).transpose(1, 2), rotary(k, self.theta).transpose(1, 2)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(b, length, d)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["hidden_dim"]
        hidden = int(d * c["mlp_ratio"])
        self.w1 = scaled_linear(d, hidden, c["weight_init_constant"])
        self.w3 = scaled_linear(d, hidden, c["weight_init_constant"])
        self.w2 = scaled_linear(hidden, d, c["weight_init_constant"])

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["hidden_dim"]
        self.norm1 = RMSNorm(d, c["rms_norm_eps"])
        self.norm2 = RMSNorm(d, c["rms_norm_eps"])
        self.attn = RoPEAttention(c)
        self.mlp = SwiGLU(c)
        self.attn_scale = nn.Parameter(torch.zeros(d))
        self.mlp_scale = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) * self.attn_scale
        return x + self.mlp(self.norm2(x)) * self.mlp_scale


class FinalProjection(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = RMSNorm(c["hidden_dim"], c["rms_norm_eps"])
        self.linear = nn.Linear(c["hidden_dim"], 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(self.norm(x))
