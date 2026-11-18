import math
import torch
from torch import nn
from models.imf.backbone import scaled_linear


class ScalarEmbedder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dim = c["scalar_embedding_dim"]
        self.max_period = c["scalar_max_period"]
        d = c["hidden_dim"]
        self.mlp = nn.Sequential(scaled_linear(self.dim, d, c["embedding_init_constant"], True), nn.SiLU(),
                                 scaled_linear(d, d, c["embedding_init_constant"], True))

    def forward(self, scalar):
        half = self.dim // 2
        freq = torch.exp(-math.log(self.max_period) * torch.arange(half, device=scalar.device, dtype=scalar.dtype) / half)
        args = scalar[:, None] * freq[None]
        emb = torch.cat((args.cos(), args.sin()), dim=-1)
        if self.dim % 2:
            emb = torch.cat((emb, torch.zeros_like(emb[:, :1])), dim=-1)
        return self.mlp(emb)


class HistoryConditioning(nn.Module):
    """Replace class tokens with time-preserving projected history tokens.

    Null history uses a trainable vector broadcast across the same T slots.
    h (not absolute t), omega and interval tokens follow the official sequence.
    """
    def __init__(self, c):
        super().__init__()
        i = c["imf"]
        d = i["hidden_dim"]
        self.condition_projector = scaled_linear(c["fusion"]["output_dim"], d, i["embedding_init_constant"], True)
        self.null_context = nn.Parameter(torch.empty(1, 1, d))
        nn.init.normal_(self.null_context, std=i["embedding_init_constant"] / math.sqrt(d))
        for name, n in (("omega", i["cfg_tokens"]), ("min", i["interval_tokens"]),
                        ("max", i["interval_tokens"]), ("h", i["time_tokens"])):
            setattr(self, name + "_embedder", ScalarEmbedder(i))
            tokens = nn.Parameter(torch.empty(1, n, d))
            nn.init.normal_(tokens, std=i["token_init_constant"] / math.sqrt(d))
            setattr(self, name + "_tokens", tokens)

    def forward(self, condition, h, omega, t_min, t_max, drop_mask=None):
        context = self.condition_projector(condition)
        if drop_mask is not None:
            context = torch.where(drop_mask[:, None, None], self.null_context.expand_as(context), context)
        pieces = [context]
        for name, scalar in (("omega", 1 - 1 / omega), ("min", t_min), ("max", t_max), ("h", h)):
            pieces.append(getattr(self, name + "_tokens") + getattr(self, name + "_embedder")(scalar)[:, None, :])
        return torch.cat(pieces, dim=1)
