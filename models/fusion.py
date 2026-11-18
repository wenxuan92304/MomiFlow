import torch
from torch import nn


class TemporalConcatFusion(nn.Module):
    def __init__(self, traj_dim, video_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(traj_dim + video_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, z, v):
        if z.shape[:2] != v.shape[:2]:
            raise ValueError("Trajectory/video time axes do not match")
        return self.mlp(torch.cat((z, v), dim=-1))


class TemporalCrossAttentionFusion(nn.Module):
    """Trajectory-time queries attend to all observed video-time tokens."""

    def __init__(self, traj_dim, video_dim, output_dim, num_heads):
        super().__init__()
        self.query_projector = nn.Linear(traj_dim, output_dim)
        self.attention = nn.MultiheadAttention(output_dim, num_heads, kdim=video_dim,
                                               vdim=video_dim, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, z, v):
        if z.shape[:2] != v.shape[:2]:
            raise ValueError("Trajectory/video time axes do not match")
        query = self.query_projector(z)
        attended, _ = self.attention(query, v, v, need_weights=False)
        return self.norm(query + attended)
