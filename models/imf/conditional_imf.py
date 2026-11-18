import torch
from torch import nn
from models.imf.backbone import TransformerBlock, FinalProjection
from models.imf.conditioning import HistoryConditioning
from models.imf.imf_core import ImprovedMeanFlowCore


class ConditionalIMF(nn.Module):
    def __init__(self, c):
        super().__init__()
        i = c["imf"]
        self.pred_len = c["data"]["pred_len"]
        self.obs_len = c["data"]["obs_len"]
        self.trajectory_state_encoder = nn.Linear(2, i["hidden_dim"])
        nn.init.xavier_uniform_(self.trajectory_state_encoder.weight)
        nn.init.zeros_(self.trajectory_state_encoder.bias)
        self.conditioning = HistoryConditioning(c)
        self.shared_blocks = nn.ModuleList([TransformerBlock(i) for _ in range(i["num_layers"] - i["aux_head_depth"])])
        self.u_heads = nn.ModuleList([TransformerBlock(i) for _ in range(i["aux_head_depth"])])
        self.v_heads = nn.ModuleList([TransformerBlock(i) for _ in range(i["aux_head_depth"])])
        self.u_output_projection = FinalProjection(i)
        self.v_output_projection = FinalProjection(i)
        self.imf_core = ImprovedMeanFlowCore(i)

    def forward(self, state, t, h, omega, t_min, t_max, condition, drop_mask=None, u_only=False):
        if state.shape[1:] != (self.pred_len, 2) or condition.shape[1] != self.obs_len:
            raise ValueError("iMF state/context shape mismatch")
        # Official imfDiT intentionally does not embed t; only h=t-r.
        prefix = self.conditioning(condition, h, omega, t_min, t_max, drop_mask)
        seq = torch.cat((prefix, self.trajectory_state_encoder(state)), dim=1)
        for block in self.shared_blocks:
            seq = block(seq)
        u_seq = seq
        for block in self.u_heads:
            u_seq = block(u_seq)
        u = self.u_output_projection(u_seq[:, prefix.shape[1]:])
        if u_only:
            return u, None
        v_seq = seq
        for block in self.v_heads:
            v_seq = block(v_seq)
        v = self.v_output_projection(v_seq[:, prefix.shape[1]:])
        return u, v

    def training_loss(self, future_rel, condition, draws=None):
        return self.imf_core.loss(self, future_rel, condition, draws)

    def generate(self, condition, noise):
        return self.imf_core.sample(self, condition, noise)
