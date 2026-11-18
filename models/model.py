import torch
from torch import nn
from models.trajectory_encoder import TrajectoryEncoder
from models.attention_pooling import make_pooling
from models.mode_head import ModeHead
from models.fusion import TemporalConcatFusion, TemporalCrossAttentionFusion
from models.imf.conditional_imf import ConditionalIMF
from training.losses import soft_cross_entropy, hard_cross_entropy


class BimodalSurgicalIMF(nn.Module):
    def __init__(self, c, load_pretrained=True):
        super().__init__()
        self.cfg = c
        self.trajectory_encoder = TrajectoryEncoder(c)
        td, vd = c["trajectory_encoder"]["hidden_dim"], c["video_encoder"]["output_dim"]
        self.video_encoder = None
        if c["experiment"]["variant"] in ("trajectory_only", "trajectory_mode"):
            self.fusion = nn.Linear(td, c["fusion"]["output_dim"])
        else:
            from models.video_encoder import VideoEncoder
            self.video_encoder = VideoEncoder(c, load_pretrained)
            if c["fusion"]["method"] == "concat_mlp":
                self.fusion = TemporalConcatFusion(td, vd, c["fusion"]["hidden_dim"], c["fusion"]["output_dim"])
            elif c["fusion"]["method"] == "cross_attention":
                self.fusion = TemporalCrossAttentionFusion(td, vd, c["fusion"]["output_dim"],
                                                           c["fusion"]["num_heads"])
            else:
                raise ValueError("Unknown fusion method")
        # Initialize the shared generator before optional auxiliary heads so
        # common modules consume the same RNG stream as their no-mode baseline.
        imf = ConditionalIMF(c)
        self.mode_enabled = c["mode_supervision"]["enabled"]
        if self.mode_enabled:
            m = c["mode_supervision"]
            self.traj_pool = make_pooling(td, m)
            if self.video_encoder is not None:
                self.video_pool = make_pooling(vd, m)
            self.traj_mode_head = ModeHead(td, m["head_hidden_dim"], m["num_modes"])
            if self.video_encoder is not None:
                self.video_mode_head = ModeHead(vd, m["head_hidden_dim"], m["num_modes"])
        # Register after heads to preserve legacy optimizer parameter ordering.
        self.imf = imf

    def encode_branches(self, history_abs, history_video=None):
        z = self.trajectory_encoder(history_abs)
        v = None
        if self.video_encoder is not None:
            if history_video is None or history_video.shape[:2] != history_abs.shape[:2]:
                raise ValueError("Synchronized history video required")
            v = self.video_encoder(history_video)
        return z, v

    def fuse(self, z, v):
        return self.fusion(z) if v is None else self.fusion(z, v)

    def mode_losses(self, z, v, q_gt):
        if not self.mode_enabled:
            raise ValueError("Mode supervision disabled")
        scale = self.cfg["mode_supervision"]["encoder_grad_scale"]
        # The forward representation is unchanged. Only the mode-branch
        # gradient reaching the shared encoder is reduced; mode heads receive
        # the configured label loss and iMF still sees unmodified Z/V.
        z_mode = z if scale == 1 else z.detach() + scale * (z - z.detach())
        a = self.traj_mode_head(self.traj_pool(z_mode))
        tol = self.cfg["mode_supervision"]["probability_tolerance"]
        loss_fn = (soft_cross_entropy if self.cfg["mode_supervision"].get("label_type", "soft") == "soft"
                   else hard_cross_entropy)
        lt = loss_fn(a, q_gt, tol)
        if v is None:
            return lt, None, a.float().softmax(-1), None
        v_mode = v if scale == 1 else v.detach() + scale * (v - v.detach())
        b = self.video_mode_head(self.video_pool(v_mode))
        lv = loss_fn(b, q_gt, tol)
        return lt, lv, a.float().softmax(-1), b.float().softmax(-1)

    def forward(self, batch, draws=None):
        z, v = self.encode_branches(batch["history_abs"], batch.get("history_video"))
        condition = self.fuse(z, v)
        imf_loss, info = self.imf.training_loss(batch["future_rel"], condition, draws)
        loss = imf_loss
        if self.mode_enabled:
            lt, lv, qt, qv = self.mode_losses(z, v, batch["q_gt"])
            m = self.cfg["mode_supervision"]
            loss = loss + m["lambda_traj"] * lt
            info.update(mode_traj=lt.detach(),
                        q_traj_sum_error=(qt.sum(-1) - 1).abs().max().detach())
            if lv is not None:
                loss = loss + m["lambda_video"] * lv
                info.update(mode_video=lv.detach(),
                            q_video_sum_error=(qv.sum(-1) - 1).abs().max().detach())
        info["total"] = loss.detach()
        return loss, info

    @torch.no_grad()
    def generate(self, history_abs, history_video=None, noise=None):
        """History-only signature; never runs mode heads or sees future/q_GT.

        noise [B,K,H,2] -> absolute normalized predictions [B,K,H,2].
        """
        if noise is None or noise.ndim != 4 or noise.shape[0] != len(history_abs):
            raise ValueError("Provide standard Gaussian noise [B,K,H,2]")
        z, v = self.encode_branches(history_abs, history_video)
        condition = self.fuse(z, v)
        b, k, h, channels = noise.shape
        if (h, channels) != (self.cfg["data"]["pred_len"], 2):
            raise ValueError("Incorrect future noise shape")
        condition = condition[:, None].expand(-1, k, -1, -1).reshape(b * k, *condition.shape[1:])
        relative = self.imf.generate(condition, noise.reshape(b * k, h, 2)).reshape(b, k, h, 2)
        return relative + history_abs[:, -1, :][:, None, None, :]

    def generate_one(self, history_abs, history_video=None, noise=None):
        if noise is None or noise.ndim != 3:
            raise ValueError("Single-sample noise must be [B,H,2]")
        return self.generate(history_abs, history_video, noise[:, None])[:, 0]
