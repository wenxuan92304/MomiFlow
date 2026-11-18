import torch
from torch import nn
from third_party.pl_stitch_vit import vit_base
from models.positional_encoding import SinusoidalPositionEncoding
from utils.config import paths
from utils.io import file_hash


class VideoEncoder(nn.Module):
    """Independent frame CLS -> projector -> fixed temporal positions.

    Partial fine tuning never wraps the entire backbone in no_grad. Frozen
    prefix parameters cannot receive gradients; trainable blocks/norm can.
    """
    def __init__(self, c, load_pretrained=True):
        super().__init__()
        v = c["video_encoder"]
        self.cfg = v
        self.backbone = vit_base(patch_size=v["patch_size"], img_size=[v["input_resolution"]],
                                 return_all_tokens=False, masked_im_modeling=False)
        self.pretrained_sha256 = None
        if load_pretrained:
            path = paths(c)["video_checkpoint"]
            state = torch.load(path, map_location="cpu", weights_only=True)
            state = state[v["checkpoint_key"]]
            prefix = v["checkpoint_prefix"]
            backbone = {k[len(prefix):]: value for k, value in state.items() if k.startswith(prefix)}
            if not backbone:
                raise ValueError("No PL-Stitch backbone keys with prefix " + prefix)
            self.backbone.load_state_dict(backbone, strict=True)
            self.pretrained_sha256 = file_hash(path)
        self.projector = nn.Linear(768, v["output_dim"])
        self.position = SinusoidalPositionEncoding(v["output_dim"], c["data"]["obs_len"])
        self._configure_trainable()

    def _configure_trainable(self):
        mode = self.cfg["freeze_mode"]
        self.backbone.requires_grad_(mode == "full_finetune")
        if mode == "partial_finetune":
            n = self.cfg["trainable_blocks"]
            if n:
                for block in self.backbone.blocks[-n:]:
                    block.requires_grad_(True)
            if self.cfg["trainable_norm"]:
                self.backbone.norm.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.cfg["freeze_mode"] != "full_finetune":
            self.backbone.eval()
            if mode and self.cfg["freeze_mode"] == "partial_finetune":
                n = self.cfg["trainable_blocks"]
                if n:
                    for block in self.backbone.blocks[-n:]:
                        block.train()
                if self.cfg["trainable_norm"]:
                    self.backbone.norm.train()
        return self

    def forward(self, history_video):
        if history_video.ndim != 5 or history_video.shape[2] != 3:
            raise ValueError("Video input must be [B,T,3,224,224]")
        b, t, channels, height, width = history_video.shape
        frames = history_video.reshape(b * t, channels, height, width)
        if self.cfg["freeze_mode"] == "freeze":
            with torch.no_grad():
                cls = self.backbone(frames)
        else:
            cls = self.backbone(frames)
        if cls.shape != (b * t, 768):
            raise ValueError("PL-Stitch did not return per-frame CLS vectors")
        return self.position(self.projector(cls.reshape(b, t, 768)))
