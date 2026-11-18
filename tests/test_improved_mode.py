import copy
import sys
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from models.model import BimodalSurgicalIMF
from utils.reproducibility import sample_noise


def trajectory_mode_config(tiny_config):
    c = copy.deepcopy(tiny_config)
    c["experiment"]["variant"] = "trajectory_mode"
    c["mode_supervision"].update(enabled=True, lambda_traj=0.1,
                                  lambda_video=0.0, encoder_grad_scale=0.25)
    return c


def test_trajectory_mode_has_no_video_and_keeps_shared_initialization(tiny_config):
    torch.manual_seed(17)
    baseline = BimodalSurgicalIMF(tiny_config, load_pretrained=False)
    torch.manual_seed(17)
    with patch.dict(sys.modules, {"models.video_encoder": None,
                                  "third_party.pl_stitch_vit": None}):
        model = BimodalSurgicalIMF(trajectory_mode_config(tiny_config), load_pretrained=False)
    assert model.video_encoder is None
    assert not hasattr(model, "video_pool")
    assert not hasattr(model, "video_mode_head")
    for name in ("trajectory_encoder", "fusion", "imf"):
        old = getattr(baseline, name).state_dict()
        new = getattr(model, name).state_dict()
        assert old.keys() == new.keys()
        for key in old:
            torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)

    batch = {"history_abs": torch.rand(2, 3, 2),
             "future_rel": torch.randn(2, 2, 2),
             "q_gt": torch.tensor([[0.1, 0.6, 0.2, 0.1], [0.2, 0.1, 0.6, 0.1]])}
    loss, info = model(batch)
    assert torch.isfinite(loss) and "mode_traj" in info and "mode_video" not in info
    model.zero_grad(set_to_none=True)
    z, v = model.encode_branches(batch["history_abs"])
    assert v is None
    mode_loss, video_loss, qhat, video_qhat = model.mode_losses(z, v, batch["q_gt"])
    assert video_loss is None and video_qhat is None
    torch.testing.assert_close(qhat.sum(-1), torch.ones(2))
    mode_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.trajectory_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.traj_mode_head.parameters())
    model.eval()
    noise = sample_noise(["one", "two"], 3, 2, 42, "cpu")
    assert model.generate(batch["history_abs"], noise=noise).shape == (2, 3, 2, 2)


def test_mode_gradient_scaling_changes_encoder_gradient_only(tiny_config):
    torch.manual_seed(23)
    c = trajectory_mode_config(tiny_config)
    model = BimodalSurgicalIMF(c, load_pretrained=False)
    q = torch.tensor([[0.1, 0.6, 0.2, 0.1], [0.2, 0.1, 0.6, 0.1]])
    features = torch.randn(2, 3, c["trajectory_encoder"]["hidden_dim"])
    head_weight = model.traj_mode_head.mlp[-1].weight

    c["mode_supervision"]["encoder_grad_scale"] = 1.0
    z = features.clone().requires_grad_()
    full_loss = model.mode_losses(z, None, q)[0]
    full_z, full_head = torch.autograd.grad(full_loss, (z, head_weight))

    c["mode_supervision"]["encoder_grad_scale"] = 0.25
    z = features.clone().requires_grad_()
    scaled_loss = model.mode_losses(z, None, q)[0]
    scaled_z, scaled_head = torch.autograd.grad(scaled_loss, (z, head_weight))
    torch.testing.assert_close(full_loss, scaled_loss, rtol=0, atol=0)
    torch.testing.assert_close(scaled_z, full_z * 0.25)
    torch.testing.assert_close(scaled_head, full_head)
    assert scaled_z.abs().sum() > 0
