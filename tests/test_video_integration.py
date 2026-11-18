"""Optional actual checkpoint test; sanity_checks also exercises this path."""
import os
import pytest
torch = pytest.importorskip("torch")
from utils.config import load_config
from models.video_encoder import VideoEncoder


@pytest.mark.skipif(os.environ.get("RUN_PL_STITCH_INTEGRATION") != "1", reason="Set RUN_PL_STITCH_INTEGRATION=1; actual weights and sufficient RAM required")
def test_actual_checkpoint_temporal_positions_and_partial_gradients():
    c = load_config("configs/default.yaml", ["data.obs_len=2"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = VideoEncoder(c).to(device).train()
    assert encoder.pretrained_sha256
    expected_trainable = ("blocks.10.", "blocks.11.", "norm.")
    for name, parameter in encoder.backbone.named_parameters():
        assert parameter.requires_grad == name.startswith(expected_trainable), name
    frame = torch.randn(1, 1, 3, 224, 224, device=device)
    video = frame.expand(1, 2, -1, -1, -1).contiguous()
    observed = []
    def capture_cls(module, inputs, output):
        observed.append(output.detach())
    hook = encoder.backbone.register_forward_hook(capture_cls)
    try:
        out = encoder(video)
    finally:
        hook.remove()
    assert len(observed) == 1 and observed[0].shape == (2, 768)
    cls_tokens = observed[0].reshape(1, 2, 768)
    assert torch.isfinite(cls_tokens).all()
    torch.testing.assert_close(cls_tokens[:, 0], cls_tokens[:, 1], atol=2e-5, rtol=2e-5)
    assert out.shape == (1, 2, 128)
    base = out - encoder.position.encoding
    torch.testing.assert_close(base[:, 0], base[:, 1], atol=2e-5, rtol=2e-5)
    assert not torch.equal(out[:, 0], out[:, 1])
    out.square().mean().backward()
    for i, block in enumerate(encoder.backbone.blocks):
        if i < 10:
            assert all(not p.requires_grad and p.grad is None for p in block.parameters())
        else:
            assert all(p.requires_grad for p in block.parameters())
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())
    assert encoder.backbone.norm.weight.grad.abs().sum() > 0
    assert encoder.projector.weight.grad.abs().sum() > 0
    for name, parameter in encoder.backbone.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None, name
