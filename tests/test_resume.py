import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("numpy")
from models.model import BimodalSurgicalIMF
from utils.checkpoint import save_training, resume_training
from utils.reproducibility import seed_all


def test_exact_epoch_boundary_resume(tiny_config, tmp_path):
    c = tiny_config
    seed_all(123)
    model = BimodalSurgicalIMF(c)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    def update(m, o):
        batch = {"history_abs": torch.rand(2, 3, 2), "future_rel": torch.randn(2, 2, 2)}
        o.zero_grad(set_to_none=True)
        loss, _ = m(batch)
        loss.backward()
        o.step()
        return loss.detach().clone()
    update(model, opt)
    path = tmp_path / "checkpoint.pt"
    artifacts = {"manifest": "synthetic"}
    save_training(path, model, opt, scaler, 1, 1, .2, c, artifacts)
    expected_loss = update(model, opt)
    expected_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    restored = BimodalSurgicalIMF(c)
    opt2 = torch.optim.AdamW(restored.parameters(), lr=.001)
    resume_training(path, restored, opt2, scaler, c, artifacts)
    actual_loss = update(restored, opt2)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
