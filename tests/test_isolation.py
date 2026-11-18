import pytest
torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
from unittest.mock import patch
from models.model import BimodalSurgicalIMF
from datasets.dataset_adapter import HistoryDataset
from utils.io import write_jsonl
from utils.reproducibility import sample_noise


def test_history_view_does_not_open_targets_or_qgt(tiny_config, tmp_path):
    c = tiny_config
    c["experiment"]["output_dir"] = str(tmp_path)
    write_jsonl(tmp_path / "manifests/test_history.jsonl", [{"sample_id": "one",
        "history_abs": [[.1, .2], [.2, .3], [.3, .4]], "history_images": [], "history_wh": []}])
    # No target or q_GT files exist. Accessing the training view would fail.
    with patch("datasets.dataset_adapter.TargetStore", side_effect=AssertionError("GT access")):
        data = HistoryDataset(c, "test", images=False)
        item = data[0]
        model = BimodalSurgicalIMF(c).eval()
        noise = sample_noise(["one"], 3, 2, 1, "cpu")
        pred = model.generate(item["history_abs"][None], noise=noise)
        assert pred.shape == (1, 3, 2, 2)
        # Zero-initialized u: absolute = standard Gaussian + last history point.
        assert torch.allclose(pred, noise + torch.tensor([.3, .4]))


def test_full_inference_skips_mode_heads(tiny_config):
    # Stub ONLY the expensive image backbone. Real fusion, heads and iMF remain.
    class FakeVideo(torch.nn.Module):
        def __init__(self, c, load_pretrained):
            super().__init__()
            self.proj = torch.nn.Linear(3, c["video_encoder"]["output_dim"])
        def forward(self, x):
            return self.proj(x.mean((-1, -2)))
    c = tiny_config
    c["experiment"]["variant"] = "full"
    c["mode_supervision"]["enabled"] = True
    with patch("models.video_encoder.VideoEncoder", FakeVideo):
        model = BimodalSurgicalIMF(c).eval()
    with patch.object(model.traj_mode_head, "forward", side_effect=AssertionError("mode head")), \
         patch.object(model.video_mode_head, "forward", side_effect=AssertionError("mode head")):
        pred = model.generate(torch.rand(1, 3, 2), torch.rand(1, 3, 3, 4, 4), torch.randn(1, 3, 2, 2))
    assert pred.shape == (1, 3, 2, 2)
