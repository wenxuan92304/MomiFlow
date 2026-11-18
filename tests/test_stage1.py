import pytest
torch = pytest.importorskip("torch")
from utils.io import write_json, write_jsonl, file_hash
from datasets.manifest import PROTOCOL
from datasets.dataset_adapter import TrainingDataset


def fake_manifests(c, tmp_path):
    c["experiment"]["output_dir"] = str(tmp_path)
    out = tmp_path / "manifests"
    hs = [{"sample_id": "a", "history_abs": [[.1, .1], [.2, .2], [.8, .8]], "history_images": [], "history_wh": []},
          {"sample_id": "b", "history_abs": [[.9, .9], [.8, .8], [.2, .2]], "history_images": [], "history_wh": []}]
    ts = [{"sample_id": "a", "future_abs": [[.7, .7], [.6, .6]]},
          {"sample_id": "b", "future_abs": [[.3, .3], [.4, .4]]}]
    write_jsonl(out / "train_history.jsonl", hs)
    write_jsonl(out / "train_targets.jsonl", ts)
    write_json(out / "metadata.json", {"files": {p.name: file_hash(p) for p in out.glob("*.jsonl")}, "spec": {"protocol": PROTOCOL}})


def test_stage1_reads_only_train_and_uses_last_history_relative(tiny_config, tmp_path):
    c = tiny_config
    fake_manifests(c, tmp_path)
    # No val/test files, no images, no q_GT: Stage1 view still works.
    data = TrainingDataset(c, stage1=True)
    sample = data[0]
    torch.testing.assert_close(sample["future_rel"], torch.tensor([[-.1, -.1], [-.2, -.2]]))
    assert sample["future_rel"].min() < 0 # Relative target is NOT clipped to [0,1].
