import pytest
import copy

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from stage1.geometry import squared_trajectory_distances, soft_assignments
from stage1.fit_geometry import fit_geometry
from stage1.build_qgt_geometry import build_qgt_geometry
from stage1.artifacts import load_qgt, stage1_signature
from utils.config import load_config, paths
from utils.io import file_hash, write_json, write_jsonl
from datasets.manifest import PROTOCOL


def test_soft_assignment_uses_full_future_rel_and_keeps_soft_labels():
    trajectories = np.array([[[0., 0.], [0., 0.]], [[1., 0.], [1., 0.]]])
    distances = squared_trajectory_distances(trajectories, trajectories)
    np.testing.assert_allclose(distances, [[0., 1.], [1., 0.]])
    q = soft_assignments(distances, 0.1)
    assert q.shape == (2, 2)
    assert np.all((q > 0) & (q < 1))
    assert q[0, 0] > q[0, 1] and q[1, 1] > q[1, 0]
    np.testing.assert_allclose(q.sum(axis=1), 1, atol=1e-6)


def test_geometry_artifacts_are_train_only_soft_and_lineage_checked(tmp_path, monkeypatch):
    c = load_config("configs/cholectrack20_stage1_geometry.yaml")
    c["experiment"]["output_dir"] = str(tmp_path)
    c["data"].update(obs_len=2, pred_len=2)
    c["mode_supervision"]["num_modes"] = 2
    c["stage1"]["prototype_n_init"] = 2
    h, t = [], []
    for i in range(8):
        sid = "train_%d" % i
        h.append({"sample_id": sid, "history_abs": [[.5, .5], [.5, .5]]})
        dx = .02 + i * .001 if i < 4 else -.02 - i * .001
        t.append({"sample_id": sid, "future_abs": [[.5 + dx, .5], [.5 + 2 * dx, .5]]})
    manifest = tmp_path / "manifests"
    write_jsonl(manifest / "train_history.jsonl", h)
    write_jsonl(manifest / "train_targets.jsonl", t)
    write_json(manifest / "metadata.json", {"files": {p.name: file_hash(p) for p in manifest.glob("*.jsonl")},
                                                 "spec": {"protocol": PROTOCOL}})
    monkeypatch.setattr("stage1.geometry.ensure_manifests", lambda config, train_only: None)
    fit_geometry(c)
    build_qgt_geometry(c)
    q = load_qgt(c, [row["sample_id"] for row in h])
    assert len(q) == 8
    values = np.stack(list(q.values()))
    np.testing.assert_allclose(values.sum(1), 1, atol=1e-6)
    assert np.all(values > 0)
    assert len(set(values.argmax(1))) == 2
    changed_main_seed = copy.deepcopy(c)
    changed_main_seed["training"]["seed"] = 123
    assert load_qgt(changed_main_seed, [row["sample_id"] for row in h]).keys() == q.keys()
    changed_stage1_seed = copy.deepcopy(c)
    changed_stage1_seed["stage1"]["seed"] = 123
    with pytest.raises(ValueError, match="stage1_signature"):
        load_qgt(changed_stage1_seed, [row["sample_id"] for row in h])
    changed_assignment = copy.deepcopy(c)
    changed_assignment["stage1"]["prototype_soft_scale"] *= 2
    with pytest.raises(ValueError, match="stage1_signature"):
        load_qgt(changed_assignment, [row["sample_id"] for row in h])
    with pytest.raises(FileExistsError):
        fit_geometry(c)
    with pytest.raises(FileExistsError):
        build_qgt_geometry(c)
    p = paths(c)["prototype_artifact"]
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="producer changed"):
        load_qgt(c, [row["sample_id"] for row in h])


def test_new_mode_configs_share_exact_stage1_signature_and_artifacts():
    stage1 = load_config("configs/cholectrack20_stage1_geometry.yaml")
    trajectory = load_config("configs/cholectrack20_improved_trajectory_mode.yaml")
    full = load_config("configs/cholectrack20_improved_full.yaml")
    assert stage1_signature(stage1) == stage1_signature(trajectory) == stage1_signature(full)
    for key in ("prototype_artifact", "qgt_artifact"):
        assert paths(stage1)[key].resolve() == paths(trajectory)[key].resolve() == paths(full)[key].resolve()
