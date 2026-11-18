import pytest
import copy
pytest.importorskip("yaml")
from utils.config import load_config, model_signature, paths
from stage1.artifacts import stage1_signature


def test_variants_and_defaults():
    c = load_config("configs/default.yaml")
    assert c["video_encoder"]["freeze_mode"] == "partial_finetune"
    assert c["video_encoder"]["trainable_blocks"] == 2
    assert c["video_encoder"]["trainable_norm"] is True
    assert c["video_encoder"]["temporal_positional_encoding"] == "sinusoidal"
    for name in ("trajectory_only", "bimodal_no_mode"):
        assert not load_config("configs/" + name + ".yaml")["mode_supervision"]["enabled"]
    phakir = load_config("configs/phakir.yaml")
    assert phakir["evaluation"]["pixel_metrics"] is True
    assert phakir["evaluation"]["pixel_space"] == "actual_image"
    assert phakir["evaluation"]["metric_prediction_representation"] == "absolute_normalized_xy"
    assert phakir["evaluation"]["metric_target_representation"] == "absolute_normalized_xy"


@pytest.mark.parametrize("override", ["unknown=1", "data.image_index_column=frame",
    "imf.noise_type=mixture", "stage1.input_representation=absolute", "evaluation.K=3",
    "experiment.variant=trajectory_only", "evaluation.phakir_pixel_protocol_confirmed=true"])
def test_reject_conflicting_or_unsupported_config(override):
    with pytest.raises(ValueError):
        load_config("configs/default.yaml", [override])


def test_four_group_configs_and_stage1_sharing():
    trajectory = load_config("configs/trajectory_only.yaml")
    bimodal = load_config("configs/bimodal_no_mode.yaml")
    trajectory_mode = load_config("configs/cholectrack20_improved_trajectory_mode.yaml")
    full = load_config("configs/cholectrack20_improved_full.yaml")
    assert [c["experiment"]["variant"] for c in (trajectory, trajectory_mode, bimodal, full)] == [
        "trajectory_only", "trajectory_mode", "bimodal_no_mode", "full"]
    assert [c["mode_supervision"]["enabled"] for c in (trajectory, trajectory_mode, bimodal, full)] == [
        False, True, False, True]
    assert paths(trajectory_mode)["qgt_artifact"].resolve() == paths(full)["qgt_artifact"].resolve()
    stage1 = load_config("configs/cholectrack20_stage1_geometry.yaml")
    assert paths(trajectory_mode)["qgt_artifact"].resolve() == (
        paths(stage1)["qgt_artifact"].resolve())
    assert paths(trajectory_mode)["prototype_artifact"].resolve() == paths(stage1)["prototype_artifact"].resolve()
    assert trajectory_mode["stage1"]["method"] == full["stage1"]["method"] == "geometry_prototypes"
    assert paths(trajectory_mode)["output"].resolve() != paths(full)["output"].resolve()
    assert paths(trajectory_mode)["output"].resolve() not in [
        paths(c)["output"].resolve() for c in (trajectory, bimodal)]
    assert trajectory_mode["mode_supervision"]["encoder_grad_scale"] == full["mode_supervision"]["encoder_grad_scale"]
    assert trajectory_mode["mode_supervision"]["lambda_traj"] == full["mode_supervision"]["lambda_traj"]
    assert trajectory_mode["mode_supervision"]["lambda_video"] == 0
    for key in ("data", "trajectory_encoder", "imf", "training", "evaluation"):
        assert trajectory_mode[key] == trajectory[key]
    for key in ("data", "trajectory_encoder", "video_encoder", "fusion", "imf", "training", "evaluation"):
        assert full[key] == bimodal[key]


def test_existing_full_signature_still_matches_saved_config():
    current = load_config("configs/default.yaml")
    old_schema = copy.deepcopy(current)
    del old_schema["mode_supervision"]["encoder_grad_scale"]
    assert model_signature(old_schema) == model_signature(current)


def test_legacy_stage1_signature_ignores_new_method_defaults():
    current = load_config("configs/default.yaml")
    old_schema = copy.deepcopy(current)
    for key in ("method", "seed", "prototype_artifact", "prototype_n_init", "prototype_max_iter",
                "prototype_tol", "prototype_soft_scale"):
        del old_schema["stage1"][key]
    assert stage1_signature(old_schema) == stage1_signature(current)
