import argparse
import copy
import math
from pathlib import Path
import yaml
from utils.io import ROOT, resolve, digest


def merge(base, update):
    for key, value in update.items():
        if key not in base:
            raise ValueError("Unknown configuration key: " + key)
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError("Expected mapping: " + key)
            merge(base[key], value)
        else:
            base[key] = value
    return base


def _read(path, stack=()):
    path = path.resolve()
    if path in stack:
        raise ValueError("Circular YAML inheritance")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = data.pop("extends", None)
    if parent:
        return merge(_read(path.parent / parent, stack + (path,)), data)
    return data


def load_config(path, overrides=()):
    cfg = _read(resolve(path))
    schema = _read(ROOT / "configs/default.yaml")
    cfg = merge(schema, cfg)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError("Unknown override: " + key)
            node = node[part]
        if parts[-1] not in node:
            raise ValueError("Unknown override: " + key)
        node[parts[-1]] = yaml.safe_load(value)
    validate(cfg)
    return cfg


def validate(c):
    fixed = {
        "data.history_coordinate_mode": "absolute_normalized_xy",
        "data.future_source_coordinate_mode": "absolute_normalized_xy",
        "data.future_target_mode": "relative_to_last_history",
        "data.relative_reference": "last_history_point", "data.input_columns": ["x", "y"],
        "data.image_index_column": "orig_frame", "data.sequence_index_column": "frame",
        "data.gap_policy": "split", "data.invalid_coordinate_policy": "reject_window_with_report",
        "data.split_source": "existing_csv_directories",
        "stage1.input_representation": "future_relative_to_last_history",
        "stage1.reference_point": "last_history_point",
        "stage1.reconstruction_reduction": "mean", "stage1.gmm.covariance_type": "full",
        "trajectory_encoder.input_dim": 2, "trajectory_encoder.positional_encoding": "sinusoidal",
        "video_encoder.model": "pl_stitch_vit_base", "video_encoder.patch_size": 16,
        "video_encoder.return_all_tokens": False, "video_encoder.masked_im_modeling": False,
        "video_encoder.input_resolution": 224, "video_encoder.temporal_adapter": "per_frame_cls",
        "video_encoder.temporal_positional_encoding": "sinusoidal",
        "video_encoder.spatial_transform": "resize_full_frame", "video_encoder.cache_features": False,
        "imf.implementation": "official_jax_math_pytorch_port",
        "imf.state_representation": "future_relative_to_last_history",
        "imf.output_representation": "future_relative_to_last_history",
        "imf.conditioning_method": "in_context_prefix", "imf.attention_backend": "math",
        "imf.noise_type": "gaussian", "evaluation.save_all_predictions": True,
        "evaluation.restore_absolute_with": "last_history_point",
        "evaluation.metric_prediction_representation": "absolute_normalized_xy",
        "evaluation.metric_target_representation": "absolute_normalized_xy",
        "evaluation.pixel_space": "actual_image", "evaluation.ade_fde_reduction": "mean_over_generated_samples",
    }
    for path, expected in fixed.items():
        v = c
        for key in path.split("."):
            v = v[key]
        if v != expected:
            raise ValueError("Unsupported plan change %s=%r; expected %r" % (path, v, expected))
    variant = c["experiment"]["variant"]
    if variant not in ("full", "bimodal_no_mode", "trajectory_only", "trajectory_mode"):
        raise ValueError("Unknown variant")
    if c["mode_supervision"]["enabled"] != (variant in ("full", "trajectory_mode")):
        raise ValueError("variant and mode_supervision.enabled conflict")
    stage1_method = c["stage1"]["method"]
    if stage1_method not in ("vae_gmm", "geometry_prototypes"):
        raise ValueError("Unknown Stage 1 method")
    if not isinstance(c["stage1"]["seed"], int) or isinstance(c["stage1"]["seed"], bool) or c["stage1"]["seed"] < 0:
        raise ValueError("stage1.seed must be a nonnegative integer")
    if stage1_method == "geometry_prototypes" and c["mode_supervision"]["num_modes"] < 2:
        raise ValueError("Geometry-prototype Stage 1 requires at least two modes")
    expected_representation = "posterior_mean" if stage1_method == "vae_gmm" else "direct_future_rel"
    if c["stage1"]["latent_representation"] != expected_representation:
        raise ValueError("Stage 1 representation/method mismatch")
    if any(not isinstance(c["stage1"][key], int) or isinstance(c["stage1"][key], bool)
           or c["stage1"][key] <= 0 for key in ("prototype_n_init", "prototype_max_iter")) or any(
           not isinstance(c["stage1"][key], (int, float)) or isinstance(c["stage1"][key], bool)
           or not math.isfinite(c["stage1"][key]) or c["stage1"][key] <= 0
           for key in ("prototype_tol", "prototype_soft_scale")):
        raise ValueError("Invalid geometry-prototype Stage 1 settings")
    v = c["video_encoder"]
    if v["freeze_mode"] not in ("freeze", "partial_finetune", "full_finetune"):
        raise ValueError("Invalid freeze mode")
    if not 0 <= v["trainable_blocks"] <= 12:
        raise ValueError("PL-Stitch has 12 blocks")
    if variant == "full" and (v["freeze_mode"], v["trainable_blocks"], v["trainable_norm"]) != ("partial_finetune", 2, True):
        raise ValueError("The agreed Full protocol requires partial_finetune, last 2 blocks, final norm")
    # Other variants retain all three freeze modes for explicit YAML ablations.
    if v["interpolation"] not in ("bicubic", "bilinear"):
        raise ValueError("Unsupported image interpolation")
    for name in ("trajectory_encoder", "imf"):
        d, h = c[name]["hidden_dim"], c[name]["num_heads"]
        if min(d, h) <= 0 or d % h:
            raise ValueError("hidden_dim must be divisible by num_heads")
    im = c["imf"]
    if (im["hidden_dim"] // im["num_heads"]) % 2:
        raise ValueError("RoPE requires even head dimension")
    if not 0 < im["aux_head_depth"] <= im["num_layers"]:
        raise ValueError("Invalid auxiliary head depth")
    if im["num_samples"] != c["evaluation"]["K"]:
        raise ValueError("imf.num_samples and evaluation.K must match")
    if im["sampling_omega"] < 1 or im["norm_eps"] <= 0 or im["time_std"] <= 0 or im["cfg_s_max"] < 0:
        raise ValueError("Invalid iMF distribution/numerical parameters")
    for key in ("data_proportion", "condition_dropout"):
        if not 0 <= im[key] <= 1:
            raise ValueError("Invalid probability: " + key)
    for key in ("cfg_interval_min_range", "cfg_interval_max_range"):
        a, b = im[key]
        if not 0 <= a <= b <= 1:
            raise ValueError("Invalid CFG interval")
    if not 0 <= im["sampling_t_min"] <= im["sampling_t_max"] <= 1:
        raise ValueError("Invalid sampling CFG interval")
    for section, keys in {"data": ["obs_len", "pred_len", "sample_stride", "expected_orig_frame_step"],
                          "stage1": ["latent_dim", "batch_size", "epochs"],
                          "training": ["batch_size", "epochs", "checkpoint_every_epochs", "validate_every_epochs"],
                          "evaluation": ["K", "batch_size"], "mode_supervision": ["num_modes"],
                          "imf": ["num_steps", "num_layers", "scalar_embedding_dim", "time_tokens", "cfg_tokens", "interval_tokens"]}.items():
        for key in keys:
            if not isinstance(c[section][key], int) or c[section][key] <= 0:
                raise ValueError("Expected positive integer: %s.%s" % (section, key))
    if c["data"]["dataset_name"] not in ("cholectrack20", "phakir"):
        raise ValueError("Unknown dataset")
    if c["mode_supervision"]["pooling"] not in ("attention", "mean"):
        raise ValueError("Unknown pooling")
    if c["mode_supervision"].get("label_type", "soft") not in ("soft", "hard"):
        raise ValueError("Unknown mode label type")
    fusion = c["fusion"]
    if fusion["method"] not in ("concat_mlp", "cross_attention"):
        raise ValueError("Unknown fusion method")
    if fusion["method"] == "cross_attention":
        fusion_heads = fusion.get("num_heads", 4)
        if not isinstance(fusion_heads, int) or isinstance(fusion_heads, bool) or fusion_heads <= 0:
            raise ValueError("fusion.num_heads must be a positive integer")
        if fusion["output_dim"] % fusion_heads:
            raise ValueError("fusion.output_dim must be divisible by fusion.num_heads")
        if variant != "full":
            raise ValueError("Cross-attention fusion requires the Full bimodal variant")
    if im["scalar_embedding_dim"] < 2:
        raise ValueError("Scalar sinusoidal embedding requires at least two dimensions")
    if c["training"]["num_workers"] < 0 or min(v["rgb_std"]) <= 0:
        raise ValueError("Invalid workers or RGB standard deviation")
    if any(c["mode_supervision"][k] < 0 for k in ("lambda_traj", "lambda_video")):
        raise ValueError("Mode loss weights cannot be negative")
    scale = c["mode_supervision"]["encoder_grad_scale"]
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not 0 < scale <= 1:
        raise ValueError("mode_supervision.encoder_grad_scale must be in (0, 1]")
    if variant == "trajectory_mode" and (c["mode_supervision"]["lambda_traj"] <= 0 or
                                          c["mode_supervision"]["lambda_video"] != 0):
        raise ValueError("trajectory_mode requires positive trajectory loss and zero video loss")
    if c["stage1"]["optimizer"] != "adam" or c["training"]["optimizer"] != "adamw":
        raise ValueError("Implemented optimizers: Stage1 adam, Stage2 adamw")
    for section in ("stage1", "training", "sanity"):
        s = c[section]
        if len(s["betas"]) != 2 or not all(0 <= beta < 1 for beta in s["betas"]):
            raise ValueError("Invalid optimizer betas in " + section)
        if s["eps"] <= 0 or s["learning_rate"] <= 0 or s["weight_decay"] < 0:
            raise ValueError("Invalid optimizer settings in " + section)


def paths(c):
    out = resolve(c["experiment"]["output_dir"])
    result = {"output": out, "manifests": out / "manifests", "dataset_root": resolve(c["data"]["dataset_root"])}
    for key in ("vae_checkpoint", "train_latents", "gmm_artifact", "qgt_artifact"):
        p = Path(c["stage1"][key])
        result[key] = p if p.is_absolute() else out / p
    p = Path(c["stage1"]["prototype_artifact"])
    result["prototype_artifact"] = p if p.is_absolute() else out / p
    result["video_checkpoint"] = resolve(c["video_encoder"]["checkpoint"])
    return result


def model_signature(c):
    spec = copy.deepcopy({k: c[k] for k in ("data", "trajectory_encoder", "video_encoder", "mode_supervision", "fusion", "imf")})
    spec["data"].pop("dataset_root")
    # Defaults preserve signatures of checkpoints saved before these ablations.
    if spec["mode_supervision"].get("label_type", "soft") == "soft":
        spec["mode_supervision"].pop("label_type", None)
    if spec["fusion"].get("method") == "concat_mlp":
        spec["fusion"].pop("num_heads", None)
    # Preserve signatures of already saved checkpoints, whose configuration
    # predates this identity-valued gradient control.
    if spec["mode_supervision"].get("encoder_grad_scale") == 1.0:
        spec["mode_supervision"].pop("encoder_grad_scale")
    for key in ("checkpoint", "source_root"):
        spec["video_encoder"].pop(key)
    for key in ("num_samples", "num_steps", "sampling_omega", "sampling_t_min", "sampling_t_max"):
        spec["imf"].pop(key)
    spec["variant"] = c["experiment"]["variant"]
    return digest(spec)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return p
