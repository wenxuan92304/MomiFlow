import numpy as np
from utils.config import paths
from utils.io import read_json, write_json, file_hash, digest
from datasets.dataset_adapter import check_probabilities


def stage1_seed(c):
    # Old resolved configs have no dedicated Stage 1 seed. Current configs do,
    # so changing only the downstream training seed cannot stale fixed labels.
    return c["stage1"].get("seed", c["training"]["seed"])


def stage1_signature(c):
    method = c["stage1"].get("method", "vae_gmm")
    if method == "vae_gmm":
        # Keep the pre-improvement signature byte-for-byte compatible with
        # already saved VAE/GMM/q_GT artifacts.
        excluded = ("method", "vae_checkpoint", "train_latents", "gmm_artifact", "qgt_artifact",
                    "prototype_artifact", "prototype_n_init", "prototype_max_iter",
                    "prototype_tol", "prototype_soft_scale", "seed")
        s = {k: v for k, v in c["stage1"].items() if k not in excluded}
    elif method == "geometry_prototypes":
        s = {k: c["stage1"][k] for k in ("method", "input_representation", "reference_point",
               "latent_representation",
               "prototype_n_init", "prototype_max_iter", "prototype_tol", "prototype_soft_scale")}
    else:
        raise ValueError("Unknown Stage 1 method: " + method)
    return digest({"stage1": s, "seed": stage1_seed(c), "K": c["mode_supervision"]["num_modes"]})


def base_metadata(c):
    p = paths(c)
    manifest = read_json(p["manifests"] / "metadata.json")
    meta = {"schema": 1, "dataset": c["data"]["dataset_name"], "split": "train",
            "train_history_sha256": manifest["files"]["train_history.jsonl"],
            "train_targets_sha256": manifest["files"]["train_targets.jsonl"],
            "stage1_signature": stage1_signature(c),
            "input_representation": "future_relative_to_last_history", "reference_point": "last_history_point",
            "latent_representation": c["stage1"]["latent_representation"], "obs_len": c["data"]["obs_len"],
            "pred_len": c["data"]["pred_len"], "num_modes": c["mode_supervision"]["num_modes"],
            "protocol": manifest["spec"]["protocol"]}
    if c["stage1"].get("method", "vae_gmm") == "geometry_prototypes":
        meta["stage1_method"] = "geometry_prototypes"
    return meta


def save_npz(path, arrays, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".tmp", "wb") as f:
        np.savez_compressed(f, **arrays)
    import os
    os.replace(str(path) + ".tmp", str(path))
    write_json(str(path) + ".json", dict(metadata, artifact_sha256=file_hash(path)))


def load_npz(path, c):
    meta = read_json(str(path) + ".json")
    expected = base_metadata(c)
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError("Stale Stage1 metadata: " + key)
    if meta["artifact_sha256"] != file_hash(path):
        raise ValueError("Corrupt artifact: " + str(path))
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    return arrays, meta


def load_qgt(c, expected_ids):
    p = paths(c)
    arrays, meta = load_npz(p["qgt_artifact"], c)
    if meta.get("component_order") != list(range(c["mode_supervision"]["num_modes"])):
        raise ValueError("q_GT component order mismatch")
    if c["stage1"].get("method", "vae_gmm") == "geometry_prototypes":
        producers = (("prototype_sha256", p["prototype_artifact"]),)
    else:
        producers = (("vae_sha256", p["vae_checkpoint"]), ("gmm_sha256", p["gmm_artifact"]))
    for key, path in producers:
        if meta[key] != file_hash(path):
            raise ValueError("q_GT producer changed: " + key)
    ids, q = arrays["sample_ids"].tolist(), arrays["q_gt"]
    if len(set(ids)) != len(ids) or set(ids) != set(expected_ids) or len(q) != len(ids):
        raise ValueError("q_GT sample_id coverage mismatch")
    check_probabilities(q, c["mode_supervision"]["num_modes"], c["mode_supervision"]["probability_tolerance"])
    return dict(zip(ids, q.astype(np.float32)))
