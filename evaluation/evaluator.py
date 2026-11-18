import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.dataset_adapter import HistoryDataset, TargetStore
from datasets.manifest import png_size
from evaluation.metrics import per_sample_metrics, with_mean_k_aliases
from utils.config import paths
from utils.io import file_hash, read_json, write_json, write_jsonl, digest
from utils.reproducibility import sample_noise, seed_worker


def generate_predictions(model, c, split, output_dir, checkpoint_sha=None, sample_ids=None, noise_sampler=None):
    """No TargetStore, Stage1 or q_GT access anywhere in generation."""
    if noise_sampler is None:
        noise_sampler = sample_noise
    model.eval()
    device = next(model.parameters()).device
    p = paths(c)
    manifest = read_json(p["manifests"] / "metadata.json")
    history_path = p["manifests"] / (split + "_history.jsonl")
    if file_hash(history_path) != manifest["files"][history_path.name]:
        raise ValueError("History manifest changed")
    dataset = HistoryDataset(c, split, images=model.video_encoder is not None, sample_ids=sample_ids)
    if not len(dataset):
        raise ValueError("Empty inference dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=c["evaluation"]["batch_size"], shuffle=False,
                        num_workers=c["training"]["num_workers"], worker_init_fn=seed_worker)
    entries = []
    for part, batch in enumerate(loader):
        history = batch["history_abs"].to(device)
        video = batch.get("history_video")
        video = video.to(device) if video is not None else None
        noise = noise_sampler(batch["sample_id"], c["evaluation"]["K"], c["data"]["pred_len"], c["evaluation"]["seed"], device)
        with torch.no_grad():
            prediction = model.generate(history, video, noise).cpu().numpy()
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite generated trajectory")
        name = "predictions_%06d.npz" % part
        with open(output_dir / name, "wb") as f:
            np.savez_compressed(f, sample_ids=np.asarray(batch["sample_id"]), predictions_abs=prediction)
        entries.append({"file": name, "sha256": file_hash(output_dir / name), "count": len(history)})
    index = {"schema": 1, "dataset": c["data"]["dataset_name"], "split": split,
             "generated_representation": "future_rel_to_last_history",
             "saved_representation": "absolute_normalized_xy", "K": c["evaluation"]["K"],
             "seed": c["evaluation"]["seed"], "num_steps": c["imf"]["num_steps"],
             "omega": c["imf"]["sampling_omega"], "checkpoint_sha256": checkpoint_sha,
             "config_sha256": digest(c), "history_manifest_sha256": file_hash(history_path),
             "manifest_fingerprint": manifest["fingerprint"], "shards": entries, "count": len(dataset)}
    write_json(output_dir / "predictions.json", index)
    return output_dir / "predictions.json"


def evaluate_saved(c, prediction_index):
    """Call only AFTER all predictions have been generated and saved."""
    index = read_json(prediction_index)
    split = index["split"]
    p = paths(c)
    meta = read_json(p["manifests"] / "metadata.json")
    if index["dataset"] != c["data"]["dataset_name"] or index["manifest_fingerprint"] != meta["fingerprint"]:
        raise ValueError("Prediction/data fingerprint mismatch")
    target_path = p["manifests"] / (split + "_targets.jsonl")
    if file_hash(target_path) != meta["files"][target_path.name]:
        raise ValueError("Target manifest changed")
    target_store = TargetStore(c, split)
    seen, records = set(), []
    sums = {}
    size_cache = {}
    for shard in index["shards"]:
        path = prediction_index.parent / shard["file"]
        if file_hash(path) != shard["sha256"]:
            raise ValueError("Corrupt prediction shard")
        with np.load(path, allow_pickle=False) as f:
            ids, pred = f["sample_ids"].tolist(), f["predictions_abs"]
        if len(ids) != shard["count"] or pred.shape[1] != index["K"]:
            raise ValueError("Prediction count/K mismatch")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Duplicate prediction IDs")
        seen.update(ids)
        targets = [target_store.get(sid) for sid in ids]
        truth = np.asarray([t["future_abs"] for t in targets])
        values = per_sample_metrics(pred, truth)
        if c["evaluation"]["pixel_metrics"]:
            actual_wh = []
            for target in targets:
                dims = []
                for rel, stored in zip(target["future_images"], target["future_wh"]):
                    if rel not in size_cache:
                        size_cache[rel] = png_size(p["dataset_root"] / c["data"]["dataset_name"] / rel)
                    if size_cache[rel] != stored:
                        raise ValueError("Target PNG dimensions changed: " + rel)
                    dims.append(size_cache[rel])
                actual_wh.append(dims)
            values.update(per_sample_metrics(pred, truth, actual_wh))
        # ADE/FDE already average over all K predictions; preserve legacy keys.
        values = with_mean_k_aliases(values, index["K"])
        for key, array in values.items():
            sums[key] = sums.get(key, 0.) + float(array.sum())
        for i, sid in enumerate(ids):
            records.append(dict(sample_id=sid, **{key: float(array[i]) for key, array in values.items()}))
    if len(seen) != index["count"]:
        raise ValueError("Incomplete prediction index")
    metrics = {k: v / len(seen) for k, v in sums.items()}
    write_json(prediction_index.parent / "metrics.json", {"metrics": metrics, "samples": len(seen),
               "K": index["K"], "ADE_FDE_reduction": "mean_over_generated_samples",
               "pixel_dimensions": "actual_original_PNG_WH", "config": c})
    write_jsonl(prediction_index.parent / "per_sample_metrics.jsonl", records)
    return metrics
