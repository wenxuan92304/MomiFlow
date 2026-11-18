"""Train-only future_rel geometry used by the prototype Stage 1."""
import numpy as np
from datasets.dataset_adapter import ensure_manifests, check_probabilities
from utils.config import paths
from utils.io import read_jsonl


def train_future_rel(c):
    ensure_manifests(c, train_only=True)
    p = paths(c)["manifests"]
    histories = read_jsonl(p / "train_history.jsonl")
    targets = read_jsonl(p / "train_targets.jsonl")
    by_id = {row["sample_id"]: row for row in targets}
    ids = [row["sample_id"] for row in histories]
    if (not ids or len(set(ids)) != len(ids) or len(by_id) != len(targets)
            or set(ids) != set(by_id)):
        raise ValueError("Invalid train history/target sample_id coverage")
    # Match TrainingDataset's float32 subtraction before using float64 for
    # numerically stable prototype fitting and softmax calculations.
    future = np.asarray([(np.asarray(by_id[sid]["future_abs"], dtype=np.float32)
                          - np.asarray(row["history_abs"][-1], dtype=np.float32)).astype(np.float64)
                         for sid, row in zip(ids, histories)], dtype=np.float64)
    if future.shape != (len(ids), c["data"]["pred_len"], 2) or not np.isfinite(future).all():
        raise ValueError("Invalid train future_rel shape or values")
    return np.asarray(ids), future


def squared_trajectory_distances(future_rel, centers):
    if (future_rel.ndim != 3 or centers.ndim != 3 or future_rel.shape[1:] != centers.shape[1:]
            or future_rel.shape[2] != 2):
        raise ValueError("Incompatible future_rel and prototype shapes")
    # Mean squared 2D displacement across the H future points, not latent distance.
    return np.square(future_rel[:, None] - centers[None]).sum(axis=-1).mean(axis=-1)


def soft_assignments(distances, temperature, tolerance=1e-5):
    if (distances.ndim != 2 or not np.isfinite(distances).all() or (distances < 0).any()
            or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("Invalid prototype distances or temperature")
    logits = -distances / (2.0 * temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    q = exp / exp.sum(axis=1, keepdims=True)
    # Avoid converting mathematically positive tail probabilities into hard zeros
    # when the offline supervision is saved as float32.
    q = np.maximum(q, np.finfo(np.float32).eps)
    q /= q.sum(axis=1, keepdims=True)
    q = q.astype(np.float32)
    check_probabilities(q, distances.shape[1], tolerance)
    return q
