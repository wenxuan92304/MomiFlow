import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.manifest import verify_manifest
from utils.io import read_jsonl, read_json, file_hash
from utils.config import paths


def ensure_manifests(c, train_only=False):
    from datasets.manifest import build_manifest
    p = paths(c)
    if not (p["manifests"] / "metadata.json").exists():
        if train_only:
            raise FileNotFoundError("Prepare manifests first: python scripts/prepare_data.py --config <CONFIG>. Stage1 only opens train files.")
        d = c["data"]
        build_manifest(p["dataset_root"], d["dataset_name"], p["manifests"],
                       d["obs_len"], d["pred_len"], d["sample_stride"], d["expected_orig_frame_step"])
    d = dict(c["data"], dataset_root=str(p["dataset_root"]))
    return verify_manifest(p["manifests"], d, splits=("train",) if train_only else ("train", "val", "test"))


def check_probabilities(q, k, tolerance):
    if q.ndim != 2 or q.shape[1] != k or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("Invalid soft probability array")
    if not np.allclose(q.sum(1), 1, atol=tolerance, rtol=0):
        raise ValueError("Probability rows must sum to one")


class HistoryDataset(Dataset):
    """Inference view: opens exactly one history JSONL; no target/qGT dependency."""
    def __init__(self, c, split, images=True, sample_ids=None):
        self.cfg = c
        p = paths(c)
        self.root = p["dataset_root"] / c["data"]["dataset_name"]
        self.rows = read_jsonl(p["manifests"] / (split + "_history.jsonl"))
        if sample_ids is not None:
            wanted = set(sample_ids)
            self.rows = [r for r in self.rows if r["sample_id"] in wanted]
            if {r["sample_id"] for r in self.rows} != wanted:
                raise ValueError("Unknown history sample IDs")
        self.images = images
        if images:
            from PIL import Image
            self.image_api = Image
            v = c["video_encoder"]
            self.mean = np.asarray(v["rgb_mean"], np.float32).reshape(1, 1, 3)
            self.std = np.asarray(v["rgb_std"], np.float32).reshape(1, 1, 3)
            self.resolution = v["input_resolution"]
            self.resampling = getattr(Image.Resampling, v["interpolation"].upper())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        result = {"sample_id": row["sample_id"], "history_abs": torch.tensor(row["history_abs"], dtype=torch.float32)}
        if self.images:
            frames = []
            for rel, expected_wh in zip(row["history_images"], row["history_wh"]):
                with self.image_api.open(self.root / rel) as im:
                    if list(im.size) != expected_wh:
                        raise ValueError("PNG dimensions changed: " + rel)
                    im = im.convert("RGB").resize((self.resolution, self.resolution), self.resampling)
                    arr = (np.asarray(im, dtype=np.float32) / 255 - self.mean) / self.std
                frames.append(torch.from_numpy(arr.transpose(2, 0, 1).copy()))
            result["history_video"] = torch.stack(frames)
        return result


class TargetStore:
    """Only training and post-generation evaluation may instantiate this view."""
    def __init__(self, c, split):
        rows = read_jsonl(paths(c)["manifests"] / (split + "_targets.jsonl"))
        self.rows = {r["sample_id"]: r for r in rows}
        if len(self.rows) != len(rows):
            raise ValueError("Duplicate target IDs")

    def get(self, sid):
        return self.rows[sid]


class TrainingDataset(HistoryDataset):
    def __init__(self, c, stage1=False):
        super().__init__(c, "train", images=not stage1 and c["experiment"]["variant"] not in
                         ("trajectory_only", "trajectory_mode"))
        self.targets = TargetStore(c, "train")
        if set(self.targets.rows) != {r["sample_id"] for r in self.rows}:
            raise ValueError("History/target ID coverage mismatch")
        self.qgt = None
        if not stage1 and c["mode_supervision"]["enabled"]:
            from stage1.artifacts import load_qgt
            self.qgt = load_qgt(c, [r["sample_id"] for r in self.rows])

    def __getitem__(self, index):
        result = super().__getitem__(index)
        target = self.targets.get(result["sample_id"])
        future_abs = torch.tensor(target["future_abs"], dtype=torch.float32)
        result["future_rel"] = future_abs - result["history_abs"][-1].unsqueeze(0)
        result["future_abs"] = future_abs
        if self.qgt is not None:
            result["q_gt"] = torch.from_numpy(self.qgt[result["sample_id"]].copy())
        return result
