"""Independent stdlib verification of every saved window against source CSV."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.manifest import verify_manifest, PROTOCOL, SCHEMA
from utils.io import resolve, read_json, read_jsonl, digest, write_json


def verify(dataset_root, manifest_dir):
    meta = read_json(manifest_dir / "metadata.json")
    spec = meta["spec"]
    cfg = dict(spec, dataset_name=spec["dataset"], dataset_root=str(dataset_root))
    verify_manifest(manifest_dir, cfg)
    root = dataset_root / spec["dataset"]
    originals = {}
    for rel in spec["csv_sha256"]:
        with (root / rel).open(encoding="utf-8-sig", newline="") as f:
            originals[rel] = {(r["agent_id"], int(r["orig_frame"])): r for r in csv.DictReader(f)}
    report, seen = {}, set()
    for split in ("train", "val", "test"):
        histories = read_jsonl(manifest_dir / (split + "_history.jsonl"))
        targets = read_jsonl(manifest_dir / (split + "_targets.jsonl"))
        assert len(histories) == len(targets) == meta["report"][split]["samples"]
        negative_targets = 0
        for h, y in zip(histories, targets):
            sid = h["sample_id"]
            assert sid == y["sample_id"] and sid not in seen
            seen.add(sid)
            assert h["split"] == split
            assert not any(k in h for k in ("future_abs", "future_rel", "q_gt"))
            assert len(h["history_abs"]) == spec["obs_len"]
            assert len(y["future_abs"]) == spec["pred_len"]
            frames = h["history_orig_frames"] + y["future_orig_frames"]
            assert all(b - a == spec["expected_orig_frame_step"] for a, b in zip(frames, frames[1:]))
            assert all(Path(p).stem == "%06d" % frame for p, frame in zip(h["history_images"] + y["future_images"], frames))
            identity = {"schema": SCHEMA, "dataset": spec["dataset"], "split": split,
                        "video": h["video"], "csv": h["csv"], "segment": h["segment"], "agent": h["agent_id"],
                        "orig_frames": frames, "obs_len": spec["obs_len"], "pred_len": spec["pred_len"], "protocol": PROTOCOL}
            assert digest(identity) == sid
            for orig, xy, processed in zip(frames, h["history_abs"] + y["future_abs"], h["history_frames"] + y["future_frames"]):
                source = originals[h["csv"]][(h["agent_id"], orig)]
                assert xy == [float(source["x"]), float(source["y"])]
                assert processed == int(source["frame"])
                assert len(xy) == 2 and all(0 <= v <= 1 for v in xy)
            last = h["history_abs"][-1]
            relative = [[p[j] - last[j] for j in range(2)] for p in y["future_abs"]]
            negative_targets += int(any(v < 0 for p in relative for v in p))
            assert all(abs(r[j] + last[j] - p[j]) < 1e-12 for r, p in zip(relative, y["future_abs"]) for j in range(2))
        report[split] = {"samples": len(histories), "history_shape": [spec["obs_len"], 2],
                         "future_shape": [spec["pred_len"], 2], "samples_with_negative_future_rel": negative_targets}
    return {"dataset": spec["dataset"], "passed": True, "manifest_fingerprint": meta["fingerprint"], "splits": report}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="../data")
    p.add_argument("--manifests", required=True)
    p.add_argument("--output")
    a = p.parse_args()
    report = verify(resolve(a.dataset_root), resolve(a.manifests))
    if a.output:
        write_json(resolve(a.output), report)
    print(json.dumps(report, indent=2))
