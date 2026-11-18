"""Read-only CSV adapter. This module intentionally requires only the stdlib.

History manifests contain no future coordinates or q_GT. Targets live in a
separate file, joined only by training views or the post-generation evaluator.
"""
import csv
import math
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path

from utils.io import digest, file_hash, write_json, write_jsonl, read_json

SCHEMA = 1
PROTOCOL = {
    "history": "absolute_normalized_xy",
    "target": "future_relative_to_last_history",
    "reference": "last_history_point",
    "image_and_continuity_index": "orig_frame",
}


def png_size(path):
    with open(path, "rb") as f:
        header = f.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("Invalid PNG header: %s" % path)
    w, h = struct.unpack(">II", header[16:24])
    if w <= 0 or h <= 0:
        raise ValueError("Invalid image dimensions: %s" % path)
    return [w, h]


def build_manifest(dataset_root, dataset_name, output_dir, obs_len=5, pred_len=5,
                   sample_stride=1, expected_orig_frame_step=25, check_headers=True):
    if dataset_name not in ("cholectrack20", "phakir"):
        raise ValueError("Unsupported dataset: " + dataset_name)
    if min(obs_len, pred_len, sample_stride, expected_orig_frame_step) <= 0:
        raise ValueError("Lengths, stride and orig_frame step must be positive")
    root = Path(dataset_root) / dataset_name
    out = Path(output_dir)
    sources, split_videos, report, manifests, targets = {}, {}, {}, {}, {}
    sizes = {}
    for split in ("train", "val", "test"):
        files = sorted((root / "csv" / split).glob("*.csv"))
        if not files:
            raise FileNotFoundError(root / "csv" / split)
        histories, futures, invalid = [], [], []
        videos, frame_keys, dims = set(), set(), Counter()
        n_rows = n_groups = n_gaps = 0
        for path in files:
            rel_csv = path.relative_to(root).as_posix()
            sources[rel_csv] = file_hash(path)
            match = re.fullmatch(r"(Video_\d+)_seg(\d+)", path.stem)
            if dataset_name == "phakir" and match is None:
                raise ValueError("Unexpected Phakir CSV name: " + path.name)
            video = match.group(1) if match else path.stem
            segment = match.group(2) if match else None
            videos.add(video)
            groups = defaultdict(list)
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not {"agent_id", "frame", "orig_frame", "x", "y"}.issubset(reader.fieldnames or []):
                    raise ValueError("Missing required columns: " + str(path))
                for line, row in enumerate(reader, 2):
                    n_rows += 1
                    if match and int(row["segment_id"]) != int(segment):
                        raise ValueError("Segment mismatch: " + str(path))
                    if not match and row.get("video", video).lower() != video.lower():
                        raise ValueError("Video mismatch: " + str(path))
                    point = [float(row["x"]), float(row["y"])]
                    valid = all(math.isfinite(v) and 0 <= v <= 1 for v in point)
                    item = {"frame": int(row["frame"]), "orig_frame": int(row["orig_frame"]),
                            "xy": point, "valid": valid, "line": line}
                    groups[row["agent_id"]].append(item)
                    frame_keys.add((video, item["orig_frame"]))
                    if not valid:
                        invalid.append({"csv": rel_csv, "line": line, "agent_id": row["agent_id"],
                                        "orig_frame": item["orig_frame"], "reason": "nonfinite_or_outside_unit_square"})
            for agent, points in sorted(groups.items()):
                n_groups += 1
                points.sort(key=lambda p: p["orig_frame"])
                if len({p["orig_frame"] for p in points}) != len(points):
                    raise ValueError("Duplicate agent/orig_frame: %s %s" % (path, agent))
                runs, run = [], []
                for p in points:
                    if run and p["orig_frame"] - run[-1]["orig_frame"] != expected_orig_frame_step:
                        runs.append(run)
                        run = []
                        n_gaps += 1
                    run.append(p)
                if run:
                    runs.append(run)
                for run in runs:
                    for start in range(0, len(run) - obs_len - pred_len + 1, sample_stride):
                        window = run[start:start + obs_len + pred_len]
                        if not all(p["valid"] for p in window):
                            continue
                        orig = [p["orig_frame"] for p in window]
                        identity = {"schema": SCHEMA, "dataset": dataset_name, "split": split,
                                    "video": video, "csv": rel_csv, "segment": segment, "agent": agent,
                                    "orig_frames": orig, "obs_len": obs_len, "pred_len": pred_len,
                                    "protocol": PROTOCOL}
                        sid = digest(identity)
                        image_paths = ["frames/%s/%06d.png" % (video, i) for i in orig]
                        for image_path in image_paths:
                            if image_path not in sizes:
                                full_path = root / image_path
                                if not full_path.is_file():
                                    raise FileNotFoundError(full_path)
                                sizes[image_path] = png_size(full_path) if check_headers else None
                        history = {"sample_id": sid, "dataset": dataset_name, "split": split,
                                   "video": video, "csv": rel_csv, "segment": segment, "agent_id": agent,
                                   "history_orig_frames": orig[:obs_len],
                                   "history_frames": [p["frame"] for p in window[:obs_len]],
                                   "history_abs": [p["xy"] for p in window[:obs_len]],
                                   "history_images": image_paths[:obs_len],
                                   "history_wh": [sizes[p] for p in image_paths[:obs_len]]}
                        future = {"sample_id": sid, "future_orig_frames": orig[obs_len:],
                                  "future_frames": [p["frame"] for p in window[obs_len:]],
                                  "future_abs": [p["xy"] for p in window[obs_len:]],
                                  "future_images": image_paths[obs_len:],
                                  "future_wh": [sizes[p] for p in image_paths[obs_len:]]}
                        histories.append(history)
                        futures.append(future)
        histories.sort(key=lambda r: r["sample_id"])
        futures.sort(key=lambda r: r["sample_id"])
        if len({r["sample_id"] for r in histories}) != len(histories):
            raise ValueError("Duplicate sample IDs")
        for key, wh in sizes.items():
            if key.split("/")[1] in videos and wh:
                dims["%dx%d" % tuple(wh)] += 1
        split_videos[split] = videos
        report[split] = {"csv_files": len(files), "rows": n_rows, "agent_groups": n_groups,
                         "gaps": n_gaps, "samples": len(histories), "invalid_rows": invalid,
                         "videos": sorted(videos), "referenced_unique_frames": len(frame_keys),
                         "window_image_dimensions": dict(dims)}
        manifests[split], targets[split] = histories, futures
        print("%s %s: %d rows, %d windows" % (dataset_name, split, n_rows, len(histories)), flush=True)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_videos[a] & split_videos[b]:
            raise ValueError("Video leakage between %s and %s" % (a, b))
    spec = {"schema": SCHEMA, "dataset": dataset_name, "obs_len": obs_len,
            "pred_len": pred_len, "sample_stride": sample_stride,
            "expected_orig_frame_step": expected_orig_frame_step, "protocol": PROTOCOL,
            "csv_sha256": sources, "headers_checked": check_headers}
    for split in manifests:
        write_jsonl(out / (split + "_history.jsonl"), manifests[split])
        write_jsonl(out / (split + "_targets.jsonl"), targets[split])
    hashes = {p.name: file_hash(p) for p in sorted(out.glob("*_*.jsonl"))}
    meta = {"spec": spec, "files": hashes, "report": report,
            "fingerprint": digest({"spec": spec, "files": hashes})}
    write_json(out / "metadata.json", meta)
    return meta


def verify_manifest(output_dir, data_cfg, check_sources=True, splits=("train", "val", "test")):
    """Check immutable CSV fingerprints, protocol and artifact integrity."""
    out = Path(output_dir)
    meta = read_json(out / "metadata.json")
    spec = meta["spec"]
    for key in ("obs_len", "pred_len", "sample_stride", "expected_orig_frame_step"):
        if spec[key] != data_cfg[key]:
            raise ValueError("Stale manifest: " + key)
    if spec["dataset"] != data_cfg["dataset_name"] or spec["protocol"] != PROTOCOL:
        raise ValueError("Manifest protocol/dataset mismatch")
    if not spec["headers_checked"]:
        raise ValueError("Training requires a manifest with actual PNG dimensions")
    if digest({"spec": spec, "files": meta["files"]}) != meta["fingerprint"]:
        raise ValueError("Corrupt manifest fingerprint")
    for name, sha in meta["files"].items():
        if name.split("_")[0] not in splits:
            continue
        if file_hash(out / name) != sha:
            raise ValueError("Modified manifest: " + name)
    if check_sources:
        root = Path(data_cfg["dataset_root"]) / spec["dataset"]
        current = {p.relative_to(root).as_posix(): file_hash(p)
                   for split in splits for p in sorted((root / "csv" / split).glob("*.csv"))}
        expected_sources = {key: value for key, value in spec["csv_sha256"].items() if key.split("/")[1] in splits}
        if current != expected_sources:
            raise ValueError("CSV files changed; rebuild manifests and Stage1 artifacts")
    return meta
