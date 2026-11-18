"""Save soft prototype responsibilities for every train sample_id."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from stage1.geometry import train_future_rel, squared_trajectory_distances, soft_assignments
from stage1.artifacts import load_npz, save_npz, load_qgt
from utils.config import parser, load_config, paths
from utils.io import file_hash


def build_qgt_geometry(c):
    if c["stage1"]["method"] != "geometry_prototypes":
        raise ValueError("build_qgt_geometry requires stage1.method=geometry_prototypes")
    p = paths(c)
    out = p["qgt_artifact"]
    if out.exists() or Path(str(out) + ".json").exists():
        raise FileExistsError("q_GT artifact already exists: " + str(out))
    ids, future_rel = train_future_rel(c)
    arrays, meta = load_npz(p["prototype_artifact"], c)
    k = c["mode_supervision"]["num_modes"]
    if (arrays["centers"].shape != (k, c["data"]["pred_len"], 2)
            or arrays["sample_ids"].tolist() != ids.tolist()
            or meta.get("component_order") != list(range(k))):
        raise ValueError("Prototype shape, order or train sample_id lineage mismatch")
    distances = squared_trajectory_distances(future_rel, arrays["centers"])
    temperature = float(arrays["train_nearest_distance_mean"]) * c["stage1"]["prototype_soft_scale"]
    q = soft_assignments(distances, temperature, c["mode_supervision"]["probability_tolerance"])
    labels = q.argmax(axis=1)
    sorted_distances = np.sort(distances, axis=1)
    meta.pop("artifact_sha256")
    meta.update(prototype_sha256=file_hash(p["prototype_artifact"]), component_order=list(range(k)),
                soft_assignment="softmax_negative_squared_future_rel_trajectory_distance",
                soft_temperature=temperature, hard_assignment_counts=np.bincount(labels, minlength=k).tolist(),
                mean_first_second_squared_distance_gap=float((sorted_distances[:, 1] - sorted_distances[:, 0]).mean()),
                mean_max_probability=float(q.max(axis=1).mean()),
                mean_entropy=float(-(q.astype(np.float64) * np.log(q.astype(np.float64))).sum(axis=1).mean()))
    save_npz(out, {"sample_ids": ids, "q_gt": q}, meta)
    load_qgt(c, ids.tolist())
    print("Validated soft q_GT and complete train sample_id coverage: %d x %d" % q.shape, flush=True)


if __name__ == "__main__":
    a = parser("Build train-only soft future_rel prototype supervision").parse_args()
    build_qgt_geometry(load_config(a.config, a.set))
