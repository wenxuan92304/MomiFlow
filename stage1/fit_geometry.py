"""Fit future-trajectory prototypes using only train future_rel."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.cluster import KMeans
from stage1.geometry import train_future_rel, squared_trajectory_distances
from stage1.artifacts import base_metadata, save_npz, stage1_seed
from utils.config import parser, load_config, paths


def fit_geometry(c):
    if c["stage1"]["method"] != "geometry_prototypes":
        raise ValueError("fit_geometry requires stage1.method=geometry_prototypes")
    p = paths(c)["prototype_artifact"]
    if p.exists() or Path(str(p) + ".json").exists():
        raise FileExistsError("Prototype artifact already exists: " + str(p))
    ids, future_rel = train_future_rel(c)
    k = c["mode_supervision"]["num_modes"]
    if len(ids) < k:
        raise ValueError("Fewer train trajectories than prototypes")
    s = c["stage1"]
    km = KMeans(n_clusters=k, random_state=stage1_seed(c), n_init=s["prototype_n_init"],
                max_iter=s["prototype_max_iter"], tol=s["prototype_tol"])
    labels = km.fit_predict(future_rel.reshape(len(ids), -1))
    centers = np.asarray(km.cluster_centers_, dtype=np.float64).reshape(k, c["data"]["pred_len"], 2)
    distances = squared_trajectory_distances(future_rel, centers)
    nearest = distances.min(axis=1)
    variance = float(nearest.mean())
    counts = np.bincount(labels, minlength=k)
    if not np.isfinite(centers).all() or not np.isfinite(variance) or variance <= 0 or (counts == 0).any():
        raise ValueError("Degenerate geometry prototypes")
    center_distances = np.sqrt(squared_trajectory_distances(centers, centers))
    between_center_rms = float(center_distances[np.triu_indices(k, 1)].mean())
    meta = base_metadata(c)
    meta.update(component_order=list(range(k)), train_samples=len(ids), counts=counts.tolist(),
                mean_nearest_squared_trajectory_distance=variance,
                within_rms_by_component=[float(np.sqrt(nearest[labels == i].mean())) for i in range(k)],
                mean_between_center_rms=between_center_rms,
                center_separation_to_within_rms=between_center_rms / float(np.sqrt(variance)),
                kmeans_inertia=float(km.inertia_), kmeans_n_iter=int(km.n_iter_))
    save_npz(p, {"sample_ids": ids, "centers": centers,
                 "train_nearest_distance_mean": np.asarray(variance, dtype=np.float64)}, meta)
    print("Saved %d train-only future_rel prototypes to %s" % (k, p), flush=True)


if __name__ == "__main__":
    a = parser("Fit train-only prototypes on full future_rel trajectory geometry").parse_args()
    fit_geometry(load_config(a.config, a.set))
