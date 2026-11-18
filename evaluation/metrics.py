import numpy as np


def per_sample_metrics(predictions_abs, future_abs, image_wh=None):
    p, y = np.asarray(predictions_abs, dtype=np.float64), np.asarray(future_abs, dtype=np.float64)
    if p.ndim != 4 or y.shape != (p.shape[0], p.shape[2], 2) or p.shape[-1] != 2 or p.shape[1] < 1:
        raise ValueError("Expected predictions [N,K,H,2], targets [N,H,2]")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite predictions or targets")
    delta = p - y[:, None]
    suffix = ""
    if image_wh is not None:
        wh = np.asarray(image_wh, dtype=np.float64)
        if wh.shape != y.shape or not np.isfinite(wh).all() or (wh <= 0).any():
            raise ValueError("Actual PNG sizes must be [N,H,2], ordered W,H")
        delta = delta * wh[:, None]
        suffix = "_px"
    distance = np.linalg.norm(delta, axis=-1)
    ade, fde = distance.mean(-1), distance[..., -1]
    return {"ADE" + suffix: ade.mean(-1), "FDE" + suffix: fde.mean(-1),
            "minADE@K" + suffix: ade.min(-1), "minFDE@K" + suffix: fde.min(-1)}


def compute_metrics(predictions_abs, future_abs, image_wh=None):
    return {k: float(v.mean()) for k, v in per_sample_metrics(predictions_abs, future_abs, image_wh).items()}


def with_mean_k_aliases(values, k):
    """Expose concrete-K names without changing any reductions or legacy keys."""
    named = {}
    for old in ("ADE", "FDE", "ADE_px", "FDE_px"):
        if old in values:
            base, suffix = (old[:-3], "_px") if old.endswith("_px") else (old, "")
            named["mean%s@%d%s" % (base, k, suffix)] = values[old]
    for old in ("minADE@K", "minFDE@K", "minADE@K_px", "minFDE@K_px"):
        if old in values:
            named[old.replace("@K", "@%d" % k)] = values[old]
    named.update(values)  # Compatibility aliases follow the explicit names.
    return named
