"""Show real future frames and trajectories for saved Stage 1 modes.

This script only reads Stage 1 artifacts, train manifests, and their PNGs.
It never fits prototypes, changes q_GT, or runs the prediction model.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image

from stage1.artifacts import load_npz, load_qgt
from stage1.geometry import squared_trajectory_distances, train_future_rel
from utils.config import load_config, paths
from utils.io import file_hash, read_jsonl, resolve


SAMPLE_COLORS = ("#176aa1", "#db6d21")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="The dataset's Stage 1 geometry YAML")
    parser.add_argument("--output-dir", required=True, help="New, separate visualization directory")
    parser.add_argument("--samples-per-mode", type=int, default=2,
                        help="Exactly two real trajectories from distinct video segments")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for choosing the second representative video segment")
    return parser.parse_args()


def fragment_key(row):
    """Use the source video and its actual CSV segment, never a frame guess."""
    return row["video"], row.get("segment"), row["csv"]


def select_two(members, distances, mode, histories, ids, rng):
    """Nearest member, then a near-prototype member from another fragment."""
    ranked = sorted(members.tolist(), key=lambda i: (float(distances[i, mode]), ids[i]))
    nearest_by_fragment = {}
    for index in ranked:
        nearest_by_fragment.setdefault(fragment_key(histories[index]), index)
    candidates = list(nearest_by_fragment.values())
    if len(candidates) < 2:
        raise ValueError("Mode %d has fewer than two distinct video segments" % mode)
    # Restrict the random choice to the closest four *other* fragments so
    # both displayed examples remain reasonably representative of the mode.
    second_pool = candidates[1:5]
    second = second_pool[int(rng.integers(len(second_pool)))]
    return candidates[0], second


def fixed_crop(current, future, image_wh):
    """One trajectory-containing crop for all five frames of this sample."""
    width, height = image_wh
    pixels = np.concatenate((current[None], future), axis=0) * [width, height]
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    side = max(1.45 * float(np.max(high - low)), 0.34 * min(width, height))
    if side >= min(width, height):
        return (0.0, 0.0, 1.0, 1.0)
    middle = (low + high) / 2
    left = float(np.clip(middle[0] - side / 2, 0, width - side))
    top = float(np.clip(middle[1] - side / 2, 0, height - side))
    return (left / width, top / height,
            (left + side) / width, (top + side) / height)


def draw_future_strip(fig, cell, history, target, root, color, sample_number,
                      expected_steps, expected_frame_gap):
    """At future frame t, draw Current and only future points 1..t."""
    if (len(target["future_abs"]) != expected_steps
            or len(target["future_images"]) != expected_steps
            or len(target["future_wh"]) != expected_steps
            or len(target["future_orig_frames"]) != expected_steps):
        raise ValueError("Incomplete future frames/coordinates: " + history["sample_id"])
    current = np.asarray(history["history_abs"][-1], dtype=np.float64)
    future = np.asarray(target["future_abs"], dtype=np.float64)
    frame_ids = [history["history_orig_frames"][-1]] + target["future_orig_frames"]
    if any(b - a != expected_frame_gap for a, b in zip(frame_ids[:-1], frame_ids[1:])):
        raise ValueError("Unexpected original-frame gap: " + history["sample_id"])
    crop = fixed_crop(current, future, target["future_wh"][0])
    subgrid = cell.subgridspec(1, expected_steps, wspace=0.02)
    axes = []
    for step, (relative_image, expected_wh, frame_id) in enumerate(
            zip(target["future_images"], target["future_wh"], target["future_orig_frames"]), 1):
        parts = Path(relative_image).parts
        if (len(parts) != 3 or parts[0] != "frames" or parts[1] != history["video"]
                or parts[2] != "%06d.png" % frame_id):
            raise ValueError("Future PNG does not match the sample frame: " + relative_image)
        image_path = root / relative_image
        with Image.open(image_path) as original:
            if list(original.size) != expected_wh:
                raise ValueError("Future PNG dimensions changed: " + str(image_path))
            original_width, original_height = original.size
            left, top, right, bottom = (
                int(round(crop[0] * original_width)),
                int(round(crop[1] * original_height)),
                int(round(crop[2] * original_width)),
                int(round(crop[3] * original_height)),
            )
            frame = original.crop((left, top, right, bottom)).convert("RGB")
            frame.thumbnail((512, 512), Image.Resampling.LANCZOS)
            width, height = frame.size
            pixels = np.asarray(frame)
        ax = fig.add_subplot(subgrid[0, step - 1])
        ax.imshow(pixels, interpolation="nearest")
        path = np.concatenate((current[None], future[:step]), axis=0)
        path_px = (path * [original_width, original_height] - [left, top])
        path_px *= [width / (right - left), height / (bottom - top)]
        ax.plot(path_px[:, 0], path_px[:, 1], color=color, linewidth=1.55,
                marker="o", markersize=2.8, markeredgecolor="white",
                markeredgewidth=0.3, zorder=3)
        ax.scatter([path_px[0, 0]], [path_px[0, 1]], marker="+", s=20,
                   linewidths=1.0, color="black", zorder=4)
        ax.text(0.05, 0.95, "t%d" % step, ha="left", va="top", transform=ax.transAxes,
                fontsize=7, color="white", bbox={"facecolor": "black", "alpha": 0.55,
                                               "edgecolor": "none", "pad": 1.5})
        if step == 1:
            ax.text(0.05, 0.05, "S%d" % sample_number, ha="left", va="bottom",
                    transform=ax.transAxes, fontsize=7, color="white",
                    bbox={"facecolor": color, "alpha": 0.9, "edgecolor": "none", "pad": 1.5})
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        axes.append(ax)
    return axes, crop


def main():
    args = parse_args()
    if args.samples_per_mode != 2 or args.seed < 0:
        raise ValueError("This layout requires exactly two samples per mode and a nonnegative seed")
    config = load_config(args.config)
    if config["stage1"]["method"] != "geometry_prototypes":
        raise ValueError("Use a geometry_prototypes Stage 1 configuration")
    source = paths(config)
    for key in ("prototype_artifact", "qgt_artifact"):
        if not source[key].is_file() or not Path(str(source[key]) + ".json").is_file():
            raise FileNotFoundError("Existing Stage 1 artifact and metadata required: " + str(source[key]))

    # Reuse exactly the future_rel representation and membership checks from
    # the Stage 1 fitter; do not fit or alter any Stage 1 artifact.
    sample_ids, future_rel = train_future_rel(config)
    ids = sample_ids.tolist()
    prototypes, prototype_meta = load_npz(source["prototype_artifact"], config)
    centers = prototypes["centers"]
    k = centers.shape[0]
    if (centers.ndim != 3 or centers.shape[1:] != future_rel.shape[1:]
            or k != config["mode_supervision"]["num_modes"]
            or prototypes["sample_ids"].tolist() != ids):
        raise ValueError("Saved prototypes do not match the train trajectories")
    q_by_id = load_qgt(config, ids)
    q = np.stack([q_by_id[sid] for sid in ids])
    distances = squared_trajectory_distances(future_rel, centers)
    labels = distances.argmin(axis=1)
    if not np.array_equal(labels, q.argmax(axis=1)):
        raise ValueError("Saved q_GT argmax differs from nearest saved prototype")
    if np.bincount(labels, minlength=k).tolist() != prototype_meta["counts"]:
        raise ValueError("Saved KMeans member counts differ from reconstructed train members")

    history_rows = read_jsonl(source["manifests"] / "train_history.jsonl")
    target_rows = read_jsonl(source["manifests"] / "train_targets.jsonl")
    if ([row["sample_id"] for row in history_rows] != ids
            or [row["sample_id"] for row in target_rows] != ids):
        raise ValueError("Train image/trajectory sample IDs do not match the Stage 1 order")
    output_dir = resolve(args.output_dir).resolve()
    stage1_dir = source["output"].resolve()
    if output_dir == stage1_dir or stage1_dir in output_dir.parents:
        raise ValueError("Visualization output must be separate from Stage 1 artifacts")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Output directory must be new or empty: " + str(output_dir))

    rng = np.random.default_rng(args.seed)
    panels = []
    for mode in range(k):
        members = np.flatnonzero(labels == mode)
        if not len(members):
            raise ValueError("Saved prototype has no train members: " + str(mode))
        first, second = select_two(members, distances, mode, history_rows, ids, rng)
        panels.append({"mode": mode, "member_count": int(len(members)),
                       "member_fraction": float(len(members) / len(ids)),
                       "selected_indices": (first, second)})

    # Four columns for the current four-mode Stage 1. Other saved K values
    # remain viewable in successive groups without assuming a fixed K.
    columns = min(4, k)
    groups = math.ceil(k / columns)
    fig = plt.figure(figsize=(4.8 * columns, 7.0 * groups))
    grid = fig.add_gridspec(3 * groups, columns, height_ratios=[1.5, 1.5, 3.6] * groups,
                           left=0.035, right=0.99, top=0.88, bottom=0.17,
                           wspace=0.08, hspace=0.12)
    root = source["dataset_root"] / config["data"]["dataset_name"]
    pred_len = config["data"]["pred_len"]
    frame_gap = config["data"]["expected_orig_frame_step"]
    shared_limit = max(float(np.max(np.abs(future_rel[list(panel["selected_indices"])])))
                       for panel in panels) * 1.12
    shared_limit = max(shared_limit, 1e-5)
    selected_records = []
    for panel in panels:
        mode = panel["mode"]
        row, column = 3 * (mode // columns), mode % columns
        samples = []
        first_strip = None
        for slot, index in enumerate(panel["selected_indices"]):
            history, target = history_rows[index], target_rows[index]
            axes, crop = draw_future_strip(fig, grid[row + slot, column], history, target,
                                           root, SAMPLE_COLORS[slot], slot + 1,
                                           pred_len, frame_gap)
            if slot == 0:
                first_strip = axes
            samples.append({"sample_number": slot + 1, "sample_id": ids[index],
                            "video": history["video"], "segment": history.get("segment"),
                            "csv": history["csv"], "agent_id": history["agent_id"],
                            "squared_trajectory_distance": float(distances[index, mode]),
                            "last_history_abs": history["history_abs"][-1],
                            "last_history_orig_frame": history["history_orig_frames"][-1],
                            "future_rel": future_rel[index].tolist(),
                            "future_abs": target["future_abs"],
                            "future_orig_frames": target["future_orig_frames"],
                            "future_images": target["future_images"],
                            "future_wh": target["future_wh"],
                            "display_crop_normalized_xyxy": list(crop),
                            "color": SAMPLE_COLORS[slot]})
        if fragment_key(history_rows[panel["selected_indices"][0]]) == fragment_key(
                history_rows[panel["selected_indices"][1]]):
            raise AssertionError("Selected samples share the same video segment")
        left = first_strip[0].get_position().x0
        right = first_strip[-1].get_position().x1
        top = max(ax.get_position().y1 for ax in first_strip)
        fig.text((left + right) / 2, top + 0.012,
                 "Mode %d" % mode,
                 ha="center", va="bottom", fontsize=10)

        ax = fig.add_subplot(grid[row + 2, column])
        for slot, index in enumerate(panel["selected_indices"]):
            path = np.concatenate((np.zeros((1, 2)), future_rel[index]), axis=0)
            ax.plot(path[:, 0], path[:, 1], color=SAMPLE_COLORS[slot],
                    linewidth=1.8, marker="o", markersize=3.2, label="Sample %d" % (slot + 1))
        ax.scatter([0], [0], marker="+", s=65, linewidths=1.4,
                   color="black", zorder=5)
        ax.set(xlim=(-shared_limit, shared_limit),
               ylim=(-shared_limit, shared_limit), xlabel="Relative x")
        ax.set_ylabel("Relative y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="0.88", linewidth=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(labelsize=7)
        selected_records.append({"mode": mode, "member_count": panel["member_count"],
                                 "member_fraction": panel["member_fraction"],
                                 "representative_sample_id": samples[0]["sample_id"],
                                 "representative_squared_trajectory_distance":
                                     samples[0]["squared_trajectory_distance"],
                                 "samples": samples})

    title_dataset = {"cholectrack20": "CholecTrack20", "phakir": "PhaKIR"}.get(
        config["data"]["dataset_name"], config["data"]["dataset_name"])
    fig.suptitle("Future Motion Modes in Surgical Video (%s)" % title_dataset,
                 fontsize=13, y=0.975)
    legend = [
        Line2D([], [], color=color, lw=1.8, marker="o", markersize=3.2,
               label="Sample %d" % (slot + 1))
        for slot, color in enumerate(SAMPLE_COLORS)
    ]
    legend.append(Line2D([], [], color="black", linestyle="none", marker="+",
                         markersize=7, label="Current"))
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=3, frameon=False, fontsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_dir / ("stage1_motion_modes." + extension), dpi=300)
    plt.close(fig)
    record = {
        "dataset": config["data"]["dataset_name"],
        "stage1_method": "geometry_prototypes", "split": "train",
        "coordinate_definition": "future_abs - last_history_abs; origin is last_history_abs",
        "membership_rule": "nearest saved prototype; verified against saved q_GT argmax",
        "selection_rule": "nearest member, plus a seeded choice among the closest four other video segments' nearest members",
        "seed": args.seed, "samples_per_mode_requested": args.samples_per_mode,
        "shared_axis_limit": [-shared_limit, shared_limit],
        "visualization_layout": {"columns": columns, "rows": 3 * groups,
                                 "top_rows": "two samples, five actual future PNGs each, cumulative GT overlay",
                                 "bottom_row": "both future_rel paths; shared x/y limits and equal aspect",
                                 "image_display": "one fixed trajectory-containing crop per sample, resized to at most 512 pixels per side"},
        "prototype_artifact": str(source["prototype_artifact"]),
        "prototype_sha256": file_hash(source["prototype_artifact"]),
        "qgt_artifact": str(source["qgt_artifact"]),
        "qgt_sha256": file_hash(source["qgt_artifact"]),
        "stage1_signature": prototype_meta["stage1_signature"],
        "modes": selected_records,
    }
    (output_dir / "selected_samples.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved Stage 1 mode figures and sample IDs to " + str(output_dir))


if __name__ == "__main__":
    main()
