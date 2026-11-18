"""Draw Fig. 4 from saved Stage 1 artifacts and saved visualization selections.

The saved representative and background sample IDs are used unchanged. This
script does not refit KMeans, choose new members, or alter any Stage 1 output.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import MaxNLocator

from stage1.artifacts import load_npz, load_qgt
from stage1.geometry import squared_trajectory_distances, train_future_rel
from utils.config import load_config, paths
from utils.io import file_hash, resolve


DATASETS = (("cholectrack20", "CholecTrack20"), ("phakir", "PhaKIR"))
MODE_COLORS = ("#2677a6", "#e58a38", "#4aa56b", "#d65b60")


class FourModeLinesHandler(HandlerBase):
    """Draw all four mode colors within one legend symbol."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        count = len(orig_handle)
        gap = width * 0.015
        return [Line2D([-xdescent + width * i / count + gap,
                        -xdescent + width * (i + 1) / count - gap],
                       [-ydescent + height / 2] * 2,
                       color=line.get_color(), linewidth=line.get_linewidth(),
                       alpha=line.get_alpha(), transform=trans)
                for i, line in enumerate(orig_handle)]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chole-config", required=True)
    parser.add_argument("--chole-selection", required=True,
                        help="Existing mode-visualization selected_samples.json")
    parser.add_argument("--phakir-config", required=True)
    parser.add_argument("--phakir-selection", required=True)
    parser.add_argument("--output-dir", required=True, help="New, empty output directory")
    return parser.parse_args()


def load_dataset(config_file, selection_file, expected_dataset):
    config = load_config(config_file)
    if (config["data"]["dataset_name"] != expected_dataset
            or config["stage1"]["method"] != "geometry_prototypes"):
        raise ValueError("Wrong geometry Stage 1 config for " + expected_dataset)
    source = paths(config)
    selection_file = resolve(selection_file).resolve()
    selected = json.loads(selection_file.read_text(encoding="utf-8"))
    if (selected["dataset"] != expected_dataset or selected["split"] != "train"
            or selected["coordinate_definition"]
            != "future_abs - last_history_abs; origin is last_history_abs"):
        raise ValueError("Selection dataset, split, or coordinates differ")
    for key, checksum in (("prototype_artifact", "prototype_sha256"),
                          ("qgt_artifact", "qgt_sha256")):
        if not source[key].is_file() or file_hash(source[key]) != selected[checksum]:
            raise ValueError("Saved selection and Stage 1 artifact differ: " + key)

    sample_ids, future_rel = train_future_rel(config)
    ids = sample_ids.tolist()
    index_of = {sid: index for index, sid in enumerate(ids)}
    prototypes, metadata = load_npz(source["prototype_artifact"], config)
    centers = prototypes["centers"]
    modes = len(centers)
    if (centers.ndim != 3 or centers.shape[1:] != future_rel.shape[1:]
            or prototypes["sample_ids"].tolist() != ids
            or selected["stage1_signature"] != metadata["stage1_signature"]
            or [m["mode"] for m in selected["modes"]] != list(range(modes))
            or [p["mode"] for p in selected["zoomed_prototypes"]] != list(range(modes))):
        raise ValueError("Selected modes do not match saved prototypes and training windows")
    q_by_id = load_qgt(config, ids)
    distances = squared_trajectory_distances(future_rel, centers)
    assignments = distances.argmin(axis=1)
    if not np.array_equal(assignments, np.asarray([q_by_id[sid].argmax() for sid in ids])):
        raise ValueError("Saved q_GT and nearest-prototype assignments disagree")
    counts = np.bincount(assignments, minlength=modes).tolist()
    if counts != metadata["counts"]:
        raise ValueError("Mode member counts differ from saved Stage 1 metadata")

    for mode, (entry, zoom) in enumerate(zip(selected["modes"], selected["zoomed_prototypes"])):
        if entry["member_count"] != counts[mode]:
            raise ValueError("Saved mode count changed: " + str(mode))
        if not np.allclose(zoom["prototype_future_rel"], centers[mode], rtol=0, atol=1e-8):
            raise ValueError("Saved zoomed prototype differs from KMeans center")
        representative = entry["representative_sample_id"]
        if representative not in index_of or assignments[index_of[representative]] != mode:
            raise ValueError("Saved nearest real member is missing or in another mode")
        members = np.flatnonzero(assignments == mode)
        nearest = min(members.tolist(), key=lambda i: (float(distances[i, mode]), ids[i]))
        if ids[nearest] != representative:
            raise ValueError("Saved nearest real member no longer matches Stage 1 data")
        background = entry["random_sample_ids"]
        if (len(background) != selected["samples_per_mode_requested"]
                or len(set(background)) != len(background)
                or any(sid not in index_of or assignments[index_of[sid]] != mode
                       for sid in background)):
            raise ValueError("Saved background member IDs are incomplete or misassigned")

    limit = selected["shared_axis_limit"]
    if len(limit) != 2 or limit[0] >= limit[1]:
        raise ValueError("Invalid saved shared axis range")
    return {"name": expected_dataset, "selection_file": selection_file,
            "selected": selected, "centers": centers, "future_rel": future_rel,
            "index_of": index_of, "mode_count": modes, "axis_limit": limit}


def color_for(mode):
    return MODE_COLORS[mode] if mode < len(MODE_COLORS) else plt.get_cmap("tab10")(mode % 10)


def trajectory_with_origin(future):
    return np.concatenate((np.zeros((1, 2)), np.asarray(future, dtype=np.float64)), axis=0)


def format_axis(ax, xlim, ylim, ylabel=False):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="0.88", linewidth=0.45)
    ax.tick_params(labelsize=6.4, pad=1.3, length=2.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    if ylabel:
        ax.set_ylabel("Relative y", fontsize=7.2)


def draw_prototype(ax, dataset, mode):
    zoom = dataset["selected"]["zoomed_prototypes"][mode]
    path = trajectory_with_origin(dataset["centers"][mode])
    ax.plot(path[:, 0], path[:, 1], color="#303030", linewidth=1.75, zorder=3)
    ax.scatter([0], [0], marker="+", color="black", s=32, linewidths=1.1, zorder=5)
    ax.scatter([path[-1, 0]], [path[-1, 1]], marker="D", color="#ef8a2c",
               s=20, linewidths=0, zorder=6)
    format_axis(ax, zoom["xlim"], zoom["ylim"], ylabel=True)
    ax.set_title("Mode %d (Zoomed)" % mode, fontsize=7.6, pad=2)


def draw_distribution(ax, dataset, mode):
    entry = dataset["selected"]["modes"][mode]
    color = color_for(mode)
    for sid in entry["random_sample_ids"]:
        path = trajectory_with_origin(dataset["future_rel"][dataset["index_of"][sid]])
        ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.20,
                linewidth=0.55, zorder=1)
    representative = entry["representative_sample_id"]
    path = trajectory_with_origin(dataset["future_rel"][dataset["index_of"][representative]])
    ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.45, zorder=3)
    ax.scatter([0], [0], marker="+", color="black", s=23, linewidths=0.9, zorder=5)
    format_axis(ax, dataset["axis_limit"], dataset["axis_limit"], ylabel=True)
    ax.set_title("Mode %d (n=%d)" % (mode, entry["member_count"]), fontsize=7.6, pad=2)


def main():
    args = arguments()
    data = (load_dataset(args.chole_config, args.chole_selection, DATASETS[0][0]),
            load_dataset(args.phakir_config, args.phakir_selection, DATASETS[1][0]))
    output = resolve(args.output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Output directory must be new or empty: " + str(output))
    for dataset in data:
        if output == dataset["selection_file"].parent or dataset["selection_file"].parent in output.parents:
            raise ValueError("Output must be separate from saved visualizations")

    columns = max(dataset["mode_count"] for dataset in data)
    fig = plt.figure(figsize=(1.83 * columns, 9.3), facecolor="white")
    grid = fig.add_gridspec(4, columns, left=0.08, right=0.99, top=0.928,
                            bottom=0.105, hspace=0.75, wspace=0.25)
    for dataset_index, dataset in enumerate(data):
        zoom_row, distribution_row = 2 * dataset_index, 2 * dataset_index + 1
        for mode in range(dataset["mode_count"]):
            top = fig.add_subplot(grid[zoom_row, mode])
            bottom = fig.add_subplot(grid[distribution_row, mode])
            draw_prototype(top, dataset, mode)
            draw_distribution(bottom, dataset, mode)
        for row, heading in ((zoom_row, "Representative Motion Modes Learned by K-means"),
                             (distribution_row, "Motion Mode Distribution")):
            y = grid[row, 0].get_position(fig).y1 + 0.036
            fig.text(0.5, y, "%s (%s)" % (heading, DATASETS[dataset_index][1]),
                     ha="center", va="bottom", fontsize=8.7, fontweight="semibold")
        for row in (zoom_row, distribution_row):
            y = grid[row, 0].get_position(fig).y0 - 0.025
            fig.text(0.535, y, "Relative x", ha="center", va="top", fontsize=7.2)

    training_colors = tuple(Line2D([], [], color=color, lw=0.6, alpha=0.34)
                            for color in MODE_COLORS)
    nearest_colors = tuple(Line2D([], [], color=color, lw=1.3)
                           for color in MODE_COLORS)
    legend = [
        training_colors,
        nearest_colors,
        Line2D([], [], color="#303030", lw=1.75),
        Line2D([], [], color="black", linestyle="none", marker="+", markersize=5),
        Line2D([], [], color="#ef8a2c", linestyle="none", marker="D", markersize=4),
    ]
    fig.legend(handles=legend,
               labels=["Training trajectories", "Nearest real trajectory",
                       "K-means prototype", "Current", "Endpoint"],
               handler_map={tuple: FourModeLinesHandler()},
               loc="lower center", bbox_to_anchor=(0.5, 0.033),
               ncol=5, frameon=False, fontsize=6.7, handlelength=2.5,
               columnspacing=1.15)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(output / ("fig4_motion_modes." + ext),
                    dpi=600 if ext == "png" else 300, facecolor="white")
    plt.close(fig)
    record = {"source": "saved Stage 1 prototypes and saved visualization selections",
              "row_order": ["CholecTrack20: zoomed prototypes", "CholecTrack20: member distribution",
                            "PhaKIR: zoomed prototypes", "PhaKIR: member distribution"],
              "datasets": {}}
    for dataset in data:
        selected = dataset["selected"]
        record["datasets"][dataset["name"]] = {
            "selection_file": str(dataset["selection_file"]),
            "selection_sha256": file_hash(dataset["selection_file"]),
            "prototype_sha256": selected["prototype_sha256"],
            "qgt_sha256": selected["qgt_sha256"],
            "modes": [{"mode": m["mode"], "member_count": m["member_count"],
                       "representative_sample_id": m["representative_sample_id"],
                       "background_sample_ids": m["random_sample_ids"]}
                      for m in selected["modes"]],
        }
    (output / "fig4_sources.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved Fig. 4 to " + str(output))


if __name__ == "__main__":
    main()
