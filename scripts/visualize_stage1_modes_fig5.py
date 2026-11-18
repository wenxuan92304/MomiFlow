"""Combine the two saved Stage 1 mode figures into the paper's Fig. 5 layout.

The selected_samples.json files are inputs, not outputs of a new selection.
No clustering, prototype distances, model inference, or sample ranking is run.
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
from matplotlib.ticker import MaxNLocator

from scripts.visualize_stage1_modes import SAMPLE_COLORS, draw_future_strip, fragment_key
from stage1.artifacts import load_npz, load_qgt
from utils.config import load_config, paths
from utils.io import file_hash, read_jsonl, resolve


DATASETS = (
    ("cholectrack20", "CholecTrack20"),
    ("phakir", "PhaKIR"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chole-config", required=True)
    parser.add_argument("--chole-selection", required=True,
                        help="Existing selected_samples.json; keeps its exact two samples per mode")
    parser.add_argument("--phakir-config", required=True)
    parser.add_argument("--phakir-selection", required=True)
    parser.add_argument("--output-dir", required=True, help="New, empty figure directory")
    return parser.parse_args()


def read_dataset(config_path, selection_path, dataset_name):
    config = load_config(config_path)
    if config["data"]["dataset_name"] != dataset_name or config["stage1"]["method"] != "geometry_prototypes":
        raise ValueError("Expected the %s geometry-prototype Stage 1 configuration" % dataset_name)
    source = paths(config)
    selected_path = resolve(selection_path).resolve()
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected["dataset"] != dataset_name or selected["split"] != "train":
        raise ValueError("Selection dataset/split mismatch: " + str(selected_path))
    if selected["coordinate_definition"] != "future_abs - last_history_abs; origin is last_history_abs":
        raise ValueError("Unexpected selected-sample coordinate definition")
    for key, hash_key in (("prototype_artifact", "prototype_sha256"),
                          ("qgt_artifact", "qgt_sha256")):
        if not source[key].is_file() or selected[hash_key] != file_hash(source[key]):
            raise ValueError("Selection does not match saved Stage 1 artifact: " + key)
    prototypes, meta = load_npz(source["prototype_artifact"], config)
    centers = prototypes["centers"]
    num_modes = len(centers)
    if selected["stage1_signature"] != meta["stage1_signature"]:
        raise ValueError("Stage 1 signature differs from the saved selection")
    if [item["mode"] for item in selected["modes"]] != list(range(num_modes)):
        raise ValueError("Saved selection does not cover every saved prototype in order")

    histories = read_jsonl(source["manifests"] / "train_history.jsonl")
    targets = read_jsonl(source["manifests"] / "train_targets.jsonl")
    ids = [row["sample_id"] for row in histories]
    if (ids != [row["sample_id"] for row in targets]
            or ids != prototypes["sample_ids"].tolist() or len(set(ids)) != len(ids)):
        raise ValueError("Train manifest and saved prototype sample IDs differ")
    by_id = {sid: i for i, sid in enumerate(ids)}
    q_by_id = load_qgt(config, ids)
    count_by_mode = np.bincount([int(q_by_id[sid].argmax()) for sid in ids],
                                minlength=num_modes).tolist()
    if count_by_mode != meta["counts"]:
        raise ValueError("Saved q_GT mode counts differ from the prototype metadata")

    for item in selected["modes"]:
        mode = item["mode"]
        if item["member_count"] != count_by_mode[mode] or len(item["samples"]) != 2:
            raise ValueError("Saved member count or representative count differs for mode %d" % mode)
        pair = item["samples"]
        if pair[0]["sample_id"] == pair[1]["sample_id"]:
            raise ValueError("Both representatives have the same sample ID")
        indices = []
        for slot, sample in enumerate(pair, 1):
            if sample["sample_number"] != slot:
                raise ValueError("Saved representative order changed")
            sid = sample["sample_id"]
            if sid not in by_id or int(q_by_id[sid].argmax()) != mode:
                raise ValueError("Representative is absent or belongs to another mode: " + sid)
            index = by_id[sid]
            history, target = histories[index], targets[index]
            for key in ("video", "segment", "csv", "agent_id"):
                if sample.get(key) != history.get(key):
                    raise ValueError("Representative provenance changed: " + sid + ":" + key)
            for key in ("future_abs", "future_orig_frames", "future_images", "future_wh"):
                if sample[key] != target[key]:
                    raise ValueError("Representative future/frame record changed: " + sid + ":" + key)
            if (sample["last_history_abs"] != history["history_abs"][-1]
                    or sample["last_history_orig_frame"] != history["history_orig_frames"][-1]):
                raise ValueError("Representative history endpoint changed: " + sid)
            current = np.asarray(history["history_abs"][-1], dtype=np.float32)
            future = np.asarray(target["future_abs"], dtype=np.float32)
            future_rel = (future - current).astype(np.float64)
            if not np.allclose(future_rel, sample["future_rel"], rtol=0, atol=1e-7):
                raise ValueError("Saved representative future_rel differs from train data: " + sid)
            indices.append(index)
        if fragment_key(histories[indices[0]]) == fragment_key(histories[indices[1]]):
            raise ValueError("Representatives share a video segment: mode %d" % mode)
        item["indices"] = indices

    return {"config": config, "source": source, "selected": selected,
            "selection_path": selected_path, "histories": histories,
            "targets": targets, "num_modes": num_modes}


def draw_dataset(fig, block, dataset, display_name, columns):
    config, source, selected = dataset["config"], dataset["source"], dataset["selected"]
    rows = block.subgridspec(3, columns, height_ratios=(1.05, 1.05, 2.85),
                             wspace=0.08, hspace=0.12)
    root = source["dataset_root"] / config["data"]["dataset_name"]
    shared_limit = float(selected["shared_axis_limit"][1])
    if (selected["shared_axis_limit"][0] != -shared_limit or shared_limit <= 0):
        raise ValueError("Invalid shared coordinate range in saved selection")
    for item in selected["modes"]:
        mode = item["mode"]
        first_axes = None
        for slot, index in enumerate(item["indices"]):
            axes, crop = draw_future_strip(
                fig, rows[slot, mode], dataset["histories"][index],
                dataset["targets"][index], root, SAMPLE_COLORS[slot], slot + 1,
                config["data"]["pred_len"], config["data"]["expected_orig_frame_step"])
            if not np.allclose(crop, item["samples"][slot]["display_crop_normalized_xyxy"],
                               rtol=0, atol=1e-6):
                raise ValueError("The existing representative's video crop changed")
            if slot == 0:
                first_axes = axes
        left = first_axes[0].get_position().x0
        right = first_axes[-1].get_position().x1
        top = max(ax.get_position().y1 for ax in first_axes)
        fig.text((left + right) / 2, top + 0.005, "Mode %d" % mode,
                 ha="center", va="bottom", fontsize=9)

        ax = fig.add_subplot(rows[2, mode])
        for slot, sample in enumerate(item["samples"]):
            future_rel = np.asarray(sample["future_rel"], dtype=np.float64)
            path = np.concatenate((np.zeros((1, 2)), future_rel), axis=0)
            ax.plot(path[:, 0], path[:, 1], color=SAMPLE_COLORS[slot],
                    linewidth=1.7, marker="o", markersize=2.8)
        ax.scatter([0], [0], marker="+", s=52, linewidths=1.3,
                   color="black", zorder=5)
        ax.set(xlim=(-shared_limit, shared_limit), ylim=(-shared_limit, shared_limit),
               xlabel="Relative x", ylabel="Relative y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="0.88", linewidth=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(labelsize=7, pad=1.5)

    y_top = block.get_position(fig).y1
    fig.text(0.5, y_top + 0.032, "Representative Examples of Motion Modes (%s)" % display_name,
             ha="center", va="bottom", fontsize=11, fontweight="semibold")


def main():
    args = parse_args()
    datasets = [
        read_dataset(args.chole_config, args.chole_selection, DATASETS[0][0]),
        read_dataset(args.phakir_config, args.phakir_selection, DATASETS[1][0]),
    ]
    output_dir = resolve(args.output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Figure output directory must be new or empty: " + str(output_dir))
    for dataset in datasets:
        stage1_dir = dataset["source"]["output"].resolve()
        if output_dir == stage1_dir or stage1_dir in output_dir.parents:
            raise ValueError("Figure output must be separate from Stage 1 artifacts")

    columns = max(dataset["num_modes"] for dataset in datasets)
    fig = plt.figure(figsize=(4.65 * columns, 12.3), facecolor="white")
    outer = fig.add_gridspec(2, 1, left=0.045, right=0.99, top=0.92,
                             bottom=0.085, hspace=0.24)
    for index, (dataset, (_, name)) in enumerate(zip(datasets, DATASETS)):
        draw_dataset(fig, outer[index, 0], dataset, name, columns)
    legend = [
        Line2D([], [], color=SAMPLE_COLORS[0], lw=1.7, marker="o", markersize=3,
               label="Sample 1"),
        Line2D([], [], color=SAMPLE_COLORS[1], lw=1.7, marker="o", markersize=3,
               label="Sample 2"),
        Line2D([], [], color="black", linestyle="none", marker="+", markersize=7,
               label="Current (Last observation)"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.019),
               ncol=3, frameon=False, fontsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(output_dir / ("fig5_motion_modes." + ext),
                    dpi=600 if ext == "png" else 300, facecolor="white")
    plt.close(fig)
    provenance = {"layout": "CholecTrack20 above PhaKIR; two saved representatives per mode",
                  "selection_is_reused": True, "datasets": {}}
    for dataset, (key, _) in zip(datasets, DATASETS):
        provenance["datasets"][key] = {
            "selection_file": str(dataset["selection_path"]),
            "selection_sha256": file_hash(dataset["selection_path"]),
            "prototype_sha256": dataset["selected"]["prototype_sha256"],
            "qgt_sha256": dataset["selected"]["qgt_sha256"],
            "shared_axis_limit": dataset["selected"]["shared_axis_limit"],
            "modes": [{"mode": item["mode"], "samples": [
                {key: sample[key] for key in ("sample_id", "video", "segment", "csv",
                                                 "agent_id", "future_orig_frames", "future_images")}
                for sample in item["samples"]]} for item in dataset["selected"]["modes"]],
        }
    (output_dir / "fig5_selected_samples.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved Fig. 5 and exact representative provenance to " + str(output_dir))


if __name__ == "__main__":
    main()
