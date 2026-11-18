"""Summarize completed Full mode-count/weight sweeps; never run inference."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 3407)
METRICS = ("meanADE@20", "meanFDE@20", "minADE@20", "minFDE@20")
SETTINGS = (("cluster_count", "k2", 2, 0.1),
            ("cluster_count", "k4", 4, 0.1),
            ("cluster_count", "k6", 6, 0.1),
            ("mode_weight", "weight05", 4, 0.05),
            ("mode_weight", "weight1", 4, 0.1),
            ("mode_weight", "weight2", 4, 0.2))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_run(dataset, setting, seed, modes, weight):
    baseline = setting in ("k4", "weight1")
    run = ROOT / "outputs" / ("%s_improved_full_seed%d" % (dataset, seed) if baseline else
                              "%s_full_sensitivity_%s_seed%d" % (dataset, setting, seed))
    result_path = run / "evaluation_test/metrics.json"
    index_path = run / "evaluation_test/predictions.json"
    for path in (result_path, index_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    result, index = read_json(result_path), read_json(index_path)
    checkpoint = run / "checkpoints/best_meanADE.pt"
    if baseline and dataset == "cholectrack20" and seed == 42:
        # The archived local copy has only latest.pt from the meanADE-best
        # epoch; a complete 20-epoch copy may instead have best_meanADE.pt.
        candidates = (checkpoint, run / "checkpoints/latest.pt")
        checkpoint = next((path for path in candidates if path.is_file() and
                           sha256(path) == index["checkpoint_sha256"]), None)
        if checkpoint is None:
            raise ValueError("No meanADE-selected checkpoint matches test predictions: " + str(run))
    elif not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = result["config"]
    trained_config = read_json(run / "resolved_config.json")
    mode = config["mode_supervision"]
    expected_epochs = 20 if dataset == "cholectrack20" else 67
    recorded_epochs = trained_config["training"]["epochs"]
    permitted_epochs = (3, 20) if baseline and dataset == "cholectrack20" and seed == 42 else (expected_epochs,)
    if (result["K"] != 20 or index["K"] != 20 or index["split"] != "test" or
            index["count"] != result["samples"] or
            index["seed"] != 42 or config["evaluation"]["seed"] != 42 or
            config["evaluation"]["K"] != 20 or
            config["data"]["dataset_name"] != dataset or
            config["experiment"]["variant"] != "full" or
            config["training"]["seed"] != seed or not mode["enabled"] or
            trained_config["training"]["seed"] != seed or
            recorded_epochs not in permitted_epochs or
            mode["num_modes"] != modes or
            not math.isclose(mode["lambda_traj"], weight) or
            not math.isclose(mode["lambda_video"], weight) or
            not math.isclose(mode["encoder_grad_scale"], 0.25)):
        raise ValueError("Sensitivity protocol mismatch: " + str(run))
    if "seed" in config["stage1"] and config["stage1"]["seed"] != 42:
        raise ValueError("Stage 1 seed differs from the shared fixed seed: " + str(run))
    with (run / "training_log.jsonl").open(encoding="utf-8") as handle:
        log = [json.loads(line) for line in handle if line.strip()]
    if not log or log[-1]["epoch"] != recorded_epochs:
        raise ValueError("Training did not complete the recorded epoch budget: " + str(run))
    validated = [row for row in log if "validation" in row]
    if not validated:
        raise ValueError("No validation metrics for checkpoint selection: " + str(run))
    best_row = min(validated, key=lambda row: row["validation"]["meanADE@20"])
    if checkpoint.name == "latest.pt" and best_row["epoch"] != recorded_epochs:
        raise ValueError("Legacy latest.pt is not at the lowest validation meanADE epoch")
    if index["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("Test predictions do not use the expected checkpoint: " + str(run))
    expected_stage1 = ROOT / "outputs" / ("%s_stage1_geometry" % dataset if modes == 4 else
                                           "%s_stage1_geometry_k%d" % (dataset, modes))
    for key, filename in (("prototype_artifact", "prototypes.npz"),
                          ("qgt_artifact", "qgt_train.npz")):
        value = Path(config["stage1"][key])
        actual = (value if value.is_absolute() else run / value).resolve()
        expected = (expected_stage1 / "stage1" / filename).resolve()
        if actual != expected or not actual.is_file():
            raise ValueError("Incorrect Stage 1 artifact for %s: %s" % (run, actual))
    values = result["metrics"]
    if any(not math.isfinite(float(values[name])) for name in METRICS):
        raise ValueError("Missing or nonfinite test metric: " + str(result_path))
    return {"dataset": dataset, "setting": setting, "train_seed": seed,
            "num_modes": modes, "lambda_traj": weight, "lambda_video": weight,
            "training_epochs": recorded_epochs,
            "test_samples": result["samples"],
            "manifest_fingerprint": index["manifest_fingerprint"],
            "best_validation_epoch": best_row["epoch"],
            "best_validation_meanADE@20": best_row["validation"]["meanADE@20"],
            "checkpoint": str(checkpoint),
            "test_metrics": str(result_path), **{name: float(values[name]) for name in METRICS}}


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("cholectrack20", "phakir"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out = out if out.is_absolute() else ROOT / out
    if out.exists():
        raise FileExistsError("Refusing to overwrite summary: " + str(out))
    per_seed = []
    summaries = []
    baseline_cache = {}
    for sweep, setting, modes, weight in SETTINGS:
        rows = []
        for seed in SEEDS:
            if setting == "weight1" and seed in baseline_cache:
                base = baseline_cache[seed]
                row = dict(base, setting=setting)
            else:
                row = check_run(args.dataset, setting, seed, modes, weight)
            if setting == "k4":
                baseline_cache[seed] = row
            row["sweep"] = sweep
            rows.append(row)
            per_seed.append(row)
        if (len({row["test_samples"] for row in per_seed}) != 1 or
                len({row["manifest_fingerprint"] for row in per_seed}) != 1):
            raise ValueError("Test samples or manifest fingerprint differ across Full runs")
        summary = {"dataset": args.dataset, "sweep": sweep, "setting": setting,
                   "num_modes": modes, "lambda_traj": weight, "lambda_video": weight,
                   "seeds": len(SEEDS), "test_samples": rows[0]["test_samples"]}
        for metric in METRICS:
            observations = [row[metric] for row in rows]
            summary[metric + "_mean"] = statistics.mean(observations)
            summary[metric + "_std"] = statistics.stdev(observations)
        summaries.append(summary)
    for summary in summaries:
        rows = [row for row in per_seed if row["sweep"] == summary["sweep"] and
                row["setting"] == summary["setting"]]
        summary["epoch_budget_matches_reused_full"] = all(
            row["training_epochs"] == baseline_cache[row["train_seed"]]["training_epochs"]
            for row in rows)
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "per_seed.csv", per_seed,
              ("dataset", "sweep", "setting", "train_seed", "num_modes", "lambda_traj",
               "lambda_video", "training_epochs", "test_samples", "manifest_fingerprint",
               "best_validation_epoch", "best_validation_meanADE@20", *METRICS,
               "checkpoint", "test_metrics"))
    write_csv(out / "summary.csv", summaries,
              ("dataset", "sweep", "setting", "num_modes", "lambda_traj", "lambda_video",
               "seeds", "test_samples", "epoch_budget_matches_reused_full",
               *(item for metric in METRICS
                                                for item in (metric + "_mean", metric + "_std"))))
    (out / "summary.json").write_text(json.dumps({"dataset": args.dataset, "seeds": SEEDS,
        "baseline_reused": True, "std_definition": "sample standard deviation over three training seeds",
        "metrics": METRICS, "per_seed": per_seed, "summary": summaries}, indent=2) + "\n",
        encoding="utf-8")
    print("Saved sensitivity summary to " + str(out))


if __name__ == "__main__":
    main()
