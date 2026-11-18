import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.manifest import build_manifest
from utils.config import parser, load_config, paths


if __name__ == "__main__":
    a = parser("Build read-only dataset manifests using YAML protocol").parse_args()
    c = load_config(a.config, a.set)
    p, d = paths(c), c["data"]
    build_manifest(p["dataset_root"], d["dataset_name"], p["manifests"], d["obs_len"],
                   d["pred_len"], d["sample_stride"], d["expected_orig_frame_step"])
