"""Dataset preparation usable even before installing the ML dependencies."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.manifest import build_manifest
from utils.io import resolve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="../data")
    p.add_argument("--dataset", choices=["cholectrack20", "phakir"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--obs-len", type=int, default=5)
    p.add_argument("--pred-len", type=int, default=5)
    p.add_argument("--sample-stride", type=int, default=1)
    p.add_argument("--orig-frame-step", type=int, default=25)
    args = p.parse_args()
    build_manifest(resolve(args.dataset_root), args.dataset, resolve(args.output),
                   args.obs_len, args.pred_len, args.sample_stride, args.orig_frame_step)


if __name__ == "__main__":
    main()
