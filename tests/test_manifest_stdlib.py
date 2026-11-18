"""Runs with Python stdlib only, including on the supplied Windows machine."""
import csv
import json
import struct
import tempfile
import unittest
from pathlib import Path
from datasets.manifest import build_manifest, png_size, verify_manifest
from utils.io import read_jsonl


def fixture(root, dataset="cholectrack20", gap=False, invalid=False):
    for split, video in (("train", "vid01"), ("val", "vid02"), ("test", "vid03")):
        folder = root / dataset / "csv" / split
        folder.mkdir(parents=True)
        frames = root / dataset / "frames" / video
        frames.mkdir(parents=True)
        with (folder / (video + ".csv")).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "agent_id", "frame", "orig_frame", "x", "y", "image_width", "image_height"])
            for i in range(12):
                orig = 100 + 25 * i + (25 if gap and i >= 6 else 0)
                w.writerow([video.upper(), "1", i, orig, -1 if invalid and i == 5 else i / 20, .4, 999, 999])
                (frames / ("%06d.png" % orig)).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + struct.pack(">II", 854, 480))


class ManifestTests(unittest.TestCase):
    def test_split_shapes_png_and_id_stability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            out = root / "out"
            first = build_manifest(root, "cholectrack20", out)
            second = build_manifest(root, "cholectrack20", out)
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            self.assertEqual([first["report"][s]["samples"] for s in ("train", "val", "test")], [3, 3, 3])
            hs, ts = read_jsonl(out / "train_history.jsonl"), read_jsonl(out / "train_targets.jsonl")
            for h, t in zip(hs, ts):
                self.assertEqual(h["sample_id"], t["sample_id"])
                self.assertEqual(len(h["history_abs"]), 5)
                self.assertEqual(len(t["future_abs"]), 5)
                self.assertEqual(h["history_orig_frames"][-1] + 25, t["future_orig_frames"][0])
                self.assertEqual(h["history_wh"], [[854, 480]] * 5)
                self.assertNotIn("future_abs", h)
                self.assertNotIn("q_gt", h)
                last = h["history_abs"][-1]
                rel = [[p[j] - last[j] for j in range(2)] for p in t["future_abs"]]
                for r, p in zip(rel, t["future_abs"]):
                    self.assertAlmostEqual(r[0] + last[0], p[0])
            cfg = dict(dataset_root=str(root), dataset_name="cholectrack20", obs_len=5, pred_len=5,
                       sample_stride=1, expected_orig_frame_step=25)
            verify_manifest(out, cfg)
            with (root / "cholectrack20/csv/train/vid01.csv").open("a") as f:
                f.write("\n")
            with self.assertRaisesRegex(ValueError, "CSV files changed"):
                verify_manifest(out, cfg)

    def test_orig_gap_not_processed_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, gap=True)
            meta = build_manifest(root, "cholectrack20", root / "out")
            self.assertEqual(meta["report"]["train"]["samples"], 0)
            self.assertEqual(meta["report"]["train"]["gaps"], 1)

    def test_reject_invalid_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, invalid=True)
            meta = build_manifest(root, "cholectrack20", root / "out")
            self.assertEqual(meta["report"]["train"]["samples"], 0)
            self.assertEqual(len(meta["report"]["train"]["invalid_rows"]), 1)

    def test_segment_boundary_never_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, video in (("train", "Video_01"), ("val", "Video_02"), ("test", "Video_03")):
                folder = root / "phakir/csv" / split
                folder.mkdir(parents=True)
                for seg in (1, 2):
                    with (folder / (video + "_seg%d.csv" % seg)).open("w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["agent_id", "frame", "orig_frame", "segment_id", "x", "y"])
                        for i in range((seg - 1) * 6, seg * 6):
                            w.writerow(["0_5", i, i * 25, seg, .5, .5])
            meta = build_manifest(root, "phakir", root / "out")
            self.assertEqual(meta["report"]["train"]["samples"], 0)

    def test_bad_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.png"
            path.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                png_size(path)


if __name__ == "__main__":
    unittest.main()
