"""Draw existing history coordinates on their actual synchronized PNGs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw
from utils.config import parser, load_config, paths
from utils.io import read_jsonl, write_json


if __name__ == "__main__":
    a = parser("Visual inspection of orig_frame alignment, history only").parse_args()
    c = load_config(a.config, a.set)
    p = paths(c)
    rows = read_jsonl(p["manifests"] / "train_history.jsonl")
    n = min(c["sanity"]["visualization_samples"], len(rows))
    out = p["output"] / "alignment"
    out.mkdir(parents=True, exist_ok=True)
    selected = []
    for row in rows[:n]:
        for t, (rel, xy, orig) in enumerate(zip(row["history_images"], row["history_abs"], row["history_orig_frames"])):
            with Image.open(p["dataset_root"] / c["data"]["dataset_name"] / rel) as raw:
                image = raw.convert("RGB")
            w, h = image.size
            x, y = xy[0] * w, xy[1] * h
            draw = ImageDraw.Draw(image)
            radius = max(4, int(min(w, h) * .01))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="red", width=3)
            draw.text((8, 8), "agent=%s t=%d orig_frame=%d" % (row["agent_id"], t, orig), fill="yellow", stroke_width=1, stroke_fill="black")
            image.save(out / (row["sample_id"][:12] + "_%d.png" % t))
        selected.append(row)
    write_json(out / "samples.json", selected)
    print("Inspect alignment overlays in " + str(out))
