import json
from models.model import BimodalSurgicalIMF
from evaluation.evaluator import generate_predictions, evaluate_saved
from utils.config import parser, load_config, paths, model_signature
from utils.io import resolve, file_hash, read_json
from utils.checkpoint import load_torch
from utils.reproducibility import seed_all, device_for


if __name__ == "__main__":
    p = parser("Save all history-only K-sample predictions, then evaluate absolute coordinates")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--output")
    p.add_argument("--inference-only", action="store_true", help="Never open target manifests")
    args = p.parse_args()
    c = load_config(args.config, args.set)
    seed_all(c["evaluation"]["seed"], c["training"]["deterministic"])
    checkpoint = resolve(args.checkpoint)
    state = load_torch(checkpoint)
    if state["model_signature"] != model_signature(c):
        raise ValueError("Evaluation model/config mismatch")
    meta = read_json(paths(c)["manifests"] / "metadata.json")
    if state["artifacts"]["manifest"] != meta["fingerprint"]:
        raise ValueError("Checkpoint and manifest do not belong to the same experiment")
    # The checkpoint contains the complete backbone; pretrained weights and
    # Stage1 artifacts are not needed, opened or reconstructed here.
    model = BimodalSurgicalIMF(c, load_pretrained=False).to(device_for(c))
    model.load_state_dict(state["model"], strict=True)
    out = resolve(args.output) if args.output else paths(c)["output"] / ("evaluation_" + args.split)
    index = generate_predictions(model, c, args.split, out, file_hash(checkpoint))
    if args.inference_only:
        print("Saved all predictions to " + str(index))
    else:
        print(json.dumps(evaluate_saved(c, index), indent=2))
