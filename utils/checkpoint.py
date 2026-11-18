import os
from pathlib import Path
import torch
from utils.reproducibility import rng_state, restore_rng
from utils.config import model_signature


def save_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(str(tmp), str(path))


def load_torch(path):
    # Project checkpoints include Python/NumPy RNG state: load only trusted files.
    return torch.load(path, map_location="cpu", weights_only=False)


def save_training(path, model, optimizer, scaler, epoch, step, best, c, artifacts):
    save_torch(path, {"schema": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict(), "epoch": epoch, "step": step, "best": best,
                      "rng": rng_state(), "config": c, "model_signature": model_signature(c),
                      "artifacts": artifacts})


def resume_training(path, model, optimizer, scaler, c, artifacts):
    state = load_torch(path)
    if state["model_signature"] != model_signature(c) or state["artifacts"] != artifacts:
        raise ValueError("Resume configuration/artifact mismatch")
    old = state["config"]["training"]
    for key in ("batch_size", "learning_rate", "betas", "eps", "weight_decay", "seed", "amp", "deterministic", "num_workers"):
        if old[key] != c["training"][key]:
            raise ValueError("Resume training parameter changed: " + key)
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return state
