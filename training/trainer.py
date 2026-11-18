import json
import torch
from torch.utils.data import DataLoader
from datasets.dataset_adapter import TrainingDataset, ensure_manifests
from models.model import BimodalSurgicalIMF
from evaluation.evaluator import generate_predictions, evaluate_saved
from utils.config import paths
from utils.io import write_json, file_hash
from utils.reproducibility import seed_all, seed_worker, device_for, rng_state, restore_rng
from utils.checkpoint import save_training, resume_training


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def artifact_fingerprints(c, manifest):
    result = {"manifest": manifest["fingerprint"]}
    p = paths(c)
    if c["mode_supervision"]["enabled"]:
        result["q_gt"] = file_hash(p["qgt_artifact"])
    if c["experiment"]["variant"] not in ("trajectory_only", "trajectory_mode"):
        result["pl_stitch_pretrained"] = file_hash(p["video_checkpoint"])
    return result


def train(c, resume=None):
    seed_all(c["training"]["seed"], c["training"]["deterministic"])
    manifest = ensure_manifests(c)
    device = device_for(c)
    dataset = TrainingDataset(c)
    if not len(dataset):
        raise ValueError("No training samples")
    artifacts = artifact_fingerprints(c, manifest)
    model = BimodalSurgicalIMF(c, load_pretrained=resume is None).to(device)
    t = c["training"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=t["learning_rate"], betas=tuple(t["betas"]), eps=t["eps"], weight_decay=t["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"])
    start, step, best = 0, 0, float("inf")
    if resume:
        state = resume_training(resume, model, optimizer, scaler, c, artifacts)
        start, step, best = state["epoch"], state["step"], state["best"]
    out = paths(c)["output"]
    mean_key = "meanADE@%d" % c["evaluation"]["K"]
    best_mean = float("inf")
    # Existing checkpoints keep their original schema. Rebuild the independent
    # mean-ADE selection state from completed validation log rows on resume.
    if resume and (out / "training_log.jsonl").exists():
        with (out / "training_log.jsonl").open(encoding="utf-8") as f:
            for line in f:
                past = json.loads(line)
                if past["epoch"] <= start and "validation" in past:
                    val = past["validation"]
                    best_mean = min(best_mean, val[mean_key] if mean_key in val else val["ADE"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "resolved_config.json", c)
    write_json(out / "training_environment.json", {"torch": torch.__version__, "device": str(device),
                "cuda": torch.version.cuda, "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "trainable_names": [n for n, p in model.named_parameters() if p.requires_grad],
                "artifacts": artifacts})
    for epoch in range(start, t["epochs"]):
        model.train()
        # Epoch-specific loader generator makes epoch-boundary resume exact.
        generator = torch.Generator().manual_seed(t["seed"] + epoch)
        loader = DataLoader(dataset, batch_size=t["batch_size"], shuffle=True, generator=generator,
                            num_workers=t["num_workers"], worker_init_fn=seed_worker,
                            pin_memory=device.type == "cuda", drop_last=False)
        totals = {}
        for batch in loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=t["amp"]):
                loss, metrics = model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite joint loss at step %d" % step)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite parameter gradients at step %d" % step)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.) + value.item() * len(batch["history_abs"])
        row = {"epoch": epoch + 1, "step": step, **{k: v / len(dataset) for k, v in totals.items()}}
        improved = False
        improved_mean = False
        if (epoch + 1) % t["validate_every_epochs"] == 0:
            saved_rng = rng_state()
            index = generate_predictions(model, c, "val", out / "validation" / ("epoch_%04d" % (epoch + 1)))
            val = evaluate_saved(c, index)
            restore_rng(saved_rng)
            row["validation"] = val
            improved = val["minADE@K"] < best
            best = min(best, val["minADE@K"])
            improved_mean = val[mean_key] < best_mean
            best_mean = min(best_mean, val[mean_key])
        print(json.dumps(row), flush=True)
        with (out / "training_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        save_args = (model, optimizer, scaler, epoch + 1, step, best, c, artifacts)
        save_training(out / "checkpoints/latest.pt", *save_args)
        if improved:
            save_training(out / "checkpoints/best.pt", *save_args)
        if improved_mean:
            save_training(out / "checkpoints/best_meanADE.pt", *save_args)
        if (epoch + 1) % t["checkpoint_every_epochs"] == 0:
            save_training(out / ("checkpoints/epoch_%04d.pt" % (epoch + 1)), *save_args)
    return model
