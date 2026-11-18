"""Real-data / real-PL-Stitch gradient and tiny-train-subset acceptance gate.

This performs actual optimization; no mocked encoder or modified iMF init.
Use a GPU. It writes only under the configured experiment directory.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from datasets.dataset_adapter import TrainingDataset, ensure_manifests
from models.model import BimodalSurgicalIMF
from training.trainer import to_device
from evaluation.metrics import compute_metrics
from utils.config import parser, load_config, paths
from utils.io import write_json
from utils.reproducibility import seed_all, sample_noise, device_for
from utils.checkpoint import save_torch


def gradient(module, tolerance):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise AssertionError("Missing/nonfinite gradients in " + type(module).__name__)
    total = sum(float(g.abs().sum()) for g in grads)
    if total <= tolerance:
        raise AssertionError("Zero gradient in " + type(module).__name__)
    return total


def video_gradient_audit(video, tolerance):
    report = {"projector": gradient(video.projector, tolerance)}
    cfg = video.cfg
    n = cfg["trainable_blocks"]
    for i, block in enumerate(video.backbone.blocks):
        trainable = cfg["freeze_mode"] == "full_finetune" or (cfg["freeze_mode"] == "partial_finetune" and i >= 12 - n)
        if trainable:
            report["block_%d" % i] = gradient(block, tolerance)
        elif any(p.grad is not None for p in block.parameters()):
            raise AssertionError("Frozen PL-Stitch block received a gradient")
    if cfg["freeze_mode"] == "full_finetune" or (cfg["freeze_mode"] == "partial_finetune" and cfg["trainable_norm"]):
        report["final_norm"] = gradient(video.backbone.norm, tolerance)
    for name, param in video.backbone.named_parameters():
        if not param.requires_grad and param.grad is not None:
            raise AssertionError("Frozen parameter gradient: " + name)
    return report


def run(c):
    s = c["sanity"]
    seed_all(c["training"]["seed"], c["training"]["deterministic"])
    meta = ensure_manifests(c)
    dataset = TrainingDataset(c)
    n = min(s["subset_size"], len(dataset))
    if n < 2:
        raise ValueError("Use at least two samples for iMF/FM sanity testing")
    # IDs are sorted SHA256 values, so this deterministic subset is not a block
    # of nearly identical adjacent windows from just one long track.
    batch = next(iter(DataLoader(Subset(dataset, range(n)), batch_size=n)))
    device = device_for(c)
    batch = to_device(batch, device)
    torch.testing.assert_close(batch["future_rel"] + batch["history_abs"][:, -1:, :], batch["future_abs"])
    model = BimodalSurgicalIMF(c).to(device)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=s["learning_rate"],
                                  betas=tuple(s["betas"]), eps=s["eps"], weight_decay=s["weight_decay"])
    report = {"manifest": meta["fingerprint"], "sample_ids": batch["sample_id"], "config": c, "gradients": {}}
    tolerance = s["gradient_tolerance"]
    if model.mode_enabled:
        z, v = model.encode_branches(batch["history_abs"], batch.get("history_video"))
        lt, lv, qt, qv = model.mode_losses(z, v, batch["q_gt"])
        initial_mode = [lt.item()] + ([lv.item()] if lv is not None else [])
        lt.backward()
        report["gradients"]["L_mode_traj"] = {"encoder": gradient(model.trajectory_encoder, tolerance),
             "pool": gradient(model.traj_pool, tolerance) if c["mode_supervision"]["pooling"] == "attention" else None,
             "head": gradient(model.traj_mode_head, tolerance)}
        model.zero_grad(set_to_none=True)
        torch.testing.assert_close(qt.sum(-1), torch.ones(n, device=device))
        if model.video_encoder is not None:
            # Fresh graph for independent video-only supervision.
            z, v = model.encode_branches(batch["history_abs"], batch.get("history_video"))
            _, lv, _, qv = model.mode_losses(z, v, batch["q_gt"])
            lv.backward()
            report["gradients"]["L_mode_video"] = video_gradient_audit(model.video_encoder, tolerance)
            report["gradients"]["L_mode_video"]["head"] = gradient(model.video_mode_head, tolerance)
            if c["mode_supervision"]["pooling"] == "attention":
                report["gradients"]["L_mode_video"]["pool"] = gradient(model.video_pool, tolerance)
            torch.testing.assert_close(qv.sum(-1), torch.ones(n, device=device))
        for step in range(s["mode_steps"]):
            optimizer.zero_grad(set_to_none=True)
            z, v = model.encode_branches(batch["history_abs"], batch.get("history_video"))
            lt, lv, _, _ = model.mode_losses(z, v, batch["q_gt"])
            (lt if lv is None else lt + lv).backward()
            optimizer.step()
            if (step + 1) % 10 == 0:
                print("mode overfit %d: %s" % (step + 1, " ".join("%.6f" % value for value in
                      ([lt.item()] + ([lv.item()] if lv is not None else [])))), flush=True)
        with torch.no_grad():
            z, v = model.encode_branches(batch["history_abs"], batch.get("history_video"))
            lt, lv, _, _ = model.mode_losses(z, v, batch["q_gt"])
        final_mode = [lt.item()] + ([lv.item()] if lv is not None else [])
        entropy = float(-(batch["q_gt"] * batch["q_gt"].clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1).mean())
        report["mode_overfit"] = {"initial": initial_mode, "final": final_mode, "soft_target_entropy": entropy}
        if s["require_improvement"] and not all(b < a for a, b in zip(initial_mode, final_mode)):
            raise AssertionError("Mode CE values must decrease; inspect q_GT/subset/learning rate")

    noise = sample_noise(batch["sample_id"], c["evaluation"]["K"], c["data"]["pred_len"], c["evaluation"]["seed"], device)
    def measure():
        model.eval()
        pred = model.generate(batch["history_abs"], batch.get("history_video"), noise)
        metric = compute_metrics(pred.cpu().numpy(), batch["future_abs"].cpu().numpy())
        model.train()
        return metric
    initial_metrics = measure()
    # Fixed draws expose raw-residual improvement instead of stochastic noise.
    draws = model.imf.imf_core.draw(batch["future_rel"])
    initial_loss, initial_info = model(batch, draws)
    del initial_loss
    for step in range(s["joint_steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss, info = model(batch, draws)
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite joint sanity loss")
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0:
            print("joint overfit %d: u_mse=%.6f v_mse=%.6f" % (step + 1, info["imf_u_mse"], info["imf_v_mse"]), flush=True)
    if s["joint_steps"] < s["warmup_steps"]:
        raise ValueError("Insufficient warmup before zero-init gradient audit")
    # Isolate L_imf after warmup: do not let auxiliary gradients mask detach bugs.
    model.zero_grad(set_to_none=True)
    z, v = model.encode_branches(batch["history_abs"], batch.get("history_video"))
    loss, final_info = model.imf.training_loss(batch["future_rel"], model.fuse(z, v), draws)
    loss.backward()
    imf_grad = {"imf": gradient(model.imf, tolerance), "fusion": gradient(model.fusion, tolerance),
                "trajectory": gradient(model.trajectory_encoder, tolerance)}
    if model.video_encoder is not None:
        imf_grad["video"] = video_gradient_audit(model.video_encoder, tolerance)
    report["gradients"]["L_imf"] = imf_grad
    final_metrics = measure()
    report["joint_overfit"] = {"initial_raw": {k: float(v) for k, v in initial_info.items()},
                                "final_raw": {k: float(v) for k, v in final_info.items()},
                                "initial_metrics": initial_metrics, "final_metrics": final_metrics}
    out = paths(c)["output"] / "sanity"
    write_json(out / "report.json", report)
    if s["require_improvement"]:
        if final_info["imf_u_mse"] >= initial_info["imf_u_mse"] or final_info["imf_v_mse"] >= initial_info["imf_v_mse"]:
            raise AssertionError("Raw iMF residuals did not both decrease; see sanity/report.json")
        if final_metrics["minADE@K"] >= initial_metrics["minADE@K"]:
            raise AssertionError("Tiny-subset generation did not improve; see sanity/report.json")
    model.eval()
    a = model.generate(batch["history_abs"], batch.get("history_video"), noise)
    b = model.generate(batch["history_abs"], batch.get("history_video"), noise)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    if noise.shape[1] > 1 and torch.equal(a[:, 0], a[:, 1]):
        raise AssertionError("Different noise produced identical trajectories")
    if device.type == "cuda":
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=True):
            amp_loss, _ = model(batch, draws)
        amp_loss.backward()
        gradient(model.imf, tolerance)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise AssertionError("Nonfinite encoder/generator AMP gradient")
        report["amp_finite"] = bool(torch.isfinite(amp_loss))
        if not report["amp_finite"]:
            raise AssertionError("AMP loss not finite")
    else:
        report["amp_finite"] = "not_run_no_cuda"
    report["passed"] = True
    write_json(out / "report.json", report)
    save_torch(out / "overfit.pt", {"model": model.state_dict(), "config": c})
    print(json.dumps({"passed": True, "report": str(out / "report.json")}), flush=True)


if __name__ == "__main__":
    args = parser("Actual backbone gradient checks and tiny-subset overfit; before full training").parse_args()
    run(load_config(args.config, args.set))
