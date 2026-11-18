import hashlib
import os
import random
import numpy as np
import torch


def seed_all(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def device_for(c):
    name = c["training"]["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if c["training"]["amp"] and device.type != "cuda":
        raise ValueError("AMP is supported only on CUDA in this implementation")
    return device


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def sample_noise(ids, k, h, seed, device):
    """Independent, batch-order invariant standard Gaussian draws on CPU."""
    rows = []
    for sid in ids:
        samples = []
        for j in range(k):
            key = (str(seed) + ":" + sid + ":" + str(j)).encode("utf-8")
            value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2 ** 63 - 1)
            g = torch.Generator(device="cpu").manual_seed(value)
            samples.append(torch.randn(h, 2, generator=g))
        rows.append(torch.stack(samples))
    return torch.stack(rows).to(device)
