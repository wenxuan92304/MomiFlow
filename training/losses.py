import torch
import torch.nn.functional as F


def soft_cross_entropy(logits, q_gt, tolerance=1e-5):
    if logits.shape != q_gt.shape or logits.ndim != 2:
        raise ValueError("Mode logits and q_GT must have identical [B,K_mode] shape")
    if not torch.isfinite(q_gt).all() or (q_gt < 0).any():
        raise ValueError("Invalid q_GT")
    if not torch.allclose(q_gt.sum(-1), torch.ones_like(q_gt[:, 0]), atol=tolerance, rtol=0):
        raise ValueError("q_GT must sum to 1")
    return -(q_gt * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


def hard_cross_entropy(logits, q_gt, tolerance=1e-5):
    """Use the existing soft q_GT artifact, but supervise its argmax class."""
    if logits.shape != q_gt.shape or logits.ndim != 2:
        raise ValueError("Mode logits and q_GT must have identical [B,K_mode] shape")
    if not torch.isfinite(q_gt).all() or (q_gt < 0).any():
        raise ValueError("Invalid q_GT")
    if not torch.allclose(q_gt.sum(-1), torch.ones_like(q_gt[:, 0]), atol=tolerance, rtol=0):
        raise ValueError("q_GT must sum to 1")
    return F.cross_entropy(logits.float(), q_gt.argmax(dim=-1))
