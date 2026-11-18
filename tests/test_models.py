import pytest
torch = pytest.importorskip("torch")
from models.model import BimodalSurgicalIMF
from models.imf.conditional_imf import ConditionalIMF
from models.attention_pooling import AttentionPooling
from models.mode_head import ModeHead
from training.losses import soft_cross_entropy
from utils.reproducibility import sample_noise, seed_all


def nonzero_grad(module):
    return any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in module.parameters())


def test_shapes_generation_and_gradients(tiny_config):
    seed_all(42)
    c = tiny_config
    model = BimodalSurgicalIMF(c)
    batch = {"history_abs": torch.rand(2, 3, 2), "future_rel": torch.randn(2, 2, 2)}
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    # Official zero gates/output: wait for real updates, do not alter init.
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss, info = model(batch)
        loss.backward()
        optimizer.step()
    assert torch.isfinite(loss)
    # Audit a fresh, explicitly conditioned forward after real warmup updates.
    # A random final minibatch may legitimately drop BOTH histories (B=2),
    # so its Fusion/encoder gradients cannot prove whether that path exists.
    draws = model.imf.imf_core.draw(batch["future_rel"])
    draws["drop"] = torch.zeros_like(draws["drop"])
    optimizer.zero_grad(set_to_none=True)
    loss, _ = model(batch, draws=draws)
    loss.backward()
    assert torch.isfinite(loss)
    assert nonzero_grad(model.imf)
    assert nonzero_grad(model.fusion)
    assert nonzero_grad(model.trajectory_encoder)
    # The converse must also hold: fully dropped conditions receive no
    # gradient through L_imf. Keep production dropout and zero-init unchanged.
    draws["drop"] = torch.ones_like(draws["drop"])
    optimizer.zero_grad(set_to_none=True)
    dropped_loss, _ = model(batch, draws=draws)
    dropped_loss.backward()
    for module in (model.fusion, model.trajectory_encoder):
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in module.parameters())
    model.eval()
    noise = sample_noise(["a", "b"], 3, 2, 42, "cpu")
    out = model.generate(batch["history_abs"], noise=noise)
    assert out.shape == (2, 3, 2, 2)
    assert torch.equal(out, model.generate(batch["history_abs"], noise=noise))
    assert not torch.equal(out[:, 0], out[:, 1])
    assert torch.equal(noise.flip(0), sample_noise(["b", "a"], 3, 2, 42, "cpu"))
    assert model.generate_one(batch["history_abs"], noise=noise[:, 0]).shape == (2, 2, 2)


def test_pooling_soft_targets_keep_time():
    x = torch.randn(3, 5, 8, requires_grad=True)
    pool, head = AttentionPooling(8, 4), ModeHead(8, 4, 3)
    logits = head(pool(x))
    q = torch.tensor([[.1, .2, .7]]).expand(3, -1)
    loss = soft_cross_entropy(logits, q)
    expected = -(q * logits.log_softmax(-1)).sum(-1).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert x.shape == (3, 5, 8) and x.grad.abs().sum() > 0
    assert nonzero_grad(pool) and nonzero_grad(head)
    assert torch.allclose(logits.softmax(-1).sum(-1), torch.ones(3))


def test_official_zero_initialization_and_no_explicit_t(tiny_config):
    net = ConditionalIMF(tiny_config)
    x, cond = torch.randn(2, 2, 2), torch.randn(2, 3, 8)
    zero, one = torch.zeros(2), torch.ones(2)
    u, v = net(x, zero, one, one, zero, one, cond)
    assert torch.count_nonzero(u) == 0 and torch.count_nonzero(v) == 0
    out = net.generate(cond, x)
    assert torch.equal(out, x) # e - zero, no image clipping
    assert all(torch.count_nonzero(b.attn_scale) == 0 for b in net.shared_blocks)


def test_time_sampling_and_prefix_dropout(tiny_config):
    core = ConditionalIMF(tiny_config).imf_core
    d = core.draw(torch.zeros(11, 2, 2))
    assert torch.all(d["t"] >= d["r"])
    assert torch.equal(d["t"][:5], d["r"][:5])
    assert torch.all(d["t_min"][:5] == 0) and torch.all(d["t_max"][:5] == 1)
    assert torch.all(d["omega"] >= 1) and torch.all(d["omega"] < 8)
    n = int(d["drop"].sum())
    assert torch.all(d["drop"][:n]) and not torch.any(d["drop"][n:])
