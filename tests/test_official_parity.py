"""Fixed-draw numerical/gradient parity against extracted official JAX code."""
import pytest
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp
import numpy as np
from models.imf.imf_core import ImprovedMeanFlowCore
from tests.reference.official_core import OfficialCoreReference


class ToyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(.7))

    def forward(self, x, t, h, w, lo, hi, condition, drop_mask=None, u_only=False):
        context = condition.mean((1, 2))
        if drop_mask is not None:
            context = torch.where(drop_mask, torch.zeros_like(context), context)
        context, h, w = (a[:, None, None] for a in (context, h, w))
        u = torch.sin(self.theta * x) + h.square() + context * self.theta + .03 * w
        v = .3 * self.theta * x + context + .1 * w
        return u, v


@pytest.mark.parametrize("p", [0., 1., 0.5])
def test_official_rank4_to_rank3_loss_and_gradients(tiny_config, p):
    torch.manual_seed(10)
    x, noise = torch.randn(4, 2, 2), torch.randn(4, 2, 2)
    condition = torch.randn(4, 3, 8, requires_grad=True)
    draws = {"noise": noise, "t": torch.tensor([.7, .4, .8, .3]),
             "r": torch.tensor([.7, .4, .2, .1]), "omega": torch.tensor([2., 3., 4., 2.]),
             "t_min": torch.tensor([0., 0., .1, .4]), "t_max": torch.tensor([1., 1., .9, .9]),
             "fm": torch.tensor([True, True, False, False]), "drop": torch.tensor([True, False, False, False])}
    config = dict(tiny_config["imf"], norm_p=p)
    net = ToyNet()
    loss, info = ImprovedMeanFlowCore(config).loss(net, x, condition, draws, return_debug=True)
    loss.backward()
    fixed = {key: jnp.asarray(value.numpy().reshape(4, 1, 1, 1))
             for key, value in draws.items() if key not in ("noise", "drop")}
    fixed["noise"] = jnp.asarray(noise.numpy()[:, :, None, :])
    fixed["drop_uniform"] = jnp.array([.1, .8, .8, .8])

    def objective(theta, cond):
        def jnet(state, t, h, w, lo, hi, labels):
            # Official v_fn batches [conditional B, unconditional B]. Match
            # that concatenation order, not repeat-interleave sample order.
            # Keep cond live so parameter AND condition-gradient parity apply.
            if state.shape[0] == cond.shape[0]:
                batched_cond = cond
            elif state.shape[0] == 2 * cond.shape[0]:
                batched_cond = jnp.concatenate([cond, cond], axis=0)
            else:
                raise AssertionError("Unexpected official CFG batch size")
            assert labels.shape == (state.shape[0],)
            context = jnp.where(labels == 1, 0., batched_cond.mean((1, 2)))[:, None, None, None]
            h, w = h[:, None, None, None], w[:, None, None, None]
            u = jnp.sin(theta * state) + h ** 2 + context * theta + .03 * w
            v = .3 * theta * state + context + .1 * w
            return u, v
        core = OfficialCoreReference(jnet, fixed, p, config["norm_eps"])
        images = jnp.asarray(x.numpy()[:, :, None, :])
        labels = jnp.zeros(4, dtype=jnp.int32)
        loss = core.forward(images, labels)[0]
        t, r, w, lo, hi = (fixed[k] for k in ("t", "r", "omega", "t_min", "t_max"))
        state = (1 - t) * images + t * fixed["noise"]
        vg, vc = core.guidance_fn(fixed["noise"] - images, state, t, r, labels, fixed["fm"], w, lo, hi)
        labels, vg = core.cond_drop(fixed["noise"] - images, vg, labels)
        def wrapped(s, time, start):
            return core.u_fn(s, time, time - start, w, lo, hi, labels)
        u, derivative, v = jax.jvp(wrapped, (state, t, r), (vc, jnp.ones_like(t), jnp.zeros_like(r)), has_aux=True)
        compound = u + (t - r) * derivative
        detail = {"u": u, "v": v, "v_c": vc, "v_g": vg, "du_dt": derivative, "V": compound,
                  "raw_u": ((compound - vg) ** 2).sum((1, 2, 3)), "raw_v": ((v - vg) ** 2).sum((1, 2, 3))}
        return loss, detail

    (value, detail), grads = jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)(jnp.array(.7), jnp.asarray(condition.detach().numpy()))
    np.testing.assert_allclose(loss.detach().numpy(), value, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(net.theta.grad.numpy(), grads[0], rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(condition.grad.numpy(), grads[1], rtol=3e-5, atol=3e-6)
    for key, actual in info["debug"].items():
        expected = np.asarray(detail[key]).reshape(actual.shape)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=3e-5, atol=3e-6, err_msg=key)
