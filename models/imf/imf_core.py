"""Rank-3 port of imeanflow-main/imf.py:118-399.

The JVP uses predicted conditional v, not noise-data. Guidance, JVP and
adaptive weights are stop-gradient; the ordinary u/v forward retains the
entire live history encoder graph. No second-order parameter optimization.
"""
import math
import torch


def expand(x):
    return x[:, None, None]


class ImprovedMeanFlowCore:
    def __init__(self, config):
        self.cfg = config

    def draw(self, x):
        c, b = self.cfg, x.shape[0]
        a = (torch.randn(b, device=x.device) * c["time_std"] + c["time_mean"]).sigmoid()
        z = (torch.randn(b, device=x.device) * c["time_std"] + c["time_mean"]).sigmoid()
        t, r = torch.maximum(a, z), torch.minimum(a, z)
        fm = torch.arange(b, device=x.device) < int(b * c["data_proportion"])
        r = torch.where(fm, t, r)
        lo, hi = c["cfg_interval_min_range"]
        t_min = lo + (hi - lo) * torch.rand(b, device=x.device)
        lo, hi = c["cfg_interval_max_range"]
        t_max = lo + (hi - lo) * torch.rand(b, device=x.device)
        t_min = torch.where(fm, torch.zeros_like(t_min), t_min)
        t_max = torch.where(fm, torch.ones_like(t_max), t_max)
        u = torch.rand(b, device=x.device)
        beta, max_scale = c["cfg_beta"], c["cfg_s_max"]
        if beta == 1.0:
            omega = (u * math.log1p(max_scale)).exp()
        else:
            omega = (torch.log1p(u * math.expm1((1 - beta) * math.log1p(max_scale))) / (1 - beta)).exp()
        # Official implementation samples a COUNT, then drops the batch prefix.
        count = (torch.rand(b, device=x.device) < c["condition_dropout"]).sum()
        drop = torch.arange(b, device=x.device) < count
        return {"noise": torch.randn_like(x), "t": t, "r": r, "fm": fm,
                "omega": omega, "t_min": t_min, "t_max": t_max, "drop": drop}

    def loss(self, net, future_rel, condition, draws=None, return_debug=False):
        if future_rel.ndim != 3 or future_rel.shape[-1] != 2:
            raise ValueError("iMF clean state must be [B,H,2]")
        with torch.autocast(device_type=future_rel.device.type, enabled=False):
            return self._loss(net, future_rel.float(), condition.float(), draws, return_debug)

    def _loss(self, net, x, condition, draws, return_debug):
        d = self.draw(x) if draws is None else draws
        t, r, omega, lo, hi = (d[k] for k in ("t", "r", "omega", "t_min", "t_max"))
        if d["noise"].shape != x.shape or any(v.shape != (len(x),) for v in (t, r, omega, lo, hi)):
            raise ValueError("Invalid deterministic draw shapes")
        state = (1 - expand(t)) * x + expand(t) * d["noise"]
        raw_v = d["noise"] - x
        zero, one = torch.zeros_like(t), torch.ones_like(t)
        no_drop = torch.zeros_like(d["drop"])
        with torch.no_grad():
            fixed_condition = condition.detach()
            # Match official v_fn: conditional omega and unconditional omega=1.
            vc_w = net(state, t, zero, omega, zero, one, fixed_condition, no_drop)[1]
            vu = net(state, t, zero, one, zero, one, fixed_condition, ~no_drop)[1]
            vg_fm = raw_v + expand(1 - 1 / omega) * (vc_w - vu)
            effective_w = torch.where((t >= lo) & (t <= hi), omega, one)
            vc = net(state, t, zero, effective_w, zero, one, fixed_condition, no_drop)[1]
            vg = raw_v + expand(1 - 1 / effective_w) * (vc - vu)
            vg = torch.where(expand(d["fm"]), vg_fm, vg)
            vg = torch.where(expand(d["drop"]), raw_v, vg)

        def u_fn(s, time_t, time_r):
            return net(s, time_t, time_t - time_r, omega, lo, hi, condition.detach(), d["drop"])[0]

        # create_graph=False returns a detached derivative. Main forward below
        # is essential: reusing the JVP return would detach encoder training.
        _, du_dt = torch.autograd.functional.jvp(
            u_fn, (state.detach(), t.detach(), r.detach()),
            (vc.detach(), one, zero), create_graph=False, strict=False)
        u, v = net(state, t, t - r, omega, lo, hi, condition, d["drop"])
        compound = u + expand(t - r) * du_dt.detach()
        error_u = (compound - vg.detach()).square()
        error_v = (v - vg.detach()).square()
        per_u, per_v = error_u.sum((1, 2)), error_v.sum((1, 2))
        weighted_u = per_u / (per_u + self.cfg["norm_eps"]).pow(self.cfg["norm_p"]).detach()
        weighted_v = per_v / (per_v + self.cfg["norm_eps"]).pow(self.cfg["norm_p"]).detach()
        loss = (weighted_u + weighted_v).mean()
        info = {"imf": loss.detach(), "imf_u_mse": error_u.detach().mean(),
                "imf_v_mse": error_v.detach().mean(), "imf_u_weighted": weighted_u.detach().mean(),
                "imf_v_weighted": weighted_v.detach().mean()}
        if return_debug:
            info["debug"] = {"u": u.detach(), "v": v.detach(), "v_c": vc.detach(), "v_g": vg.detach(),
                             "du_dt": du_dt.detach(), "V": compound.detach(),
                             "raw_u": per_u.detach(), "raw_v": per_v.detach()}
        return loss, info

    @torch.no_grad()
    def sample(self, net, condition, noise):
        state = noise.float()
        c = self.cfg
        b = len(state)
        schedule = torch.linspace(1, 0, c["num_steps"] + 1, device=state.device)
        omega = state.new_full((b,), c["sampling_omega"])
        lo = state.new_full((b,), c["sampling_t_min"])
        hi = state.new_full((b,), c["sampling_t_max"])
        with torch.autocast(device_type=state.device.type, enabled=False):
            for t, r in zip(schedule[:-1], schedule[1:]):
                time = t.expand(b)
                h = (t - r).expand(b)
                u, _ = net(state, time, h, omega, lo, hi, condition.float(), u_only=True)
                state = state - expand(h) * u
        return state
