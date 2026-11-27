"""Hypothesis: in RATE (X7b arms, X7c) and the DT control, the action at step t does not change
when observations or previous actions after t change; the read memory copy does not attend to
the segment."""
from _util import FAST, close
from hypothesis import given, strategies as st

import torch

import x7b
import x7c


@FAST
@given(seed=st.integers(0, 10 ** 6), arm=st.sampled_from(list(x7b.ARMS)), t=st.integers(0, 58))
def check_x7b(seed, arm, t):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = x7b.Agent(arm).eval()
    B, L = 2, 60
    obs, prev, _, _, _, _ = x7b.oracle_batch(B, g, T_max=L, T_min=40, L=L)
    extra = ntok = None
    if model.cfg["var_tok"]:
        extra, ntok = x7b.draw_extra(B, L, g)
    bounds = x7b.segment_bounds(L, model.cfg["K"], model.cfg["var_len"], g)
    obs2, prev2 = obs.clone(), prev.clone()
    obs2[:, t + 1:] = torch.randn(obs2[:, t + 1:].shape, generator=g)
    prev2[:, t + 1:] = torch.randint(0, 5, prev2[:, t + 1:].shape, generator=g)
    extra2 = None
    if extra is not None:
        extra2 = extra.clone()
        extra2[:, t + 1:] = torch.randn(extra2[:, t + 1:].shape, generator=g)
    with torch.no_grad():
        a = x7b.seq_logits(model, obs, prev, extra, ntok, bounds)
        b = x7b.seq_logits(model, obs2, prev2, extra2, ntok, bounds)
    close(a[:, :t + 1], b[:, :t + 1], f"x7b {arm} causality at t={t}")


@FAST
@given(seed=st.integers(0, 10 ** 6), t=st.integers(0, 88))
def check_x7c(seed, t):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = x7c.RateAgent(5, aux=False).eval()
    obs = x7c.episodes(2, 90, g)[0]
    obs2 = obs.clone()
    obs2[:, t + 1:] = torch.randn(obs2[:, t + 1:].shape, generator=g)
    with torch.no_grad():
        close(model(obs)[0][:, :t + 1], model(obs2)[0][:, :t + 1], f"x7c causality at t={t}")


@FAST
@given(seed=st.integers(0, 10 ** 6))
def check_read_copy(seed):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    core = x7b.Agent("rate_fixed").eval().core
    B, S = 2, 30
    mem = torch.randn(B, core.m, core.d, generator=g)
    pos = torch.arange(S)[None].expand(B, S)
    ones = torch.ones(B, S, dtype=torch.bool)
    with torch.no_grad():
        _, r1, _ = core.segment(mem, torch.randn(B, S, core.d, generator=g), pos, ones)
        _, r2, _ = core.segment(mem, torch.randn(B, S, core.d, generator=g), pos, ones)
    close(r1, r2, "read copy depends on the segment")


def run():
    check_x7b()
    check_x7c()
    check_read_copy()
    return "no step sees the future (5 X7b arms, X7c RATE); the read copy ignores the segment"
