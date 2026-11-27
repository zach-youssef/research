"""Hypothesis: for fixed observations, changing the labels (X7a dodge action, X7b oracle actions,
X7c oracle answers and cue values) leaves every model output in the training path unchanged:
labels enter only the losses. Outputs are captured with a forward hook on the model."""
from _util import FAST, close
from hypothesis import given, strategies as st

import torch

import x7a
import x7b
import x7c


def outputs_of(model, fn):
    seen = []
    h = model.register_forward_hook(lambda m, i, o: seen.append(o[0] if isinstance(o, tuple) else o))
    try:
        fn()
    finally:
        h.remove()
    return seen


@FAST
@given(seed=st.integers(0, 10 ** 6))
def check(seed):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    # X7a
    m = x7a.SetPolicy("summary").eval()
    p, o, v = x7a.fields(8, g)
    y, L = x7a.label(p, o, v), torch.randint(4, 33, (8,), generator=g)
    y2 = (y + 1 + torch.randint(0, 4, y.shape, generator=g)) % 5
    a = outputs_of(m, lambda: x7a.train_loss(m, p, o, v, y, L))
    b = outputs_of(m, lambda: x7a.train_loss(m, p, o, v, y2, L))
    close(a[0], b[0], "x7a outputs depend on the label")
    # X7b: the model's inputs are obs and its previous actions; the oracle's actions are targets
    ag = x7b.Agent("rate_fixed").eval()
    obs, prev, act, valid, T, cue = x7b.oracle_batch(4, g)
    act2 = torch.randint(0, 4, act.shape, generator=g)
    runs = [outputs_of(ag.head, lambda lab=lab: x7b.train_loss(ag, (obs, prev, lab, valid, T, cue),
                                                               torch.Generator().manual_seed(seed)))
            for lab in (act, act2)]
    close(torch.cat(runs[0], 1), torch.cat(runs[1], 1), "x7b outputs depend on the oracle actions")
    # X7c: answers and cue values are targets only (the values reach the model only through the cue observations)
    for arm in ("aux", "gru"):
        mc = x7c.build(arm, 5).eval()
        ob, ac, _, k, vals = x7c.episodes(4, 60, g)
        ac2 = torch.randint(0, 5, ac.shape, generator=g)
        vals2 = torch.randint(0, 4, vals.shape, generator=g)
        a = outputs_of(mc, lambda: x7c.losses(mc, ob, ac, k, vals))
        b = outputs_of(mc, lambda: x7c.losses(mc, ob, ac2, k, vals2))
        close(a[0], b[0], f"x7c {arm} outputs depend on labels")


def run():
    check()
    return "x7a, x7b, x7c (aux and gru) outputs independent of the labels"
