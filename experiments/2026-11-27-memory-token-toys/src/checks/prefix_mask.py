"""Hypothesis: with a prefix L, replacing the masked-out tokens (summary queries past L,
pertrunc dims past L, canon objects past L) with noise leaves the policy output unchanged."""
from _util import FAST, close
from hypothesis import given, strategies as st

import torch

import x7a


@FAST
@given(seed=st.integers(0, 10 ** 6), comp=st.sampled_from(["summary", "pertrunc", "canon"]))
def check(seed, comp):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = x7a.SetPolicy(comp).eval()
    p, o, v = x7a.fields(4, g, 8, 48)
    L = torch.randint(4, 33, (4,), generator=g)
    with torch.no_grad():
        close(model(p, o, v, L), model(p, o, v, L, perturb=torch.Generator().manual_seed(seed + 1)),
              f"x7a {comp} masked tokens")


def run():
    check()
    return "masked queries, dims and objects never reach the policy"
