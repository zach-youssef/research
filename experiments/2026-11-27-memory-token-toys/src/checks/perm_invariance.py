"""Hypothesis: the X7a set encoders are invariant to object order (summary, pertrunc, canon, full,
including mirrored objects that tie on distance to the player), and the X7b rate_var_tok step
encoding is invariant to the order of a step's extra tokens."""
from _util import FAST, close
from hypothesis import given, strategies as st

import torch

import x7a
import x7b


@FAST
@given(seed=st.integers(0, 10 ** 6), comp=st.sampled_from(["summary", "pertrunc", "canon", "full"]),
       mirror=st.booleans())
def check_set(seed, comp, mirror):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = x7a.SetPolicy(comp).eval()
    p, o, v = x7a.fields(4, g, 8, 20, pad_to=24)
    if mirror:  # second half mirrors the first around the player: equal distances, same role
        h = 12
        o[:, h:2 * h, :2] = 2 * p[:, None] - o[:, :h, :2]
        o[:, h:2 * h, 2:] = o[:, :h, 2:]
        v[:, :2 * h] = True
    L = torch.randint(4, 33, (4,), generator=g)
    perm = torch.stack([torch.randperm(24, generator=g) for _ in range(4)])
    o2 = torch.gather(o, 1, perm[..., None].expand_as(o))
    v2 = torch.gather(v, 1, perm)
    with torch.no_grad():
        close(model(p, o, v, L), model(p, o2, v2, L), f"x7a {comp} permutation")


@FAST
@given(seed=st.integers(0, 10 ** 6))
def check_var_tok(seed):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = x7b.Agent("rate_var_tok").eval()
    B, S = 3, 12
    obs, prev, _, _, _, _ = x7b.oracle_batch(B, g, T_max=S, T_min=S, L=S)
    extra, ntok = x7b.draw_extra(B, S, g)
    ntok[:, :4] = 4  # make sure some steps have 3 extra tokens to permute
    perm = torch.stack([torch.stack([torch.randperm(3, generator=g) for _ in range(S)]) for _ in range(B)])
    perm = torch.where((ntok == 4)[..., None], perm, torch.arange(3)[None, None])  # only valid extras move
    extra2 = torch.gather(extra, 2, perm[..., None].expand_as(extra))
    mem = model.core.init_memory(B)
    with torch.no_grad():
        a, _, wa = model.forward_segment(mem, obs, prev, extra, ntok)
        b, _, wb = model.forward_segment(mem, obs, prev, extra2, ntok)
    close(a, b, "x7b var_tok logits under extra-token permutation")
    close(wa, wb, "x7b var_tok write memory under extra-token permutation")


def run():
    check_set()
    check_var_tok()
    return "x7a summary/pertrunc/canon/full and x7b rate_var_tok invariant to order"
