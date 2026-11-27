"""X7a: prefix masking over an unordered object set (design readout, no claim).

Bullet fields in the unit square: a player and n objects (x, y, vx, vy, role). The label is a
scripted dodge: for each of 5 actions, roll the player and the objects forward 8 steps and take
the action with the lowest danger (sum over objects of exp(-d_min^2 / 0.05^2), plus a wall term).
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Block, CrossAttn, arm_seed, make_opt, metric, opt_step, seed_all, setup_torch, write_json

D = 64
NQ = 32  # summary queries / pertrunc dims
LS = (4, 8, 16, 32)
L_MIN, L_MAX = 4, 32
N_MIN, N_MAX = 8, 48
SPEED_OBJ, SPEED_PLAYER = 0.03, 0.02
HORIZON, SIGMA = 8, 0.05
ACTIONS = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]])  # stay up down left right


# ----------------------------------------------------------------------------- data
def fields(B, gen, n_min=N_MIN, n_max=N_MAX, pad_to=None):
    """player [B,2], objs [B,P,5] (x, y, vx, vy, role), valid [B,P]."""
    P = pad_to or n_max
    n = torch.randint(n_min, n_max + 1, (B,), generator=gen)
    player = 0.1 + 0.8 * torch.rand(B, 2, generator=gen)
    xy = torch.rand(B, P, 2, generator=gen)
    ang = 2 * np.pi * torch.rand(B, P, generator=gen)
    sp = SPEED_OBJ * torch.rand(B, P, generator=gen)
    v = torch.stack([sp * torch.cos(ang), sp * torch.sin(ang)], -1)
    role = (torch.rand(B, P, generator=gen) < 0.2).float()  # 1 = enemy, 0 = bullet
    objs = torch.cat([xy, v, role[..., None]], -1)
    valid = torch.arange(P)[None] < n[:, None]
    objs = objs * valid[..., None]
    return player, objs, valid


def danger(player, objs, valid):
    """[B,5] danger of each action (the labeller's score)."""
    steps = torch.arange(1, HORIZON + 1).float()  # [H]
    pp = player[:, None, None, :] + SPEED_PLAYER * steps[None, None, :, None] * ACTIONS[None, :, None, :]
    pp = pp.clamp(0.0, 1.0)  # [B,5,H,2]
    op = objs[:, None, :, :2] + steps[None, :, None, None] * objs[:, None, :, 2:4]  # [B,H,P,2]
    d2 = ((pp[:, :, :, None, :] - op[:, None, :, :, :]) ** 2).sum(-1)  # [B,5,H,P]
    dmin2 = d2.min(2).values  # [B,5,P]
    obj = (torch.exp(-dmin2 / SIGMA ** 2) * valid[:, None, :]).sum(-1)
    fin = pp[:, :, -1]  # [B,5,2]
    wd = torch.cat([fin, 1 - fin], -1)  # distance to the 4 walls
    wall = torch.exp(-wd ** 2 / SIGMA ** 2).sum(-1)
    return obj + wall


def label(player, objs, valid):
    return danger(player, objs, valid).argmin(-1)


def canon_order(player, objs, valid):
    """Permutation sorting objects by (distance to player, role, x, y, vx, vy); padding last.
    Stable sorts from the last key to the first give the lexicographic order."""
    dist = ((objs[..., :2] - player[:, None]) ** 2).sum(-1)
    keys = [dist, objs[..., 4], objs[..., 0], objs[..., 1], objs[..., 2], objs[..., 3]]
    B, P = valid.shape
    perm = torch.arange(P)[None].expand(B, P).clone()
    for key in reversed(keys):
        kk = torch.where(valid, key, torch.full_like(key, float("inf")))
        kk = torch.gather(kk, 1, perm)
        o = torch.sort(kk, dim=1, stable=True).indices
        perm = torch.gather(perm, 1, o)
    inv = torch.where(valid, torch.zeros_like(dist), torch.ones_like(dist))
    o = torch.sort(torch.gather(inv, 1, perm), dim=1, stable=True).indices  # padding strictly last
    return torch.gather(perm, 1, o)


# ----------------------------------------------------------------------------- model
class SetPolicy(nn.Module):
    """per-object MLP -> 2 set-transformer layers -> compression to L tokens -> 2-layer transformer
    over [player token, tokens] -> action head on the player token.
    comp in {summary, pertrunc, canon, full}."""

    def __init__(self, comp):
        super().__init__()
        self.comp = comp
        self.obj_mlp = nn.Sequential(nn.Linear(7, D), nn.ReLU(), nn.Linear(D, D))
        self.set_layers = nn.ModuleList([Block(D, 4) for _ in range(2)])
        self.player = nn.Linear(2, D)
        self.ttype = nn.Embedding(2, D)  # type 0 = player, type 1 = set token
        self.mask_emb = nn.Embedding(2, D)  # learned 0/1 mask embeddings (yan2025elastictok#A6)
        if comp == "summary":
            self.queries = nn.Parameter(torch.randn(NQ, D) * 0.02)
            self.xattn = CrossAttn(D, 4)
        if comp == "pertrunc":
            self.down = nn.Linear(D, NQ)
            self.dim_mask = nn.Parameter(torch.zeros(2, NQ))  # value of a masked / kept dim offset
            self.up = nn.Linear(NQ, D)
        self.policy = nn.ModuleList([Block(D, 4) for _ in range(2)])
        self.head = nn.Linear(D, 5)

    def encode_set(self, player, objs, valid):
        rel = objs[..., :2] - player[:, None]
        h = self.obj_mlp(torch.cat([objs, rel], -1))
        mask = valid[:, None, :] | torch.eye(valid.shape[1], dtype=torch.bool)[None]
        for blk in self.set_layers:
            h = blk(h, mask)
        return h

    def forward(self, player, objs, valid, L, perturb=None):
        """L [B] long (prefix length per sample) or None for comp == full. `perturb` (a torch
        Generator, checks only) replaces every masked-out token or dim with noise first."""
        B, P = valid.shape
        if self.comp == "canon":
            perm = canon_order(player, objs, valid)
            objs = torch.gather(objs, 1, perm[..., None].expand_as(objs))
            valid = torch.gather(valid, 1, perm)
        h = self.encode_set(player, objs, valid)
        if self.comp == "summary":
            z = self.xattn(self.queries[None].expand(B, -1, -1), h, valid)
            keep = torch.arange(NQ)[None] < L[:, None]
            if perturb is not None:
                z = torch.where(keep[..., None], z, torch.randn(z.shape, generator=perturb))
            tok = z + self.mask_emb(keep.long())
        elif self.comp == "pertrunc":
            z = self.down(h)
            km = (torch.arange(NQ)[None] < L[:, None]).float()[:, None, :]  # [B,1,NQ]
            if perturb is not None:
                z = torch.where(km.bool(), z, torch.randn(z.shape, generator=perturb))
            z = z * km + self.dim_mask[0] * (1 - km) + self.dim_mask[1] * km
            tok = self.up(z) + self.mask_emb.weight[1]
            keep = valid
        elif self.comp == "canon":
            keep = valid & (torch.arange(P)[None] < L[:, None])
            if perturb is not None:
                h = torch.where(keep[..., None], h, torch.randn(h.shape, generator=perturb))
            tok = h + self.mask_emb(keep.long())
        else:
            keep, tok = valid, h + self.mask_emb.weight[1]
        x = torch.cat([self.player(player)[:, None] + self.ttype.weight[0], tok + self.ttype.weight[1]], 1)
        kv = torch.cat([torch.ones(B, 1, dtype=torch.bool), keep], 1)
        mask = kv[:, None, :] | torch.eye(kv.shape[1], dtype=torch.bool)[None]
        for blk in self.policy:
            x = blk(x, mask)
        return self.head(x[:, 0])


def train_loss(model, player, objs, valid, y, L):
    """The model sees the observation (player, objs, valid) and the prefix L; y is the target only."""
    return F.cross_entropy(model(player, objs, valid, L), y)


ARMS = [("summary", "elastic"), ("pertrunc", "elastic"), ("canon", "elastic")] + \
       [("summary", f"fixed{L}") for L in LS] + [("full", "none")]


def draw_L(B, regime, gen):
    if regime == "elastic":
        return torch.randint(L_MIN, L_MAX + 1, (B,), generator=gen)
    if regime == "none":
        return None
    return torch.full((B,), int(regime[5:]), dtype=torch.long)


@torch.no_grad()
def accuracy(model, test, L):
    player, objs, valid, y = test
    hits = 0
    for c in range(0, len(y), 2000):
        sl = slice(c, c + 2000)
        Lb = None if L is None else torch.full((len(y[sl]),), L, dtype=torch.long)
        hits += (model(player[sl], objs[sl], valid[sl], Lb).argmax(-1) == y[sl]).sum().item()
    return hits / len(y)


def make_test(n, seed, n_min=N_MIN, n_max=N_MAX):
    g = torch.Generator().manual_seed(seed)
    p, o, v = fields(n, g, n_min, n_max)
    return p, o, v, label(p, o, v)


def run_arm(comp, regime, seed, iters, tests, log):
    name = f"{comp}_{regime}"
    sd = arm_seed("x7a", name, seed)
    seed_all(sd)
    gen = torch.Generator().manual_seed(sd)
    model = SetPolicy(comp)
    opt = make_opt(model)
    t0 = time.time()
    lh = []
    for it in range(iters):
        p, o, v = fields(64, gen)
        loss = train_loss(model, p, o, v, label(p, o, v), draw_L(64, regime, gen))
        opt_step(opt, model, loss)
        lh.append(loss.item())
        if (it + 1) % 500 == 0 or it + 1 == iters:
            log(f"x7a {name} seed {seed} it {it + 1} loss_action {np.mean(lh[-50:]):.4f} {time.time() - t0:.0f}s")
    model.eval()
    res = {"loss_action": float(np.mean(lh[-50:])), "train_s": time.time() - t0}
    evalL = [None] if regime == "none" else (LS if regime == "elastic" else [int(regime[5:])])
    for tname, test in tests.items():
        for L in evalL:
            res[f"{tname}_L{L}"] = accuracy(model, test, L)
    log(f"x7a {name} seed {seed} done {res}")
    return name, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--time-iters", type=int, default=0)
    args = ap.parse_args()
    setup_torch()
    log = lambda s: print(s, flush=True)
    if args.time_iters:
        out = {}
        for comp, regime in [("summary", "elastic"), ("pertrunc", "elastic"), ("canon", "elastic"), ("full", "none")]:
            seed_all(0); gen = torch.Generator().manual_seed(0)
            model = SetPolicy(comp); opt = make_opt(model)
            t0 = time.time()
            for _ in range(args.time_iters):
                p, o, v = fields(64, gen)
                opt_step(opt, model, F.cross_entropy(model(p, o, v, draw_L(64, regime, gen)), label(p, o, v)))
            out[comp] = (time.time() - t0) / args.time_iters
            log(f"timing x7a {comp}: {out[comp]:.3f} s/iter")
        write_json(args.out, out)
        return
    if args.smoke:
        arms, seeds, iters = [("summary", "elastic")], [0], 20
        tests = {"acc": make_test(200, 12345)}
    else:
        arms, seeds, iters = ARMS, [0, 1, 2], 3000
        tests = {"acc": make_test(20000, 12345), "acc96": make_test(5000, 54321, 96, 96)}
    raw = {}
    for comp, regime in arms:
        for s in seeds:
            name, r = run_arm(comp, regime, s, iters, tests, log)
            raw[f"{name}/{s}"] = r
    metrics = {}
    for comp, regime in arms:
        name = f"{comp}_{regime}"
        keys = [k for k in raw[f"{name}/{seeds[0]}"] if k.startswith("acc")]
        for k in keys:
            tname, Ls = k.split("_L")
            arm = comp if regime in ("elastic", "none") else "summary_fixed"
            mname = f"x7a_{tname}_{arm}" + ("" if Ls == "None" else f"_L{Ls}")
            metrics[mname] = metric([raw[f"{name}/{s}"][k] for s in seeds], len(seeds) * len(tests[tname][3]))
        metrics[f"x7a_{name}_loss_action"] = metric([raw[f"{name}/{s}"]["loss_action"] for s in seeds], len(seeds), ci=False)
    readouts = {}
    if not args.smoke:
        v = lambda n: metrics[n]["value"]
        diff = {L: v(f"x7a_acc_summary_L{L}") - v(f"x7a_acc_canon_L{L}") for L in LS}
        if diff[4] >= 0.02 and diff[8] >= 0.02:
            r1 = "summary main arm"
        elif all(d <= 0.01 for d in diff.values()):
            r1 = "canon default, summary ablation"
        else:
            r1 = "both"
        gap = max(v(f"x7a_acc_summary_fixed_L{L}") - v(f"x7a_acc_summary_L{L}") for L in LS)
        r2 = "one elastic model" if gap <= 0.02 else ("add fixed-L baselines" if gap > 0.05 else "unclear")
        readouts = {"x7a_summary_minus_canon": diff, "x7a_readout_arms": r1,
                    "x7a_fixed_minus_elastic_max": gap, "x7a_readout_elastic": r2}
    write_json(args.out, {"part": "x7a", "seeds": seeds, "iters": iters, "metrics": metrics,
                          "readouts": readouts, "raw": raw})


if __name__ == "__main__":
    main()
