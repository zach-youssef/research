"""X7c: several facts over a long episode (design readouts; the questions of pleines2025memorygym #C4, #C10).

Multi-cue T-Maze: k cues at steps 0..k-1 (cue flag, index one-hot of 8, value one-hot of 4),
distractor steps (distractor flag, random value), and k query steps at distinct random times in
[30, L-1] (query flag, index), one per cue in random order. Actions: 0 no-op, 1 + v answers
value v. The task is passive (actions never change observations), so a closed-loop rollout is a
single forward pass over the observations. Step tokens carry no previous-action embedding here:
the oracle's previous answers would carry cue values into the input.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import RATECore, arm_seed, make_opt, metric, opt_step, seed_all, setup_torch, write_json

N_IDX, N_VAL = 8, 4
OBS_DIM = 3 + N_IDX + N_VAL
N_ACT = 1 + N_VAL
KS = (1, 2, 4, 8)
MS = (1, 5, 20)
K_SEG = 30
QUERY_START = 30
AUX_W = 0.1


def episodes(B, L, gen, k=None, values=None):
    """Returns obs [B,L,15], act [B,L], query [B,L] bool, k [B], values [B,8].
    The structure (k, distractors, query times and order) and the cue values come from separate
    draws, so `values` can be replaced while everything else stays fixed (see checks/oracles)."""
    if k is None:
        k = torch.tensor(KS)[torch.randint(0, len(KS), (B,), generator=gen)]
    elif isinstance(k, int):
        k = torch.full((B,), k, dtype=torch.long)
    dist_val = torch.randint(0, N_VAL, (B, L), generator=gen)
    qkeys = torch.rand(B, L - QUERY_START, generator=gen)
    order_keys = torch.rand(B, N_IDX, generator=gen)
    vals_draw = torch.randint(0, N_VAL, (B, N_IDX), generator=gen)
    values = vals_draw if values is None else values
    t = torch.arange(L)[None].expand(B, L)
    is_cue = t < k[:, None]
    rank = torch.argsort(torch.argsort(qkeys, 1), 1)  # random rank of each candidate time
    is_q = torch.zeros(B, L, dtype=torch.bool)
    is_q[:, QUERY_START:] = rank < k[:, None]
    perm = torch.argsort(torch.where(torch.arange(N_IDX)[None] < k[:, None], order_keys, torch.full_like(order_keys, 2.0)), 1)
    qrank = (torch.cumsum(is_q.long(), 1) - 1).clamp(min=0)
    q_idx = torch.gather(perm, 1, qrank.clamp(max=N_IDX - 1))  # cue index asked at each query step
    is_d = ~is_cue & ~is_q
    idx = torch.where(is_cue, t.clamp(max=N_IDX - 1), q_idx)
    obs = torch.zeros(B, L, OBS_DIM)
    obs[:, :, 0] = is_cue.float()
    obs[:, :, 1] = is_d.float()
    obs[:, :, 2] = is_q.float()
    obs[:, :, 3:3 + N_IDX] = F.one_hot(idx, N_IDX).float() * (is_cue | is_q)[:, :, None].float()
    cue_val = torch.gather(values, 1, t.clamp(max=N_IDX - 1))
    shown_val = torch.where(is_cue, cue_val, dist_val)
    obs[:, :, 3 + N_IDX:] = F.one_hot(shown_val, N_VAL).float() * (is_cue | is_d)[:, :, None].float()
    act = torch.where(is_q, 1 + torch.gather(values, 1, q_idx), torch.zeros_like(t))
    query = is_q
    return obs, act, query, k, values


class StepEmbed(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.obs = nn.Linear(OBS_DIM, d)
        self.rtg = nn.Linear(1, d)
        self.ttype = nn.Embedding(2, d)
        self.pos = nn.Embedding(K_SEG, d)

    def forward(self, obs, pos=None):
        B, S, _ = obs.shape
        x = self.obs(obs) + self.rtg(torch.ones(B, S, 1)) + self.ttype.weight[0]
        return x if pos is None else x + self.pos(pos)


class RateAgent(nn.Module):
    def __init__(self, m, aux, d=64, layers=3, heads=4):
        super().__init__()
        self.aux = aux
        self.emb = StepEmbed(d)
        self.core = RATECore(d, layers, heads, m)
        self.head = nn.Linear(d, N_ACT)
        if aux:
            self.aux_head = nn.Sequential(nn.Linear(m * d, 256), nn.ReLU(), nn.Linear(256, N_IDX * N_VAL))

    def forward(self, obs):
        """obs [B,L,15] -> action logits [B,L,5], list of per-segment memories after the write."""
        B, L, _ = obs.shape
        mem = self.core.init_memory(B)
        logits, mems = [], []
        for s in range(0, L, K_SEG):
            o = obs[:, s:s + K_SEG]
            S = o.shape[1]
            pos = torch.arange(S)[None].expand(B, S)
            out, _, w = self.core.segment(mem, self.emb(o, pos), pos, torch.ones(B, S, dtype=torch.bool))
            logits.append(self.head(out))
            mem = self.core.mrv(mem, w)
            mems.append(mem)
        return torch.cat(logits, 1), mems

    def aux_logits(self, mem):
        return self.aux_head(mem.reshape(mem.shape[0], -1)).view(-1, N_IDX, N_VAL)


class GruAgent(nn.Module):
    def __init__(self, d=64, hidden=512):
        super().__init__()
        self.aux = False
        self.emb = StepEmbed(d)
        self.gru = nn.GRU(d, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden, N_ACT)

    def forward(self, obs):
        h, _ = self.gru(self.emb(obs))
        return self.head(h), []


def losses(model, obs, act, k, values):
    logits, mems = model(obs)
    la = F.cross_entropy(logits.reshape(-1, N_ACT), act.reshape(-1))
    lx = torch.zeros(())
    if model.aux:
        shown = (torch.arange(N_IDX)[None] < k[:, None]).float()  # every cue is shown in segment 1
        for mem in mems:
            ce = F.cross_entropy(model.aux_logits(mem).reshape(-1, N_VAL), values.reshape(-1), reduction="none")
            lx = lx + (ce * shown.reshape(-1)).sum() / shown.sum()
        lx = lx / len(mems)
    return la, lx


@torch.no_grad()
def recall(model, L, k, n_ep, gen, chunk=250):
    hits = tot = 0
    for c in range(0, n_ep, chunk):
        obs, act, query, _, _ = episodes(min(chunk, n_ep - c), L, gen, k=k)
        pred = model(obs)[0].argmax(-1)
        hits += ((pred == act) & query).sum().item()
        tot += query.sum().item()
    return hits / tot


def build(arm, m):
    return GruAgent() if arm == "gru" else RateAgent(m, aux=(arm == "aux"))


def run_arm(arm, m, seed, iters, n_ep, eval_L, ks, log):
    name = f"{arm}_m{m}" if arm != "gru" else "gru"
    sd = arm_seed("x7c", name, seed)
    seed_all(sd)
    gen = torch.Generator().manual_seed(sd)
    model = build(arm, m)
    opt = make_opt(model)
    t0 = time.time()
    la_h, lx_h = [], []
    for it in range(iters):
        obs, act, _, k, values = episodes(64, 150, gen)
        la, lx = losses(model, obs, act, k, values)
        opt_step(opt, model, la + AUX_W * lx)
        la_h.append(la.item()); lx_h.append(lx.item())
        if (it + 1) % 200 == 0 or it + 1 == iters:
            log(f"x7c {name} seed {seed} it {it + 1} loss_action {np.mean(la_h[-50:]):.4f} "
                f"loss_aux {np.mean(lx_h[-50:]):.4f} {time.time() - t0:.0f}s")
    train_s = time.time() - t0
    model.eval()
    egen = torch.Generator().manual_seed(arm_seed("x7c-eval", name, seed))
    res = {"loss_action": float(np.mean(la_h[-50:])), "loss_aux": float(np.mean(lx_h[-50:])), "train_s": train_s}
    for L in eval_L:
        for k in ks:
            res[f"L{L}_k{k}"] = recall(model, L, k, n_ep, egen)
    res["eval_s"] = time.time() - t0 - train_s
    log(f"x7c {name} seed {seed} done {res}")
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
        for arm, m in [("act", 1), ("act", 5), ("act", 20), ("aux", 20), ("gru", 0)]:
            seed_all(0); gen = torch.Generator().manual_seed(0)
            model = build(arm, m); opt = make_opt(model)
            t0 = time.time()
            for _ in range(args.time_iters):
                obs, act, _, k, values = episodes(64, 150, gen)
                la, lx = losses(model, obs, act, k, values)
                opt_step(opt, model, la + AUX_W * lx)
            out[f"{arm}_m{m}"] = (time.time() - t0) / args.time_iters
            t0 = time.time(); recall(model.eval(), 150, 8, 500, gen); out[f"{arm}_m{m}_eval"] = time.time() - t0
            log(f"timing x7c {arm} m{m}: {out[f'{arm}_m{m}']:.3f} s/iter, eval 500 ep {out[f'{arm}_m{m}_eval']:.1f}s")
        write_json(args.out, out)
        return
    if args.smoke:
        grid, seeds, iters, n_ep, eval_L, ks = [("act", 5)], [0], 20, 10, [150], [1, 8]
    else:
        grid = [(a, m) for a in ("act", "aux") for m in MS] + [("gru", 0)]
        seeds, iters, n_ep, eval_L, ks = [0, 1, 2], 2000, 500, [150, 450], list(KS)
    raw = {}
    for arm, m in grid:
        for s in seeds:
            name, r = run_arm(arm, m, s, iters, n_ep, eval_L, ks, log)
            raw[f"{name}/{s}"] = r
            write_json(args.out + ".partial", {"raw": raw})
    names = sorted({k.split("/")[0] for k in raw})
    metrics = {}
    for name in names:
        for L in eval_L:
            for k in ks:
                suf = "" if L == 150 else f"_l{L}"
                metrics[f"facts_recalled_{name}_k{k}{suf}"] = metric([raw[f"{name}/{s}"][f"L{L}_k{k}"] for s in seeds],
                                                                    len(seeds) * n_ep * k)
        metrics[f"{name}_loss_action"] = metric([raw[f"{name}/{s}"]["loss_action"] for s in seeds], len(seeds), ci=False)
        if name.startswith("aux"):
            metrics[f"{name}_loss_aux"] = metric([raw[f"{name}/{s}"]["loss_aux"] for s in seeds], len(seeds), ci=False)
    readouts = {}
    if not args.smoke:
        k4 = [k for k in ks if k >= 4]
        per_seed = [np.mean([raw[f"aux_m{m}/{s}"][f"L150_k{k}"] - raw[f"act_m{m}/{s}"][f"L150_k{k}"]
                             for m in MS for k in k4]) for s in seeds]
        metrics["aux_gain_k4plus"] = metric(per_seed, len(seeds) * n_ep * sum(k4) * len(MS))
        metrics["aux_gain_k4plus_min_seed"] = {"value": float(min(per_seed)), "unit": "1", "n": len(seeds)}
        best = max(MS, key=lambda m: metrics[f"facts_recalled_act_m{m}_k8"]["value"])
        g = [raw[f"gru/{s}"]["L150_k8"] - raw[f"act_m{best}/{s}"]["L150_k8"] for s in seeds]
        metrics["gru_minus_rate_k8"] = metric(g, len(seeds) * n_ep * 8)
        metrics["gru_minus_rate_k8"]["best_act_m"] = best
        for arm in ("act", "aux"):
            for k in ks:
                ok = [m for m in MS if metrics[f"facts_recalled_{arm}_m{m}_k{k}"]["value"] >= 0.9]
                readouts[f"m_needed_{arm}_k{k}"] = ok[0] if ok else "none"
    write_json(args.out, {"part": "x7c", "seeds": seeds, "iters": iters, "n_ep": n_ep, "eval_L": eval_L,
                          "metrics": metrics, "readouts": readouts, "raw": raw})
    if os.path.exists(args.out + ".partial"):
        os.remove(args.out + ".partial")


if __name__ == "__main__":
    main()
