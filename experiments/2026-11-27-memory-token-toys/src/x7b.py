"""X7b: RATE on passive T-Maze with fixed and variable segments (cherepanov2026recurrent #C1, #C8, #C12).

Corridor length T in agent steps: the agent starts at x = 0, the junction is at x = T - 1, so the
oracle goes right T - 1 times and turns at step T - 1. Observation (cue_up, cue_down,
junction_flag, 1), the cue only at t = 0. Actions: 0 right, 1 left, 2 up, 3 down; up/down at the
junction end the episode (success if it matches the cue), elsewhere they do nothing. Time limit
T + 10 steps.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import RATECore, arm_seed, make_opt, metric, opt_step, seed_all, setup_torch, write_json

T_TRAIN_MAX = 90
N_ACT = 4
START = 4  # previous-action id at t = 0
MAX_POS = 90
SLOTS = 4  # rate_var_tok: up to 4 tokens per step (1 observation + up to 3 extra type-1 tokens)
NT_CHOICES = (1, 2, 4)
VAR_LENS = (15, 30, 60)

ARMS = {
    "rate_fixed": dict(kind="rate", K=30, var_tok=False, var_len=False),
    "rate_n1k90": dict(kind="rate", K=90, var_tok=False, var_len=False),
    "dt": dict(kind="dt", K=90, var_tok=False, var_len=False),
    "rate_var_tok": dict(kind="rate", K=30, var_tok=True, var_len=False),
    "rate_var_len": dict(kind="rate", K=None, var_tok=False, var_len=True),
}


# ----------------------------------------------------------------------------- task
def tmaze_obs(t, x, T, cue):
    """Batched observation. t int, x [B] long, T [B] long, cue [B] long (0 up, 1 down)."""
    B = x.shape[0]
    o = torch.zeros(B, 4)
    if t == 0:
        o[torch.arange(B), cue] = 1.0
    o[:, 2] = (x == T - 1).float()
    o[:, 3] = 1.0
    return o


class TMaze:
    def __init__(self, T, cue):
        self.T, self.cue = T, cue
        B = T.shape[0]
        self.x = torch.zeros(B, dtype=torch.long)
        self.t = 0
        self.done = torch.zeros(B, dtype=torch.bool)
        self.success = torch.zeros(B, dtype=torch.bool)
        self.junction = torch.zeros(B, dtype=torch.bool)

    def obs(self):
        return tmaze_obs(self.t, self.x, self.T, self.cue)

    def step(self, a):
        live = ~self.done
        at_j = self.x == self.T - 1
        right = live & (a == 0)
        left = live & (a == 1)
        self.x = torch.where(right, torch.minimum(self.x + 1, self.T - 1), self.x)
        self.x = torch.where(left, torch.clamp(self.x - 1, min=0), self.x)
        turn = live & at_j & ((a == 2) | (a == 3))
        self.success |= turn & ((a - 2) == self.cue)
        self.done |= turn
        self.junction |= self.x == self.T - 1
        self.t += 1
        self.done |= self.t >= self.T + 10


def oracle_batch(B, gen, T_max=T_TRAIN_MAX, T_min=9, L=T_TRAIN_MAX):
    """Oracle trajectories, padded to L steps. Returns obs [B,L,4], prev [B,L], act [B,L], valid [B,L], T, cue."""
    T = torch.randint(T_min, T_max + 1, (B,), generator=gen)
    cue = torch.randint(0, 2, (B,), generator=gen)
    t = torch.arange(L)[None].expand(B, L)
    obs = torch.zeros(B, L, 4)
    obs[torch.arange(B), 0, cue] = 1.0
    obs[:, :, 2] = (t == (T[:, None] - 1)).float()
    obs[:, :, 3] = 1.0
    act = torch.where(t == T[:, None] - 1, (2 + cue)[:, None].expand(B, L), torch.zeros_like(t))
    prev = torch.cat([torch.full((B, 1), START), act[:, :-1]], 1)
    valid = t < T[:, None]
    return obs, prev, act, valid, T, cue


# ----------------------------------------------------------------------------- model
class StepEmbed(nn.Module):
    """Observation token: obs projection + previous-action + return-to-go (constant 1) +
    token type + within-segment step position. Extra tokens: projection + type 1 + position."""

    def __init__(self, d, obs_dim=4, extra_dim=4):
        super().__init__()
        self.obs = nn.Linear(obs_dim, d)
        self.extra = nn.Linear(extra_dim, d)
        self.prev = nn.Embedding(N_ACT + 1, d)
        self.rtg = nn.Linear(1, d)
        self.ttype = nn.Embedding(2, d)
        self.pos = nn.Embedding(MAX_POS, d)

    def obs_tok(self, obs, prev, pos):
        rtg = self.rtg(torch.ones(*obs.shape[:-1], 1))
        return self.obs(obs) + self.prev(prev) + rtg + self.ttype.weight[0] + self.pos(pos)

    def extra_tok(self, feat, pos):
        return self.extra(feat) + self.ttype.weight[1] + self.pos(pos)


class Agent(nn.Module):
    def __init__(self, arm, d=128, layers=3, heads=4, m=5):
        super().__init__()
        self.cfg = ARMS[arm]
        self.emb = StepEmbed(d)
        self.core = RATECore(d, layers, heads, 0 if self.cfg["kind"] == "dt" else m)
        self.head = nn.Linear(d, N_ACT)

    def seg_tokens(self, obs, prev, extra=None, ntok=None):
        """Build one segment's tokens. obs [B,S,4], prev [B,S]; for var_tok extra [B,S,SLOTS-1,4] and
        ntok [B,S]. Returns tok [B,S*slots,d], step_idx, valid, index of each step's obs token."""
        B, S, _ = obs.shape
        pos = torch.arange(S)[None].expand(B, S)
        ot = self.emb.obs_tok(obs, prev, pos)
        if extra is None:
            return ot, pos, torch.ones(B, S, dtype=torch.bool), torch.arange(S)
        et = self.emb.extra_tok(extra, pos[:, :, None].expand(B, S, SLOTS - 1))
        tok = torch.cat([ot[:, :, None], et], 2).reshape(B, S * SLOTS, -1)
        step_idx = pos[:, :, None].expand(B, S, SLOTS).reshape(B, S * SLOTS)
        slot = torch.arange(SLOTS)[None, None].expand(B, S, SLOTS)
        valid = (slot < ntok[:, :, None]).reshape(B, S * SLOTS)
        return tok, step_idx, valid, torch.arange(S) * SLOTS

    def forward_segment(self, mem, obs, prev, extra=None, ntok=None, want_write=True):
        tok, step_idx, valid, oidx = self.seg_tokens(obs, prev, extra, ntok)
        out, read_out, write_out = self.core.segment(mem, tok, step_idx, valid, want_write)
        logits = self.head(out[:, oidx])
        return logits, read_out, write_out


def segment_bounds(L, K, var_len, gen):
    b, s = [], 0
    while s < L:
        k = VAR_LENS[int(torch.randint(0, len(VAR_LENS), (1,), generator=gen))] if var_len else K
        b.append((s, min(s + k, L)))
        s += k
    return b


def draw_extra(B, S, gen):
    extra = torch.randn(B, S, SLOTS - 1, 4, generator=gen)
    ntok = torch.tensor(NT_CHOICES)[torch.randint(0, len(NT_CHOICES), (B, S), generator=gen)]
    return extra, ntok


def seq_logits(model, obs, prev, extra, ntok, bounds):
    """Teacher-forced logits [B,L,4] over a whole sequence split at `bounds` (memory carried)."""
    mem = model.core.init_memory(obs.shape[0]) if model.core.m else None
    out = []
    for s, e in bounds:
        logits, _, w = model.forward_segment(
            mem, obs[:, s:e], prev[:, s:e],
            None if extra is None else extra[:, s:e], None if ntok is None else ntok[:, s:e])
        out.append(logits)
        if mem is not None:
            mem = model.core.mrv(mem, w)
    return torch.cat(out, 1)


def train_loss(model, batch, gen):
    """batch = (obs, prev, act, valid, T, cue); the model sees obs and prev only."""
    obs, prev, act, valid, _, _ = batch
    B, L, _ = obs.shape
    cfg = model.cfg
    extra = ntok = None
    if cfg["var_tok"]:
        extra, ntok = draw_extra(B, L, gen)
    logits = seq_logits(model, obs, prev, extra, ntok, segment_bounds(L, cfg["K"], cfg["var_len"], gen))
    ce = F.cross_entropy(logits.reshape(-1, N_ACT), act.reshape(-1), reduction="none")
    return (ce * valid.reshape(-1).float()).sum() / valid.sum()


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, T_len, n_ep, gen, noise=False):
    """Closed loop on corridors of length T_len, cue balanced. Returns (success, junction) rates."""
    cfg = model.cfg
    cue = torch.arange(n_ep) % 2
    env = TMaze(torch.full((n_ep,), T_len, dtype=torch.long), cue)
    H = T_len + 10
    prev_a = torch.full((n_ep,), START, dtype=torch.long)
    if cfg["kind"] == "dt":
        obs_h, prev_h = [], []
        for t in range(H):
            obs_h.append(env.obs()); prev_h.append(prev_a)
            o = torch.stack(obs_h[-cfg["K"]:], 1); p = torch.stack(prev_h[-cfg["K"]:], 1)
            logits, _, _ = model.forward_segment(None, o, p, want_write=False)
            a = logits[:, -1].argmax(-1)
            env.step(a); prev_a = a
            if env.done.all():
                break
        return env.success.float().mean().item(), env.junction.float().mean().item()
    bounds = segment_bounds(H, cfg["K"], cfg["var_len"], gen)
    mem = model.core.init_memory(n_ep)
    for si, (s, e) in enumerate(bounds):
        if noise and si > 0:
            mem = torch.randn(mem.shape, generator=gen)
        obs_h, prev_h, ex_h, nt_h = [], [], [], []
        for t in range(s, e):
            obs_h.append(env.obs()); prev_h.append(prev_a)
            if cfg["var_tok"]:
                ex, nt = draw_extra(n_ep, 1, gen); ex_h.append(ex); nt_h.append(nt)
            o = torch.stack(obs_h, 1); p = torch.stack(prev_h, 1)
            ex = torch.cat(ex_h, 1) if ex_h else None
            nt = torch.cat(nt_h, 1) if nt_h else None
            logits, _, _ = model.forward_segment(mem, o, p, ex, nt, want_write=False)
            a = logits[:, -1].argmax(-1)
            env.step(a); prev_a = a
            if env.done.all():
                return env.success.float().mean().item(), env.junction.float().mean().item()
        _, _, w = model.forward_segment(mem, o, p, ex, nt, want_write=True)
        mem = model.core.mrv(mem, w)
    return env.success.float().mean().item(), env.junction.float().mean().item()


# ----------------------------------------------------------------------------- driver
def run_arm(arm, seed, iters, eval_T, n_ep, log):
    seed_all(arm_seed("x7b", arm, seed))
    gen = torch.Generator().manual_seed(arm_seed("x7b", arm, seed))
    model = Agent(arm)
    opt = make_opt(model)
    t0 = time.time()
    losses = []
    for it in range(iters):
        loss = train_loss(model, oracle_batch(64, gen), gen)
        opt_step(opt, model, loss)
        losses.append(loss.item())
        if (it + 1) % 200 == 0 or it + 1 == iters:
            log(f"x7b {arm} seed {seed} it {it + 1} loss_action {np.mean(losses[-50:]):.4f} {time.time() - t0:.0f}s")
    train_s = time.time() - t0
    model.eval()
    egen = torch.Generator().manual_seed(arm_seed("x7b-eval", arm, seed))
    res = {"loss_action": float(np.mean(losses[-50:])), "train_s": train_s,
           "params": sum(p.numel() for p in model.parameters())}
    for T in eval_T:
        res[f"success_t{T}"], res[f"junction_t{T}"] = evaluate(model, T, n_ep, egen)
    if arm == "rate_fixed":
        for T in [T for T in (90, 900) if T in eval_T] or eval_T[:1]:
            res[f"noise_success_t{T}"], res[f"noise_junction_t{T}"] = evaluate(model, T, n_ep, egen, noise=True)
    res["eval_s"] = time.time() - t0 - train_s
    log(f"x7b {arm} seed {seed} done {res}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--time-iters", type=int, default=0, help="time this many iterations per arm and exit")
    args = ap.parse_args()
    setup_torch()
    log = lambda s: print(s, flush=True)
    if args.time_iters:
        out = {}
        for arm in ARMS:
            seed_all(0); gen = torch.Generator().manual_seed(0)
            model = Agent(arm); opt = make_opt(model)
            t0 = time.time()
            for _ in range(args.time_iters):
                opt_step(opt, model, train_loss(model, oracle_batch(64, gen), gen))
            out[arm] = (time.time() - t0) / args.time_iters
            t0 = time.time(); evaluate(model.eval(), 90, 200, gen); out[arm + "_eval90"] = time.time() - t0
            log(f"timing x7b {arm}: {out[arm]:.3f} s/iter, eval T=90 {out[arm + '_eval90']:.1f}s")
        write_json(args.out, out)
        return
    if args.smoke:
        arms, seeds, iters, eval_T, n_ep = ["rate_fixed"], [0], 20, [30], 10
    else:
        arms, seeds, iters, eval_T, n_ep = list(ARMS), [0, 1, 2], 2000, [30, 60, 90, 180, 450, 900], 200
    raw = {}
    for arm in arms:
        for s in seeds:
            raw[f"{arm}/{s}"] = run_arm(arm, s, iters, eval_T, n_ep, log)
            write_json(args.out + ".partial", {"raw": raw})
    metrics = {}
    for arm in arms:
        keys = [k for k in raw[f"{arm}/{seeds[0]}"] if k.startswith(("success_", "junction_", "noise_"))]
        for k in keys:
            name = f"{arm}_{k}" if not k.startswith("noise_") else f"{arm}_noise_{k[6:]}"
            metrics[name] = metric([raw[f"{arm}/{s}"][k] for s in seeds], len(seeds) * n_ep)
        in_T = [T for T in (30, 60, 90) if T in eval_T]
        metrics[f"{arm}_success_in"] = metric(
            [np.mean([raw[f"{arm}/{s}"][f"success_t{T}"] for T in in_T]) for s in seeds], len(seeds) * n_ep * len(in_T))
        metrics[f"{arm}_loss_action"] = metric([raw[f"{arm}/{s}"]["loss_action"] for s in seeds], len(seeds), ci=False)
    write_json(args.out, {"part": "x7b", "seeds": seeds, "iters": iters, "eval_T": eval_T, "n_ep": n_ep,
                          "metrics": metrics, "raw": raw})
    if os.path.exists(args.out + ".partial"):
        os.remove(args.out + ".partial")


if __name__ == "__main__":
    main()
