"""Shared pieces: seeding, transformer blocks, the RATE core, bootstrap CIs, JSON output.

Every model here consumes a *token set per agent step*: each token is a feature projection plus
a learned token-type embedding (type 0 = observation, type 1 = an extra stream such as detail or
audio tokens) plus a within-segment step position. Segment boundaries are counted in agent
steps, never in tokens. Attention is block-causal by step index: a token sees every token whose
step index is <= its own.
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THREADS = int(os.environ.get("X7_THREADS", "3"))


def setup_torch():
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def arm_seed(part, arm, seed):
    """A stable integer seed per (part, arm, seed); no Python hash() (it is salted per process)."""
    h = 0
    for ch in f"{part}/{arm}":
        h = (h * 131 + ord(ch)) % 1_000_003
    return seed * 1_000_003 + h


class Block(nn.Module):
    """Pre-LN transformer block; mask is bool [B or 1, L, L], True = may attend."""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1))
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.ln2(x))


class CrossAttn(nn.Module):
    """Multi-head attention with separate query and key/value sequences (no residual)."""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, xq, xkv, kv_valid=None):
        B, Lq, D = xq.shape
        Lk = xkv.shape[1]
        q = self.q(xq).view(B, Lq, self.h, D // self.h).transpose(1, 2)
        k, v = self.kv(xkv).view(B, Lk, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        mask = None
        if kv_valid is not None:
            mask = kv_valid[:, None, None, :].expand(B, 1, Lq, Lk)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(a.transpose(1, 2).reshape(B, Lq, D))


def block_causal_mask(step_idx, valid):
    """[B,S] step indices and validity -> [B,S,S] bool: a sees b iff step[b] <= step[a] and b valid
    (every token always sees itself, so no row is empty)."""
    m = (step_idx[:, None, :] <= step_idx[:, :, None]) & valid[:, None, :]
    eye = torch.eye(step_idx.shape[1], dtype=torch.bool, device=step_idx.device)[None]
    return m | eye


class RATECore(nn.Module):
    """RATE (cherepanov2026recurrent Eq. 1-3) without Transformer-XL caching (mem_len = 0).

    A segment is the sequence [M_n (read copy), S_n, M_n (write copy)]. The read copy attends only
    to itself; the segment's tokens attend to the read copy and block-causally to each other; the
    write copy attends to everything. The write copy's outputs W go through the Memory Retention
    Valve MRV-CA-2: M_{n+1} = FFN(MultiHead(Query=M_n, Key=W, Value=W)) (Eq. 3).
    With m = 0 the core is a plain block-causal transformer (the memoryless control).
    """

    def __init__(self, d, layers, heads, m):
        super().__init__()
        self.d, self.m = d, m
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        if m > 0:
            self.mem0 = nn.Parameter(torch.randn(m, d))  # M_0 ~ N(0, 1)
            self.read_emb = nn.Parameter(torch.zeros(m, d))
            self.write_emb = nn.Parameter(torch.zeros(m, d))
            nn.init.normal_(self.read_emb, std=0.02)
            nn.init.normal_(self.write_emb, std=0.02)
            self.mrv_attn = CrossAttn(d, 1)  # 1 MRV head, the paper's default
            self.mrv_ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

    def init_memory(self, B):
        return self.mem0[None].expand(B, -1, -1)

    def segment(self, mem, tok, step_idx, valid, want_write=True):
        """mem [B,m,d] or None, tok [B,S,d], step_idx [B,S] long, valid [B,S] bool.
        Returns (step_out [B,S,d], read_out [B,m,d] or None, write_out [B,m,d] or None)."""
        B, S, _ = tok.shape
        if self.m == 0 or mem is None:
            mask = block_causal_mask(step_idx, valid)
            x = tok
            for blk in self.blocks:
                x = blk(x, mask)
            return self.ln_f(x), None, None
        m = self.m
        nw = m if want_write else 0
        L = m + S + nw
        dev = tok.device
        allow = torch.zeros(B, L, L, dtype=torch.bool, device=dev)
        allow[:, :m, :m] = True  # read copy: only itself
        allow[:, m:m + S, :m] = True  # segment tokens read memory
        allow[:, m:m + S, m:m + S] = block_causal_mask(step_idx, valid)
        if want_write:
            allow[:, m + S:, :m] = True
            allow[:, m + S:, m:m + S] = valid[:, None, :]
            allow[:, m + S:, m + S:] = True
        parts = [mem + self.read_emb, tok]
        if want_write:
            parts.append(mem + self.write_emb)
        x = torch.cat(parts, 1)
        for blk in self.blocks:
            x = blk(x, allow)
        x = self.ln_f(x)
        return x[:, m:m + S], x[:, :m], (x[:, m + S:] if want_write else None)

    def mrv(self, mem, write_out):
        return self.mrv_ffn(self.mrv_attn(mem, write_out))


def bootstrap_ci(per_seed, reps=2000, level=0.95, seed=0):
    """Seed bootstrap of the mean of per-seed values."""
    a = np.asarray(per_seed, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(reps, len(a)))
    means = a[idx].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"low": float(lo), "high": float(hi), "level": level}


def metric(per_seed, n, unit="1", ci=True):
    out = {"value": float(np.mean(per_seed)), "unit": unit, "n": int(n),
           "per_seed": [float(v) for v in per_seed]}
    if ci and len(per_seed) > 1:
        out["ci"] = bootstrap_ci(per_seed)
    return out


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def make_opt(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-4)


def opt_step(opt, model, loss):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
