"""Exhaustive over small cases: the X7a labeller returns the argmin of its danger score, which
matches a scalar reference; the T-Maze and multi-cue oracles succeed on every episode; and in both
T-Maze tasks the observations after the cue phase are independent of the cue values, so only
memory can solve them."""
import itertools
import math

from _util import close  # noqa: F401  (sets the import path)

import torch

import x7a
import x7b
import x7c


def danger_ref(player, objs, valid, a):
    """Scalar reference for one field and one action."""
    px, py = float(player[0]), float(player[1])
    ax, ay = [float(v) for v in x7a.ACTIONS[a]]
    traj = []
    for s in range(1, x7a.HORIZON + 1):
        traj.append((min(max(px + x7a.SPEED_PLAYER * s * ax, 0.0), 1.0),
                     min(max(py + x7a.SPEED_PLAYER * s * ay, 0.0), 1.0)))
    total = 0.0
    for j in range(objs.shape[0]):
        if not valid[j]:
            continue
        x, y, vx, vy = [float(v) for v in objs[j, :4]]
        d2 = min((tx - (x + s * vx)) ** 2 + (ty - (y + s * vy)) ** 2 for s, (tx, ty) in zip(range(1, 9), traj))
        total += math.exp(-d2 / x7a.SIGMA ** 2)
    fx, fy = traj[-1]
    total += sum(math.exp(-w ** 2 / x7a.SIGMA ** 2) for w in (fx, fy, 1 - fx, 1 - fy))
    return total


def check_x7a():
    g = torch.Generator().manual_seed(7)
    p, o, v = x7a.fields(40, g)
    dz = x7a.danger(p, o, v)
    for b in range(40):
        ref = torch.tensor([danger_ref(p[b], o[b], v[b], a) for a in range(5)], dtype=dz.dtype)
        if not torch.allclose(ref, dz[b], rtol=1e-4, atol=1e-5):
            raise AssertionError(f"x7a danger differs from the scalar reference: {ref} vs {dz[b]}")
    if not torch.equal(x7a.label(p, o, v), dz.argmin(-1)):
        raise AssertionError("x7a label is not the argmin of danger")


def check_tmaze():
    for T in list(range(9, 91)) + [180, 900]:
        for cue in (0, 1):
            env = x7b.TMaze(torch.tensor([T]), torch.tensor([cue]))
            for t in range(T):
                assert not env.done.any()
                env.step(torch.tensor([0 if t < T - 1 else 2 + cue]))
            if not (env.done.all() and env.success.all() and env.junction.all()):
                raise AssertionError(f"T-Maze oracle fails at T={T}, cue={cue}")
    g = torch.Generator().manual_seed(3)
    obs, prev, act, valid, T, cue = x7b.oracle_batch(64, g)
    for b in range(64):  # the oracle trajectories of training are the env's
        env = x7b.TMaze(T[b:b + 1], cue[b:b + 1])
        for t in range(int(T[b])):
            if not torch.equal(env.obs()[0], obs[b, t]):
                raise AssertionError("oracle_batch observation differs from the env")
            env.step(act[b, t:t + 1])
        assert env.success.all()
    o0 = x7b.oracle_batch(8, torch.Generator().manual_seed(5))[0]
    gg = torch.Generator().manual_seed(5)
    T2 = torch.randint(9, 91, (8,), generator=gg)
    # flipping the cue changes only step 0
    for b in range(8):
        e1 = x7b.TMaze(T2[b:b + 1], torch.tensor([0])); e2 = x7b.TMaze(T2[b:b + 1], torch.tensor([1]))
        for t in range(int(T2[b])):
            if t > 0 and not torch.equal(e1.obs(), e2.obs()):
                raise AssertionError("T-Maze observation after t=0 depends on the cue")
            e1.step(torch.tensor([0])); e2.step(torch.tensor([0]))
    del o0


def check_multicue():
    for k in x7c.KS:
        obs, act, query, kk, values = x7c.episodes(200, 150, torch.Generator().manual_seed(k), k=k)
        if not (query.sum(1) == k).all():
            raise AssertionError("not every cue is queried once")
        for b in range(200):
            shown = {int(obs[b, t, 3:11].argmax()): int(obs[b, t, 11:].argmax()) for t in range(k)}
            asked = [int(obs[b, t, 3:11].argmax()) for t in query[b].nonzero().flatten()]
            if sorted(asked) != list(range(k)):
                raise AssertionError("queries are not a permutation of the cues")
            for t in query[b].nonzero().flatten():
                if int(act[b, t]) != 1 + shown[int(obs[b, t, 3:11].argmax())]:
                    raise AssertionError("multi-cue oracle answer is wrong")
            if (query[b].nonzero().min() < x7c.QUERY_START) or (act[b][~query[b]] != 0).any():
                raise AssertionError("query timing or no-op labels wrong")
    for k, L in ((2, 150), (8, 450)):
        combos = list(itertools.product(range(4), repeat=k)) if k == 2 else \
            [tuple((i * 7 + j * 3) % 4 for j in range(8)) for i in range(16)]
        ref = None
        for c in combos:
            vals = torch.zeros(1, 8, dtype=torch.long)
            vals[0, :k] = torch.tensor(c)
            obs = x7c.episodes(1, L, torch.Generator().manual_seed(11), k=k, values=vals)[0]
            if ref is None:
                ref = obs[:, k:]
            elif not torch.equal(ref, obs[:, k:]):
                raise AssertionError("multi-cue observations after the cue phase depend on the values")


def run():
    check_x7a()
    check_tmaze()
    check_multicue()
    return "labeller matches scalar reference (40 fields x 5 actions); oracles succeed (T 9..90, 180, 900; k 1..8 x 200); post-cue observations cue-independent"
