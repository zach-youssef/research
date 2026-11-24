"""OCCAM/HackAtari Mac smoke test driver. See ../plan.md.

Loads the released OCAtariWrappers checkpoints through upstream's own loader, plays them in
HackAtari on Pong, Freeway and Breakout (original and modified), and writes results.json,
results.md and a log. Smoke mode: one short Pong episode, writes only under runs/<id>/smoke/.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

EXP_ID = "2026-11-24-occam-hackatari-mac-smoke"
CLAIMS = ["delfosse2025intelligence#C8", "delfosse2025intelligence#C9", "delfosse2025intelligence#C11"]
N_EPISODES = 10
MAX_STEPS = 10000
OBS_FRAMES = 200  # observations kept per agent for the MPS microbenchmark

# game -> (conditions, file name of the pixel agent)
GAMES = {
    "Pong": (["orig", "lazy_enemy", "hidden_enemy"], "ppo"),
    "Freeway": (["orig", "stop_all_cars"], "pixel_ppo"),
    "Breakout": (["orig", "red"], "ppo"),
}
# Freeway: HackAtari's current name for the paper-era "stop_all_cars" (same function)
MODIFS = {
    "orig": [], "lazy_enemy": ["lazy_enemy"], "hidden_enemy": ["hidden_enemy"],
    "stop_all_cars": ["stop_all_cars_tunnel"], "red": ["color_player_and_ball_red"],
}
# agent -> (file stem or None for the game's pixel agent, obs_mode, wrapper as scripts/eval.py pairs it)
AGENTS = {
    "pixel": (None, "dqn", None),
    "binary": ("ppo_binary", "dqn", "binary"),
    "classes": ("ppo_classes", "dqn", "classes"),
    "object": ("ppo_object", "dqn", "pixels"),
    "planes": ("ppo_planes", "dqn", "planes"),
    "objvec": ("obj_vector", "obj", None),
}


def sh(*cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_path(wrappers_dir, game, agent):
    stem = AGENTS[agent][0] or GAMES[game][1]
    return Path(wrappers_dir) / "models" / game / "0" / f"{stem}.cleanrl_model"


def check_weights(wrappers_dir, exp_dir):
    """Assert every weight file's sha256 equals the committed list; return {relpath: sha}."""
    expected = {}
    for line in (Path(exp_dir) / "src" / "weights.sha256").read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            expected[rel.strip()] = digest
    got = {}
    for game in GAMES:
        for agent in AGENTS:
            p = weight_path(wrappers_dir, game, agent)
            rel = str(p.relative_to(wrappers_dir))
            got[rel] = sha256(p)
    assert got == expected, "weights differ from src/weights.sha256: %s" % {
        k: (got.get(k), expected.get(k)) for k in set(got) | set(expected) if got.get(k) != expected.get(k)}
    return got


class FireReset:
    """gym.Wrapper pressing FIRE (and action 2) after reset, as stable-baselines3's FireResetEnv."""

    @staticmethod
    def wrap(env):
        import gymnasium as gym

        class _FireReset(gym.Wrapper):
            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                for a in (1, 2):
                    obs, _, term, trunc, info = self.env.step(a)
                    if term or trunc:
                        obs, info = self.env.reset(**kwargs)
                return obs, info

        return _FireReset(env)


def make_env(game, cond, agent, render=False):
    from hackatari import HackAtari
    import ocatari_wrappers as ow
    obs_mode, wrapper = AGENTS[agent][1], AGENTS[agent][2]
    env = HackAtari(
        game, MODIFS[cond], None, dopamine_pooling=False, game_mode=0, difficulty=0,
        render_mode="rgb_array" if render else None, obs_mode=obs_mode, mode="ram", hud=False,
        render_oc_overlay=False, buffer_window_size=4, frameskip=4,
        repeat_action_probability=0.25, full_action_space=False)
    if MODIFS[cond]:  # HackAtari silently ignores unknown modification names
        hooks = (len(env.step_modifs) + len(env.reset_modifs) + len(env.post_detection_modifs)
                 + len(getattr(env, "inpainting_modifs", [])) + len(getattr(env, "place_above_modifs", [])))
        assert hooks > 0, f"modification {MODIFS[cond]} registered no hooks for {game}"
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireReset.wrap(env)
    cls = {"binary": ow.BinaryMaskWrapper, "pixels": ow.PixelMaskWrapper,
           "classes": ow.ObjectTypeMaskWrapper, "planes": ow.ObjectTypeMaskPlanesWrapper}
    if wrapper:
        env = cls[wrapper](env)
    return env


def run_episode(env, policy, seed, cap, keep_obs=None):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    obs, _ = env.reset(seed=seed)
    ret, steps, done, trunc = 0.0, 0, False, False
    pol_s = env_s = 0.0
    n_pol = 0
    for _ in range(int(rng.randint(1, 31))):  # random NOOP start
        obs, r, te, tr, _ = env.step(0)
        ret += r
        if te or tr:
            done = True
            break
    while not done:
        if keep_obs is not None and len(keep_obs) < OBS_FRAMES:
            keep_obs.append(np.array(obs, dtype=np.float32))
        t0 = time.perf_counter()
        with torch.no_grad():
            a = int(policy(torch.Tensor(obs).unsqueeze(0))[0])
        t1 = time.perf_counter()
        obs, r, te, tr, _ = env.step(a)
        env_s += time.perf_counter() - t1
        pol_s += t1 - t0
        n_pol += 1
        ret += r
        steps += 1
        done = te or tr
        if steps >= cap and not done:
            trunc, done = True, True
    return dict(seed=seed, ret=ret, steps=steps, truncated=bool(trunc), pol_s=pol_s, env_s=env_s, n_pol=n_pol)


def run_cell(game, cond, agent, wrappers_dir, n_eps, cap, runs_dir, load_ms, keep_obs=False):
    from load_agent import load_agent
    env = make_env(game, cond, agent)
    t0 = time.perf_counter()
    model, policy = load_agent(str(weight_path(wrappers_dir, game, agent)), env, "cpu")
    load_ms.setdefault(f"{game}/{agent}", (time.perf_counter() - t0) * 1000)
    model.eval()
    kept = [] if keep_obs else None
    eps = []
    for seed in range(n_eps):
        ep = run_episode(env, policy, seed, cap, keep_obs=kept if seed == 0 else None)
        ep.update(game=game, cond=cond, agent=agent)
        eps.append(ep)
    if keep_obs:
        np.savez(Path(runs_dir) / f"obs_{game}_{agent}.npz", obs=np.stack(kept),
                 n_actions=env.action_space.n)
    env.close()
    return eps


def smoke(args, wrappers_dir):
    from load_agent import load_agent
    out = Path(args.runs_dir) / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    for cond in ("orig", "lazy_enemy"):  # also checks that the modification registers hooks
        env = make_env("Pong", cond, "planes")
        model, policy = load_agent(str(weight_path(wrappers_dir, "Pong", "planes")), env, "cpu")
        first = model.network[0]
        assert first.weight.shape[1] == env.observation_space.shape[0], "obs/model channel mismatch"
        ep = run_episode(env, policy, 0, 300)
        assert ep["steps"] > 0
        print("smoke", cond, ep)
        env.close()
    (out / "smoke.json").write_text(json.dumps(dict(ok=True), indent=1))
    print("smoke ok")


def write_videos(wrappers_dir, cache_dir, wr_steps=500):
    import cv2
    from load_agent import load_agent
    vdir = Path(cache_dir) / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for game, (conds, _) in GAMES.items():
        mod = conds[1]
        envs = [make_env(game, c, "pixel", render=True) for c in ("orig", mod)]
        pols = [load_agent(str(weight_path(wrappers_dir, game, "pixel")), e, "cpu")[1] for e in envs]
        torch.manual_seed(0)
        obss = [e.reset(seed=0)[0] for e in envs]
        done = [False, False]
        frames = [e.render() for e in envs]
        path = vdir / f"{game}_pixel_orig_vs_{mod}.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 420))
        for _ in range(wr_steps):
            for i, e in enumerate(envs):
                if done[i]:
                    continue
                with torch.no_grad():
                    a = int(pols[i](torch.Tensor(obss[i]).unsqueeze(0))[0])
                obss[i], _, te, tr, _ = e.step(a)
                done[i] = te or tr
                frames[i] = e.render()
            both = np.hstack([cv2.resize(f, (320, 420), interpolation=cv2.INTER_NEAREST) for f in frames])
            vw.write(cv2.cvtColor(both, cv2.COLOR_RGB2BGR))
            if all(done):
                break
        vw.release()
        for e in envs:
            e.close()
        paths.append(str(path))
    return paths


def mps_bench(runs_dir, wrappers_dir, reps=200, warm=20):
    from types import SimpleNamespace
    from load_agent import load_agent
    assert torch.backends.mps.is_available(), "MPS not available"
    res = {}
    for agent in AGENTS:
        d = np.load(Path(runs_dir) / f"obs_Pong_{agent}.npz")
        obs = d["obs"]
        stub = SimpleNamespace(observation_space=SimpleNamespace(shape=obs.shape[1:]),
                               action_space=SimpleNamespace(n=int(d["n_actions"])),
                               obs_mode=AGENTS[agent][1])
        model, _ = load_agent(str(weight_path(wrappers_dir, "Pong", agent)), stub, "cpu")
        model.eval()

        def logits(m, x):
            h = m.network(x / 255.0) if hasattr(m, "get_action_and_value") and AGENTS[agent][1] == "dqn" else m.network(x)
            return m.actor(h)

        def bench(dev):
            m = model.to(dev)
            outs, ts = [], []
            for i in range(warm + reps):
                x = torch.from_numpy(obs[i % len(obs)]).unsqueeze(0).to(dev)
                t0 = time.perf_counter()
                with torch.no_grad():
                    y = logits(m, x)
                    y = y.cpu()  # forces synchronisation
                dt = time.perf_counter() - t0
                if i >= warm:
                    ts.append(dt * 1000)
                    outs.append(y)
            return statistics.mean(ts), torch.cat(outs)

        cpu_ms, cpu_out = bench("cpu")
        mps_ms, mps_out = bench("mps")
        res[agent] = dict(cpu_ms=cpu_ms, mps_ms=mps_ms, max_abs_diff=float((cpu_out - mps_out).abs().max()),
                          n=reps)
        print("mps", agent, res[agent])
    (Path(runs_dir) / "mps_bench.json").write_text(json.dumps(res, indent=1))
    return res


def bootstrap_ci(x, seed=0, level=0.95, n=1000):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def mean(m, key):
    return m[key]["value"]


def evaluate_criteria(m):
    """The plan's criteria applied to the metrics: returns {claim: (branch, evidence)}."""
    g = lambda k: mean(m, k)
    out = {}
    po, pl, ao, al = g("pong_pixel_orig_mean"), g("pong_pixel_lazy_enemy_mean"), g("pong_planes_orig_mean"), g("pong_planes_lazy_enemy_mean")
    if po >= 10 and pl <= po - 15 and al >= ao - 8:
        b = "supports"
    elif pl >= po - 5:
        b = "contradicts"
    else:
        b = "inconclusive"
    out[CLAIMS[0]] = (b, f"pong pixel {po:.1f} -> lazy_enemy {pl:.1f}; planes {ao:.1f} -> {al:.1f}")
    bo, br = g("breakout_pixel_orig_mean"), g("breakout_pixel_red_mean")
    masks = {a: (g(f"breakout_{a}_orig_mean"), g(f"breakout_{a}_red_mean")) for a in ("binary", "classes", "planes")}
    if bo >= 10 and br <= 0.5 * bo and all(r >= 0.7 * o for o, r in masks.values()):
        b = "supports"
    elif br >= 0.8 * bo:
        b = "contradicts"
    else:
        b = "inconclusive"
    ev = f"breakout pixel {bo:.1f} -> red {br:.1f}; " + "; ".join(f"{a} {o:.1f} -> {r:.1f}" for a, (o, r) in masks.items())
    out[CLAIMS[1]] = (b, ev)
    fo, fs = g("freeway_pixel_orig_mean"), g("freeway_pixel_stop_all_cars_mean")
    if fo >= 15 and fs <= 12:
        b = "supports"
    elif fs >= 25:
        b = "contradicts"
    else:
        b = "inconclusive"
    out[CLAIMS[2]] = (b, f"freeway pixel {fo:.1f} -> stop_all_cars {fs:.1f}")
    return out


def finalize(args, eps, load_ms, weights, videos, wall, mps):
    exp_dir = Path(args.exp_dir)
    repo = Path(sh("git", "rev-parse", "--show-toplevel", cwd=exp_dir))
    rel = f"experiments/{EXP_ID}"
    m = {}
    cells = {}
    for e in eps:
        cells.setdefault((e["game"], e["agent"], e["cond"]), []).append(e)
    for (game, agent, cond), es in cells.items():
        rets = [e["ret"] for e in es]
        lo, hi = bootstrap_ci(rets)
        m[f"{game.lower()}_{agent}_{cond}_mean"] = dict(
            value=float(np.mean(rets)), unit="1", n=len(rets), ci=dict(low=lo, high=hi, level=0.95))
    tot_pol = sum(e["pol_s"] for e in eps)
    tot_env = sum(e["env_s"] for e in eps)
    tot_n = sum(e["n_pol"] for e in eps)
    m["policy_ms_per_step_cpu"] = dict(value=tot_pol / tot_n * 1000, unit="ms", n=tot_n)
    m["env_ms_per_step"] = dict(value=tot_env / tot_n * 1000, unit="ms", n=tot_n)
    m["truncated_episodes"] = dict(value=sum(e["truncated"] for e in eps), unit="1", n=len(eps))
    for k, v in load_ms.items():
        m[f"load_ms_{k.replace('/', '_').lower()}"] = dict(value=v, unit="ms", n=1)
    if mps:
        for agent, r in mps.items():
            m[f"mps_ms_pong_{agent}"] = dict(value=r["mps_ms"], unit="ms", n=r["n"])
            m[f"cpu_fwd_ms_pong_{agent}"] = dict(value=r["cpu_ms"], unit="ms", n=r["n"])
            m[f"mps_cpu_max_abs_diff_pong_{agent}"] = dict(value=r["max_abs_diff"], unit="1", n=r["n"])
    crit = evaluate_criteria(m)
    verdicts = {c: dict(verdict="inconclusive",
                        evidence=f"exploratory run, capped at inconclusive; the criterion's branch is {b}: {ev}")
                for c, (b, ev) in crit.items()}
    inputs = {p: sh("git", "rev-parse", f"HEAD:{rel}/{p}", cwd=repo) for p in ("src", "run.sh", "requirements.txt")}
    freeze = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    outputs = tuple(f"{rel}/{o}" for o in ("results.json", "results.md", "figures"))
    dirty = any(not l[3:].startswith(outputs) for l in sh("git", "status", "--porcelain", cwd=repo).splitlines())
    res = dict(
        schema=1, experiment=EXP_ID, date=datetime.date.today().isoformat(),
        commits=[dict(repo="research", sha=sh("git", "rev-parse", "HEAD", cwd=repo), dirty=dirty)],
        inputs=inputs, exploratory=True, tests=CLAIMS, verdicts=verdicts, metrics=m,
        seeds=list(range(N_EPISODES)), wall_clock_s=round(wall, 1),
        env=dict(python=platform.python_version(), os=f"macOS {platform.mac_ver()[0]} {platform.machine()}",
                 packages=freeze, gpu=("Apple MPS (torch %s)" % torch.__version__) if mps else None),
        provenance=dict(
            ocatariwrappers=dict(url="https://github.com/VanillaWhey/OCAtariWrappers",
                                 commit=sh("git", "rev-parse", "HEAD", cwd=args.wrappers_dir), license="MIT"),
            hackatari=dict(url="https://github.com/k4ntz/HackAtari",
                           commit=sh("git", "rev-parse", "HEAD", cwd=args.hackatari_dir), license="MIT"),
            weights_sha256=weights, max_steps_per_episode=MAX_STEPS, episodes_per_cell=N_EPISODES,
            videos=videos, episodes_file="runs/%s/episodes.jsonl" % EXP_ID))
    (exp_dir / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    write_results_md(exp_dir, res, eps, crit)


def write_results_md(exp_dir, res, eps, crit):
    m = res["metrics"]
    L = [f"# OCCAM/HackAtari agents in the real ALE on a Mac: results", "",
         f"Experiment `{EXP_ID}`, run on {res['date']} at research `{res['commits'][0]['sha'][:7]}` "
         f"(code id `inputs.src` `{res['inputs']['src'][:7]}`). Plan: [plan.md](plan.md). "
         "Machine-readable results: [results.json](results.json). Generated by `src/run_agents.py`; "
         "the sections after `## Checks` were written by hand.", "",
         "## Verdicts", "", "| Claim | Verdict | Evidence |", "|---|---|---|"]
    for c, v in res["verdicts"].items():
        L.append(f"| {c} | {v['verdict']} | {v['evidence']} |")
    L += ["", "## Metrics", "", "Return table: mean game-score return over 10 episodes (seeds 0 to 9), "
          "95% bootstrap CI in brackets, per agent and condition.", "",
          "| Game | Condition | " + " | ".join(AGENTS) + " |", "|---|---|" + "---|" * len(AGENTS)]
    for game, (conds, _) in GAMES.items():
        for cond in conds:
            row = []
            for a in AGENTS:
                x = m[f"{game.lower()}_{a}_{cond}_mean"]
                row.append(f"{x['value']:.1f} [{x['ci']['low']:.1f}, {x['ci']['high']:.1f}]")
            L.append(f"| {game} | {cond} | " + " | ".join(row) + " |")
    L += ["", "| Metric | Value | Unit | n |", "|---|---|---|---|"]
    for k, x in m.items():
        if k.endswith("_mean") and "ci" in x:
            continue
        L.append(f"| {k} | {x['value']:.4g} | {x['unit']} | {x['n']} |")
    L += ["", "Per-episode returns, lengths and truncation flags: `runs/%s/episodes.jsonl` (git-ignored)." % EXP_ID, ""]
    (exp_dir / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--wrappers-dir", required=True)
    ap.add_argument("--hackatari-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mps-only", action="store_true", help="only the MPS microbenchmark (needs the GPU slot)")
    ap.add_argument("--finalize-only", action="store_true", help="rebuild results from saved episodes and benchmark")
    args = ap.parse_args()
    torch.set_num_threads(1)
    t_start = time.time()
    weights = check_weights(args.wrappers_dir, args.exp_dir)
    print("weights ok:", len(weights), "files")
    if args.smoke:
        return smoke(args, args.wrappers_dir)
    runs = Path(args.runs_dir)
    ep_file, state_file, mps_file = runs / "episodes.jsonl", runs / "eval_state.json", runs / "mps_bench.json"
    if args.mps_only:
        mps_bench(runs, args.wrappers_dir)
        return
    if not args.finalize_only:
        ep_file.unlink(missing_ok=True)
        mps_file.unlink(missing_ok=True)
        eps, load_ms = [], {}
        for game, (conds, _) in GAMES.items():
            for cond in conds:
                for agent in AGENTS:
                    t0 = time.time()
                    es = run_cell(game, cond, agent, args.wrappers_dir, N_EPISODES, MAX_STEPS, runs, load_ms,
                                  keep_obs=(game == "Pong" and cond == "orig"))
                    with ep_file.open("a") as f:
                        for e in es:
                            f.write(json.dumps(e) + "\n")
                    eps += es
                    print(f"{game} {cond} {agent}: mean {np.mean([e['ret'] for e in es]):.1f} "
                          f"steps {sum(e['steps'] for e in es)} trunc {sum(e['truncated'] for e in es)} "
                          f"({time.time() - t0:.1f}s)", flush=True)
        videos = write_videos(args.wrappers_dir, args.cache_dir)
        state_file.write_text(json.dumps(dict(load_ms=load_ms, videos=videos, wall=time.time() - t_start)))
    eps = [json.loads(l) for l in ep_file.read_text().splitlines()]
    st = json.loads(state_file.read_text())
    mps = json.loads(mps_file.read_text()) if mps_file.exists() else None
    finalize(args, eps, st["load_ms"], weights, st["videos"], st["wall"], mps)
    print("done")


if __name__ == "__main__":
    main()
