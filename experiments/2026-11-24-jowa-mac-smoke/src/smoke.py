"""JOWA Mac smoke test: load tokenizer + JOWA-{40M,150M}, play capped Atari episodes with and
without beam-search planning on CPU and MPS, record timing and sanity metrics. See ../plan.md."""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from importlib import metadata
from pathlib import Path

EXP_ID = "2026-11-24-jowa-mac-smoke"
CLAIM = "cheng2025jointly#C16"
JOWA_COMMIT = "8cb02c5e6768a948c566a9012ac8a4fc420c6970"
GAMES = ["Assault", "Seaquest"]
SIZES = ["JOWA_40M", "JOWA_150M"]
SEEDS = [0, 1]
STEPS_NOPLAN, STEPS_PLAN = 300, 60
CACHE = Path.home() / ".cache" / "jowa"
DRIVE_FOLDER = "1YHaCemhobchJWE5zt28TUrKp2tp8yysF"
FILES = {  # name: (sha256 of the official Drive file, bytes)
    "tokenizer.pt": ("4f624acf7ae82cfdeb6bc3b11651d073bc168fa5300f70a3b8ba029b07c68a4a", 75162490),
    "JOWA_40M.pt": ("c1c3d8242141725a2a4bfa61e899e18ff8153a2600a2803d7aa98405366fd771", 76485274),
    "JOWA_150M.pt": ("3f60776688ead94a684b359f23709327600af5b20a109b48e649cedaee37c5b1", 511212618),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def fetch_weights(names):
    """Official Google Drive folder from the README, one file at a time (GitHub LFS is disabled)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    missing = [n for n in names if not (CACHE / n).exists() or (CACHE / n).stat().st_size != FILES[n][1]]
    if missing:
        import gdown
        listing = gdown.download_folder(id=DRIVE_FOLDER, skip_download=True, quiet=True, use_cookies=False)
        ids = {Path(x.path).name: x.id for x in listing}
        for n in missing:
            gdown.download(id=ids[n], output=str(CACHE / n), quiet=True)
    out = {}
    for n in names:
        got = sha256(CACHE / n)
        assert got == FILES[n][0], f"{n}: sha256 {got} != expected"
        out[n] = got
    return out


def sync(device):
    import torch
    if device == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--jowa-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--devices", default=None, help="comma list (default: cpu,mps; cpu in --smoke)")
    ap.add_argument("--sizes", default=",".join(SIZES))
    ap.add_argument("--no-write", action="store_true", help="dev run: skip results.json")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    runs, exp, jowa = Path(args.runs_dir), Path(args.exp_dir), Path(args.jowa_dir)
    t_start = time.time()

    sys.path.insert(0, str(exp / "src"))
    import jowa_patch
    state = jowa_patch.install(str(jowa / "src"))
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from action_tokenizer import ATARI_NUM_ACTIONS, GAME_NAMES, LIMITED_ACTION_TO_FULL_ACTION
    from agent import Agent
    from envs import AtariEnvWrapper, SingleProcessEnv
    from game import AgentEnv
    from models.jowa_model import JOWAModel
    from utils import set_seed

    smoke = args.smoke
    sizes = ["JOWA_40M"] if smoke else args.sizes.split(",")
    devices = (args.devices or ("cpu" if smoke else "cpu,mps")).split(",")
    devices = sorted(devices, key=lambda d: d != "mps")  # MPS first: frees the GPU slot early
    games = GAMES[:1] if smoke else GAMES
    seeds = SEEDS[:1] if smoke else SEEDS
    steps_noplan, steps_plan = (20, 5) if smoke else (STEPS_NOPLAN, STEPS_PLAN)
    if "cpu" in devices:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    if "mps" in devices:
        assert torch.backends.mps.is_available(), "MPS not available"

    weights = fetch_weights(["tokenizer.pt"] + [f"{s}.pt" for s in sizes])
    plan_hp = {s: json.load(open(jowa / "eval" / f"{s}_raw_scores.json")) for s in sizes}

    def make_agent(m, game, device, buffer_size, plan, bw, hz):
        dev = torch.device(device)
        a = Agent(m["tokenizer"], m["model"], GAME_NAMES.index(game), 1, "float32", buffer_size, dev,
                  should_plan=plan, beam_width=bw, horizon=hz).to(dev)
        return a.eval()

    # --- load models on CPU: upstream's own loader, key check, load time, sizes ----------------------
    models = {}
    for size in sizes:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(jowa / "config"), version_base="1.1"):
            cfg = compose(config_name="eval", overrides=[f"transformer={size}", f"critic_head={size}"])
        tokenizer = instantiate(cfg.tokenizer)
        model = JOWAModel(obs_vocab_size=cfg.tokenizer.vocab_size, act_vocab_size=ATARI_NUM_ACTIONS,
                          config_transformer=instantiate(cfg.transformer), config_critic_arch=cfg.critic_head,
                          config_critic_train=cfg.action, device=torch.device("cpu"))
        m = dict(cfg=cfg, tokenizer=tokenizer, model=model)
        agent0 = make_agent(m, games[0], "cpu", 8, False, 1, 0)
        t0 = time.time()
        skipped = jowa_patch.load_checkpoint(agent0, str(CACHE), size)  # KeyError on any other missing key
        load_s = time.time() - t0
        missing = [k for k in skipped if not k.endswith(".attn.mask")]  # none expected
        m.update(load_s=load_s, skipped_mask_keys=len(skipped), nparams=sum(p.numel() for p in model.parameters()),
                 ntok=sum(p.numel() for p in tokenizer.parameters()), missing=missing,
                 tokens_per_frame=model.config_transformer.tokens_per_block - 1,
                 vocab=int(tokenizer.vocab_size), layers=model.config_transformer.num_layers,
                 embed=model.config_transformer.embed_dim)
        models[size] = m
        print(f"{size}: load {load_s:.2f}s, params {m['nparams']:,}, tokenizer params {m['ntok']:,}, "
              f"missing keys {len(missing)} (+{len(skipped)} attn.mask buffers built at init), tokens/frame {m['tokens_per_frame']}, vocab {m['vocab']}", flush=True)

    # --- episodes -----------------------------------------------------------------------------------
    cells, nonfinite, bad_actions, episodes_ok, exceptions = [], 0, 0, 0, []
    for device in devices:
        for size in sizes:
            m = models[size]
            m["tokenizer"].to(device)
            m["model"].to(device)
            # plain tensor attributes (not buffers) that .to() does not move; created on CPU at init
            for attr in ("atoms_support", "log_alpha"):
                if hasattr(m["model"], attr):
                    setattr(m["model"], attr, getattr(m["model"], attr).to(device))
            for game in games:
                hp = plan_hp[size][game]
                legal = set(LIMITED_ACTION_TO_FULL_ACTION[game].tolist())
                for planning in (False, True):
                    for seed in seeds:
                        buf = int(hp["buffer_size"][0]) if planning else 8
                        bw, hz = int(hp["beam_width"]), int(hp["planning_horizon"])
                        cap = (steps_plan // 2 if device == "mps" else steps_plan) if planning else steps_noplan  # MPS planning is slow: half cap
                        cell = dict(device=device, size=size, game=game, planning=planning, seed=seed,
                                    buffer_size=buf, beam_width=bw if planning else None,
                                    horizon=hz if planning else None, step_cap=cap)
                        h = None
                        try:
                            state["seed"] = seed
                            set_seed(seed)
                            env = SingleProcessEnv(AtariEnvWrapper(game).create_env)
                            agent = make_agent(m, game, device, buf, "beam_search" if planning else False, bw, hz)
                            aenv = AgentEnv(agent, env, "atari", False, verbose=False)
                            nf = {"n": 0}

                            def hook(mod, inp, out, nf=nf):
                                q = getattr(out, "logits_q", None)
                                if q is not None:
                                    nf["n"] += int((~torch.isfinite(q)).sum())
                            h = m["model"].register_forward_hook(hook)
                            act_times, acts = [], []
                            orig_act = agent.act

                            def timed_act(obs, orig_act=orig_act, device=device):
                                sync(device)
                                t = time.perf_counter()
                                a = orig_act(obs)
                                sync(device)
                                act_times.append(time.perf_counter() - t)
                                acts.append(int(a[0]))
                                return a
                            agent.act = timed_act
                            aenv.reset()
                            t_ep = time.time()
                            done, steps = False, 0
                            while steps < cap and not done:
                                _, _, d, info = aenv.step()
                                done = bool(d[0])
                                steps += 1
                            ep_s = time.time() - t_ep
                            env.env._env.close()  # upstream AtariEnvWrapper has no close()
                            bad = sum(1 for a in acts if a not in legal)
                            nonfinite += nf["n"]
                            bad_actions += bad
                            episodes_ok += int(bad == 0)
                            ts = sorted(act_times)
                            cell.update(steps=steps, game_over=done, ret=float(info["return"][0]),
                                        clipped_return=float(info["clipped_return"][0]),
                                        ms_per_step_median=1000 * ts[len(ts) // 2],
                                        ms_per_step_mean=1000 * sum(ts) / len(ts),
                                        ms_first_step=1000 * act_times[0], episode_s=ep_s,
                                        nonfinite=nf["n"], illegal_actions=bad,
                                        distinct_actions=len(set(acts)), ok=bad == 0)
                        except Exception as e:  # recorded; the episode counts as not completed
                            import traceback
                            traceback.print_exc()
                            cell.update(ok=False, error=repr(e))
                            exceptions.append(f"{device}/{size}/{game}/plan={planning}/seed={seed}: {e!r}")
                        finally:
                            if h is not None:
                                h.remove()
                        cells.append(cell)
                        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                          for k, v in cell.items()}), flush=True)
            m["tokenizer"].to("cpu")
            m["model"].to("cpu")
            if device == "mps":
                torch.mps.empty_cache()

    # --- tokenizer sanity: reconstruct real frames after 100 seeded random legal actions (CPU) --------
    import numpy as np
    tok = models[sizes[0]]["tokenizer"].to("cpu").eval()
    recon = {}
    for g in games:
        state["seed"] = 0
        e = AtariEnvWrapper(g).create_env()
        obs = e.reset()
        rng = np.random.RandomState(0)
        for _ in range(100):
            obs, *_ = e.step(int(rng.choice(LIMITED_ACTION_TO_FULL_ACTION[g])))
        x = torch.from_numpy(obs).permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            enc = tok.encode(x, should_preprocess=True)
            rec = tok.decode(enc.z_quantized, should_postprocess=True).clamp(0, 1)
        recon[g] = dict(tokens_shape=list(enc.tokens.shape), distinct_tokens=int(enc.tokens.unique().numel()),
                        recon_mse=float(((rec - x) ** 2).mean()), frame_variance=float(x.var()))
        e._env.close()
    print("tokenizer recon:", recon, flush=True)

    planned = len(cells)

    def agg(sel, key):
        v = [c[key] for c in cells if sel(c) and key in c]
        return sum(v) / len(v) if v else float("nan")
    n_loaded = sum(1 for s in sizes if not models[s]["missing"] and
                   all(c.get("ok") for c in cells if c["size"] == s))
    metrics = {"models_loaded": (n_loaded, len(sizes)), "nonfinite_count": (nonfinite, planned),
               "episodes_completed": (episodes_ok, planned), "episodes_planned": (planned, planned),
               "illegal_action_count": (bad_actions, planned)}
    for s in sizes:
        tag = s.split("_")[1].lower()
        metrics[f"params_{tag}"] = (models[s]["nparams"], 1)
        metrics[f"load_s_{tag}"] = (models[s]["load_s"], 1)
        for d in devices:
            for p in (False, True):
                sel = lambda c, s=s, d=d, p=p: c["size"] == s and c["device"] == d and c["planning"] == p
                metrics[f"ms_per_step_{tag}_{d}_{'plan' if p else 'noplan'}"] = (
                    agg(sel, "ms_per_step_median"), sum(1 for c in cells if sel(c)))
    metrics["tokenizer_params"] = (models[sizes[0]]["ntok"], 1)
    metrics["tokens_per_frame"] = (models[sizes[0]]["tokens_per_frame"], 1)
    metrics["tokenizer_vocab"] = (models[sizes[0]]["vocab"], 1)
    for g, r in recon.items():
        metrics[f"tokenizer_recon_mse_{g.lower()}"] = (r["recon_mse"], 1)
        metrics[f"frame_variance_{g.lower()}"] = (r["frame_variance"], 1)
    metrics["mean_clipped_return_capped"] = (agg(lambda c: True, "clipped_return"), planned)

    wall = time.time() - t_start
    claim_ok = n_loaded == len(sizes) and nonfinite == 0 and episodes_ok == planned
    print(f"wall clock {wall:.1f}s; models_loaded {n_loaded}/{len(sizes)}, nonfinite {nonfinite}, "
          f"episodes ok {episodes_ok}/{planned}, illegal actions {bad_actions}", flush=True)
    if smoke or args.no_write:
        out = runs / "smoke"
        out.mkdir(parents=True, exist_ok=True)
        (out / "cells.json").write_text(json.dumps(cells, indent=1))
        sys.exit(0 if claim_ok else 1)

    # --- results.json -----------------------------------------------------------------------------------
    def git(*a):
        return subprocess.run(["git", *a], cwd=exp, check=True, capture_output=True, text=True).stdout.strip()
    sha = git("rev-parse", "HEAD")
    eid = f"experiments/{EXP_ID}"
    outputs = tuple(f"{eid}/{o}" for o in ("results.json", "results.md", "figures"))
    dirty = any(not l[3:].startswith(outputs) for l in git("status", "--porcelain").splitlines())
    inputs = {"src": git("rev-parse", f"HEAD:{eid}/src"), "run.sh": git("rev-parse", f"HEAD:{eid}/run.sh"),
              "requirements.txt": git("rev-parse", f"HEAD:{eid}/requirements.txt")}
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    evidence = (f"exploratory run, capped at inconclusive (the observed values alone would read as "
                f"{'supports' if claim_ok else 'not supports'}): models_loaded {n_loaded}/{len(sizes)}, "
                f"nonfinite_count {nonfinite}, episodes_completed {episodes_ok}/{planned}; the weights are not "
                f"in the GitHub repo (Git LFS disabled) and come from the README's Google Drive folder")
    res = {
        "schema": 1, "experiment": EXP_ID, "date": datetime.date.today().isoformat(),
        "commits": [{"repo": "research", "sha": sha, "dirty": dirty}],
        "inputs": inputs, "exploratory": True, "tests": [CLAIM],
        "verdicts": {CLAIM: {"verdict": "inconclusive", "evidence": evidence}},
        "metrics": {k: {"value": v, "unit": "ms" if k.startswith("ms_") else "s" if k.startswith("load_s") else "1",
                        "n": n} for k, (v, n) in metrics.items()},
        "seeds": seeds, "wall_clock_s": round(wall, 1),
        "env": {"python": platform.python_version(), "os": f"macOS {platform.mac_ver()[0]} {platform.machine()}",
                "packages": packages,
                "gpu": f"Apple MPS (torch {torch.__version__})" if "mps" in devices else None},
        "provenance": {
            "jowa_commit": JOWA_COMMIT, "weights_source": f"Google Drive folder {DRIVE_FOLDER} (README link)",
            "weights_sha256": weights, "weights_bytes": {n: FILES[n][1] for n in weights},
            "loads": {s: {**{k: models[s][k] for k in ("load_s", "nparams", "ntok", "tokens_per_frame",
                                                       "vocab", "layers", "embed", "skipped_mask_keys")},
                          "missing_keys": models[s]["missing"]} for s in sizes},
            "tokenizer_check": recon, "cells": cells, "exceptions": exceptions,
            "steps_noplan_cap": steps_noplan, "steps_plan_cap": steps_plan,
        },
    }
    (exp / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    print("wrote results.json", flush=True)


if __name__ == "__main__":
    main()
