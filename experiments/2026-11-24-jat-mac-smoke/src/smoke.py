"""JAT Mac smoke test: load the released checkpoint, check the token layout, run closed-loop
episodes in Atari and BabyAI on CPU and MPS, record returns and time per step. See ../plan.md."""
import argparse
import datetime
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

EXP_ID = "2026-11-24-jat-mac-smoke"
CLAIMS = ["gallouedec2024jack#C12", "gallouedec2024jack#C4"]
REPO_ID, REV = "jat-project/jat", "911dd7447b1c56bece63424c888d2217828c981e"
SAFETENSORS_OID = "5b66502d0c5687998593d89582dc18697d3d144871b0905c345126e632c22508"  # hub LFS oid
GAMES = ["atari-pong", "atari-breakout", "babyai-go-to-obj"]
SEEDS = [0, 1, 2]
STEP_CAP = {"atari": 2000, "babyai": 64}
CONTEXT = {"atari": 32, "babyai": 256}  # as scripts/eval_jat.py
NUM_THREADS = 4


def git(*a, cwd):
    return subprocess.check_output(["git", *a], cwd=cwd, text=True).strip()


def fetch_model():
    from huggingface_hub import snapshot_download
    t = time.time()
    path = snapshot_download(REPO_ID, revision=REV, allow_patterns=[
        "*.json", "*.py", "*.txt", "model.safetensors"], ignore_patterns=["evaluations.json"])
    return path, time.time() - t


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    t = time.time()
    model, info = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, output_loading_info=True)
    model = model.to(device).eval()
    proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    return model, proc, info, time.time() - t


class Probe:
    """Forward hooks: count non-finite action logits and record the token layout per call."""

    def __init__(self, model):
        import torch
        self.nonfinite = 0
        self.layout = []  # (timesteps in the processed input, sequence positions into the transformer)
        self._steps = None
        self._torch = torch

        def pre_model(mod, args, kwargs):
            for k in ("image_observations", "discrete_observations", "continuous_observations"):
                if kwargs.get(k) is not None:
                    self._steps = kwargs[k].shape[1]
        model.register_forward_pre_hook(pre_model, with_kwargs=True)

        def pre_tr(mod, args, kwargs):
            self.layout.append((self._steps, kwargs["inputs_embeds"].shape[1]))
        model.transformer.register_forward_pre_hook(pre_tr, with_kwargs=True)

        def post_dec(mod, args, out):
            self.nonfinite += int((~self._torch.isfinite(out)).sum())
        model.single_discrete_decoder.register_forward_hook(post_dec)


def sync(device):
    import torch
    if device == "mps":
        torch.mps.synchronize()


def rollout(model, proc, make, task, seed, device, step_cap):
    """One episode, mirroring scripts/eval_jat.py (KV cache reset, fake-done handling)."""
    import torch
    domain = task.split("-")[0]
    kw = {"clip_reward": False} if domain == "atari" else {}
    env = make(task, **kw)
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    reward, rewards, steps, done, capped = None, [], 0, False, False
    model_t, env_t = [], []
    model.reset_rl()
    while not done:
        t0 = time.perf_counter()
        action = model.get_next_action(proc, **obs, reward=reward, action_space=env.action_space,
                                       context_window=CONTEXT[domain])
        sync(device)
        t1 = time.perf_counter()
        obs, reward, term, trunc, info = env.step(action)
        t2 = time.perf_counter()
        model_t.append(t1 - t0)
        env_t.append(t2 - t1)
        done = term or trunc
        if done and domain == "atari" and "episode" not in info:  # life lost, not game over
            obs, info = env.reset()
            done = False
        rewards.append(reward)
        steps += 1
        if steps >= step_cap and not done:
            capped = True
            break
    env.close()
    return dict(task=task, seed=seed, ret=float(sum(rewards)), steps=steps, capped=capped,
                model_ms=1000 * statistics.mean(model_t), env_ms=1000 * statistics.mean(env_t),
                model_ms_median=1000 * statistics.median(model_t))


def run_device(device, path, games, seeds, caps, jat_dir):
    import torch
    import ale_py  # noqa: F401  (registers ALE/* with gymnasium 0.29.1)
    from jat.eval.rl import make
    torch.set_num_threads(NUM_THREADS)
    model, proc, info, load_s = load(path, device)
    _, _, _, warm_load_s = None, None, None, None
    probe = Probe(model)
    eps = []
    for task in games:
        for seed in seeds:
            e = rollout(model, proc, make, task, seed, device, caps[task.split("-")[0]])
            print(f"[{device}] {task} seed {seed}: return {e['ret']:.1f} steps {e['steps']}"
                  f"{' (capped)' if e['capped'] else ''} model {e['model_ms']:.1f} ms env {e['env_ms']:.2f} ms",
                  flush=True)
            eps.append(e)
    return dict(device=device, episodes=eps, load_s=load_s, nonfinite=probe.nonfinite,
                layout=probe.layout[:3], layout_all_ratio=sorted({t // s if s else None
                                                                  for s, t in probe.layout}),
                missing=list(info["missing_keys"]), unexpected=list(info["unexpected_keys"]),
                mismatched=[list(map(str, m)) for m in info.get("mismatched_keys", [])],
                params=sum(p.numel() for p in model.parameters()),
                max_position_embeddings=model.config.max_position_embeddings,
                window_size=model.config.window_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--jat-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--devices", default="cpu,mps")
    ap.add_argument("--finalize", action="store_true", help="only build results from runs/out/*.json")
    args = ap.parse_args()
    runs, exp = Path(args.runs_dir), Path(args.exp_dir)
    out = runs / ("smoke" if args.smoke else "out")
    out.mkdir(parents=True, exist_ok=True)
    if args.finalize:
        return finalize(runs, exp, Path(args.jat_dir))
    t_start = time.time()
    path, dl_s = fetch_model()
    print(f"model snapshot {path} ({dl_s:.1f}s)", flush=True)
    st_sha = sha256(Path(path) / "model.safetensors")
    print("safetensors sha256", st_sha, "match", st_sha == SAFETENSORS_OID, flush=True)
    meta = dict(snapshot=path, download_s=dl_s, safetensors_sha256=st_sha,
                sha_match=int(st_sha == SAFETENSORS_OID),
                code_sha256={n: sha256(Path(path) / n) for n in
                             ("modeling_jat.py", "configuration_jat.py", "processing_jat.py", "config.json")})
    if args.smoke:
        r = run_device("cpu", path, ["atari-pong"], [0], {"atari": 20, "babyai": 20}, args.jat_dir)
        assert r["nonfinite"] == 0 and r["layout_all_ratio"] == [2], r
        (out / "smoke.json").write_text(json.dumps(dict(meta=meta, run=r), indent=1))
        print("smoke PASS", f"{time.time() - t_start:.0f}s")
        return
    for d in args.devices.split(","):
        if d == "mps":
            import torch
            assert torch.backends.mps.is_available(), "MPS not available"
        t_dev = time.time()
        r = run_device(d, path, GAMES, SEEDS, STEP_CAP, args.jat_dir)
        r["meta"] = meta
        r["wall_s"] = time.time() - t_dev  # this device's episodes plus model load
        (out / f"{d}.json").write_text(json.dumps(r, indent=1))
    if (out / "cpu.json").exists():
        finalize(runs, exp, Path(args.jat_dir))


def normalized(task, scores, jat_dir):
    d = json.loads((jat_dir / "jat/eval/rl/scores_dict.json").read_text())[task]
    lo, hi = d["random"]["mean"], d["expert"]["mean"]
    return [(s - lo) / (hi - lo) for s in scores]


def finalize(runs, exp, jat_dir):
    out = runs / "out"
    res = {d: json.loads((out / f"{d}.json").read_text()) for d in ("cpu", "mps") if (out / f"{d}.json").exists()}
    cpu = res["cpu"]
    eps = cpu["episodes"]
    metrics = {}

    def put(k, v, n, unit="1"):
        metrics[k] = dict(value=v, unit=unit, n=n)

    ratio = cpu["layout_all_ratio"]
    load_ok = int(not cpu["missing"] and not cpu["unexpected"])
    put("load_ok", load_ok, 1)
    put("weights_sha256_match", cpu["meta"]["sha_match"], 1)
    put("episodes_completed", sum(1 for e in eps if not e["capped"]), len(eps))
    put("nonfinite_count", cpu["nonfinite"], sum(e["steps"] for e in eps))
    tps = ratio[0] if len(ratio) == 1 else -1
    put("tokens_per_timestep", tps, len(cpu["layout"]))
    put("window_timesteps", cpu["max_position_embeddings"] // tps if tps > 0 else -1, 1)
    put("param_count", cpu["params"], 1)
    put("model_load_s_cpu", cpu["load_s"], 1, "s")
    for d, r in res.items():
        put(f"nonfinite_count_{d}", r["nonfinite"], sum(e["steps"] for e in r["episodes"]))
        for task in GAMES:
            es = [e for e in r["episodes"] if e["task"] == task]
            short = task.split("-", 1)[1].replace("-", "")
            rets = [e["ret"] for e in es]
            put(f"return_{short}_{d}", statistics.mean(rets), len(rets))
            put(f"norm_return_{short}_{d}", statistics.mean(normalized(task, rets, jat_dir)), len(rets))
            put(f"steps_{short}_{d}", statistics.mean(e["steps"] for e in es), len(es))
            put(f"ms_model_{short}_{d}", statistics.mean(e["model_ms"] for e in es), len(es), "ms")
            put(f"ms_env_{short}_{d}", statistics.mean(e["env_ms"] for e in es), len(es), "ms")
        if d != "cpu":
            put(f"model_load_s_{d}", r["load_s"], 1, "s")

    ev = {}
    ev[CLAIMS[0]] = (f"exploratory run, capped at inconclusive. load_ok {load_ok}, weights_sha256_match "
                     f"{cpu['meta']['sha_match']}, episodes_completed {metrics['episodes_completed']['value']} of "
                     f"{len(eps)}, nonfinite_count {cpu['nonfinite']}: the checkpoint loads with the hub's custom code "
                     f"and plays closed-loop episodes")
    ev[CLAIMS[1]] = (f"exploratory run, capped at inconclusive. tokens_per_timestep {tps} (sequence positions per "
                     f"processed timestep, all calls: {ratio}), window_timesteps {metrics['window_timesteps']['value']} "
                     f"(max_position_embeddings {cpu['max_position_embeddings']})")
    exp_rel = f"experiments/{EXP_ID}"
    repo = Path(git("rev-parse", "--show-toplevel", cwd=exp))
    sha = git("rev-parse", "HEAD", cwd=repo)
    dirty_lines = [l for l in git("status", "--porcelain", cwd=repo).splitlines()
                   if not any(l[3:].startswith(p) for p in (f"{exp_rel}/results.", f"{exp_rel}/figures/"))]
    inputs = {n: git("rev-parse", f"HEAD:{exp_rel}/{n}", cwd=repo) for n in ("src", "run.sh", "requirements.txt")}
    import torch
    pk = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    results = {
        "schema": 1, "experiment": EXP_ID, "date": datetime.date.today().isoformat(),
        "commits": [{"repo": "research", "sha": sha, "dirty": bool(dirty_lines)}],
        "inputs": inputs, "exploratory": True, "tests": CLAIMS,
        "verdicts": {c: {"verdict": "inconclusive", "evidence": ev[c]} for c in CLAIMS},
        "metrics": metrics, "seeds": SEEDS,
        "wall_clock_s": round(sum(r["wall_s"] for r in res.values()), 1),  # devices ran as separate invocations; excludes env setup
        "env": {"python": platform.python_version(),
                "os": f"macOS {platform.mac_ver()[0]} {platform.machine()}", "packages": pk,
                "gpu": ("Apple MPS (torch %s)" % torch.__version__) if "mps" in res else None},
        "provenance": {"jat_commit": git("rev-parse", "HEAD", cwd=jat_dir), "model": f"{REPO_ID}@{REV}",
                       "model_safetensors_sha256": cpu["meta"]["safetensors_sha256"],
                       "remote_code_sha256": cpu["meta"]["code_sha256"],
                       "download_s": round(cpu["meta"]["download_s"], 1),
                       "missing_keys": cpu["missing"], "unexpected_keys": cpu["unexpected"],
                       "layout_first_calls": cpu["layout"], "step_cap": STEP_CAP,
                       "episodes": {d: r["episodes"] for d, r in res.items()}},
    }
    (exp / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    rows = "\n".join(f"| {k} | {m['value']:.6g} | {m['unit']} | {m['n']} |  |" for k, m in metrics.items())
    vrows = "\n".join(f"| {c} | inconclusive | {ev[c]} |" for c in CLAIMS)
    eprow = "\n".join(f"| {d} | {e['task']} | {e['seed']} | {e['ret']:.1f} | {e['steps']} | {'yes' if e['capped'] else 'no'} "
                      f"| {e['model_ms']:.1f} | {e['env_ms']:.2f} |" for d, r in res.items() for e in r["episodes"])
    (exp / "results.generated.md").write_text(f"""# JAT Mac smoke test: results

Experiment `{EXP_ID}`, run on {results['date']} at research `{sha[:7]}` (code id `inputs.src` `{inputs['src'][:7]}`). Plan:
[plan.md](plan.md). Machine-readable results: [results.json](results.json).

## Verdicts

| Claim | Verdict | Evidence |
|---|---|---|
{vrows}

## Metrics

| Metric | Value | Unit | n | CI |
|---|---|---|---|---|
{rows}

## Episodes

| device | task | seed | return | agent steps | capped | model ms/step | env ms/step |
|---|---|---|---|---|---|---|---|
{eprow}
""")
    print("wrote results.json, results.generated.md")


if __name__ == "__main__":
    main()
