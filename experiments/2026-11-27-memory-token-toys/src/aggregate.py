"""Combine the three parts' outputs into results.json, the generated blocks of results.md, and figures.

Full mode writes into the experiment directory; smoke mode only under runs/<id>/smoke/.
results.md keeps its hand-written prose: only the blocks between
`<!-- generated:<name> begin -->` and `<!-- generated:<name> end -->` are rewritten.
"""
import argparse
import datetime
import json
import os
import platform
import re
import subprocess
from importlib import metadata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP_ID = "2026-11-27-memory-token-toys"
CLAIMS = ["cherepanov2026recurrent#C1", "cherepanov2026recurrent#C8", "cherepanov2026recurrent#C12"]
SEEDS = [0, 1, 2]
TOLERANCES = {  # restated from plan.md "## Success criteria"; used only for results.md's Reproduce text
    "rate_fixed_success_in": "0.05", "rate_fixed_success_t900": "0.08", "dt_success_t900": "0.08",
    "rate_fixed_success_t90": "0.05", "rate_fixed_noise_success_t90": "0.08",
    "rate_fixed_noise_junction_t90": "0.05", "rate_n1k90_success_t900": "0.08",
}


def git(*a, cwd):
    return subprocess.check_output(["git", *a], cwd=cwd, text=True).strip()


def verdicts(m):
    v = lambda k: m[k]["value"]
    out = {}
    need = {"cherepanov2026recurrent#C1": ["rate_fixed_success_in", "rate_fixed_success_t900", "dt_success_t900"],
            "cherepanov2026recurrent#C8": ["rate_fixed_success_t90", "rate_fixed_noise_success_t90",
                                           "rate_fixed_noise_junction_t90"],
            "cherepanov2026recurrent#C12": ["rate_fixed_success_t900", "rate_n1k90_success_t900"]}
    for c, keys in need.items():
        if not all(k in m for k in keys):
            out[c] = {"verdict": "inconclusive", "evidence": "metrics missing (smoke run)"}
    if "cherepanov2026recurrent#C1" not in out:
        a, b, d = v("rate_fixed_success_in"), v("rate_fixed_success_t900"), v("dt_success_t900")
        verdict = ("supports" if a >= 0.95 and b >= 0.7 and d <= 0.6 else
                   "contradicts" if a < 0.8 or b < d + 0.1 else "inconclusive")
        out["cherepanov2026recurrent#C1"] = {"verdict": verdict, "evidence": (
            f"rate_fixed_success_in {a:.3f} (>= 0.95), rate_fixed_success_t900 {b:.3f} (>= 0.7), "
            f"dt_success_t900 {d:.3f} (<= 0.6); contradicts if success_in < 0.8 or t900 < dt + 0.1")}
    if "cherepanov2026recurrent#C8" not in out:
        a, b, c = v("rate_fixed_success_t90"), v("rate_fixed_noise_success_t90"), v("rate_fixed_noise_junction_t90")
        verdict = ("supports" if a >= 0.9 and b <= 0.65 and c >= 0.9 else
                   "contradicts" if a >= 0.9 and b >= a - 0.1 else "inconclusive")
        out["cherepanov2026recurrent#C8"] = {"verdict": verdict, "evidence": (
            f"rate_fixed_success_t90 {a:.3f} (>= 0.9), rate_fixed_noise_success_t90 {b:.3f} (<= 0.65), "
            f"rate_fixed_noise_junction_t90 {c:.3f} (>= 0.9); contradicts if t90 >= 0.9 and noise >= t90 - 0.1")}
    if "cherepanov2026recurrent#C12" not in out:
        a, b = v("rate_fixed_success_t900"), v("rate_n1k90_success_t900")
        verdict = ("supports" if a - b >= 0.3 and b <= 0.55 else "contradicts" if a - b <= 0.1 else "inconclusive")
        out["cherepanov2026recurrent#C12"] = {"verdict": verdict, "evidence": (
            f"rate_fixed_success_t900 {a:.3f} - rate_n1k90_success_t900 {b:.3f} = {a - b:.3f} (>= 0.3), "
            f"rate_n1k90_success_t900 <= 0.55; contradicts if the difference <= 0.1")}
    return out


def x7b_readout(m):
    k = ["rate_var_tok_success_t900", "rate_var_len_success_t900", "rate_fixed_success_t900"]
    if not all(x in m for x in k):
        return "n/a (smoke)"
    a, b, f = (m[x]["value"] for x in k)
    if a >= 0.85 and b >= 0.85 and f >= 0.9:
        return "feed variable token counts straight into the memory"
    if a < 0.6 or b < 0.6:
        return "fixed per-segment token budget needed between compression and memory"
    return "unclear: X8 tests both"


def x7c_readouts(m):
    if "gru_minus_rate_k8" not in m:
        return {}
    g, a, amin = m["gru_minus_rate_k8"]["value"], m["aux_gain_k4plus"]["value"], m["aux_gain_k4plus_min_seed"]["value"]
    return {"x7c_readout_gru": ("GRU stays X8's baseline to beat" if g >= 0.1 else
                                "memory tokens hold more; GRU only a reference" if g <= -0.1 else "unclear"),
            "x7c_readout_aux": ("X8 adds the auxiliary reconstruction head" if a >= 0.05 and amin > 0 else
                                "X8 uses the action loss only" if a <= 0 else "X8 runs both")}


def figures(parts, figdir):
    os.makedirs(figdir, exist_ok=True)
    b = parts["x7b"]
    Ts = b["eval_T"]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=100)
    for arm in ["rate_fixed", "rate_n1k90", "dt", "rate_var_tok", "rate_var_len"]:
        if f"{arm}_success_t{Ts[0]}" in b["metrics"]:
            ax.plot(Ts, [b["metrics"][f"{arm}_success_t{T}"]["value"] for T in Ts], marker="o", label=arm)
    if "rate_fixed_noise_success_t90" in b["metrics"]:
        nT = [T for T in (90, 900) if f"rate_fixed_noise_success_t{T}" in b["metrics"]]
        ax.plot(nT, [b["metrics"][f"rate_fixed_noise_success_t{T}"]["value"] for T in nT], "kx", label="rate_fixed, noised memory")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("corridor length T (agent steps; trained up to 90)")
    ax.set_ylabel("success"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.set_title("X7b T-Maze")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "x7b_success.png")); plt.close(fig)

    c = parts["x7c"]
    ks = [1, 2, 4, 8]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=100)
    for name in sorted({k[len("facts_recalled_"):].rsplit("_k", 1)[0] for k in c["metrics"]
                        if k.startswith("facts_recalled_") and "_l450" not in k}):
        ys = [c["metrics"].get(f"facts_recalled_{name}_k{k}", {}).get("value") for k in ks]
        kk = [k for k, y in zip(ks, ys) if y is not None]
        ax.plot(kk, [y for y in ys if y is not None], marker="o", label=name, ls="--" if name.startswith("aux") else "-")
    ax.axhline(0.25, color="grey", lw=0.8, ls=":")
    ax.set_xscale("log", base=2); ax.set_xlabel("facts k"); ax.set_ylabel("facts_recalled (150 steps)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7, ncol=2); ax.set_title("X7c several facts")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "x7c_recall.png")); plt.close(fig)

    a = parts["x7a"]
    Ls = [4, 8, 16, 32]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=100)
    for arm in ["summary", "pertrunc", "canon", "summary_fixed"]:
        ys = [a["metrics"].get(f"x7a_acc_{arm}_L{L}", {}).get("value") for L in Ls]
        if any(y is not None for y in ys):
            ax.plot([L for L, y in zip(Ls, ys) if y is not None], [y for y in ys if y is not None], marker="o", label=arm)
    if "x7a_acc_full" in a["metrics"]:
        ax.axhline(a["metrics"]["x7a_acc_full"]["value"], color="k", lw=0.8, ls="--", label="full")
    ax.set_xscale("log", base=2); ax.set_xlabel("prefix L"); ax.set_ylabel("dodge accuracy")
    ax.legend(fontsize=7); ax.set_title("X7a set prefix")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "x7a_prefix.png")); plt.close(fig)


def render_blocks(res, readouts, checks):
    fmt = lambda x: f"{x:.4g}" if isinstance(x, float) else str(x)
    blocks = {}
    blocks["verdicts"] = "| Claim | Verdict | Evidence |\n|---|---|---|\n" + "\n".join(
        f"| {c} | {res['verdicts'][c]['verdict']} | {res['verdicts'][c]['evidence']} |" for c in res["tests"])
    rows = []
    for k, m in res["metrics"].items():
        ci = f"[{m['ci']['low']:.3f}, {m['ci']['high']:.3f}] at {m['ci']['level']}" if "ci" in m else ""
        rows.append(f"| {k} | {m['value']:.4g} | {m['unit']} | {m['n']} | {ci} |")
    blocks["metrics"] = "| Metric | Value | Unit | n | CI |\n|---|---|---|---|---|\n" + "\n".join(rows)
    blocks["readouts"] = "| Readout | Value |\n|---|---|\n" + "\n".join(
        f"| {k} | {json.dumps(v) if isinstance(v, dict) else fmt(v)} |" for k, v in readouts.items())
    blocks["checks"] = "\n".join(f"- {line}" for line in checks) or "- (no checks log)"
    blocks["run"] = (f"Experiment `{res['experiment']}`, run on {res['date']} at research "
                     f"`{res['commits'][0]['sha'][:7]}` (code id `inputs.src` `{res['inputs']['src'][:7]}`), "
                     f"wall clock {res['wall_clock_s'] / 60:.0f} min. Plan: [plan.md](plan.md). "
                     f"Machine-readable results: [results.json](results.json).")
    blocks["tolerances"] = "\n".join(f"- `{k}` within {v}" for k, v in TOLERANCES.items())
    return blocks


def apply_blocks(text, blocks):
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- generated:{name} begin -->\n).*?(<!-- generated:{name} end -->)", re.S)
        if pat.search(text):
            text = pat.sub(lambda mm: mm.group(1) + body + "\n" + mm.group(2), text)
    return text


SKELETON = """# Memory-token toys for the shmup stack (X7): results

<!-- generated:run begin -->
<!-- generated:run end -->

## Verdicts

<!-- generated:verdicts begin -->
<!-- generated:verdicts end -->

## Metrics

<!-- generated:metrics begin -->
<!-- generated:metrics end -->

Design readouts (no claim; decision rules fixed in plan.md):

<!-- generated:readouts begin -->
<!-- generated:readouts end -->

## Checks

<!-- generated:checks begin -->
<!-- generated:checks end -->

## Deviations from the plan

- (to be written)

## Figures

- ![X7b success against corridor length](figures/x7b_success.png)
- ![X7c facts recalled against fact count](figures/x7c_recall.png)
- ![X7a dodge accuracy against prefix length](figures/x7a_prefix.png)

## Reproduce

```sh
experiments/2026-11-27-memory-token-toys/run.sh
```

<!-- generated:tolerances begin -->
<!-- generated:tolerances end -->

## Notes

- (to be written)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--wall-s", type=float, required=True)
    ap.add_argument("--render-only", action="store_true", help="re-render results.md from results.json")
    args = ap.parse_args()
    exp = os.path.abspath(args.exp_dir)
    out_dir = args.parts_dir if args.smoke else exp
    parts = {p: json.load(open(os.path.join(args.parts_dir, f"part_{p}.json"))) for p in ("x7a", "x7b", "x7c")}
    checks = []
    cl = os.path.join(args.parts_dir, "checks.log")
    if os.path.exists(cl):
        checks = [line.strip()[len("check "):] for line in open(cl) if line.startswith("check ")]
    if args.render_only:
        res = json.load(open(os.path.join(out_dir, "results.json")))
    else:
        repo = git("rev-parse", "--show-toplevel", cwd=exp)
        rel = os.path.relpath(exp, repo)
        sha = git("rev-parse", "HEAD", cwd=repo)
        outputs = (f"{rel}/results.json", f"{rel}/results.md", f"{rel}/figures/")
        dirty = [l for l in git("status", "--porcelain", "--untracked-files=all", cwd=repo).splitlines()
                 if not l[3:].startswith(outputs)]
        inputs = {}
        for n in ("src", "run.sh", "requirements.txt"):
            try:
                inputs[n] = git("rev-parse", f"HEAD:{rel}/{n}", cwd=repo)
            except subprocess.CalledProcessError:  # not committed yet: only an exploratory (dirty) run
                inputs[n] = "0" * 40
        metrics, per_seed = {}, {}
        for p in ("x7b", "x7c", "x7a"):
            for k, m in parts[p]["metrics"].items():
                metrics[k] = {kk: m[kk] for kk in ("value", "unit", "n", "ci") if kk in m}
                if "per_seed" in m:
                    per_seed[k] = m["per_seed"]
        exploratory = bool(dirty) or args.smoke
        vd = verdicts(metrics)
        if exploratory:
            vd = {c: {"verdict": "inconclusive", "evidence": "exploratory run, capped at inconclusive. " + v["evidence"]}
                  for c, v in vd.items()}
        res = {
            "schema": 1, "experiment": EXP_ID, "date": datetime.date.today().isoformat(),
            "commits": [{"repo": "research", "sha": sha, "dirty": bool(dirty)}],
            "inputs": inputs, "exploratory": exploratory, "tests": CLAIMS, "verdicts": vd,
            "metrics": metrics, "seeds": parts["x7b"]["seeds"], "wall_clock_s": round(args.wall_s, 1),
            "env": {"python": platform.python_version(), "os": f"macOS {platform.mac_ver()[0]} {platform.machine()}",
                    "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()),
                    "gpu": None},
            "readouts": {"x7b_readout_variable": x7b_readout(metrics), **parts["x7a"].get("readouts", {}),
                         **parts["x7c"].get("readouts", {}), **x7c_readouts(metrics),
                         "x7c_gru_minus_rate_best_act_m": parts["x7c"]["metrics"].get("gru_minus_rate_k8", {}).get("best_act_m")},
            "per_seed": per_seed,
            "threads_per_process": int(os.environ.get("X7_THREADS", "3")),
        }
        if dirty:
            res["dirty_paths"] = dirty
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(res, f, indent=2)
            f.write("\n")
        figures(parts, os.path.join(out_dir, "figures"))
    md_path = os.path.join(out_dir, "results.md")
    text = open(md_path).read() if os.path.exists(md_path) else SKELETON
    with open(md_path, "w") as f:
        f.write(apply_blocks(text, render_blocks(res, res["readouts"], checks)))
    print(f"wrote {out_dir}/results.json, results.md, figures/; verdicts: "
          + ", ".join(f"{c} {v['verdict']}" for c, v in res["verdicts"].items()))


if __name__ == "__main__":
    main()
