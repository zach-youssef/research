"""NitroGen Mac smoke test: load the released checkpoint, run in-process inference on a few frames,
check output shape, finiteness and variation, and time CPU and MPS. See ../plan.md."""
import argparse
import datetime
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
from importlib import metadata
from pathlib import Path

import numpy as np

EXP_ID = "2026-11-24-nitrogen-mac-smoke"
CLAIM = "magne2026nitrogen#C8"
VIDEO_URL = "https://nitrogen.minedojo.org/assets/videos/rollouts.mp4"
VIDEO_BYTES = 21549899
FRAME_IDX = [150, 450, 750, 1050, 1350]
CKPT_REPO, CKPT_REV = "nvidia/NitroGen", "584c8dded734d032f07a4bcc0ccb330e703298c4"
SIGLIP_REPO, SIGLIP_REV = "google/siglip2-large-patch16-256", "787800c8990e6f058423089178e718139608408c"
HORIZON_DOC = 16  # the plan's documented chunk length (paper Sec. 2.3)
SEED = 0
N_TIMED = 3


def fetch_frames(runs: Path, idxs):
    import av
    from PIL import Image
    data = runs / "data"
    data.mkdir(parents=True, exist_ok=True)
    video = data / "rollouts.mp4"
    if not video.exists() or video.stat().st_size != VIDEO_BYTES:
        urllib.request.urlretrieve(VIDEO_URL, video)
    assert video.stat().st_size == VIDEO_BYTES, "unexpected video size"
    want, frames = set(idxs), {}
    with av.open(str(video)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                frames[i] = np.asarray(f.to_image().convert("RGB"))
            if i >= max(idxs):
                break
    assert sorted(frames) == sorted(idxs), "frame decode failed"
    return [frames[i] for i in idxs], hashlib.sha256(video.read_bytes()).hexdigest()


def fetch_weights(runs: Path):
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download(CKPT_REPO, "ng.pt", revision=CKPT_REV)
    sig = [hf_hub_download(SIGLIP_REPO, n, revision=SIGLIP_REV, local_dir=runs / "siglip2")
           for n in ("config.json", "preprocessor_config.json")]
    return ckpt, str(Path(sig[0]).parent)


def run_device(device, model, tok, proc, frames, n_timed, bf16=False):
    import torch
    m = model.to(device)
    per_frame, raws, decs, med = [], [], [], []
    for fr in frames:
        predict(m, tok, proc, fr, device, SEED, bf16)  # warm-up (kernel compile, caches)
        ts = []
        for _ in range(n_timed):
            raw, dec, dt = predict(m, tok, proc, fr, device, SEED, bf16)
            ts.append(dt)
        raws.append(raw)
        decs.append(dec)
        med.append(statistics.median(ts))
        per_frame.append(ts)
    return raws, decs, med, per_frame


def predict(*a):
    import mac_patch
    return mac_patch.predict(*a)


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--nitrogen-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--devices", default="cpu,mps", help="comma list; the committed results need cpu,mps")
    ap.add_argument("--no-write", action="store_true", help="skip results.json/results.md (dev runs)")
    args = ap.parse_args()
    runs, exp = Path(args.runs_dir), Path(args.exp_dir)
    t_start = time.time()

    import torch
    import mac_patch
    torch.set_num_threads(os.cpu_count())

    idxs = FRAME_IDX[:1] if args.smoke else FRAME_IDX
    frames, video_sha = fetch_frames(runs, idxs)
    ckpt, siglip_dir = fetch_weights(runs)
    h = hashlib.sha256()
    with open(ckpt, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    ckpt_sha = h.hexdigest()
    t0 = time.time()
    model, tok, proc, cfg = mac_patch.load_model(ckpt, siglip_dir)
    load_s = time.time() - t0
    mc = cfg.model_cfg
    horizon, adim = mc.action_horizon, mc.action_dim
    nparams = sum(p.numel() for p in model.parameters())
    print(f"loaded in {load_s:.1f}s: {nparams:,} params, action_horizon={horizon}, action_dim={adim}, "
          f"vision={mc.vision_encoder_name}, inference_steps={mc.num_inference_timesteps}", flush=True)

    devices = ["cpu"] if args.smoke else args.devices.split(",")
    n_timed = 1 if args.smoke else N_TIMED
    res = {}
    for d in devices:
        if d == "mps":
            assert torch.backends.mps.is_available(), "MPS not available"
        raws, decs, med, per_frame = run_device(d, model, tok, proc, frames, n_timed)
        res[d] = dict(raws=raws, decs=decs, med=med, per_frame=per_frame)
        print(f"{d}: median ms per chunk per frame: {[round(x*1000) for x in med]}", flush=True)

    cpu = res["cpu"]
    B = adim - 4
    shapes_ok = True
    for raw, dec in zip(cpu["raws"], cpu["decs"]):
        shapes_ok &= tuple(raw.shape) == (1, HORIZON_DOC, B + 4)
        shapes_ok &= tuple(dec["buttons"].shape) == (1, HORIZON_DOC, B)
        shapes_ok &= tuple(dec["j_left"].shape) == (1, HORIZON_DOC, 2)
        shapes_ok &= tuple(dec["j_right"].shape) == (1, HORIZON_DOC, 2)
    stack = torch.stack([r[0] for r in cpu["raws"]])  # (F, T, D)
    nonfinite = int((~torch.isfinite(stack)).sum())
    std_across = float(stack.std(dim=0, unbiased=False).mean()) if len(frames) > 1 else float("nan")
    buttons = torch.stack([d["buttons"][0] for d in cpu["decs"]])
    sticks = torch.cat([torch.stack([d["j_left"][0] for d in cpu["decs"]]),
                        torch.stack([d["j_right"][0] for d in cpu["decs"]])], dim=-1)
    distinct = len({tuple(b.flatten().tolist()) for b in buttons})
    metrics = {
        "chunk_shape_ok": (int(shapes_ok), 1),
        "nonfinite_count": (nonfinite, int(stack.numel())),
        "action_std_across_frames": (std_across, len(frames)),
        "action_horizon_found": (horizon, 1),
        "action_dim_found": (adim, 1),
        "button_width_found": (B, 1),
        "button_on_fraction": (float(buttons.mean()), len(frames)),
        "stick_abs_mean": (float(sticks.abs().mean()), len(frames)),
        "distinct_button_chunks": (distinct, len(frames)),
        "ms_per_chunk_cpu": (1000 * statistics.mean(cpu["med"]), len(frames)),
    }
    if "mps" in res:
        diff = max(float((a - b).abs().max()) for a, b in zip(res["mps"]["raws"], cpu["raws"]))
        metrics["ms_per_chunk_mps"] = (1000 * statistics.mean(res["mps"]["med"]), len(frames))
        metrics["mps_cpu_max_abs_diff"] = (diff, len(frames))
        metrics["mps_nonfinite_count"] = (int(sum((~torch.isfinite(r)).sum() for r in res["mps"]["raws"])), len(frames))
        try:
            _, _, med_bf, _ = run_device("mps", model, tok, proc, frames, n_timed, bf16=True)
            metrics["ms_per_chunk_mps_bf16"] = (1000 * statistics.mean(med_bf), len(frames))
        except Exception as e:  # reported, not fatal
            print("mps bf16 autocast failed:", repr(e)[:200], flush=True)
    for k, (v, n) in metrics.items():
        print(f"metric {k} = {v} (n={n})", flush=True)

    ok = shapes_ok and nonfinite == 0 and (len(frames) < 2 or std_across > 0.0001)
    if args.smoke:
        out = runs / "smoke"
        out.mkdir(parents=True, exist_ok=True)
        (out / "smoke.json").write_text(json.dumps({k: v[0] for k, v in metrics.items()}, indent=1))
        print("smoke", "PASS" if ok else "FAIL (shape per plan: horizon %d; found %d)" % (HORIZON_DOC, horizon))
        # The plan's shape criterion (horizon 16) is a recorded finding, not a smoke failure:
        sys.exit(0 if nonfinite == 0 else 1)
    if args.no_write:
        return

    # ---- results.json / results.md (exploratory: every verdict is inconclusive) ----
    exp_rel = f"experiments/{EXP_ID}"
    repo = Path(git("rev-parse", "--show-toplevel", cwd=exp))
    sha = git("rev-parse", "HEAD", cwd=repo)
    outputs = {f"{exp_rel}/results.json", f"{exp_rel}/results.md"}
    dirty_lines = [l for l in git("status", "--porcelain", cwd=repo).splitlines()
                   if not any(l[3:].startswith(p) for p in (f"{exp_rel}/results.", f"{exp_rel}/figures/"))]
    dirty = bool(dirty_lines)
    inputs = {
        "src": git("rev-parse", f"HEAD:{exp_rel}/src", cwd=repo),
        "run.sh": git("rev-parse", f"HEAD:{exp_rel}/run.sh", cwd=repo),
        "requirements.txt": git("rev-parse", f"HEAD:{exp_rel}/requirements.txt", cwd=repo),
    }
    shape_found = f"horizon {horizon} x dim {adim}"
    evidence = (f"exploratory run, capped at inconclusive. chunk_shape_ok {metrics['chunk_shape_ok'][0]}: the "
                f"released ng.pt outputs {shape_found} ({B} buttons, BUTTON_ACTION_TOKENS in nitrogen/shared.py), "
                f"while the paper states 16-step chunks over 16 buttons + 2 sticks; this paper-vs-release gap is "
                f"unexplained and bears on A11's chunk length, not on C8's content. nonfinite_count {nonfinite}, "
                f"action_std_across_frames {std_across:.4f}")
    pk = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    elapsed = time.time() - t_start
    results = {
        "schema": 1, "experiment": EXP_ID, "date": datetime.date.today().isoformat(),
        "commits": [{"repo": "research", "sha": sha, "dirty": dirty}],
        "inputs": inputs, "exploratory": True, "tests": [CLAIM],
        "verdicts": {CLAIM: {"verdict": "inconclusive", "evidence": evidence}},
        "metrics": {k: {"value": v, "unit": ("ms" if k.startswith("ms_") else "1"), "n": n}
                    for k, (v, n) in metrics.items()},
        "seeds": [SEED], "wall_clock_s": round(elapsed, 1),
        "env": {"python": platform.python_version(),
                "os": f"macOS {platform.mac_ver()[0]} {platform.machine()}",
                "packages": pk, "gpu": "Apple MPS (torch %s)" % torch.__version__},
        "provenance": {"nitrogen_commit": git("rev-parse", "HEAD", cwd=Path(args.nitrogen_dir)),
                       "ckpt": f"{CKPT_REPO}@{CKPT_REV}", "ckpt_sha256": ckpt_sha, "vision_encoder_config": f"{SIGLIP_REPO}@{SIGLIP_REV}",
                       "video": VIDEO_URL, "video_sha256": video_sha, "frame_indices": FRAME_IDX,
                       "load_seconds": round(load_s, 1), "params": nparams,
                       "per_frame_ms": {d: [[round(t * 1000, 1) for t in ts] for ts in res[d]["per_frame"]] for d in res}},
    }
    (exp / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    rows = "\n".join(f"| {k} | {v:.6g} | {results['metrics'][k]['unit']} | {n} |  |" for k, (v, n) in metrics.items())
    (exp / "results.md").write_text(f"""# NitroGen Mac smoke test: results

Experiment `{EXP_ID}`, run on {results['date']} at research `{sha[:7]}` (code id `inputs.src` `{inputs['src'][:7]}`). Plan:
[plan.md](plan.md). Machine-readable results: [results.json](results.json).

## Verdicts

| Claim | Verdict | Evidence |
|---|---|---|
| {CLAIM} | inconclusive | {evidence} |

## Metrics

| Metric | Value | Unit | n | CI |
|---|---|---|---|---|
{rows}

## Checks

- none (inference smoke test: no derivation or custom gradient; the shape, finiteness and variation
  assertions above are the run's own checks)

## Deviations from the plan

@@DEVIATIONS@@

## Figures

- none (the frames come from a video of unclear licence and are not committed)

## Reproduce

```sh
experiments/{EXP_ID}/run.sh
```

Expected wall clock about {elapsed / 60:.0f} min with a warm HF cache and venv (first run adds the
2 GB weight download and the environment). Tolerances from plan.md: `chunk_shape_ok` within 0;
`nonfinite_count` within 0; `action_std_across_frames` within 10%. A rerun reproduces when every
criterion's verdict is unchanged and each toleranced metric is within its tolerance.

## Notes

@@NOTES@@
""")
    print("wrote results.json, results.md")


if __name__ == "__main__":
    main()
