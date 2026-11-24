---
id: 2026-11-24-nitrogen-mac-smoke
title: NitroGen released checkpoint on an Apple-silicon Mac: output shape, variation and latency smoke test
claims: [magne2026nitrogen#C8]
assumptions: [magne2026nitrogen#A3, magne2026nitrogen#A5, magne2026nitrogen#A6, magne2026nitrogen#A8, magne2026nitrogen#A10, magne2026nitrogen#A11, magne2026nitrogen#A13]
status: approved
gpu: yes
budget_wall_min: 90
budget_downloads_mb: 3500
ci_smoke: false
approved: user 2026-11-24
---

# NitroGen released checkpoint on an Apple-silicon Mac: output shape, variation and latency smoke test

## Hypothesis

The released NitroGen checkpoint (`nvidia/NitroGen` `ng.pt`, code `MineDojo/NitroGen`) loads on a
Mac with a minimal device patch (MPS or CPU instead of CUDA) and, given single gameplay frames,
returns 16-step action chunks of the documented layout: button logits thresholded to 0/1 plus two
sticks in [-1, 1], finite and varying with the frame (assumption A11 and the "16-action chunks"
part of C8; A3 for the layout). This is an architecture and shape check of the open weights, not
a test of C8's actual content (that chunking improves temporal consistency) and not a
performance claim. The paper's text says 16 binary buttons plus 4 stick dims; the released code
has its own width, which the run records.

## Success criteria

The run is exploratory (it clones and patches third-party code at run time, and inference is
stochastic), so any recorded verdict is capped at `inconclusive`; the criterion below states
what the observed values would otherwise mean, and the observed outcome is stated in the
verdict's evidence.

- magne2026nitrogen#C8: supports if chunk_shape_ok == 1 and nonfinite_count == 0 and action_std_across_frames > 0.01; contradicts if chunk_shape_ok == 0 or nonfinite_count > 0 or action_std_across_frames < 0.0001; otherwise inconclusive. (Exploratory: recorded as inconclusive whichever way it falls.)
- tolerance chunk_shape_ok: 0
- tolerance nonfinite_count: 0
- tolerance action_std_across_frames: 10%

Metric definitions (CPU float32 run, fixed noise seed per frame): `chunk_shape_ok` is 1 if every
decoded output has the shapes buttons (16, B), j_left (16, 2), j_right (16, 2) with B the
checkpoint's button width, and the raw sampled tensor is (1, 16, B + 4), else 0;
`nonfinite_count` counts NaN or inf in the raw sampled tensors over all frames;
`action_std_across_frames` is the mean, over the 16 x (B + 4) raw entries, of the standard
deviation across the frames, with the same noise seed for every frame (so the variation comes from
the image).

## Method

- **Inputs.** Five frames from `https://nitrogen.minedojo.org/assets/videos/rollouts.mp4` (21.5 MB,
  the project page's rollout video of the agent in commercial games), taken at decoded frame
  indices 150, 450, 750, 1050 and 1350 (30 fps, 1280 x 720). Fetched by `run.sh` into `runs/`; not
  committed (licence unclear). The frames carry the video's caption overlays, which the real
  observation wouldn't (A5); this affects only realism, not the shape check. Single frame per
  call (context length 1, as `serve.py`'s default `--ctx 1`), unconditional game id (A13: no goal
  input).
- **Code.** `run.sh` clones `https://github.com/MineDojo/NitroGen` at commit
  `32608444660950ffda95e1e57c79632ad65bea10` into `runs/<id>/NitroGen/` (git-ignored; not vendored;
  the licence is NVIDIA non-commercial research) and applies `src/mac_patch.py` at runtime: device
  selection, replacing the hard-coded `cuda` and CUDA bf16 autocast, building the SigLIP vision
  tower from its config instead of downloading its ~1 GB weights (the checkpoint overwrites them),
  and skipping the tokenizer's game-mapping parquet read if the files are absent. The model is
  called in-process through `InferenceSession`-equivalent code; `serve.py` and ZeroMQ are never run.
- **Weights.** `ng.pt` from `nvidia/NitroGen` (1.97 GB) into the HF cache outside the repo, plus
  `google/siglip-large-patch16-256` config and preprocessor files only (under 1 MB).
- **Environment.** CPython 3.12 (the README's tested version) via a pinned `uv`, a venv under
  `runs/`, dependencies resolved as of the repo commit's date (2026-01-25) and pinned with `==`.
- **Measurements.** CPU float32: for each frame, 1 warm-up call and 3 timed calls (the noise seed
  is reset per call, so the output is identical across repeats); the shape and variation metrics
  come from this device. MPS float32: the same, after the output of the first call is compared to
  the CPU output (max absolute difference, reported). An MPS bf16 autocast timing is attempted and
  reported if it works. Metrics: `ms_per_chunk_cpu`, `ms_per_chunk_mps` (mean over frames of the
  per-frame median of the timed calls, including image preprocessing and the 16 Euler steps,
  synchronised), `mps_cpu_max_abs_diff`, `button_on_fraction`, `stick_abs_mean`, button width and
  the layout found. Seeds: noise seed 0 for all calls (`torch.manual_seed(0)` before each).
- **Smoke mode.** `run.sh --smoke` runs one frame on CPU with one warm-up and one timed call and
  writes under `runs/<id>/smoke/` only. It needs the venv and the HF-cached weights; loading a
  2 GB pickle may push it past a minute on a cold cache (reported if so).
- **Assumptions.** A3, A5, A11, A13: met as stated (shared layout, single frame, the released
  architecture, no goal input), with the A5 caveat above. A6, A8, A10: training-side, not
  exercised by an inference smoke test.
- **Correctness checks.** none (inference smoke test, no derivation or custom gradient; the shape
  and finiteness assertions are the run's own checks).
- **Domain-repo code.** none.

## Budget

- Approved scope (user, 2026-11-24, through the supervisor): Mac only, about 1 h, 3-4 GB of
  downloads.
- Downloads (about 2.1 GB per fresh run): ng.pt 1.97 GB, rollouts.mp4 21.5 MB, Python 3.12 and uv
  about 40 MB, torch and other wheels about 150 MB, SigLIP config files under 1 MB. The
  reproduction run reuses the HF and uv caches (outside the checkout) and costs about 0.1 GB
  (video, wheels); 3500 MB covers the first run plus that, with margin, under the 4 GB scope. If
  the weights cache is cleared before the reproduction, the total reaches about 4.2 GB, which
  needs the user's call.
- Wall clock: run about 25 min (environment 5, weight loads 2 x 1 min, CPU timing of 5 frames x 4
  calls at an estimated 1-5 s each, MPS the same, debugging excess), reproduction about 15 min,
  slack to 90.
- GPU: yes, the Mac's MPS device only, one process for a few minutes; the CPU steps need no slot.
