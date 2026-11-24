---
id: 2026-11-24-jowa-mac-smoke
title: JOWA-40M and JOWA-150M released checkpoints on an Apple-silicon Mac: load, tokenize, plan and play Atari
claims: [cheng2025jointly#C16]
assumptions: [cheng2025jointly#A1, cheng2025jointly#A5, cheng2025jointly#A6]
status: approved
gpu: yes
budget_wall_min: 150
budget_downloads_mb: 1500
ci_smoke: false
approved: user 2026-11-24
---

# JOWA-40M and JOWA-150M released checkpoints on an Apple-silicon Mac: load, tokenize, plan and play Atari

## Hypothesis

The released JOWA checkpoints (VQ-VAE tokenizer, JOWA-40M, JOWA-150M; code `CJReinforce/JOWA`)
load into the released code on a Mac with a minimal device patch (CPU or MPS instead of CUDA),
and drive a real Atari environment (ported from atari-py/gym 0.19 to ale-py/gymnasium) for whole
episode prefixes with and without beam-search planning, producing finite Q-values and legal
actions (C16: code and weights are released and usable; A1: 18-action, 84x84 grayscale, no sticky
actions; A5: 36 tokens per frame, vocabulary 2048; A6: planning at horizon 2). This is an
interface and plumbing check, not a test of the paper's performance claims: returns of capped
episodes are recorded for reference only and are not compared with the paper.

## Success criteria

The run is exploratory (clones and patches third-party code at run time, ports the environment,
caps episodes), so any recorded verdict is capped at `inconclusive`; the criterion states what the
observed values would otherwise mean.

- cheng2025jointly#C16: supports if models_loaded == 2 and nonfinite_count == 0 and episodes_completed == episodes_planned; contradicts if models_loaded == 0 or nonfinite_count > 0; otherwise inconclusive. (Exploratory: recorded as inconclusive whichever way it falls.)
- tolerance models_loaded: 0
- tolerance nonfinite_count: 0
- tolerance episodes_completed: 0

Metric definitions: `models_loaded` is the number of model sizes (of 40M, 150M) whose tokenizer
and model state dicts load with every key present and every checked output finite;
`nonfinite_count` counts NaN or inf in the Q-value logits seen at every agent step of every
episode; `episodes_completed` is the number of capped episodes that ran to their step cap or game
end without an exception or an action outside the game's legal set; `episodes_planned` is the
number scheduled (a metric too, equal for every full run).

## Method

- **Code.** `run.sh` shallow-fetches `https://github.com/CJReinforce/JOWA` at commit
  `8cb02c5e6768a948c566a9012ac8a4fc420c6970` (2024-12-02) into `runs/<id>/JOWA/` (git-ignored, not
  vendored; GPL-3.0) and applies patches at runtime from `src/` (monkeypatching, no edits of the
  clone): device selection instead of CUDA, `torch.cuda.amp.autocast` replaced by a
  device-generic float32 path, CUDA RNG calls guarded, and the environment port below.
- **Environment port.** `create_atari_environment` (gym 0.19 `{Game}NoFrameskip-v4` +
  atari-py) is replaced by gymnasium `ALE/{Game}-v5` with `frameskip=1`,
  `repeat_action_probability=0`, `full_action_space=False` (same minimal action set and order as
  atari-py), wrapped in a shim exposing the gym 0.19 API (`reset()`, 4-tuple `step`, `.ale`) so
  the repo's own `AtariPreprocessing` (frame skip 4, max-pool, 84x84, INTER_AREA), action mapping
  and `Game`/`AgentEnv` loops run unchanged.
- **Weights.** The GitHub repo's `checkpoints/JOWA/*.pt` are Git LFS pointers and the repository
  has LFS disabled (LFS batch API: 403), so the weights come from the README's Google Drive
  folder (`1YHaCemhobchJWE5zt28TUrKp2tp8yysF`) via `gdown`, into `~/.cache/jowa/` (outside the
  checkout). The 40M file's sha256 equals the LFS pointer's oid in the repo; every file's sha256 is
  recorded. Loaded with the repo's own `torch.load` (pickle; official source only), `map_location=cpu`.
- **Environment.** CPython 3.8 (upstream's version) via a pinned `uv` into a venv under `runs/`;
  torch 2.2.2 (last release for 3.8 on macOS arm64), gymnasium 0.29.1, ale-py 0.9.1, all pinned
  in `requirements.txt` (upstream's own pins, numpy 1.19, atari-py, gym 0.19, do not install on
  arm64).
- **Games and seeds.** Assault and Seaquest (both pretraining games). Seeds 0 and 1 per cell
  (ALE seed, torch/numpy seed; the only stochasticity with sticky actions off is the repo's
  epsilon 0.001 random action and the ALE seed). `buffer_size` (context steps) 8 for no planning;
  planning hyperparameters from the repo's `eval/JOWA_<size>_raw_scores.json` per game and model
  (beam width, horizon, buffer size).
- **Runs.** For each of model in (40M, 150M) x game x seed: planning off and beam-search planning
  on, on CPU (all cores but two, `torch.set_num_threads`) and on MPS (float32; MPS bf16 autocast
  attempted only if cheap). Episodes capped at 300 agent steps without planning and 60 with
  planning (one agent step is 4 emulator frames; the cap is far below the paper's 108k-frame
  episodes, so returns are not comparable). If planning on a device is too slow, the cap for that
  cell is cut and recorded. No multi-process env: a single env (the repo's own planning needs
  `num_envs == 1`).
- **Measurements.** Load time (tokenizer, model; wall clock, from the cached file), parameter counts,
  state-dict key check, per-step inference time (`agent.act`, synchronised, median and mean over
  all steps of a cell, env stepping excluded), total episode wall clock, return and clipped
  return of the capped episode, legal-action check, finite-Q check; one tokenizer
  encode/decode reconstruction MSE on real frames as a sanity check. Metrics named in
  the criterion are above; all others are reported but not compared.
- **Smoke mode.** `run.sh --smoke` runs 40M on CPU, one game, 20 steps without planning and 5
  with planning, writing under `runs/<id>/smoke/` only; it needs the venv and the cached weights
  (cold venv and clone take longer than a minute and are reported).
- **Assumptions.** A1: met (18-action full set mapped to the game's minimal set, 84x84 grayscale,
  no sticky actions, 4 frame skip). A5: checked (token count 36 per frame and vocabulary 2048
  read from the loaded tokenizer). A6: planning horizon 2 as in the repo's configs; accuracy of
  the dynamics is not tested.
- **Correctness checks.** none (inference and plumbing smoke test with no derivation or custom
  gradient; the run's own assertions on state-dict keys, action legality and finiteness are its
  checks).
- **Domain-repo code.** none.

## Budget

- Approved scope (user, 2026-11-24, through the supervisor): Mac only, a couple of hours,
  downloads within 1.5 GB, 150M added to 40M.
- Downloads: weights 40M 76 MB + tokenizer 75 MB + 150M 511 MB = 662 MB; JOWA clone about 130 MB
  (full history at first, a shallow fetch is smaller); wheels (torch 2.2.2 about 60 MB plus others)
  and Python 3.8 about 250 MB; total about 1.05 GB for the first run. (The first fetch also pulled
  the unused 70M file, 213 MB, by taking the whole Drive folder; it was deleted and run.sh
  fetches single files.) The reproduction reuses the weights cache and uv cache outside the
  checkout and costs about 0.2 GB (clone, wheels); if the caches are cleared it costs the full
  1.05 GB again, which would exceed 1.5 GB combined and needs the user's call.
- Wall clock: run about 60 min (environment 5, loads 2, 32 cells at an estimated 2 to 120 s each,
  debugging excess), reproduction about 40 min, slack to 150.
- GPU: yes, the Mac's MPS device only; one process for the MPS cells (about 15 min), under the
  supervisor's GPU slot. CPU cells need none.
