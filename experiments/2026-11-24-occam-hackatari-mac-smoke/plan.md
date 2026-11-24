---
id: 2026-11-24-occam-hackatari-mac-smoke
title: OCCAM/HackAtari agents in the real ALE on a Mac: Pong, Freeway and Breakout, original vs modified
claims: [delfosse2025intelligence#C8, delfosse2025intelligence#C9, delfosse2025intelligence#C11]
assumptions: [delfosse2025intelligence#A2, delfosse2025intelligence#A3, delfosse2025intelligence#A5, delfosse2025intelligence#A6]
status: approved
gpu: yes
budget_wall_min: 120
budget_downloads_mb: 500
ci_smoke: false
approved: user 2026-11-24
---

# OCCAM/HackAtari agents in the real ALE on a Mac: Pong, Freeway and Breakout, original vs modified

## Hypothesis

The released OCCAM/PPO checkpoints (`VanillaWhey/OCAtariWrappers`, seed 0, six agents per game:
pixel PPO, binary masks, class masks, object masks, planes, object vector) load and play in the
real ALE through HackAtari on an Apple-silicon Mac with no or minimal patches. Evaluated on the
original game and on HackAtari modifications they were not trained on, they show the direction
of the paper's modified-vs-original drops: the pixel PPO agent collapses on Pong `lazy_enemy`
while the Planes agent holds (C8), the pixel PPO agent collapses on Breakout with a red paddle
and ball while the Binary, Class and Planes mask agents hold (C9), and the pixel PPO agent
scores low on Freeway `stop_all_cars` (C11, paper PPO 7.6). This is an exploratory smoke test
of the released weights and the environment stack (the claims' architecture and interface
assumptions A2 and A6), not a performance reproduction: n is small and the checkpoints are one
seed each.

## Success criteria

The run is exploratory (it clones third-party code and weights at run time, and the agents act
stochastically), so every recorded verdict is capped at `inconclusive`; the criteria below state
what the observed values would otherwise mean, and the verdict's evidence states which branch
the numbers fell in. Metric `<game>_<agent>_<condition>_mean` is the mean undiscounted
game-score return over n = 10 episodes (seeds 0 to 9); agents are `pixel`, `binary`, `classes`,
`object`, `planes`, `objvec`; conditions are `orig`, `lazy_enemy`, `hidden_enemy` (Pong),
`stop_all_cars` (Freeway), `red` (Breakout, HackAtari `color_player_and_ball_red`).

- delfosse2025intelligence#C8: supports if pong_pixel_orig_mean >= 10 and pong_pixel_lazy_enemy_mean <= pong_pixel_orig_mean - 15 and pong_planes_lazy_enemy_mean >= pong_planes_orig_mean - 8; contradicts if pong_pixel_lazy_enemy_mean >= pong_pixel_orig_mean - 5; otherwise inconclusive. (Exploratory: recorded as inconclusive whichever way it falls.)
- delfosse2025intelligence#C9: supports if breakout_pixel_orig_mean >= 10 and breakout_pixel_red_mean <= 0.5 * breakout_pixel_orig_mean and breakout_binary_red_mean >= 0.7 * breakout_binary_orig_mean and breakout_classes_red_mean >= 0.7 * breakout_classes_orig_mean and breakout_planes_red_mean >= 0.7 * breakout_planes_orig_mean; contradicts if breakout_pixel_red_mean >= 0.8 * breakout_pixel_orig_mean; otherwise inconclusive. (Exploratory: recorded as inconclusive.)
- delfosse2025intelligence#C11: supports if freeway_pixel_orig_mean >= 15 and freeway_pixel_stop_all_cars_mean <= 12; contradicts if freeway_pixel_stop_all_cars_mean >= 25; otherwise inconclusive. (Exploratory: recorded as inconclusive.)
- tolerance pong_pixel_orig_mean: 5
- tolerance pong_pixel_lazy_enemy_mean: 5
- tolerance pong_planes_orig_mean: 5
- tolerance pong_planes_lazy_enemy_mean: 5
- tolerance breakout_pixel_orig_mean: 25%
- tolerance breakout_pixel_red_mean: 25%
- tolerance breakout_binary_orig_mean: 25%
- tolerance breakout_binary_red_mean: 25%
- tolerance breakout_classes_orig_mean: 25%
- tolerance breakout_classes_red_mean: 25%
- tolerance breakout_planes_orig_mean: 25%
- tolerance breakout_planes_red_mean: 25%
- tolerance freeway_pixel_orig_mean: 3
- tolerance freeway_pixel_stop_all_cars_mean: 3

## Method

- **Code, pinned and not vendored.** `run.sh` clones into `~/.cache/occam-hackatari/` (outside the
  checkout): `VanillaWhey/OCAtariWrappers` (MIT; wrappers, agent loader and the weights, which
  live in the repo under `models/`) at commit `057afc4e53f66ae1c12eaa72c321575be05534f0`
  (2026-04-21), fetched blobless with a sparse checkout of the 18 weight files for Pong,
  Freeway and Breakout (about 134 MB; the whole repo is about 340 MB); and `k4ntz/HackAtari`
  (MIT) at commit `3b5fc32` (2025-05-23, full hash recorded by run.sh), chosen because that is
  the state at the paper's date and still uses the paper's modification names: at current
  HEAD Freeway `stop_all_cars` was renamed `stop_all_cars_tunnel` (checked), and an unknown
  modification name is silently ignored, so the driver asserts that every requested
  modification registered hooks. OCAtari 2.1.0 from PyPI (MIT, as the upstream requirements pin).
  Both are put on `PYTHONPATH`, not installed. Weights are sha256-recorded in `results.json`
  (inputs of the `weights` field) and are loaded with `torch.load` (torch 2.7.1 default
  `weights_only=True`) from the official repo only.
- **Environment.** CPython 3.11 (numpy 1.24.4, upstream's pin, needs <= 3.12) via a pinned
  `uv` in a venv under `runs/`; torch 2.7.1, gymnasium 0.29.1, ale-py 0.9.1 (ROMs bundled; the
  paper used 0.8.1, which needs a separate ROM install), ocatari 2.1.0, opencv 4.10, numpy
  1.24.4, with `==` pins for everything and no stable-baselines3/rliable (the driver
  re-implements the FIRE-on-reset step and the aggregation).
- **Evaluation loop (own driver, `src/run_agents.py`).** HackAtari `obs_mode="dqn"`
  (or `"obj"` for the object-vector agent), `mode="ram"`, frameskip 4, sticky actions 0.25,
  minimal action set, 84x84 grey with 4 stacked frames, the mask wrappers from
  OCAtariWrappers, paired with the agent files by name as the README's `eval_example.sh` does (pixel and object-vector: none;
  object: `pixels`; binary: `binary`; classes: `classes`; planes: `planes`), sampled actions
  from the agent's policy as upstream `draw_action` does (no epsilon greedy, since these are
  PPO policies), a random number (1 to 30) of NOOPs after reset (paper protocol, App. D), FIRE
  on reset where the game has it, an episode cap of 10,000 agent steps recorded as `truncated`.
  Seeds 0 to 9 per cell: `env.reset(seed=s)`, `torch.manual_seed(s)`, `numpy` seeded with s
  (so the NOOP count is reproducible).
- **Cells.** Per game, six agents x conditions: Pong orig, `lazy_enemy`, `hidden_enemy`;
  Freeway orig, `stop_all_cars`; Breakout orig, `color_player_and_ball_red`. 7 conditions x 6
  agents x 10 episodes = 420 episodes, CPU, one thread, one process.
- **Recorded.** Per-episode return, length, truncation, seed; per-agent load time (file read,
  `torch.load`, model build); CPU per-step inference time (forward pass only, from the
  episodes) and env-step time; an MPS vs CPU forward-pass microbenchmark (200 steps each of
  recorded observations, MPS synchronised, bit-difference of the logits) for each of the six
  Pong agents; Pong, Freeway and Breakout video (pixel agent, original and modified side by
  side, 500 agent steps each) saved under `~/.cache/occam-hackatari/videos/`, not committed.
- **Assumptions.** A2 (zero-shot, trained on the original only): met, as the agents are used
  untouched. A6 (objects from OCAtari's RAM extraction): met, the wrappers use OCAtari. A3 and
  A5 concern the human study and the choice of variants: not exercised (no humans); C11 is
  tested for the agents only. Differences from the paper that bear on comparing numbers:
  these are seed-0 checkpoints with the repository's own evaluation settings, the paper's
  tables use 30 episodes, epsilon 0.001 and the paper's agent set (which includes IMPALA, DQN
  and others we do not run); the mask agents trained for 40M frames, not 200M.
- **Correctness checks.** `src/checks/` is not used: no derivation or custom gradient. The
  driver asserts, before any episode, that the weights' sha256 values equal the recorded ones
  and, for every cell, that every requested modification registered at least one hook; the
  observation-shape check against the first layer runs in `--smoke` (upstream's
  `load_state_dict` also fails on a mismatch).
- **Domain-repo code.** none.

## Budget

- Approved scope (user, 2026-11-24, through the supervisor): Mac only, a couple of hours,
  downloads within 500 MB in total.
- Downloads: weights 134 MB; wheels (torch 2.7.1 about 70 MB, opencv about 55 MB, scipy,
  pandas, scikit-image, matplotlib, sklearn, numpy, others: about 250 MB in total); Python 3.11
  and uv about 40 MB; git metadata under 10 MB. About 430 MB for the first run, and the
  reproduction reuses `~/.cache/occam-hackatari` and the uv cache (both outside the checkout)
  for about 30 MB more; 500 MB covers both. If the caches are cleared before the reproduction,
  the total passes the budget and needs the user's call.
- Wall clock: about 1 ms per agent step on CPU (probe), 420 episodes of up to 10,000 steps:
  15 to 30 min worst case, plus environment setup 5 min, MPS microbenchmark and videos 5 min:
  run about 40 min; reproduction about 35 min; slack to 120.
- GPU: yes, the Mac's MPS device only for the forward-pass microbenchmark (about 2 minutes, after
  the supervisor grants the slot); the evaluation itself is CPU.
