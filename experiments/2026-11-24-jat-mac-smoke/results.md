# JAT Mac smoke test: results

Experiment `2026-11-24-jat-mac-smoke`, run on 2026-11-24 at research `cb70638` (code id `inputs.src` `f13012d`). Plan:
[plan.md](plan.md). Machine-readable results: [results.json](results.json).

## Verdicts

| Claim | Verdict | Evidence |
|---|---|---|
| gallouedec2024jack#C12 | inconclusive | exploratory run, capped at inconclusive. load_ok 1, weights_sha256_match 1, episodes_completed 9 of 9, nonfinite_count 0: the checkpoint loads with the hub's custom code and plays closed-loop episodes |
| gallouedec2024jack#C4 | inconclusive | exploratory run, capped at inconclusive. tokens_per_timestep 2 (sequence positions per processed timestep, all calls: [2]), window_timesteps 256 (max_position_embeddings 512) |

## Metrics

| Metric | Value | Unit | n | CI |
|---|---|---|---|---|
| load_ok | 1 | 1 | 1 |  |
| weights_sha256_match | 1 | 1 | 1 |  |
| episodes_completed | 9 | 1 | 9 |  |
| nonfinite_count | 0 | 1 | 4888 |  |
| tokens_per_timestep | 2 | 1 | 3 |  |
| window_timesteps | 256 | 1 | 1 |  |
| param_count | 1.92702e+08 | 1 | 1 |  |
| model_load_s_cpu | 0.189395 | s | 1 |  |
| nonfinite_count_cpu | 0 | 1 | 4888 |  |
| return_pong_cpu | 7.33333 | 1 | 3 |  |
| norm_return_pong_cpu | 0.672475 | 1 | 3 |  |
| steps_pong_cpu | 1384 | 1 | 3 |  |
| ms_model_pong_cpu | 26.4548 | ms | 3 |  |
| ms_env_pong_cpu | 0.608783 | ms | 3 |  |
| return_breakout_cpu | 4 | 1 | 3 |  |
| norm_return_breakout_cpu | 0.00327975 | 1 | 3 |  |
| steps_breakout_cpu | 240 | 1 | 3 |  |
| ms_model_breakout_cpu | 32.0726 | ms | 3 |  |
| ms_env_breakout_cpu | 0.731438 | ms | 3 |  |
| return_gotoobj_cpu | 0.925 | 1 | 3 |  |
| norm_return_gotoobj_cpu | 0.99348 | 1 | 3 |  |
| steps_gotoobj_cpu | 5.33333 | 1 | 3 |  |
| ms_model_gotoobj_cpu | 36.2095 | ms | 3 |  |
| ms_env_gotoobj_cpu | 0.245984 | ms | 3 |  |
| nonfinite_count_mps | 0 | 1 | 5969 |  |
| return_pong_mps | 21 | 1 | 3 |  |
| norm_return_pong_mps | 1.00032 | 1 | 3 |  |
| steps_pong_mps | 1647.33 | 1 | 3 |  |
| ms_model_pong_mps | 28.5714 | ms | 3 |  |
| ms_env_pong_mps | 0.680499 | ms | 3 |  |
| return_breakout_mps | 6.33333 | 1 | 3 |  |
| norm_return_breakout_mps | 0.00660704 | 1 | 3 |  |
| steps_breakout_mps | 337 | 1 | 3 |  |
| ms_model_breakout_mps | 31.7034 | ms | 3 |  |
| ms_env_breakout_mps | 0.827664 | ms | 3 |  |
| return_gotoobj_mps | 0.925 | 1 | 3 |  |
| norm_return_gotoobj_mps | 0.99348 | 1 | 3 |  |
| steps_gotoobj_mps | 5.33333 | 1 | 3 |  |
| ms_model_gotoobj_mps | 88.0086 | ms | 3 |  |
| ms_env_gotoobj_mps | 0.244602 | ms | 3 |  |
| model_load_s_mps | 0.973361 | s | 1 |  |

## Episodes

| device | task | seed | return | agent steps | capped | model ms/step | env ms/step |
|---|---|---|---|---|---|---|---|
| cpu | atari-pong | 0 | -20.0 | 852 | no | 25.3 | 0.59 |
| cpu | atari-pong | 1 | 21.0 | 1652 | no | 26.0 | 0.61 |
| cpu | atari-pong | 2 | 21.0 | 1648 | no | 28.0 | 0.63 |
| cpu | atari-breakout | 0 | 7.0 | 335 | no | 27.5 | 0.67 |
| cpu | atari-breakout | 1 | 3.0 | 198 | no | 31.5 | 0.69 |
| cpu | atari-breakout | 2 | 2.0 | 187 | no | 37.2 | 0.83 |
| cpu | babyai-go-to-obj | 0 | 1.0 | 3 | no | 45.2 | 0.29 |
| cpu | babyai-go-to-obj | 1 | 0.9 | 6 | no | 32.2 | 0.22 |
| cpu | babyai-go-to-obj | 2 | 0.9 | 7 | no | 31.2 | 0.24 |
| mps | atari-pong | 0 | 21.0 | 1642 | no | 27.3 | 0.61 |
| mps | atari-pong | 1 | 21.0 | 1652 | no | 26.4 | 0.67 |
| mps | atari-pong | 2 | 21.0 | 1648 | no | 32.0 | 0.76 |
| mps | atari-breakout | 0 | 3.0 | 202 | no | 31.6 | 0.82 |
| mps | atari-breakout | 1 | 5.0 | 279 | no | 30.8 | 0.82 |
| mps | atari-breakout | 2 | 11.0 | 530 | no | 32.7 | 0.84 |
| mps | babyai-go-to-obj | 0 | 1.0 | 3 | no | 201.6 | 0.34 |
| mps | babyai-go-to-obj | 1 | 0.9 | 6 | no | 29.7 | 0.19 |
| mps | babyai-go-to-obj | 2 | 0.9 | 7 | no | 32.7 | 0.20 |

## Checks

- none (inference smoke test: no derivation or custom gradient; the load, layout, finiteness and
  episode assertions above are the run's own checks)

## Deviations from the plan

- CPU and MPS ran as two invocations of `run.sh` (`--devices cpu`, then `--devices mps`), because the GPU
  slot had to be requested between them; `--finalize` then built `results.json` from both. A plain
  `./run.sh` runs both in one go. `wall_clock_s` is the sum of the two devices' episode time plus model
  load and hashing, and excludes the environment setup and the model download.
- Both device runs (CPU and MPS) were produced by the commit before cb70638 ("wall clock is the sum of
  per-device times"), which changed only the wall-clock bookkeeping (`wall_s` per device and its sum in
  `finalize`). `commits` and `inputs` in `results.json` are captured at `--finalize` time (HEAD cb70638),
  not at the device runs; the code is otherwise identical. The run is exploratory, so `inputs` is not
  checked.
- The Atari step cap in the code is 2000 agent steps per episode, not the 4000 the plan's Method says
  (decided while writing the code; not caught before the run). It did not bind: the longest episode is
  1652 agent steps (Pong), so no episode was capped and `episodes_completed` is 9 of 9. A close Pong game
  would exceed 2000 steps and be counted as not completed.
- Assumption A6: the plan says the run checks that an Atari step's model input has the reward. No such
  check exists in the code (the probe records only shapes and finiteness). Reading the hub's remote code
  instead shows that A6 does not hold for Atari: `embed_rl` (modeling_jat.py:445, :457) concatenates the
  reward to continuous and discrete observations but not to image observations (lines 458-460), which agrees
  with the image bullet on p. 4 of the paper (the general sentence there writes phi(s, r)). The plan's A6
  statement was wrong for image observations.
- The plan's claim C12 covers the model, the dataset and the training code; the run exercised only the
  released model and its inference code, not the dataset or the training code.
- Model load time is for a warm HF cache (the weights had been fetched by the first smoke attempt, so
  `download_s` in `results.json` is not the real download time). Real download: model snapshot about
  745 MB on disk in the HF cache, plus about 300 MB of wheels: about 1.05 GB, within the 1.5 GB budget.
- MuJoCo and Meta-World were skipped, as the brief says; BabyAI GoToObj installed cleanly (minigrid 2.3.1).

## Figures

- none

## Reproduce

```sh
experiments/2026-11-24-jat-mac-smoke/run.sh
```

Expected wall clock about 6 min with a warm HF cache and venv (first run adds about 750 MB of
weights, the wheels and the Python bootstrap). The MPS step needs the supervisor's GPU slot.
Tolerances from plan.md: `load_ok` within 0; `weights_sha256_match` within 0; `episodes_completed`
within 0; `nonfinite_count` within 0; `tokens_per_timestep` within 0; `window_timesteps` within 0. A
rerun reproduces when every criterion's verdict is unchanged and each toleranced metric is within its
tolerance. Returns and timings are reported but not compared (sampling is stochastic and differs
between CPU and MPS).

## Notes

- Loading works on Mac with transformers 4.36.2 (the version in the checkpoint's config), torch 2.2.2 and
  `trust_remote_code=True`, with no patch: no missing or unexpected state-dict keys. The hub's remote code
  was read first (imports only torch, numpy, gymnasium, transformers, torchvision). Env wrappers come from
  the upstream repo (commit pinned in `results.json` provenance). Only `import ale_py` (ale-py 0.9.1; 0.8.1
  doesn't register `ALE/*` with gymnasium 0.29.1) and opencv for gymnasium's resize wrapper were needed.
- Layout (C4): every transformer call has two sequence positions per processed timestep (the first call
  has 1 timestep, 2 positions; later calls carry the previous and the current timestep, 4 positions).
  `window_timesteps` 256 is computed from the config (`max_position_embeddings` 512 // 2, the processor's
  truncation limit in `get_next_action`, modeling_jat.py:717), not a window the run exercised. The KV cache
  is trimmed with `context_window` counted in sequence positions (modeling_jat.py:809-811), so upstream's
  eval settings (32 for Atari, 256 for BabyAI, which this run copied) mean 16 and 128 timesteps of context.
  The C4 evidence is about the layout, not about the 19x ratio against Gato.
- A6: see Deviations. For Atari the released model's input has no reward channel; the run did not test the
  effect. The wording of A6 in the paper note ("reward is encoded in the observation embedding") is general
  and may need narrowing to vector and discrete tasks: a follow-up for the note's reader.
- Code remarks: `smoke.py` has a dead line (`_, _, _, warm_load_s = None, ...`), and the plan's cold load
  time was not measured (only the warm-cache load is reported). The layout ratio uses floor division
  (`t // s`), so a layout such as 5 positions per 2 timesteps would also be reported as 2; the recorded
  first calls ([1, 2], [2, 4], [2, 4]) show exact 2 per timestep. Code left as is (no rerun).
- Behaviour (not a claim under test): Pong reaches 21 (expert level) in 5 of 6 episodes across devices,
  Breakout scores only a few points (expert-normalised about 0.003 to 0.007), BabyAI GoToObj solves all six
  episodes in 3 to 7 steps. CPU and MPS differ in Pong seed 0 (-20 on CPU, 21 on MPS) because sampling
  paths diverge from numerically different logits; with only 3 episodes per cell these are anecdotes.
- Speed: about 25 to 37 ms per Atari step for the model on CPU (4 threads) and about 26 to 33 ms on MPS (per-game means 28.6 ms Pong, 31.7 ms Breakout), so
  MPS gives no speed-up at batch 1 with a 32-position (16-timestep) Atari cache; the first MPS BabyAI episode averaged 202 ms per step over its 3 steps (warm-up of the whole episode, not one call).
  The env step is under 1 ms.
- Going further: a longer evaluation (100 episodes per game as in the paper) is cheap on the Mac for Atari
  (about 1 to 3 min per Pong episode at these speeds); Meta-World needs the `qgallouedec/Metaworld` fork and
  MuJoCo, untested here. Training needs the 8-V100-class setup of the paper (9 days); fine-tuning on
  TheaterPC's CUDA GPU is plausible (weights 0.77 GB) but needs the jat-dataset (not downloaded).
- Licences: upstream code Apache-2.0 (LICENSE in the repo); the hub model card says "License: Apache 2.0"
  and the hub metadata carries no licence tag.
