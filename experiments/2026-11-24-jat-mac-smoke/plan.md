---
id: 2026-11-24-jat-mac-smoke
title: JAT released checkpoint on an Apple-silicon Mac: load, interleaved-token layout and Atari/BabyAI episodes
claims: [gallouedec2024jack#C12, gallouedec2024jack#C4]
assumptions: [gallouedec2024jack#A1, gallouedec2024jack#A2, gallouedec2024jack#A6]
status: approved
gpu: yes
budget_wall_min: 120
budget_downloads_mb: 1500
ci_smoke: false
approved: user 2026-11-24
---

# JAT released checkpoint on an Apple-silicon Mac: load, interleaved-token layout and Atari/BabyAI episodes

## Hypothesis

The released JAT checkpoint (`jat-project/jat`, custom code run with `trust_remote_code`) loads on a
Mac with an older pinned `transformers` and runs closed-loop episodes through ALE (Atari) and
BabyAI on CPU and MPS (C12: model and code are openly released and usable; A1, A2: pixel Atari
observations, discrete actions, behaviour cloning without return conditioning; A6: reward rides
in the observation embedding). It also shows the interleaved layout of C4: one timestep costs two
embeddings (observation, action). This is an architecture and interface smoke test with an
unknown, probably weak, policy (the model card reports an Atari IQM of 0.14 of expert), not a test
of any performance claim, and not a test of C4's 19x ratio against Gato.

## Success criteria

The run is exploratory (it clones third-party code and runs repo-supplied Python at run time,
policy sampling is stochastic), so any recorded verdict is capped at `inconclusive`; the criteria
state what the observed values would otherwise mean, and the observed outcome is stated in the
verdict's evidence.

- gallouedec2024jack#C12: supports if load_ok == 1 and weights_sha256_match == 1 and episodes_completed >= 6 and nonfinite_count == 0; contradicts if load_ok == 0 or weights_sha256_match == 0; otherwise inconclusive. (Exploratory: recorded as inconclusive whichever way it falls.)
- gallouedec2024jack#C4: supports if tokens_per_timestep == 2 and window_timesteps == 256; contradicts if tokens_per_timestep != 2; otherwise inconclusive. (Exploratory: recorded as inconclusive whichever way it falls.)
- tolerance load_ok: 0
- tolerance weights_sha256_match: 0
- tolerance episodes_completed: 0
- tolerance nonfinite_count: 0
- tolerance tokens_per_timestep: 0
- tolerance window_timesteps: 0

Metric definitions: `load_ok` is 1 if `AutoModelForCausalLM` and `AutoProcessor` load the pinned
revision with `trust_remote_code=True` and no missing or unexpected state-dict keys;
`weights_sha256_match` is 1 if the sha256 of `model.safetensors` equals the hub's LFS oid;
`episodes_completed` counts finished (not step-capped) episodes over the games below on CPU;
`nonfinite_count` counts NaN or inf in the action logits over all steps; `tokens_per_timestep` is
the number of sequence positions the model adds per environment step for an Atari observation
(measured from the processed model input of a 3-step rollout); `window_timesteps` is
`max_position_embeddings / tokens_per_timestep` as used by `get_next_action` (config 512 / 2).

## Method

- **Code.** `run.sh` clones `https://github.com/huggingface/jat` (Apache-2.0) at commit
  `2e1ac33216adf48910bb6bc3bbb27bf9a94132aa` into `runs/<id>/jat/` (git-ignored, not vendored) and
  puts it on `PYTHONPATH` only for its env wrappers (`jat.eval.rl.make`) and score tables;
  the model code itself is the hub's own `modeling_jat.py`, `configuration_jat.py` and
  `processing_jat.py` at the pinned model revision (read before running; they import only torch,
  numpy, gymnasium, transformers, torchvision). Patches, if needed, live in `src/` and are applied at
  run time.
- **Weights.** `jat-project/jat` at revision `911dd7447b1c56bece63424c888d2217828c981e`
  (`model.safetensors` 771 MB, tokenizer and code files; `replay.mp4` and `evaluations.json` are not
  fetched), into the HF cache outside the repo. Safetensors, so no pickle. sha256 recorded.
- **Environment.** CPython 3.11 in a uv venv (pinned `uv`), `transformers` pinned in the 4.36 to
  4.40 range (the config says 4.36.1), `gymnasium==0.29.1` (upstream's pin), ale-py of that era,
  minigrid for BabyAI, all pinned with `==`. MuJoCo and MetaWorld are skipped.
- **Episodes.** Games: Atari Pong and Breakout (`ALE/<Game>-v5`, via upstream's `make`, no reward
  clipping, as in upstream's `eval_jat.py`), and BabyAI GoToObj if minigrid installs cleanly.
  3 episodes per game and device, env seeds 0, 1, 2 (`env.reset(seed=...)` on the first reset and
  `torch.manual_seed(seed)` before each episode; actions sampled from the softmax as upstream does),
  agent-step cap per episode (Atari 4000, BabyAI 64) so a stuck policy can't run away; capped episodes
  are reported separately. The rollout loop follows `scripts/eval_jat.py` (KV cache reset per
  episode, `context_window` 32 for Atari and 256 otherwise, the fake-done handling for episodic life).
- **Measurements.** Raw return per episode, expert-normalised return from upstream's
  `scores_dict.json`, steps per episode, mean ms per agent step (model call plus env step, and the
  model call alone, synchronised) on CPU and on MPS, model load time (wall, cold and warm), CPU
  threads limited to 4. CPU is the reference device for verdict metrics; MPS runs the same seeds and
  is compared for returns and step time (sampling differs across devices, so returns aren't expected
  to match). Also recorded: the licence statements found, patches applied, parameter count.
- **Smoke mode.** `run.sh --smoke` runs one game for 20 agent steps on CPU and writes under
  `runs/<id>/smoke/` only; it needs the venv and the cached weights.
- **Assumptions.** A1 (behaviour cloning, no return conditioning), A2 (84x84 pixel Atari, discrete
  actions) and A6 (reward in the observation embedding) are met by running the released model as
  designed through upstream's wrappers; the run checks that the model's input for an Atari step has the
  reward and an 84x84 image and that actions are discrete indices, but doesn't test training.
- **Correctness checks.** none (inference smoke test; the shape, finiteness and layout assertions are
  the run's own).

## Budget

Downloads: model snapshot about 775 MB, wheels about 300 MB (torch for macOS arm64 is the bulk),
upstream repo a few MB; about 1.1 GB for my run. The reproduction run starts with an empty `runs/`
but the HF cache and uv cache are outside the checkout, so it re-downloads little; budget 1500 MB
covers both with margin, below the 1 GB-per-step approval line only for the model itself. Wall clock:
setup 10 min, CPU episodes 3 games x 3 episodes at an expected 20 to 100 ms per step, MPS the same:
about 30 min per full run, 60 for mine with iteration, 120 including the reproduction. GPU: MPS is
used for the timing comparison, with the supervisor's slot; everything else is CPU.
