---
id: 2026-11-24-uitars-mac-smoke
title: UI-TARS-1.5-7B (4-bit MLX) on an Apple-silicon Mac: closed-loop browser 2048 smoke test of the screenshot-to-key-action interface
claims: [wang2025gametars#C8]
assumptions: [wang2025gametars#A1, wang2025gametars#A2, wang2025gametars#A3, wang2025gametars#A4, wang2025gametars#A5, wang2025gametars#A6, wang2025gametars#A7, wang2025gametars#A8, wang2025gametars#A9, wang2025gametars#A10, wang2025gametars#A11, wang2025gametars#A12, wang2025gametars#A13, wang2025gametars#A14, wang2025gametars#A17, wang2025gametars#A18, wang2025gametars#A21, wang2025gametars#A26]
status: done
gpu: yes
budget_wall_min: 150
budget_downloads_mb: 6000
ci_smoke: false
approved: user 2026-11-24
---

# UI-TARS-1.5-7B (4-bit MLX) on an Apple-silicon Mac: closed-loop browser 2048 smoke test of the screenshot-to-key-action interface

## Hypothesis

UI-TARS-1.5-7B, the open GUI-agent model that Game-TARS uses as its reference baseline (A23, C4-C6),
runs on a 16 GB Mac in 4-bit MLX and can drive a keyboard-only web game closed-loop in a paused
browser sandbox (A17, A18, A3: one screenshot per step, a few named keys, no real-time latency):
its official prompt and action space produce output that parses into key actions which Playwright
executes and which change the 2048 board. This is an interface and feasibility check of the open
model, not a test of C8's content (Game-TARS vs humans on Poki games) and not a performance
claim; Game-TARS itself is not released, and 2048 is not one of C8's four games. The assumptions
listed are C8's; only A3, A17 and A18 (and the device-level key action space of A1) are exercised;
the rest concern training and the Game-TARS model and are not exercised.

## Success criteria

The run is exploratory (it clones third-party code at run time, uses a third-party 4-bit
conversion and a different model from the claim's), so any recorded verdict is capped at
`inconclusive`; the criterion states what the observed values would otherwise mean.

- wang2025gametars#C8: supports if parse_rate >= 0.9 and board_change_rate >= 0.5 and mean_score_model > mean_score_random; contradicts if parse_rate < 0.5 or board_change_rate < 0.1; otherwise inconclusive. (Exploratory: recorded as inconclusive whichever way it falls.)
- tolerance parse_rate: 10%
- tolerance board_change_rate: 20%
- tolerance mean_score_model: 30%
- tolerance mean_score_random: 0

Metric definitions: per step the model gets one screenshot and decodes greedily (temperature 0).
`parse_rate` is the fraction of steps whose output `ui_tars.action_parser` parses into a known
action of the COMPUTER_USE action space; `board_change_rate` the fraction of steps whose executed
action changed the 2048 board (a key slid or merged tiles); `mean_score_model` the mean final game
score over the model's episodes; `mean_score_random` the mean final score of a uniform random
arrow-key policy with the same game seeds and step cap (no GPU).

## Method

- **Model.** `mlx-community/UI-TARS-1.5-7B-4bit` (4-bit conversion of `ByteDance-Seed/UI-TARS-1.5-7B`,
  Apache-2.0 both; conversion repo revision `cca0e360e25322cdf480cf841e301b6d55bce9e0`, one 4.87 GB
  `model.safetensors`, sha256 `75cc3578...07d3` listed by the hub, re-verified after download). Used
  as is, no self-conversion: a community conversion of the official weights, safetensors only (no
  pickle). The bf16 official weights (16 GB) are never downloaded. Weights go in the HF cache outside the repo.
- **Runtime.** `mlx-vlm` on the Mac GPU (Metal), uv venv with CPython 3.12 (upstream needs >=3.10),
  dependencies pinned with `==` as resolved on 2026-11-24. `PYTHONDONTWRITEBYTECODE=1`.
- **Prompt and parser.** `run.sh` clones `https://github.com/bytedance/UI-TARS` (Apache-2.0) at commit
  `582f3a7ea5d285ee8ed9e2e84048d1ab01453c49` into `runs/<id>/UI-TARS/` (git-ignored, not vendored), and uses
  its `ui_tars.prompt` COMPUTER_USE template (language English, instruction naming the 2048 goal and
  the arrow keys) and `ui_tars.action_parser` (`parse_action_to_structure_output`, `smart_resize`). The model's text is decoded with special tokens skipped, as the official TGI deployment path does.
  Single screenshot per step, no action history in the prompt (A3).
- **Game.** `https://github.com/gabrielecirulli/2048` (MIT) at commit
  `478b6ec346e3787f589e4af751378d06ded4cbbc`, cloned into `runs/`, served from a local static HTTP
  server; headless Chromium via Playwright (pinned; browser binary from Playwright's cache outside
  the repo). Viewport 728 x 812 px (multiples of 28, so screenshot coordinates equal model coordinates; 812 px shows the whole page including the board's bottom row). The game's
  `Math.random` is replaced by a seeded generator (init script) so tile spawns follow the seed; the game
  is turn-based, so it is paused while the model thinks, as in the paper's sandbox (A17).
- **Action execution.** `hotkey(key=...)` with left/right/up/down (or arrow-prefixed) presses the arrow
  key; click, left_double and right_single click at the mapped point; wait, scroll and type are
  no-ops that still count as parsed; `finished` ends the episode. Anything unparsable is a
  no-op counted as a parse failure.
- **Protocol.** 3 model episodes with game seeds 0, 1, 2, at most 30 steps each (an episode also ends at
  game over or `finished`), then the random baseline on the same seeds and cap. Recorded: score per
  episode, time per step (screenshot to action, MLX generation incl. prefill) and tokens per step, peak
  memory (MLX peak and process RSS), parse and board-change rates, load time, weights provenance. One
  frame strip (PNG of the screenshots) per episode is saved under `runs/` (git-ignored); the three small strips are also copied to `figures/`.
- **Smoke mode.** `run.sh --smoke` loads the model and plays 1 model step plus a 10-step random
  game, writing under `runs/<id>/smoke/` only; it may exceed a minute on a cold cache (reported).
- **Assumptions.** A3, A17, A18 met as stated (single screenshot, paused sandbox, keyboard-only game
  with four named keys, though UI-TARS gets no per-game key list beyond the instruction); A1 is met
  by UI-TARS's own (GUI, not relative-mouse) action space, which differs from Game-TARS's. Others
  (training, memory, thinking, Minecraft) are not exercised.
- **Correctness checks.** none (inference smoke test with no derivation or custom gradient; the
  run asserts that the parser's output action names are in the documented action space and that the
  weights hash matches).
- **Domain-repo code.** none.

## Budget

- Scope (user, 2026-11-24, through the supervisor): Mac only, a couple of hours, downloads within 6 GB.
- Downloads about 5.2 GB for a fresh run: model 4.87 GB plus tokenizer files, Chromium headless shell
  about 0.1-0.2 GB, wheels (mlx, transformers, opencv, ...) about 0.3 GB, uv and Python 3.12 about 40 MB.
  Run's share about 5.2 GB. The reproduction (not required for exploratory results) would reuse the HF
  and Playwright caches and cost about 0.3 GB; 6000 MB covers both.
- Wall clock: run about 100 min (environment and downloads 20, model load 1, 90 model steps at an
  estimated 10-30 s, debugging slack), reproduction about 30 min; 150 total.
- GPU: yes, Metal through MLX, one process during the model steps; granted by the supervisor first.
