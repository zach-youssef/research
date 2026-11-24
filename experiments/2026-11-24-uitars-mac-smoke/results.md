# UI-TARS-1.5-7B (4-bit MLX) 2048 smoke test: results

Experiment `2026-11-24-uitars-mac-smoke`, run on 2026-11-24 at research `14446b2` (code id `inputs.src` `2db566d`; the sha is the
pre-rebase head, informational per docs/experiments.md "Branch rewrites"; `inputs` are what bind the results). Plan:
[plan.md](plan.md). Machine-readable results: [results.json](results.json). Exploratory run (verdict capped at inconclusive).

## Verdicts

| Claim | Verdict | Evidence |
|---|---|---|
| wang2025gametars#C8 | inconclusive | exploratory run, capped at inconclusive; parse_rate 1.000, board_change_rate 0.167, mean_score_model 13.3 vs mean_score_random 156.0 (n=3 seeds, 90 steps); criterion on these values would be inconclusive. The criterion is not met for supports (board_change_rate 0.167 below 0.5, mean_score_model below random) and not for contradicts (parse_rate 1.0, board_change_rate above 0.1), so on these values it would also read inconclusive; exploratory runs are capped at inconclusive in any case. |

## Metrics

| Metric | Value | Unit | n |
|---|---|---|---|
| parse_rate | 1 | 1 | 90 |
| board_change_rate | 0.166667 | 1 | 90 |
| arrow_action_rate | 0.5 | 1 | 90 |
| click_action_rate | 0.5 | 1 | 90 |
| parse_rate_after_box_token_strip | 1 | 1 | 90 |
| mean_score_model | 13.3333 | 1 | 3 |
| mean_score_random | 156 | 1 | 3 |
| ms_per_step_mean | 11784.2 | ms | 90 |
| ms_per_step_median | 11236.7 | ms | 90 |
| gen_tokens_per_step_mean | 122.089 | 1 | 90 |
| prompt_tokens_per_step_mean | 1143 | 1 | 90 |
| gen_tokens_per_second_mean | 23.0355 | 1 | 90 |
| load_time_s | 2.15107 | s | 1 |
| peak_mem_mlx_gb | 5.62784 | GB | 1 |
| peak_rss_gb | 1.40711 | GB | 1 |

Per-episode scores (seeds 0, 1, 2): model 36, 4, 0 (max tiles 8, 4, 2; none game over); uniform-random arrow
keys with the same seeds and 30-step cap: 176, 192, 100. `parse_rate_after_box_token_strip` and
`click_action_rate` are diagnostics, not criterion metrics.

## Checks

- none (inference smoke test: no derivation or custom gradient; the run asserts the screenshot size is a
  fixed point of `smart_resize` and that the weights' sha256 equals the hub's listed value, which it did)

## Deviations from the plan

- Two harness defects were found in review of a first full run and fixed, then the run was repeated; the numbers above are
  the repeat's. (1) The first run decoded with mlx-vlm's default, which keeps special tokens, so click
  coordinates came back as `start_box='<|box_start|>(349,641)<|box_end|>'` and the official parser rejected 57 of 90 outputs
  (parse_rate 0.367). The official TGI deployment path decodes with special tokens skipped; the repeat does that
  (`skip_special_tokens=True`), and all 90 outputs parse with the unmodified official parser. (2) The first run's 728 x 728 viewport cut off the
  bottom row of the board; the repeat uses 728 x 812 (a `smart_resize` fixed point, whole page visible). The first run's
  numbers (parse_rate 0.367, board_change_rate 0.044, scores 0, 4, 4) are therefore a harness artefact and are not used.
- No parser patch is applied: the official parser is used as is. `parse_rate_after_box_token_strip` (1.0) equals `parse_rate` here
  because no output contains box tokens any more.
- The pinned conversion repo's weights are from 2025-04-25 and loaded unchanged in mlx-vlm 0.8.1; no self-conversion was needed.
- The model played with one screenshot per step and no action history in the prompt (plan, A3), although the COMPUTER_USE template's
  first line says it is given an action history.
- Clicks were executed (as clicks on the page); all 45 click actions landed on board cells (none on "New Game"), and clicking a tile does not move 2048 tiles,
  so they count as parsed, board-unchanged steps.
- Nothing else shared the GPU during the final run.

## Figures

- `figures/strip_seed{0,1,2}.png`: frame strips (screenshots every 2-3 steps, left to right) of the three model episodes. Full per-step logs with the model's
  raw text are in `runs/2026-11-24-uitars-mac-smoke/out/play_log.json` (git-ignored).

## Reproduce

```sh
experiments/2026-11-24-uitars-mac-smoke/run.sh
```

Expected wall clock about 19 min with a warm HF cache, venv and Playwright cache (1130 s recorded; a first run adds the
4.9 GB weight download and the environment). Tolerances from plan.md: `parse_rate` within 10%;
`board_change_rate` within 20%; `mean_score_model` within 30%; `mean_score_random` within 0. A rerun reproduces
when every criterion's verdict is unchanged and each toleranced metric is within its tolerance. (Exploratory
results skip the supervisor's reproduction run.)

## Notes

- Interface: with the official COMPUTER_USE prompt (decoded as the official path does) UI-TARS-1.5-7B emits `Thought: ... Action: ...` text
  that the unmodified parser accepts on 90 of 90 steps. Its coordinates came as `start_box='(x,y)'` (the v1 form),
  not the template's `<point>x y</point>`, so the template and the 1.5-7B checkpoint do not agree on syntax, though the parser handles both.
  Keys came as `hotkey(key='arrowleft')`, `'left'`, etc. (the executor's `norm_key` in `src/play.py` normalises them); half of the steps
  (45 of 90) are clicks on tiles, although the instruction names the arrow keys.
- Behaviour: 45 key steps, 15 of them changed the board. The same `arrowleft` (30 times) and the same two clicks (27 and 17 times) repeat,
  so the model often loops on an action that does nothing. Scores are 36, 4 and 0 against 100-192 for random keys. Candidate causes, not separated
  by this run: no action history in the prompt, a template and checkpoint that don't match on syntax, a prompt that is not game-specific, 4-bit quantisation,
  greedy decoding re-seeing an unchanged screen. This is a feasibility check of the stack, not evidence about Game-TARS or C8 (a different, unreleased
  model; different games; no history; no key list; best-of-5 and human baselines not reproduced).
- Cost on an M4 / 16 GB: model load about 2 s warm (weights 4.87 GB, hash re-verified, about 15 s of the run); about 11.8 s per step
  (about 1,143 prompt tokens for one 728 x 812 screenshot, about 122 generated tokens at 23 tokens/s); MLX peak memory 5.6 GB, process peak
  RSS 1.4 GB (MLX memory is outside RSS). The full run took 19 min.
- Next: an action history in the prompt, a per-game key list as Game-TARS does (A11), the 6-bit conversion, sampling with temperature; a CUDA or TheaterPC route is only
  needed for bf16 or fine-tuning.
