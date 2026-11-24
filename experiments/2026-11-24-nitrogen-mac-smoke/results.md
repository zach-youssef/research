# NitroGen Mac smoke test: results

Experiment `2026-11-24-nitrogen-mac-smoke`, run on 2026-11-24 at research `aa0ee37` (code id `inputs.src` `070d6fd`). Plan:
[plan.md](plan.md). Machine-readable results: [results.json](results.json).

## Verdicts

| Claim | Verdict | Evidence |
|---|---|---|
| magne2026nitrogen#C8 | inconclusive | exploratory run, capped at inconclusive. chunk_shape_ok 0: the released ng.pt outputs horizon 18 x dim 25 (21 buttons, BUTTON_ACTION_TOKENS in nitrogen/shared.py), while the paper states 16-step chunks over 16 buttons + 2 sticks; this paper-vs-release gap is unexplained and bears on A11's chunk length, not on C8's content. nonfinite_count 0, action_std_across_frames 0.0374 |

## Metrics

| Metric | Value | Unit | n | CI |
|---|---|---|---|---|
| chunk_shape_ok | 0 | 1 | 1 |  |
| nonfinite_count | 0 | 1 | 2250 |  |
| action_std_across_frames | 0.037361 | 1 | 5 |  |
| action_horizon_found | 18 | 1 | 1 |  |
| action_dim_found | 25 | 1 | 1 |  |
| button_width_found | 21 | 1 | 1 |  |
| button_on_fraction | 0.0153439 | 1 | 5 |  |
| stick_abs_mean | 0.113183 | 1 | 5 |  |
| distinct_button_chunks | 4 | 1 | 5 |  |
| ms_per_chunk_cpu | 949.244 | ms | 5 |  |
| ms_per_chunk_mps | 477.907 | ms | 5 |  |
| mps_cpu_max_abs_diff | 0.000112474 | 1 | 5 |  |
| mps_nonfinite_count | 0 | 1 | 5 |  |
| ms_per_chunk_mps_bf16 | 470.255 | ms | 5 |  |

## Checks

- none (inference smoke test: no derivation or custom gradient; the shape, finiteness and variation
  assertions above are the run's own checks)

## Deviations from the plan

- The shape criterion fails: the released checkpoint outputs 18-step chunks of 25 dims (`action_horizon` 18,
  `action_dim` 25 in its config; 21 buttons from `BUTTON_ACTION_TOKENS` in `nitrogen/shared.py`, plus 4 stick
  dims), while the plan's criterion (from the paper: 16-step chunks over 16 buttons + 2 sticks, or
  16 x 24 in App. A.1) expects 16 x 25 (B taken from the checkpoint). So `chunk_shape_ok` is 0. The gap
  between paper and release is unexplained; it bears on A11's chunk length, not on C8's content, and is
  recorded in the note's Implementation notes and open questions.
- The vision encoder named in the checkpoint is `google/siglip2-large-patch16-256` (SigLIP 2, as in
  the paper), not the siglip-large default of the repo's config class. Its weights were not downloaded: the
  tower is built from the config and the checkpoint supplies the weights (strict load; the checkpoint holds
  all 400 vision-encoder keys), which saves about 1 GB.
- Smoke mode takes about 30 s with a warm cache, and it exits 0 despite the horizon finding (it fails
  only on non-finite outputs).
- Download total about 2.1 GB (ng.pt 1.97 GB, video 21.5 MB, wheels and Python). The run was repeated
  once after review (code changes only for logging and the ckpt hash); these are the repeat's numbers.

## Figures

- none (the frames come from a video of unclear licence and are not committed)

## Reproduce

```sh
experiments/2026-11-24-nitrogen-mac-smoke/run.sh
```

Expected wall clock about 1 min with a warm HF cache and venv (first run adds the
2 GB weight download and the environment). Tolerances from plan.md: `chunk_shape_ok` within 0;
`nonfinite_count` within 0; `action_std_across_frames` within 10%. A rerun reproduces when every
criterion's verdict is unchanged and each toleranced metric is within its tolerance.

## Notes

- Mac inference is workable: about 0.95 s per chunk on CPU (fp32) and 0.48 s on MPS (fp32), 0.47 s
  with bf16 autocast (no real gain). MPS matches CPU to 1e-4 (max abs difference on the raw chunk,
  with the same CPU-drawn noise seed), so MPS is safe to use. Frame-to-frame variation is small
  (std 0.037 across five frames, same noise) but nonzero: 4 of 5 frames give distinct button
  chunks, and few buttons fire (1.5% on), consistent with a sparse chunk on single frames.
- This checkpoint is game-unconditional (`game_mapping_cfg` is null).
- The video has caption overlays on the frames (not in the training observation), and the 5 frames
  come from two or three games; this is a shape and latency check, not a behaviour check.
- Run log: `runs/<id>/logs/` (git-ignored). ng.pt sha256 is in `results.json` provenance.
- Next: stage 1 on TheaterPC (game plus clock hook).
