# gthbaselines

Baseline policies for the game-transformer-harness evaluation runner (`gth evaluate`) that live
here rather than in the harness, because their code or licence must stay out of it. Each one
implements `gtharness.policies.Policy` and is named on the runner's command line as
`gthbaselines.<module>:<Class>` with `lib/` on `PYTHONPATH`.

This package is not one of the repo's `tools/`: it has pinned third-party dependencies
(`requirements.txt`, Python 3.12), and the stdlib-only, Python 3.9 rule doesn't apply to it.

## Licences

| Part | Licence | Where it lives |
|---|---|---|
| This package's code | the research repo's (no LICENSE file yet) | here |
| Upstream JOWA code ([CJReinforce/JOWA](https://github.com/CJReinforce/JOWA) at `8cb02c5`) | GPL-3.0 | a clone under `runs/gthbaselines/JOWA`, fetched by `check.sh`, never vendored or edited |
| JOWA weights (`tokenizer.pt`, `JOWA_40M.pt`, `JOWA_150M.pt`, from the README's Google Drive folder) | **unknown**: the upstream repo states none for them | `$JOWA_WEIGHTS_DIR` (default `~/.cache/jowa`), outside the checkout |

`jowa_policy.py` imports and runs the GPL-3.0 upstream code, so a program that loads it is a
GPL-3.0 combination: keep it in this repo, and don't import it from the harness (decision 6a of
the baseline-comparison plan). Never commit or upload the weights, and don't publish anything
that redistributes them; results and scores are fine.

## JOWA (`jowa_policy.JowaPolicy`)

JOWA-40M or -150M (Cheng et al. 2025) with the paper's evaluation settings as explicit
arguments. Every argument except `beam_width` and `horizon` is required, because each changes
what is played. The values that reproduce upstream's evaluation:

| Argument | Upstream's value | Other values |
|---|---|---|
| `size` | `JOWA_40M` or `JOWA_150M` | |
| `task_game` | the game's `GAME_NAMES` entry (`Assault`) | any entry, for games JOWA never saw |
| `valid_actions` | `task_game` (the ALE minimal set) | `all` (the 18) |
| `context_steps` | per game: `buffer_size` in `eval/JOWA_<size>_raw_scores.json` (Assault: 3 for 40M, 4 for 150M) | 1..8; 8 is only `config/eval.yaml`'s default, which the jowa smoke's greedy runs used |
| `epsilon` | 0.001 | any in [0, 1) |
| `q_dropout` | `upstream` (the Q heads' dropout is live at evaluation) | `off` (deterministic heads) |
| `planning` | `beam_search`, with `beam_width` and `horizon` from `eval/JOWA_<size>_raw_scores.json` (it needs `valid_actions="task_game"`: upstream's world model plans over the minimal set only) | `greedy` |
| `precision` | `bfloat16_autocast` (upstream's `torch.cuda.amp.autocast`, bfloat16 on CUDA, a no-op elsewhere) | `float32`; `bfloat16_autocast` runs `torch.autocast` on the device (works on the CPU and MPS) |
| `device`, `threads` | | `cpu` or `mps`; torch's CPU threads |

Observations: 84x84 gray, max-pooled, frame skip 4, and actions: ALE's 18 (`act` returns
ALE action numbers, and the runner maps them onto the pad with `GameConfig.from_atari`), so
the protocol's action set must be `atari_full`, as `jowa2025`'s is. JOWA reads only the newest frame of the stack and keeps its own context of image
and action tokens, so the policy asks for no particular stack depth. The weights are checked
against `jowa_weights.FILES` (SHA-256) before loading, and `checkpoint_sha256` is the model
file's. The policy id names every setting that changes play, the device included (CPU and MPS
differ numerically).

JOWA's published scores, the held-out games' included, are with planning (beam search, paper
Sec. 5.1; `kb/papers/cheng2025jointly.md` C17), so greedy runs compare with them only as
baselines, not as reproductions.

Speed (Mac, M-series, JOWA-40M, greedy, 8-step context, the testcore smoke, 199 steps):

| Device | `float32` | `bfloat16_autocast` |
|---|---|---|
| CPU | about 35 ms a step | about 450 ms a step |
| MPS (`check.sh --mps`, 2026-11-27) | 27.2 ms | 49.3 ms |

bfloat16 autocast works on MPS (the tokenizer's layers run in bfloat16 there, checked with a
forward hook) but is slower than float32 on both Mac devices; it matches upstream's precision,
which is what the PC (CUDA) runs should use. Beam search costs one world-model rollout per beam and step (1-2 s a
step on the Mac, the `2026-11-24-jowa-mac-smoke` experiment), so it's for the PC.

Fetch the weights (151 MB for the tokenizer and 40M, plus 511 MB for 150M):

    PYTHONPATH=lib runs/gthbaselines/.venv/bin/python -m gthbaselines.jowa_weights ~/.cache/jowa JOWA_40M

`jowa_smoke.py` runs `gth evaluate` on the harness's test core under the `jowa2025` protocol
(only the game, episode length and count changed), as the end-to-end check.

## Testing

The research repo has no CI, so `check.sh` is the check: run it on every commit that touches
`lib/gthbaselines`, from the repo root:

    lib/gthbaselines/check.sh           # set up if needed, then the CPU tests (~25 s once set up)
    lib/gthbaselines/check.sh --mps     # also the testcore smoke on MPS: needs the Mac GPU slot
    lib/gthbaselines/check.sh --setup   # set up only

It builds everything under the git-ignored `runs/gthbaselines/`: a pinned uv, the venv from
`requirements.txt` (rebuilt when the file changes), upstream JOWA and the harness at their
pinned commits (`GTH_HARNESS_URL` overrides the harness's remote, for a local clone), the
harness's extension module and test core. It fetches the tokenizer and JOWA-40M into
`$JOWA_WEIGHTS_DIR` when missing. The first run takes about 10 minutes and 1.5 GB.

The tests (`tests/`) check the weights' hashes, loading upstream JOWA, the argument refusals,
and the real JOWA-40M on the CPU: the same actions as upstream's own `Agent.act` loop,
determinism, the seeded epsilon and dropout draws, the action mask, the sliding
context, beam search, bfloat16 autocast, and the testcore smoke through `gth evaluate`.
`--mps` runs the smoke on MPS in both precisions.

To move a pin: change `JOWA_COMMIT` in both `check.sh` and `_jowa_upstream.py`, or
`HARNESS_COMMIT` in `check.sh`, and rerun `check.sh`. To change a dependency, edit the direct
list in `requirements.txt`'s header and regenerate it with the command written there.
