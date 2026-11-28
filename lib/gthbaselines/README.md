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

## JOWA's upstream code and weights

`_jowa_upstream.py` loads upstream JOWA from the pinned clone, with the import-time patches
inference on a Mac needs (each listed there with its reason); `jowa_weights.py` lists the
released weights' SHA-256s and fetches them (151 MB for the tokenizer and 40M, plus 511 MB for
150M):

    PYTHONPATH=lib runs/gthbaselines/.venv/bin/python -m gthbaselines.jowa_weights ~/.cache/jowa JOWA_40M

## Testing

The research repo has no CI, so `check.sh` is the check: run it on every commit that touches
`lib/gthbaselines`, from the repo root:

    lib/gthbaselines/check.sh           # set up if needed, then the tests
    lib/gthbaselines/check.sh --setup   # set up only

It builds everything under the git-ignored `runs/gthbaselines/`: a pinned uv, the venv from
`requirements.txt` (rebuilt when the file changes), upstream JOWA and the harness at their
pinned commits (`GTH_HARNESS_URL` overrides the harness's remote, for a local clone), the
harness's extension module and test core. It fetches the tokenizer and JOWA-40M into
`$JOWA_WEIGHTS_DIR` when missing. The first run takes about 10 minutes and 1.5 GB.

The tests (`tests/`) check the weights' hashes.

To move a pin: change `JOWA_COMMIT` in both `check.sh` and `_jowa_upstream.py`, or
`HARNESS_COMMIT` in `check.sh`, and rerun `check.sh`. To change a dependency, edit the direct
list in `requirements.txt`'s header and regenerate it with the command written there.
