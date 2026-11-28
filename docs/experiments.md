# Experiments

An experiment tests claims recorded in the knowledge base (`docs/knowledge-base.md`). Its plan,
including the success criterion, is approved by the user before anything runs, and its results
record a verdict for every claim it tests.

## Lifecycle

1. **Propose:** write `plan.md` only (no code, no runs, no downloads), with `status: proposed`
   and `approved:` empty.
2. **Approve:** the user approves the plan, or asks for changes. The first commit after that
   fills `approved:`, sets `status: approved`, and makes only the changes the user asked for.
   From then on the hypothesis, claims, assumptions, criteria, tolerances and metrics don't
   change.
3. **Run:** implement in `src/`, with the correctness checks that apply (below), commit, then
   run from the clean tree.
4. **Record:** write `results.json` and `results.md` from the run's output, set
   `status: done`, set `status: verified` on the `read` paper notes that gain a
   qualifying latest verdict from a non-exploratory run (the rule is in `docs/knowledge-base.md`,
   "Statuses"), and run `kb.py index` so the tested notes' `Experiments` blocks list the
   result.

A plan that won't be run is set to `status: abandoned`, with the reason under `## Method`, and
stays in the repo.

## Layout

```
experiments/<YYYY-MM-DD>-<slug>/
  plan.md            the approved plan (format below)
  src/               the experiment's code
  src/checks/        correctness checks, run by run.sh before anything else (below)
  run.sh             entry point: ./run.sh (full run) and ./run.sh --smoke
  requirements.txt   pinned dependencies, one name==version per line (may be empty)
  results.json       machine-readable results (schema below)
  results.md         the human summary of results.json
  figures/           PNGs made by the run, at most 500 KB each
  eval/              gth evaluate's outputs the results rest on (below, "The eval block")
runs/<experiment>/   git-ignored: .venv/, logs/, smoke/, large outputs
```

The experiment id is the directory name: the date the plan was proposed, then a lowercase
kebab-case slug (`2026-11-02-jittered-grid`). Its branch is `exp-<slug>`. `runs/` is relative
to the checkout that runs it, so a worktree's runs go away with the worktree: everything the
results need must be reproducible from `run.sh`.

No notebooks: they diff, review and run in CI badly. Plain Python modules in `src/`.

## plan.md

`templates/experiment-plan.md` is the starting point. Frontmatter, in the restricted subset of
`docs/knowledge-base.md`, every key present, in this order:

| Key | Value |
|---|---|
| `id` | The experiment id; equals the directory name. |
| `title` | One line. |
| `claims` | Inline list of the full claim ids under test: `[example2000jittered#C1, example2000jittered#C2]`. Each must exist and, while the plan is `proposed` or `approved`, must not be withdrawn (`done` and `abandoned` plans keep their claims as history). |
| `assumptions` | Inline list of the full assumption ids the experiment relies on (`docs/knowledge-base.md`, "Assumptions"): every assumption the tested claims use, plus any other note's that the setup relies on (a method borrowed from another paper brings its conditions). Same existence and withdrawal rules as `claims`; while the plan is `proposed` or `approved`, `kb.py check` also fails when it misses an assumption a tested claim uses (a claim's list may grow when its paper is re-read, and `done` and `abandoned` plans keep theirs as history). `[]` when there are none. `## Method` says how the setup meets each one, or which one it deliberately breaks and why. |
| `status` | `proposed`, `approved`, `done` or `abandoned`. |
| `gpu` | `yes` or `no`: whether any step uses the GPU. |
| `budget_wall_min` | Wall-clock budget in minutes for all runs, smoke runs included. |
| `budget_downloads_mb` | Download budget in MB: datasets, models and packages all count. |
| `ci_smoke` | `true` or `false`: whether CI runs `./run.sh --smoke`. |
| `approved` | Empty until approved, then `<user> <YYYY-MM-DD>` (`zach-youssef 2026-11-03`). |

`gpu`, `budget_wall_min` and `budget_downloads_mb` are the plan's budget. Sections, in this
order:

| Section | Contents |
|---|---|
| `## Hypothesis` | What we expect and why, in a few lines, citing the claims. |
| `## Success criteria` | Fixed before the run. One criterion line per claim in `claims`, then one tolerance line per metric the criteria name (formats below). |
| `## Method` | Inputs, parameters, seeds, repetitions, what is measured and how; how the setup meets the listed assumptions; the correctness checks (below); domain-repo code needed, if any; anything vendored, with its URL, commit and license. |
| `## Budget` | How the budget numbers were estimated. |

A criterion maps every outcome to a verdict:
```
- <claim id>: supports if <metric> <comparison>; contradicts if <metric> <comparison>; otherwise inconclusive.
```
The metric names are the ones `results.json` will use. After the criteria, a tolerance line for
every metric a criterion names says how far a rerun's value may differ from the recorded one
and still reproduce it, either absolute or relative (with `%`):
```
- tolerance <metric>: <number>
- tolerance <metric>: <number>%
```
The tolerance is fixed with the criteria, before the run, so reproduction isn't judged after the
numbers are known. `kb.py check` rejects a plan missing one. Metrics that no criterion names
are reported but not compared when reproducing.

## results.json

Written by the run (never by hand-copying numbers), and validated by `kb.py check`:

```json
{
  "schema": 1,
  "experiment": "2026-11-02-jittered-grid",
  "date": "2026-11-03",
  "commits": [
    {"repo": "research", "sha": "<40 hex>", "dirty": false}
  ],
  "inputs": {"src": "<40 hex>", "run.sh": "<40 hex>", "requirements.txt": "<40 hex>"},
  "exploratory": false,
  "tests": ["example2000jittered#C1", "example2000jittered#C2"],
  "verdicts": {
    "example2000jittered#C1": {"verdict": "supports", "evidence": "time_slope 1.01, criterion [0.9, 1.1]"},
    "example2000jittered#C2": {"verdict": "contradicts", "evidence": "min_dist_ratio 0.998, criterion >= 1"}
  },
  "metrics": {
    "time_slope": {"value": 1.01, "unit": "1", "n": 5, "ci": {"low": 0.97, "high": 1.05, "level": 0.95}},
    "min_dist_ratio": {"value": 0.998, "unit": "1", "n": 5}
  },
  "seeds": [0, 1, 2, 3, 4],
  "wall_clock_s": 41.2,
  "env": {
    "python": "3.9.6",
    "os": "macOS 26.0 arm64",
    "packages": ["numpy==2.0.2"],
    "gpu": null
  }
}
```

| Field | Meaning |
|---|---|
| `schema` | `1`. |
| `experiment` | The experiment id. |
| `date` | Date of the run, `YYYY-MM-DD`. |
| `commits` | Every repo whose code the run used: `repo` (directory name under `~/code`), the full `sha`, and whether its tree was `dirty`. `research` comes first. The `research` sha is informational once the branch is rewritten (below); domain repos' shas are merged commits, so they stay valid. |
| `inputs` | The fingerprint of the experiment's code at the run: the git object ids of `src` (a tree), `run.sh` and `requirements.txt` (blobs), from `git rev-parse <sha>:experiments/<id>/<path>`. If `src/` is missing or empty (git doesn't track an empty directory; the code may live in a domain repo), `src` is git's empty-tree id, `4b825dc642cb6eb9a060e54bf8d69288fbee4904`. |
| `exploratory` | `true` allows dirty trees; every verdict must then be `inconclusive`, because the run can't be reproduced, so an exploratory run never makes a note `verified`. `kb.py check` skips the `inputs` comparison for it. |
| `tests` | The claim ids tested; the same set as the plan's `claims`. |
| `verdicts` | One entry per id in `tests`: `verdict` is `supports`, `contradicts` or `inconclusive`, as the plan's criterion maps the metrics; `evidence` names the metrics and values it rests on. |
| `metrics` | By name: `value` (number), `unit` (`"1"` for dimensionless), `n` (sample count), and optionally `ci` (`low`, `high`, `level`). |
| `seeds` | Every seed used, as a list. |
| `wall_clock_s` | Total wall clock of the full run, in seconds. |
| `env` | `python` version; `os` (name, version, architecture); `packages`, the venv's `pip freeze` lines; `gpu`, the device and driver when the plan says `gpu: yes` (from `vulkaninfo --summary` when present), otherwise `null`. |
| `eval` | Present when any result comes from `gth evaluate` runs: one block per evaluation protocol (below, "The eval block"). Absent otherwise. |

`kb.py check` rejects a `results.json` that cites a claim that doesn't exist, lacks a verdict
for a tested claim, has a dirty commit without `exploratory: true`, belongs to a plan
without `approved:`, or, unless `exploratory: true`, whose `inputs` differ from the current
commit's (`git rev-parse HEAD:experiments/<id>/<path>`): code changed after the run, so the
results must be rerun.

### The eval block

`templates/results.json` is the starting point for an experiment that evaluates policies with
the harness (`game-transformer-harness`). Its `eval` block records where every score came
from, so a number can be recomputed and compared. It mirrors the harness's outputs and
doesn't redefine them: the run directory (`protocol.toml`, `run.json`, `episodes.jsonl`) is in
`game-transformer-harness/docs/evaluation.md`, "Results"; normalisation, the aggregates,
`run_scores` and its `seeds_are`, and the stratified bootstrap are in
`game-transformer-harness/docs/metrics.md`; the human reference file is in
`docs/evaluation.md`, "Human reference". The numbers in the template are illustrative.

`eval` maps a protocol key to one block. The key is the preset's name (`jowa2025`) when the
runs used `gth evaluate --protocol <preset>`, and `file:<stem>` for a protocol file of the
experiment's own, so a modified copy of a preset can't pass for it. A block holds only runs
under that protocol, so its aggregates can't mix protocols:

| Field | Meaning |
|---|---|
| `protocol_sha1` | The protocol's SHA-1 (`run.json`'s `protocol_sha1`). Every run of the block has it. Under a preset key, it is the preset's canonical SHA-1 at the runs' `harness.git_sha`. |
| `runs` | By arm name (the policy evaluated: `model`, `random-v1`, a baseline's name): one `gth evaluate` run each, below. |
| `normalisation` | How the block's `normalised` scores were made, or `null` when it reports raw scores only (below). One table per block. |
| `aggregates` | By metric name: `statistic` (a `gtharness.metrics` function: `iqm`, `median`, `mean`, `optimality_gap`, `probability_of_improvement`), `params` (its required keyword arguments, such as `{"threshold": 1}` for the optimality gap; `{}` when none), `runs` (the arms it reads: one, or two for `probability_of_improvement`, first arm compared against the second), `scores` (`normalised` or `raw`), `point`, and `ci`. Every aggregate is over all of the protocol's games. |
| `ci` | The interval on `point`: `method` (`stratified_bootstrap_percentile`, `stratified_bootstrap_ci`'s), `low`, `high`, `confidence`, `resamples`, and `rng_seed`, the seed of the `numpy.random.default_rng(rng_seed)` made fresh for this interval alone, so it reproduces. `null` when the bootstrap is refused: one run per game (a single checkpoint), which `results.md` then says. |
| `published` | The published numbers this block's aggregates are set against, or `[]`. Per number: `claim` (the claim id that states it), `model` (the paper's model as the claim names it), `protocol_sha1` (the protocol the number is matched with: the block's), `normalisation`, `aggregate` and `score` (the paper's), `kind` (`reproduction` when the arm claims to be the published model, `baseline` when our arm is set against it), `arm` (the run set against it) and `compare` (the block's aggregate, one that reads `arm`). |

A run (`runs.<arm>`):

| Field | Meaning |
|---|---|
| `dir` | The run's directory, relative to the experiment, as `gth evaluate --out` wrote it: `eval/<key>/<arm>` (`eval/file-<stem>/<arm>` for a `file:` key). Committed, so the scores can be recomputed from `episodes.jsonl` with `gtharness.evaluate.check_results` and `run_scores`, without rerunning; its three files are small (a few MB at most). The scored run is made without `--record` (`run.json`'s `recorded` is `false`), since `check_results` refuses a recorded run whose recordings are missing; recordings, when wanted, come from a separate run under `runs/`. |
| `sha256` | The SHA-256 of `protocol.toml`, `run.json` and `episodes.jsonl`. |
| `variant` | What was played: the model and the settings that change its play (planning, beam width, sampling), such as `JOWA-150M no planning`, or our model's name and settings. Required, since those settings live in the policy's kwargs, not in the protocol. |
| `policy` | `run.json`'s `policy`, verbatim: `spec`, `kwargs`, and `seeds`, per policy seed, its `id` and `checkpoint_sha256`. Paths in `kwargs` are relative to the checkout's root (`runs/<experiment>/checkpoints/seed0.pt`), which assumes `gth evaluate` ran from there (the policy gets its kwargs as given); never absolute, which `kb.py check` refuses. |
| `harness` | `run.json`'s `harness` (the build: `git_sha`, `git_dirty`, `profile`, `rustc`, `target`, `version`), verbatim. Its `git_sha` is the `game-transformer-harness` entry of `commits`, and `git_dirty` is `"false"` unless the run is exploratory. |
| `seeds_are` | The `seeds_are` passed to `run_scores`: `independent_runs` or `sampling_seeds`. With `sampling_seeds`, intervals cover the evaluation's randomness for one model only, and `results.md` says so. |
| `games` | Per game of the protocol: `episodes` (played), `truncated` (how many hit the cap, which `results.md` flags when not 0), `raw` (`run_scores`' per-run scores, in the protocol's policy-seed order) and `normalised` (the same runs normalised with the block's table; omitted when `normalisation` is `null`). |

The normalisation (`normalisation`):

| Field | Meaning |
|---|---|
| `name` | `hns` for human-normalised (the HNS analogue on our games), `dns` for DQN-normalised, another name for another pair. |
| `low_source`, `high_source` | Where each reference comes from: a claim id for a published table (`mnih2015human#C8` and `badia2020agent57#C10` differ for some games), an arm of the block (`runs.random-v1`: the reference is the mean of that arm's `raw` runs), or a `gth reference-scores` output committed in the block's directory (`eval/<key>/human-reference.json`). |
| `high_statistic` | With a `gth reference-scores` source, which of its per-game values is `high`: `mean` or `median`. Required then, `null` otherwise. |
| `sha256` | By path, the SHA-256 of every source that is a file in the experiment; `{}` when none is. |
| `games` | Per game of the protocol, `low` and `high`. |

A metric that a criterion names, taken from an aggregate, also appears in `metrics` under the
same name, with `value` the aggregate's `point`, `n` the number of scores it reads summed over
its arms (runs times games per arm: 12 for a probability of improvement between two arms of 3
runs on 2 games), and `ci` the aggregate's `low`, `high` and `confidence` as `level`.

`kb.py check` (once written) also rejects an `eval` block whose files don't match their
`sha256`; whose `run.json` has another `protocol_sha1` or `harness`, has `recorded: true`, or
has an absolute path in `kwargs`; whose `protocol_sha1`, under a preset key, isn't the
preset's at `harness.git_sha`; whose `raw` scores differ from `run_scores` over the run;
whose `low` or `high` differ from their source (an arm's mean, or a reference file's
`high_statistic`, whose `protocol.sha1` must be the block's `protocol_sha1`); whose
`normalised` scores differ from the table's rescale; or whose `published` entries cite a
claim that doesn't exist, name another `protocol_sha1`, or, for a `reproduction`, an arm
whose `variant` isn't the entry's `model`.

## results.md

`templates/results.md` is the starting point. Sections, in this order: `## Verdicts` and
`## Metrics` (tables repeating `results.json`, with the same numbers), `## Checks` (the
correctness checks, below), `## Deviations from the plan` (what was done differently and why),
`## Figures`, `## Reproduce` (the command, the expected wall clock, and the tolerances restated
from the plan), and `## Notes`
(anything surprising, experiments worth running next). It doesn't repeat the plan.

## Reproducibility

- `requirements.txt` pins every dependency with `==`. `run.sh` creates
  `runs/<experiment>/.venv` from it with the Python it was tested with, and records that
  version in `env.python`.
- Every source of randomness is seeded from `seeds`, and the seeds are recorded.
- The full run is from a clean, committed tree; `record_env` (in `tools/lab`, not written yet)
  captures the commits, the dirty flags, the `inputs` fingerprint and the environment. A tree
  is `dirty` when `git status --porcelain` lists anything outside the experiment's own outputs
  (`results.json`, `results.md`, `figures/`, `eval/`), so a rerun over committed results is
  still clean.
- **Branch rewrites.** The working branch is rebased onto `main` and fixups are folded in before
  merge, so the recorded `research` sha may be unreachable afterwards. What binds the results to
  the code is `inputs`: a rewrite that leaves `src/`, `run.sh` and `requirements.txt` unchanged
  keeps them valid, and one that changes them makes `kb.py check` fail until the run is
  repeated. Keep everything that affects the results inside those three paths (or in a domain
  repo at a merged commit).
- **Reproducing** (a reviewer or test-runner): from a checkout whose `inputs` match, run
  `./run.sh` and compare with `git show HEAD:experiments/<id>/results.json`. The rerun
  reproduces when every criterion's verdict is the same as the recorded one and each
  toleranced metric is within its tolerance. Then `git restore` the outputs and `git clean --
  experiments/<id>/figures/` (a rerun's new untracked figures aren't removed by restore).
- **Symlinked paths.** Call `run.sh` through a resolved path (`"$(cd <dir> && pwd -P)"/run.sh`),
  and write `run.sh` to resolve its own directory with `pwd -P` and the aggregation step to use
  `os.path.realpath`. `git rev-parse --show-toplevel` returns the real path, so a logical path
  through a symlink (macOS `$TMPDIR` is under `/var` -> `/private/var`) makes the
  experiment-relative path leave the repository, and `git rev-parse HEAD:<path>` fails. A
  fingerprint that can't be computed must fail the run on a clean tree (a placeholder such as
  forty zeros is allowed only when the tree is dirty and the run is exploratory). A zeroed
  `inputs` on a clean tree is a bug: don't record it.
- `./run.sh --smoke` finishes in under a minute on CPU with no downloads beyond the pinned
  requirements, writes only under `runs/<experiment>/smoke/` (never the committed results),
  and exits non-zero on failure. CI runs it for plans with `ci_smoke: true`.
- Figures are made by the run from its own data, as PNGs of at most 500 KB under `figures/`.
  Larger outputs stay in `runs/`.
- Experiments that need domain code (ZGAL in zgal, vulkan_splatting) get it on a working
  branch in that repo, through that repo's normal review and merge. The experiment runs against
  the merged commit and records its sha in `commits`.

## Correctness checks

Mechanical checks catch the errors a reviewer reading code misses. Where they apply, an
experiment includes them:
- **SymPy** for a derivation the experiment depends on (a closed-form gradient, a bound, an
  expected value the criteria compare against): the check derives or simplifies it and asserts
  it equals the formula the code uses.
- **Finite-difference gradient checks** for every custom loss and custom gradient op (a
  gradient reversal layer, a hand-written backward): compare the analytic gradient with central
  differences in float64, on small random inputs.
- **Property-based tests (Hypothesis)** for every invariance the experiment or a tested claim
  states (a set encoder is permutation-invariant, a state embedding ignores entity order).
- **Z3** for discrete properties (an action-space mapping between games is total and
  unambiguous, a tokenisation round-trips).
- **Lean** only for a genuinely new theoretical statement, never to re-prove a paper's
  theorem, and only when `## Method` in `plan.md` states the theorem and asks for it, so the
  user's approval of the plan covers it.

The checks live in `src/checks/` (so `inputs` fingerprints them), one module per check, with
their tools pinned with `==` in `requirements.txt` (`sympy`, `hypothesis`, `z3-solver`; a Lean
check pins its toolchain in `src/checks/lean-toolchain`). `./run.sh` and `./run.sh --smoke` run
them first and stop with a non-zero exit if one fails, so recorded results always passed them;
seed Hypothesis (`derandomize=True`) so a rerun checks the same examples. `plan.md`'s
`## Method` lists the planned checks, and `results.md` reports them under `## Checks`, one line
per check:
```
- <module> (<kind>): <what it checks>; <outcome, e.g. max relative error 3e-7, 500 examples>
```
or `- none (<why no check applies>)`. They aren't recorded in `results.json`: a failing
check means there is no result to record.

## Budgets and the GPU

- The run is held to the plan's budget. At **2×** the wall clock or the downloads, stop and
  report what was spent and what's left; raising the budget is the user's call.
- The user also approves, before it happens: GPU use when the plan says `gpu: no`, and
  downloads over 1 GB in total.
- The Mac has one GPU, shared with the other repos' work. A step that uses it waits for the
  supervisor to grant the GPU slot (`## GPU slot` in `~/code/.supervisor/ledger.md`) and
  releases it when done. Without a grant, don't run it.
