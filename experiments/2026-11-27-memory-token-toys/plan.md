---
id: 2026-11-27-memory-token-toys
title: Memory-token toys for the shmup stack (X7) - set-prefix compression, variable segments, several facts
claims: [cherepanov2026recurrent#C1, cherepanov2026recurrent#C8, cherepanov2026recurrent#C12, pleines2025memorygym#C4, pleines2025memorygym#C10]
assumptions: [cherepanov2026recurrent#A1, cherepanov2026recurrent#A2, cherepanov2026recurrent#A3, cherepanov2026recurrent#A4, cherepanov2026recurrent#A5, cherepanov2026recurrent#A7, pleines2025memorygym#A1, pleines2025memorygym#A2, pleines2025memorygym#A3, pleines2025memorygym#A6, pleines2025memorygym#A7, yan2025elastictok#A2, yan2025elastictok#A6]
status: proposed
gpu: no
budget_wall_min: 360
budget_downloads_mb: 250
ci_smoke: true
approved:
---

# Memory-token toys for the shmup stack (X7) - set-prefix compression, variable segments, several facts

Three CPU toys, no emulator, run by one `run.sh`. They de-risk the seams of the candidate stack
(object encoder -> ElasticTok-style compression -> RATE memory tokens -> QueST skill tokens)
before X8 and X9 spend emulator time:
- **X7a (set prefix):** how prefix masking applies to an unordered object set. Design readout,
  no claim (no KB claim covers sets; topic E2).
- **X7b (variable segments):** RATE on T-Maze, the paper's segmentation result reproduced at
  small scale, then with per-step token counts and segment lengths that vary.
  Claims cherepanov2026recurrent#C1, #C8, #C12.
- **X7c (several facts):** a multi-cue T-Maze, memory size m against fact count k, with an
  action-loss-only arm, an auxiliary cue-reconstruction arm and a GRU baseline. Claims
  pleines2025memorygym#C4, #C10, plus a design readout (the m needed per k).

## Hypothesis

- X7b: with recurrent memory tokens and segments of K = 30 steps (N = 3 over a 90-step training
  sequence), RATE keeps near-perfect success in distribution and well above chance at 10x the
  training length, while a memoryless transformer with a 90-step context falls to chance
  (cherepanov2026recurrent#C1). The same 90 steps as one segment (N = 1, K = 90) collapses on
  long corridors, because the memory is never trained to carry across a boundary (#C12).
  Replacing memory tokens with noise at inference drops success to about chance while the
  agent still reaches the junction (#C8). For the stack (design readout): we expect the memory
  read/write to tolerate per-step token counts and segment lengths that vary, since nothing in
  Eq. 1-3 depends on the segment's token count.
- X7c: on a task where k facts are shown once and queried later, an auxiliary head that
  reconstructs the cues from the memory tokens (labels as targets only) raises recall at
  k >= 4 over the action loss alone, as pleines2025memorygym#C10's ground-truth head lifted
  TrXL; and a GRU holds more facts than the action-loss transformer memory
  (pleines2025memorygym#C4's GRU-over-transformer result), here under behaviour cloning.
- X7a (readout only): a learned set of ordered summary tokens (prefix = first L queries)
  beats a distance-to-player canonical order with prefix truncation at small L, and one
  elastic model trained over all L costs at most a couple of points against models trained
  at one fixed L.

## Success criteria

Metric definitions (all means over 3 seeds; T is the corridor length in agent steps; success is
the share of evaluation episodes that end with the correct turn; `_in` averages T in
{30, 60, 90}): `rate_fixed_*` is RATE with N = 3, K = 30; `rate_n1k90_*` RATE with N = 1,
K = 90; `dt_*` the memoryless transformer with a 90-step sliding context;
`rate_fixed_noise_success_t90` and `rate_fixed_noise_junction_t90` are the success and
junction-reached rates of `rate_fixed` when its memory tokens are replaced by N(0, 1) noise at
every segment boundary; `aux_gain_k4plus` is the mean over k in {4, 8} and m in {1, 5, 20} of
(aux-arm minus action-only-arm) `facts_recalled`, `aux_gain_k4plus_min_seed` the smallest of
its three per-seed values; `gru_minus_rate_k8` is the GRU's `facts_recalled` at k = 8 minus the
best action-only RATE arm's (best m) at k = 8.

- cherepanov2026recurrent#C1: supports if rate_fixed_success_in >= 0.95 and rate_fixed_success_t900 >= 0.7 and dt_success_t900 <= 0.6; contradicts if rate_fixed_success_in < 0.8 or rate_fixed_success_t900 < dt_success_t900 + 0.1; otherwise inconclusive.
- cherepanov2026recurrent#C8: supports if rate_fixed_success_t90 >= 0.9 and rate_fixed_noise_success_t90 <= 0.65 and rate_fixed_noise_junction_t90 >= 0.9; contradicts if rate_fixed_success_t90 >= 0.9 and rate_fixed_noise_success_t90 >= rate_fixed_success_t90 - 0.1; otherwise inconclusive.
- cherepanov2026recurrent#C12: supports if rate_fixed_success_t900 - rate_n1k90_success_t900 >= 0.3 and rate_n1k90_success_t900 <= 0.55; contradicts if rate_fixed_success_t900 - rate_n1k90_success_t900 <= 0.1; otherwise inconclusive.
- pleines2025memorygym#C4: supports if gru_minus_rate_k8 >= 0.1; contradicts if gru_minus_rate_k8 <= -0.1; otherwise inconclusive.
- pleines2025memorygym#C10: supports if aux_gain_k4plus >= 0.05 and aux_gain_k4plus_min_seed > 0; contradicts if aux_gain_k4plus <= 0; otherwise inconclusive.
- tolerance rate_fixed_success_in: 0.05
- tolerance rate_fixed_success_t900: 0.08
- tolerance dt_success_t900: 0.08
- tolerance rate_fixed_success_t90: 0.05
- tolerance rate_fixed_noise_success_t90: 0.08
- tolerance rate_fixed_noise_junction_t90: 0.05
- tolerance rate_n1k90_success_t900: 0.08
- tolerance gru_minus_rate_k8: 0.08
- tolerance aux_gain_k4plus: 0.04
- tolerance aux_gain_k4plus_min_seed: 0.05

The tolerances are wide enough for a rerun on another Mac (thread scheduling and BLAS can change
float sums even with fixed seeds and thread counts) and narrow against the criteria's margins.
The design readouts below are reported and their decisions recorded in `results.md`; they name
no claim, so they are not compared when reproducing.

## Method

### Shared setup

- PyTorch on CPU only (no MPS), `torch.set_num_threads` fixed per process and recorded,
  `torch.use_deterministic_algorithms(True)`; seeds 0, 1, 2 for every arm (numpy, torch and the
  data generators are seeded from the arm seed). Dependencies pinned with `==` at
  implementation: torch (CPU wheel), numpy, matplotlib, hypothesis.
- `run.sh` creates `runs/2026-11-27-memory-token-toys/.venv` with the system `python3` (3.9),
  runs the checks, then the three parts as three parallel processes (3 threads each), then an
  aggregation step that writes `results.json` and the figures. `--part a|b|c` runs one part
  (for development; the recorded run is all three).
- **Audio stays open (topic 0.2):** every model consumes a **token set per agent step**: each
  token is a feature projection plus a learned token-type embedding (type 0 = observation; X7b's
  extra tokens are type 1, the slot an audio token would take), plus a within-segment step
  position. Segment boundaries are counted in agent steps, never in tokens. Losses are logged
  **per stream** (`loss_action`, `loss_aux`), never as one total only.
- **Observations only as inputs:** the models' forward takes observation tensors (and their own
  previous actions); labels (dodge action, oracle action, cue values) enter only the losses.
  A check enforces it (below).
- Training is behaviour cloning on freshly generated oracle data (cherepanov2026recurrent#A1:
  offline, supervised action loss, successful trajectories; the return-to-go input is kept as
  a constant 1 embedded into each step token, since every oracle episode succeeds). AdamW,
  lr 3e-4, batch 64, gradient clipping 1.0; iteration counts below. Gradients flow back through
  the memory across all segments of a training sequence (the note says the paper doesn't state
  this depth; full BPTT over 3 or 5 segments is the choice here); no Transformer-XL cached
  states (mem_len = 0, the paper's recommendation for sparse tasks, cherepanov2026recurrent
  implementation notes).

### X7a: prefix masking over an unordered object set (design readout, no claim)

- **Data:** synthetic bullet fields in the unit square: a player position and n ~ U{8..48}
  objects, each (x, y, vx, vy, role), role in {bullet, enemy}, speeds up to 0.03 per step.
  The label is a scripted dodge: for each of 5 actions (stay, up, down, left, right; player
  speed 0.02 per step) roll the player and all objects forward 8 steps linearly and score
  danger = sum over objects of exp(-d_min^2 / 0.05^2) plus a wall term; the label is the
  action with the lowest danger. Training data are generated on the fly (fresh per batch,
  seeded); the test set is 20,000 fields from a separate seed, plus 5,000 fields with n = 96
  (twice the training maximum, informational).
- **Encoder:** per-object MLP to d = 64, two set-transformer (self-attention, no positional
  encoding) layers, then one of three compressions to L tokens, then a 2-layer transformer
  over [player token, L tokens] and an action head:
  - (i) `summary`: 32 learned queries cross-attend to the objects; prefix = first L queries
    (ordered summary tokens, the topic's recommended reading of yan2025elastictok#A2 for sets);
  - (ii) `pertrunc`: every object kept, each embedding projected to 32 dims and truncated to
    its first L dims (a control: it keeps every object, so it never drops one, and its
    information budget is n x L, not L x d);
  - (iii) `canon`: objects sorted by distance to the player, then role, and the first L kept.
- **Elastic training** follows yan2025elastictok#A2: per sample, L ~ U{4..32} and the prefix
  mask keeps the first L (dims for `pertrunc`); the mask is given to the encoder as learned
  0/1 embeddings (#A6); M_min = 4 (yan2025elastictok#C11's lower bound, here the smallest L
  evaluated). Fixed-L baselines: `summary` trained at a single L, one model per L in
  {4, 8, 16, 32}; a `full` reference with no compression (all objects to the policy).
- **Runs:** 8 models per seed (3 elastic, 4 fixed, 1 full) x 3 seeds, 3,000 iterations each.
- **Metrics:** `x7a_acc_<arm>_L<L>` (test accuracy against the scripted label), L in
  {4, 8, 16, 32}, arm in {summary, pertrunc, canon, summary_fixed}, plus `x7a_acc_full` and
  the n = 96 variants (`x7a_acc96_*`). Seed-bootstrap 95% CIs.
- **Readout (fixed now):** (1) if `summary` beats `canon` by >= 0.02 at both L = 4 and L = 8,
  X9 keeps learned summary tokens as its main arm; if `canon` is within 0.01 of or above
  `summary` at every L, X9 makes the canonical order the default and learned summary tokens
  an ablation; otherwise X9 runs both. (2) If max over L of (`summary_fixed` - `summary`)
  <= 0.02, one elastic model is enough for the stack; above 0.05, X9 adds fixed-L baselines.
- **Shmup relevance:** this is the compression step between the object encoder and the memory
  tokens. Bullet counts swing from a handful to hundreds within a stage, and objects have no
  order, so ElasticTok's prefix rule must be given a meaning on a set before X9 measures
  `tokens_at_target_nll` on real play; the readout picks X9's arms.

### X7b: RATE on T-Maze with fixed and variable segments (#C1, #C8, #C12)

- **Task:** passive T-Maze as RATE uses it (cherepanov2026recurrent#A3: a memory puzzle with a
  cue at the first step). Corridor length T; observation (cue_up, cue_down, junction_flag,
  1), the cue only at t = 0; 4 discrete actions (right moves along the corridor; up/down at the
  junction end the episode, success if it matches the cue; #A7 discrete actions); time limit
  T + 10 steps. Training corridors T ~ U{9..90} (within #A2's T_train range), padded to 90
  steps with a loss mask, sequences starting at t = 0 so the cue is always in them (#A2).
- **Model:** RATE (Eq. 1-3 of the note's Method): 3 layers, d = 128, 4 heads, 5 memory tokens
  (the paper's default), read copy before and write copy after each segment, MRV-CA-2 memory
  update, M_0 ~ N(0, 1); one observation token per step (observation, return-to-go and previous
  action embeddings summed, plus the type and step-position embeddings). 2,000 iterations.
- **Arms** (3 seeds each):
  - `rate_fixed`: N = 3, K = 30 (the paper's main split, #A2);
  - `rate_n1k90`: N = 1, K = 90, same K_eff = 90 (#C12's comparison); at evaluation it runs
    K = 90 segments with memory carried, as RATE infers on long sequences;
  - `dt`: the same transformer without memory tokens, a sliding 90-step context (the
    memoryless control of #C1);
  - `rate_var_tok` (readout): N = 3, K = 30 steps, but each step adds n_t - 1 extra type-1
    tokens, n_t ~ U{1, 2, 4} per step, whose features are i.i.d. N(0, 1) (content-free, like
    detail or audio tokens that carry nothing about the cue), so a segment holds 30 to 120
    step tokens; the action is read from the step's observation token;
  - `rate_var_len` (readout): one token per step, segment lengths drawn per segment from
    {15, 30, 60} steps (still on step boundaries), in training and evaluation.
  This breaks #A2's fixed N = 3 split on purpose in the two readout arms; the claims are
  judged on `rate_fixed`, `rate_n1k90` and `dt` only.
- **Evaluation:** closed loop, 200 episodes per T per seed (cue balanced 100/100), T in
  {30, 60, 90, 180, 450, 900}; metrics `<arm>_success_t<T>`, `<arm>_junction_t<T>` (reached the
  junction) and `<arm>_success_in`. Memory corruption (#C8): `rate_fixed` at T = 90 and 900
  with the memory tokens replaced by N(0, 1) noise at every segment boundary after the first
  (`rate_fixed_noise_success_t<T>`, `rate_fixed_noise_junction_t<T>`). With mem_len = 0 the
  memory tokens are the only path from the cue to a decision in a later segment. Seed-bootstrap
  95% CIs; n = 3 seeds x 200 episodes.
- **Scale versus the claims:** #C1 is stated for corridors trained up to 900; here training
  stops at 90 and evaluation goes to 10x (900), the same ratio at a tenth of the length, to fit
  a CPU. #C12's Fig. 18 setup (K_eff = 90) is met at its own scale. The model meets #A4's
  layers and embedding (3 layers, d = 128) but has about 0.6M parameters, under the paper's
  1.6-6.8M, because the task's inputs are 4 numbers. #A5 (action loss only) holds in all X7b
  arms.
- **Readout (fixed now, no claim):** if `rate_var_tok_success_t900` and
  `rate_var_len_success_t900` are each >= 0.85 while `rate_fixed_success_t900` >= 0.9, the
  stack feeds variable token counts straight into the memory; if either is < 0.6, the stack
  needs a fixed per-segment token budget (padding or a cap) between compression and memory;
  otherwise the result is unclear and X8 tests both.
- **Shmup relevance:** ElasticTok-style compression and a changing object count give each agent
  step a different number of tokens, and audio would add more; RATE was only shown with fixed
  segments. This toy decides whether the memory tokens can take that stream as it comes, and
  #C1 / #C12 / #C8 confirm at small scale that the memory mechanism the stack relies on works
  as the paper says before it is built on.

### X7c: several facts over a long episode (#C4, #C10)

- **Task:** a multi-cue T-Maze (mine, after zhang2024beacon#C5's multi-needle test): episodes
  of 150 steps; k ~ U{1, 2, 4, 8} cues at steps 0..k-1, cue j showing (cue flag, index j
  one-hot of 8, value one-hot of 4); then distractor steps showing a random value with a
  distractor flag; k query steps at distinct random times in [30, 149] (so every query is at
  least one segment boundary after the cues), each showing (query flag, index j) for each cue
  once, in random order. Actions: no-op plus 4 answers; the oracle answers each query with its
  cue's value and no-ops otherwise. `facts_recalled` = share of query steps answered
  correctly (chance 0.25). Evaluation: 500 episodes per k per seed, also at 450 steps (queries
  in [30, 449], informational).
- **Arms** (3 seeds each, all trained on mixed k):
  - `act`: RATE (3 layers, d = 64, 4 heads, N = 5, K = 30, MRV) with memory sizes m in
    {1, 5, 20}, action loss only (cherepanov2026recurrent#A5);
  - `aux`: the same plus an auxiliary head: an MLP on the flattened write memory after each
    segment predicts the value of every cue shown so far (cross-entropy, masked to cues
    already shown), weight 0.1 (Memory Gym's reconstruction coefficient, pleines2025memorygym
    implementation notes). Cue values are targets only. This breaks #A5 on purpose: that is
    the comparison;
  - `gru`: one GRU layer, hidden 512, trained on the full 150-step sequence (pleines2025memorygym
    #A2's GRU setting), same step tokens and action loss.
  2,000 iterations each: 21 models in all.
- **Metrics:** `facts_recalled_<arm>_m<m>_k<k>` (and `facts_recalled_gru_k<k>`),
  `aux_gain_k4plus`, `aux_gain_k4plus_min_seed`, `gru_minus_rate_k8`, per-stream final losses.
  Seed-bootstrap 95% CIs; n = 3 seeds x 500 episodes per k.
- **How the Memory Gym assumptions are met or broken:** #A1 (PPO) is broken on purpose: this is
  behaviour cloning, since PPO to 800M steps (#A6, also broken) is far outside a CPU budget;
  the claims are tested in transfer form, as "an extra supervised signal on what memory should
  hold helps a transformer memory" (#C10) and "a GRU holds accumulated once-shown facts better
  than a transformer memory" (#C4). #A2 is met for the GRU (one layer, 512, full-length
  sequences); the transformer is RATE with memory tokens, not TrXL with a cached window, so
  #A2's TrXL setting is not met. #A3 is met in part (3 seeds, not 5; many evaluation
  episodes). #A7 (Memory Gym's 84x84 RGB environments) is broken: the toy keeps only the
  structure of Endless Mortar Mayhem (facts shown once, queried later). A supports or
  contradicts verdict therefore says the claim's mechanism carries over to this setting, and
  `results.md` says so next to it.
- **Readout (fixed now, no claim):** `m_needed_<arm>_k<k>` = the smallest m in {1, 5, 20} with
  `facts_recalled` >= 0.9 (or "none"), for act and aux; reported as a table, and used to size
  X8's memory per fact count.
- **Shmup relevance:** a shmup run holds several long-lived facts at once (weapon tier, bombs,
  rank, a shop or route choice) that are shown once and matter much later; the paper's tasks
  have one cue. This sizes the memory tokens per fact, says whether X8 needs an auxiliary head
  on offline RAM labels (allowed as targets, never as inputs), and whether a GRU should stay as
  the baseline the memory tokens must beat.

### Correctness checks (`src/checks/`, run first in both modes)

- `perm_invariance` (Hypothesis, derandomized): the X7a encoder's output is invariant to the
  order of the objects for `summary`, `pertrunc` and `canon` (ties in the canonical sort broken
  by the role and then the coordinates, so the sort is a function of the set), and the X7b
  `rate_var_tok` step encoding is invariant to the order of a step's extra tokens.
- `prefix_mask` (Hypothesis): with a prefix L, changing the masked tokens (queries, dims or
  objects past L) leaves the policy output unchanged.
- `causality` (Hypothesis): in RATE and the DT control, the action at step t does not change
  when observations after t are changed; the read memory copy does not attend to the segment.
- `labels_not_inputs` (Hypothesis): for fixed observations, changing the labels (dodge action,
  oracle actions, cue values) leaves every model output unchanged, so labels enter only the
  losses.
- `oracles` (exhaustive over small cases): the X7a labeller returns the argmin of its own
  danger score over the 5 actions; the T-Maze and multi-cue oracles succeed on every episode;
  and in both T-Maze tasks the observation at any step after the cue phase is independent of
  the cue values (checked by enumeration), so only memory can solve them.
- No finite-difference gradient check: every loss is a standard cross-entropy through autograd,
  with no custom backward. No SymPy, Z3 or Lean check applies.

### Smoke mode

`./run.sh --smoke` runs the checks, then each part with one seed, the smallest arm grid
(X7a `summary` elastic only; X7b `rate_fixed` only; X7c `act` with m = 5 only), 20 iterations
and 10 evaluation episodes at the shortest T, writing only under
`runs/2026-11-27-memory-token-toys/smoke/`, in under a minute on CPU once the venv exists.
`ci_smoke: true` because it needs no GPU, ROM or data download, only the pinned wheels;
the research repo has no CI yet, so nothing runs it today.

### Domain code and vendoring

None. All code is written for this experiment in `src/`; nothing is vendored.

## Budget

- **GPU:** none (`gpu: no`). The models are 0.15-1M parameters on tiny inputs; MPS would gain
  little and would need the GPU slot and give less deterministic results.
- **Wall clock, estimated per full run (CPU, 3 parallel processes of 3 threads):**
  - X7a: 24 models x 3,000 iterations of a small set model at batch 64 (about 20 ms each)
    is about 25 min, plus evaluation about 2 min;
  - X7b: 15 models x 2,000 iterations; a RATE iteration runs 3 segments of 40 tokens
    (`rate_var_tok` up to 130) forward and backward through memory at d = 128, about
    0.15-0.25 s, so 5-8 min per model (about 2.5x for `rate_var_tok`), about 120 min; closed-loop
    evaluation (1,710 steps x 200 episodes per model, plus the noise runs) about 2 min per
    model, 30 min; about 150 min in all, the longest part;
  - X7c: 18 RATE models x 2,000 iterations of 5 segments at d = 64, about 0.1-0.15 s each, so
    3-5 min each, about 75 min, plus 3 GRUs (about 3 min each) and evaluation (about 15 min),
    about 100 min;
  - the parts run in parallel, so the full run is bounded by X7b: about 150 min, estimated
    at 120-180 min because CPU speed for these sizes is a guess until the smoke run measures
    it.
- **Experimenter's share: 200 min** = venv creation and smoke runs while implementing (about
  20 min), the full run (about 160 min, the estimate with a margin), and per-part reruns of
  short pieces during development (about 20 min). The run is stopped at 2x this share. Before
  the full run, the experimenter times 50 training iterations per part and stops (reports
  `blocked`) if the projected full run exceeds 2x 160 min, rather than shrinking the grid.
- **Reproduction's share: 160 min** = venv creation and the smoke run (about 5 min) and the
  full run (about 155 min). The total, 360 min, is under the 480-minute ceiling.
- **Downloads:** the pinned wheels only (torch CPU for macOS arm64 about 70 MB, numpy about 5,
  matplotlib with its dependencies about 25, hypothesis about 1): about 100 MB per venv. The
  experimenter's run is 100 MB; the reproduction's clone starts with an empty `runs/`, so it is
  counted again (100 MB) even though pip's cache outside the checkout will likely serve it.
  250 MB in all, with a margin, under the 300 MB ceiling and far under 1 GB. No datasets,
  models, ROMs or recordings.
