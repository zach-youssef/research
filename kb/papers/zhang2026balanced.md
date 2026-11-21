---
id: zhang2026balanced
title: "A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control"
authors: [Octi Zhang, Mateo Guaman Castro, Patrick Yin, Ignacio Dagnino, Abhishek Gupta, Rosario Scalise, Byron Boots]
year: 2026
venue: CoRL 2026 conference paper
url: https://arxiv.org/abs/2610.12465
arxiv: 2610.12465
doi:
pdf_sha256: 1017cc0863353b02ecf237d00437e83998ee0bee67d9e22c352df281653a0ee1
tags: [reinforcement-learning, curriculum-learning, task-sampling, exploration, massively-parallel-simulation, on-policy, robotics, sim-to-real]
status: read
read: 2026-11-20
---

# A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control

## TL;DR

SGS samples each PPO episode's task configuration by a Beta-shaped score of its rolling success rate,
peaking at moderate success; at 1M parallel envs it reaches 73% on 13-terrain quadruped locomotion
(PLR 54%) and 70% on Franka nut-and-bolt (uniform 6%), and gains as envs grow where baselines stall.

## Problem

- Sim-to-real RL for robots relies on per-task shaped rewards, hand-tuned curricula,
  demonstrations, or per-task experts distilled into a generalist; each task is a fresh
  engineering project (Sec. 1, p. 2).
- Diverse simulator resets (OmniReset) plus massively parallel RL remove much of that, but
  uniform sampling over the reset distribution spends a growing share of environments on
  configurations already mastered or not yet attemptable, so adding parallel environments gives
  little; the paper finds this limits scaling on precise or dynamic tasks (Abstract, p. 1;
  Sec. 1, p. 2; Sec. 3.3, p. 5).
- The authors say they do not claim a new curriculum principle: SGS is a simple variant of
  success-based sampling, closest to Sampling for Learnability (SFL, p(1−p) weighting) (Sec. 2,
  p. 3).

## Method

- **Setup.** Goal-conditioned MDP with initial-state/goal distribution ρ; a task configuration
  is τ = (s0, g, e): initial state, goal, environment configuration. RL trains on "compact
  Lagrangian states" with PPO; the manipulation teacher also sees a point cloud (Sec. 3.1,
  pp. 3-4; App. A.3.2, p. 15).
- **Configuration set.** Before training, a fixed set of N configurations is drawn from the
  task distribution (Sec. 3.2, p. 4). Locomotion: ρ is categorical over terrain type (13),
  level (10), start state (20) and goal (20), with 2 seeds per terrain, giving 104,000
  configurations (App. A.1, p. 14). Manipulation: a buffer of 32,768 collision-checked resets,
  balanced over three reset strategies from OmniReset: Near-Goal (object on its assembly path),
  Stable Grasp (object grasped above the target) and Reaching (object anywhere, gripper
  aperture random, so it may fall) (Sec. 3.1, p. 4; App. A.2, p. 14).
- **Success estimate.** Per configuration, a circular buffer of the last H = 100 Boolean
  outcomes; p_i is their mean (Sec. 3.2, p. 4; App. B.2, p. 16).
- **Score.** Beta kernel in mode-concentration form, Beta(1 + κt, 1 + κ(1−t)) with target
  success t and concentration κ: w_i = (p_i + ε)^(κt) (1 − p_i + ε)^(κ(1−t)), logit
  ℓ_i = log(w_i + ε) (Eq. 1, p. 4). The implementation floors w_i at ε before the log and
  uses T_eff = max(T, 1) (App. B.2, p. 16). ε keeps every configuration's probability nonzero
  so unsolved ones keep getting re-measured (p. 4).
- **Sampling.** At each episode end, the next configuration is drawn i.i.d. from
  softmax(ℓ_i / T) over all N (Eq. 2, p. 4). Fig. 2 (p. 4) plots kernels for (t, κ) =
  (0.5, 2), (0.5, 10), (0.66, 5): larger κ is greedier, smaller gives more coverage.
- **Rewards.** One reward per domain: a Boolean terminal success term plus small per-step
  regularizers (Eq. 3, p. 5). Locomotion (Eq. 4, p. 15): success +50, failure (drop or base
  contact) −50, L1 mechanical power −1e-3, L1 posture deviation −5e-2, squared touchdown
  velocity −0.25, undesired hard contacts −5e-2, clamped forward-alignment +0.3 (off for the
  first 500 steps). Manipulation (Eq. 5, pp. 15-16): success 100, termination penalty 1
  (joint velocity over 2× limits, or object out of workspace), squared action magnitude 1e-5,
  action rate 1e-4 and arm joint velocity 1e-4, each squared norm clamped at 1e4.
- **Scaling recipe.** PPO mini-batch count held constant, so batch size grows with the number
  of environments; all other PPO and optimizer settings fixed across scales (Sec. 4.3, p. 7).
- **Tasks.** Locomotion: one ANYmal-D policy (system-identified actuators) trained from scratch
  on all 13 terrains (slopes, pits, gaps, stairs, beams, stepping stones, plus five harder
  procedural ones: mazes, contours, floating islands joined by stepping stones or jump boxes)
  (Sec. 4.1, p. 6). Observations: base velocities, projected gravity, joint state, previous
  action, goal command, local height scan; 12-D joint-offset action (App. A.3.1, p. 15).
  Manipulation: a separate policy per task; Franka nut-and-bolt (joint-position control,
  25 Hz) and six UR5e NIST-board tasks (operational-space control, 10 Hz) (Sec. 4.2, p. 6;
  App. A.2, p. 14). Teacher observes joint positions, end-effector pose, previous action and a
  gripper-frame point cloud; 7-D action (App. A.3.2, p. 15).
- **Baselines.** Uniform sampling from ρ; a "linear" terrain curriculum (promote near the
  goal, demote on little progress); PLR with absolute GAE as the score plus a staleness bonus;
  SFL with its default hyperparameters on locomotion only (Sec. 4.3, p. 7; App. B.1, p. 16).
- **Transfer.** State/point-cloud teachers distilled into RGB students (side and wrist camera,
  one ResNet-18 each) with online DAgger, one step per iteration, 6.67% of environments held out
  for evaluation; deployed zero-shot on a UR5e (Sec. 4.5, p. 8; App. C.3, pp. 19-20).

## Claims

- C1: On 13-terrain ANYmal-D locomotion at 1M (2^20) parallel environments, SGS reaches 73% mean success versus 54% for PLR, means over three training seeds; in panel a, uniform sampling and the linear curriculum are near zero at 256K and 1M after peaking around 32K (Fig. 5, Sec. 4.3, p. 7).
- C2: On Franka nut-and-bolt assembly (Fig. 5 panel b) at 1M parallel environments, SGS reaches 70% success versus 6% for uniform sampling and 5% for PLR (Fig. 5, Sec. 4.3, p. 7).
- C3: SGS's success rises monotonically with the number of parallel environments (4K, 32K, 256K, 1M) on locomotion and Franka nut-and-bolt, while the other methods struggle to benefit from scale or collapse (Fig. 5, Sec. 4.3, p. 7).
- C4: On the easier UR5e rod-in-hole insertion (Fig. 5 panel c), SGS gets nonzero success at 32K environments where uniform sampling and PLR get zero, and all methods reach about 98% at 256K (Fig. 5, Sec. 4.3, p. 7).
- C5: Against SFL on locomotion (3 seeds, mean ± 95% CI), SGS final success is 0.456 ± 0.019 at 4K and 0.494 ± 0.113 at 32K; SFL final is 0.255 ± 0.051 and 0.051 ± 0.142, SFL best checkpoint 0.289 ± 0.015 and 0.421 ± 0.032; SFL degrades as training continues, most at the larger scale, while SGS improves monotonically (Table 1, p. 8, App. B.1, p. 16).
- C6: With 4,096 parallel environments on locomotion, SGS gets nonzero success on a large share of the 13 terrains while uniform sampling, the linear curriculum and PLR do not learn at all (Fig. 8, Sec. 4.4, pp. 8-9).
- C7: Varying one SGS hyperparameter at a time (t, κ, T, H, ε) on locomotion at 32K environments with two seeds each gives mean success from 0.39 to 0.53 against 0.49 nominal; the lowest is H = 25 (0.39) and the highest T = 4 (0.53); the authors read this as low sensitivity, though a small search helps (Table 3, App. B.2, pp. 16-17).
- C8: Replacing the Beta kernel with a Normal kernel gives similar locomotion success at 4K environments, 0.45 versus 0.46 over three seeds (Table 3, App. B.2, pp. 16-17).
- C9: More task configurations help up to a point: on locomotion at 32K environments, raising spawn/goal patches per tile p from 10 to 20 (52,000 to 104,000 configurations) raises success from 0.38 to 0.49, and p = 40 (208,000) gives 0.48 (Table 3, App. B.2, pp. 16-17).
- C10: In UR5e rod-in-hole insertion (3 seeds), training on all three reset strategies gives 99.5 / 96.0 / 100.0% success from Stable Grasp / Reaching / Near-Goal; dropping Near-Goal gives 0.0% on both others; dropping Reaching leaves the others essentially unchanged (99.4, 100.0); dropping Stable Grasp lowers Reaching to 63.8% and Near-Goal to 81.3%, with one of three seeds collapsing (Table 4, App. B.3, p. 17).
- C11: The gravity curriculum used in the main manipulation runs (start at zero gravity, raise it once success exceeds 80%) is unnecessary in this setting: a single-seed SGS run on Franka nut-and-bolt at 1M environments without it reaches 0.694 ± 0.040 Reaching-reset success versus 0.702 ± 0.039 with it (CIs over 512 evaluation episodes) (Table 5, App. B.4, pp. 17-18).
- C12: On six UR5e tasks evaluated from Reaching resets, SGS reaches 90.04% on nut-and-bolt at 256K environments versus 48.83% uniform and 59.77% PLR, and is comparable to uniform on the five simpler insertion tasks at 64K (84.08-96.09% SGS, 83.40-95.90% uniform) (Table 6, App. B.5, pp. 17-18).
- C13: RGB students distilled from SGS teachers keep simulated success close to the teacher (Reaching resets): nut-and-bolt 90.04 to 89.05%, rod-in-hole 95.70 to 93.36%, gear mesh 96.09 to 96.42% (Table 2, Sec. 4.5, p. 8).
- C14: Zero-shot on UR5e hardware, the RGB students succeed in 37.5% (18/48) of nut-and-bolt, 61% (30/49) of rod-in-hole and 94% (47/50) of gear-mesh trials, with first-try success 12.5%, 28.57% and 76.0% and throughput 0.23, 0.71 and 3.08 successes per minute (Table 2, Sec. 4.5, p. 8).
- C15: Under SGS the sampled locomotion configurations shift over training: near-uniform early, concentrated on jumps between level islands at mid-training, then on reaching the top of the tall center pillar late (Fig. 7, p. 8).
- C16: Training UR5e rod-in-hole from scratch at fixed gripper-pad friction μ = 0.5, 1, 1.5 and 2, only μ = 2 trains stably: 0.5 and 1 never learn and 1.5 peaks near 25% grasp-in-air success then declines, while OmniReset's μ = 100 allows physically implausible grasps (Fig. 10, Fig. 9, App. C.2, p. 19).
- C17: On OmniReset peg insertion, a point-cloud teacher with eight-orientation symmetric resets and rewards raises the distilled RGB student's simulated success from 55.45% (OmniReset's pose-based teacher) to 98% (Table 7, p. 21, App. C.3, p. 20).
- C18: In online DAgger distillation of nut-and-bolt, success in held-out environments lags success in the training pool early and catches up with continued training, one run (Fig. 11, App. C.3, p. 20).

## Implementation notes

- Sampler: N = 104,000 (locomotion), 32,768 (manipulation); H = 100; T = 2; locomotion t = 0.66, κ = 5, ε = 1e-8; manipulation t = 0.5, κ = 1, ε = 1e-4 (source: Sec. 3.2, p. 5).
- The manipulation values come from a grid over κ ∈ {1, 5, 10}, t ∈ {0.33, 0.5, 0.66} at T = 2, 32K environments; T is normally left alone because it is tied to κ (source: App. B.2, p. 17).
- Sampling is per episode reset, i.i.d. over the whole set, so many environments can start from the same configuration (source: Eq. 2, Sec. 3.2, p. 4); whether duplicates are avoided across environments is not stated (source: assumption that they are not).
- Outcome per episode is the task's Boolean success predicate; how timeouts and early terminations count (as failures) is not stated (source: assumption).
- Locomotion: 5K PPO iterations, up to 32,768 environments per GPU, about 36 h; Franka nut-and-bolt: 7.5K iterations, up to 16,384 per GPU, about 128 h; L40S and H200 GPUs; GPU counts per run not given (source: Sec. 4.3, p. 7). UR5e nut-and-bolt takes about 335 h, since 10 Hz with 20 physics steps per action costs 2.5× the physics steps per transition of Franka's 25 Hz with 8 (source: App. A.2, p. 14).
- Physics step 5 ms (200 Hz) for both arms (source: App. A.2, p. 14).
- PPO hyperparameters (clip, epochs, learning rate, GAE λ, mini-batch count), network architecture and the simulator are not stated; a GPU simulator in the Isaac Lab/Orbit family is likely given the citations and the OmniReset pipeline (source: assumption); project page lists code as coming soon (source: https://sgs-rl.github.io/, fetched 2026-11-20).
- PLR uses absolute GAE as the score plus a staleness bonus; its other settings are not given (source: Sec. 4.3, p. 7). SFL uses its paper's defaults (source: App. B.1, p. 16).
- Evaluation resets are freshly sampled from the training distribution, not reused training configurations (source: Sec. 4.3, p. 7).
- Distillation: about two days on four L40S GPUs with 480 environments each; student gets visual randomization and image augmentation from OmniReset; 1 mm per-step point-cloud noise helped rod-in-hole retries (source: Sec. 4.5, p. 8; App. C.3, p. 20; App. C.4, p. 21).
- Gripper pad friction μ = 2 with max combine mode for all runs (source: App. C.2, p. 19).

## Limitations / open questions

- SGS keeps estimates over a discrete set of cells; continuous or much larger task spaces would need a learned or hierarchical scoring representation (Sec. 5, p. 9).
- SGS reweights a given distribution but does not build it: it relies on OmniReset-style reset diversity, and which configurations to offer is open (Sec. 5, p. 9); Table 4 shows that, with SGS fixed, removing Near-Goal resets stops learning (C10).
- A sim-to-real gap remains (hardware 37.5-94% versus 89-96% in simulation) (Sec. 5, p. 9; Table 2, p. 8).
- Open question (reader): seeds are few: three per scaling point, two per sensitivity setting, one for the gravity ablation (Sec. 4.3, p. 7; App. B.2, p. 16; Table 5, p. 18); the 32K SFL and SGS CIs are wide (C5).
- On the simpler insertion tasks SGS is only comparable to uniform at enough scale (C4, C12): the gain is concentrated on long-horizon, unforgiving tasks, which the authors attribute, as a hypothesis, to contact dynamics and horizon (App. B.5, p. 18).
- Open question: the note's teacher observations and 7-D action follow App. A.3.2, but App. C.3 lists different teacher observations, and a 7-D action doesn't obviously match the Franka's joint-position control (p. 15, p. 20).
- Open question: Sec. 5 gives the discrete set as K = 32,768, the manipulation size; locomotion uses N = 104,000 (p. 5, p. 9).
- Open question: Table 3 bolds ε = 1e-4 (0.46) as nominal while Sec. 3.2 gives ε = 1e-8 for locomotion, which Table 3 scores 0.49 (p. 5, p. 17).
- Open question (survey lens): for multi-game RL with a transformer policy, the "configuration" could be (game, level, start state); SGS needs a Boolean success signal per configuration, which many games lack (score or progress thresholds would have to stand in), and its per-cell tables don't generalize across unseen games.
- Open question (survey lens): SGS and PLR were compared on PPO with state or point-cloud inputs and an unstated architecture; whether the scaling gains hold with pixel or embedding inputs and a large transformer policy is untested here.

## Related

- uses [[schulman2017proximal]]: all RL training is PPO, scaled to 2^20 environments with a fixed mini-batch count (Sec. 3.3, p. 5; Sec. 4.3, p. 7).
- related [[serrano2024crossdomain]]: cross-domain transfer survey for the same lens; SGS trains one policy across 13 terrain types but transfers between tasks only through shared training, not across domains.
- related [[vaswani2017attention]]: the survey's transformer policy would be trained on data balanced by a sampler like SGS; this paper does not use transformers for the RL policy.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
