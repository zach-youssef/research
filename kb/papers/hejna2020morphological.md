---
id: hejna2020morphological
title: Hierarchically Decoupled Imitation for Morphological Transfer
authors: [Donald J. Hejna, Pieter Abbeel, Lerrel Pinto]
year: 2020
venue: ICML 2020
url: https://arxiv.org/abs/2003.01709
arxiv: 2003.01709
doi:
pdf_sha256: 8bcee201027c252ab0869dc6cb0e4fc02b21a0bdd91ad6508e161be8dd173537
tags: [reinforcement-learning, transfer-learning, hierarchical-rl, morphology-transfer, imitation-learning, cross-embodiment]
status: read
read: 2026-11-21
---

# Hierarchically Decoupled Imitation for Morphological Transfer

## TL;DR

Transfers a two-level policy from a simple agent to a complex one: the goal-emitting high level is
reused, the per-morphology low level is retrained with a discriminator reward that makes its goal
reaching look like the simple agent's, and the high level is fine-tuned with a KL anchor to the source.

## Problem

Training long-horizon behaviour on a complex robot from scratch costs millions of samples, and
policies can't be copied across morphologies because state and action spaces differ (Sec. 1,
pp. 1-2). Morphology-conditioned embeddings need hundreds of training morphologies (p. 2, p. 8),
whereas only two or three are realistic in practice (Sec. 5.1, p. 8). High-level policies are
tied to the low level they were trained with, so a sub-goal the new agent's low level can't reach
or reaches differently breaks zero-shot transfer (Sec. 1, p. 2). The authors say they are the
first work to focus concretely on morphological transfer (p. 2, Sec. 5.2, p. 9).

## Method

- **Decoupled hierarchy** (Sec. 2.2, pp. 2-3, Fig. 2, p. 3): the state splits into a global,
  morphology-independent s^hi (e.g. torso or end-effector x, y, block or goal positions) and a
  proprioceptive s^lo. The high level π^hi(s^hi) emits a sub-goal a^hi in a shared goal space G
  (the x, y change of the torso or end effector); the low level π^lo(s^lo, a^hi) outputs torques.
  So π^hi's input and output are morphology-independent and can be copied from A to B (Sec. 3.1,
  p. 3). Low levels are pre-trained on uniformly sampled goals from G, independently of any task,
  and reused across tasks of the same type (Sec. 3.1, p. 3, Sec. 4.2, p. 5).
- **Low-level alignment** (Sec. 3.2, pp. 3-4): minimise I(morphology; behaviour), where the
  behaviour τ_t = s^hi_{t-k:t} is a window of global states (Eq. 1, p. 4). With H(M) fixed this is
  maximising H(M|τ), lower-bounded variationally with a binary discriminator q_φ(m|τ) (Eq. 2,
  p. 4; derivation App. A, p. 14, which assumes a uniform prior over morphologies and conditions
  on the goal). B's low-level reward becomes the goal reward minus λ0^λ1 log q_φ(M=B|τ^B), the
  temperature annealed over training (Eq. 3, p. 4); policy and discriminator alternate as in
  GAIL (Alg. 1, p. 4).
- **KL-regularized high-level fine-tuning** (Sec. 3.3, p. 4): B's copied high level is trained
  with the RL gradient plus α∇KL(π^hi_B || π^hi_A) on B's high-level states (Eq. 4, p. 4), to stop
  catastrophic forgetting; the KL weight is annealed to zero (Sec. 4.2, p. 5).
- Full pipeline: Alg. 2 (App. B, p. 15); the experiments ablate the two parts separately (p. 14).
- **Setup** (Sec. 4.1-4.2, p. 5, App. C, pp. 14-16): MuJoCo, SAC from Stable Baselines.
  Navigation agents PointMass (2-D action), Ant, 3-Leg Ant, 2-Leg Ant, Quadruped (12-D action);
  tasks Waypoint and a sparse-reward U-shaped Maze (evaluated as Maze Sample and Maze End).
  Manipulation agents PointMass, 2/3/4-Link arms; tasks Block Push 1 and 2, Peg Insert. Five
  seeds, ten for the high-level transfers and fine-tunings of the selected navigation tasks.

## Assumptions

- A1: The state splits into a global, morphology-independent part s^hi (torso or end-effector position, object and goal positions) and a proprioceptive, morphology-dependent part s^lo (Sec. 2.2, pp. 2-3).
- A2: Source and target share a sub-goal space G that both can be commanded in: the desired x, y change of the torso (within 4 m) for navigation or of the end effector (within 0.07 m) for manipulation (Sec. 2.2, p. 2, Sec. 4.1, p. 5, App. C.1, p. 14, App. C.2, p. 16).
- A3: The policy is a two-level hierarchy whose high level picks a sub-goal every fixed number of steps: 32 (64 for the Quadruped) in navigation, 35 in manipulation (Sec. 2.2, p. 2, App. C.1, p. 14, App. C.2, p. 16).
- A4: Each agent's low level is pre-trained on goals sampled uniformly from G with a generic goal-reaching reward, independently of the task and the high level, and reused across tasks of the same type (Sec. 3.1, p. 3, Sec. 4.2, p. 5).
- A5: A trained hierarchical policy for the source agent A on the task is available, and A is the simpler agent (Sec. 3, p. 3).
- A6: Transfer is between two morphologies, one source and one target (Sec. 3, p. 3, Sec. 5.1, p. 8).
- A7: Source and target belong to the same family for a task type: navigation agents (PointMass, Ant, 3- and 2-Leg Ant, Quadruped) or planar manipulators (PointMass, 2-, 3- and 4-Link arms), all torque-controlled in MuJoCo (Sec. 4.1, p. 5, App. C, pp. 14-16).
- A8: The alignment discriminator sees goal-relative global-state transitions (g − s^hi_t, g − s^hi_{t+1}), defined for both agents; its derivation assumes a uniform prior over the two morphologies (Sec. 3.2, p. 4, Sec. 4.2, p. 5, App. A, p. 14).
- A9: The discriminator's data for A is collected offline by running A's low level on uniformly sampled goals (Sec. 4.2, p. 5, App. D, p. 16).
- A10: The high-level policies are diagonal Gaussians, so the KL between B's and A's high levels is computed in closed form on B's states (Sec. 4.2, p. 5, Eq. 4, Sec. 3.3, p. 4).
- A11: Imitating the source's high level helps only when the optimal high-level solution is similar for both agents (Sec. 4.7, p. 8).
- A12: All policies are trained with SAC from Stable Baselines, over five seeds (ten for the high-level transfers and fine-tunings of the selected navigation tasks) (Sec. 4.2, p. 5).
- A13: All experiments are in simulation (Sec. 6, p. 9).
- A14: Task rewards are dense L2 terms plus a sparse completion bonus for Waypoint, Block Push and Peg Insert, and sparse only for the Maze (Sec. 4.1, p. 5, App. C.1, pp. 14-15, App. C.2, p. 16).
- A15: The navigation agents observe their low-dimensional state: joint positions and velocities, the vector to the sub-goal, and contact forces (all but the PointMass); the Ant in the step maze also gets 16 rangefinder sensors. App. C.2 doesn't state the manipulation agents' low-level observations (App. C.1, pp. 14-15, App. C.2, p. 16).
- A16: The target's low level can reach, within one high-level step, the sub-goals the source's high level issues; a high level doesn't choose sub-goals its low level can't reach, and different morphologies afford different low-level behaviour (Sec. 1, p. 2, Sec. 3.2, p. 3).

## Claims

- C1: On Waypoint, a PointMass high level zero-shot on the Ant's low level scores 483 ± 39, the same within confidence bounds as the Ant's own high level (476 ± 19) [A1, A2, A3, A4, A5, A6, A7, A11, A12, A13, A14, A15, A16] (Table 1, Sec. 4.3, p. 6).
- C2: On sparse Maze End, a PointMass high level zero-shot on the Ant succeeds 0.30 ± 0.08 of the time, better than the Ant's own high level (0.16 ± 0.14), which the authors attribute to the PointMass's faster exploration (the Ant topples into walls); this is far below the PointMass's own 0.80 [A1, A2, A3, A4, A5, A6, A7, A11, A12, A13, A14, A15, A16] (Table 1, Sec. 4.3, p. 6).
- C3: High-level policies overfit to the morphology they were trained with: the PointMass's Waypoint return drops from 1e3 with its own high level to 577-716 with the others', and on Block Push 1 a PointMass high level on the 4-Link arm loses about 74% of the 4-Link's own performance (0.24 vs 0.89) [A1, A2, A3, A4, A5, A6, A7, A12, A13, A14, A15, A16] (Table 1, Sec. 4.3, p. 6).
- C4: Discriminative low-level imitation improves zero-shot high-level transfer in nearly all tested pairs, most in manipulation: e.g. PM→3Link Block Push 1 0.17 → 0.34, PM→Quad Waypoint 169 → 338, PM→Ant Maze End 0.30 → 0.55, 3Link→4Link Insert 0.80 → 0.93; PM→Quad Maze End stays at 0.40 and 2Link→3Link Block Push 2 drops from 0.61 to 0.42 [A1, A2, A3, A4, A5, A6, A7, A8, A9, A11, A12, A13, A14, A15, A16] (Table 2, Sec. 4.4, pp. 6-7).
- C5: One aligned low level, trained once, improves transfer for both Block Push tasks and across several source high levels, often doubling Block Push success, except 2Link→3Link on Block Push 2 (0.61 → 0.42); the authors argue this makes it useful when high-level training is expensive or many tasks are targeted, which is not measured [A1, A2, A3, A4, A5, A6, A7, A8, A9, A11, A12, A13, A14, A15, A16] (Table 2, Sec. 4.4, pp. 6-7).
- C6: The gains in C4 come from imitation, not from a better low level: final low-level rewards with the discriminator are near those without it, slightly lower for the Ant and learned slightly faster for the 3- and 4-Link arms [A1, A2, A3, A4, A5, A6, A7, A8, A9, A12, A13, A14, A15] (Fig. 4, Sec. 4.4, p. 6).
- C7: With discriminative imitation the Quadruped's zero-shot Waypoint return beat training its high level from scratch, for the limited number of low-level steps used in training [A1, A2, A3, A4, A5, A6, A7, A8, A9, A11, A12, A13, A14, A15, A16] (Sec. 4.4, p. 6).
- C8: In most cases, fine-tuning the transferred high level beats training the high level from scratch and reaches substantially higher performance than zero-shot, but plain fine-tuning can be unstable, e.g. 2-Link from PointMass on Block Push, with large fluctuations across seeds the authors attribute to catastrophic forgetting [A1, A2, A3, A4, A5, A6, A7, A11, A12, A13, A14, A15, A16] (Fig. 6, Fig. 7, Sec. 4.5, pp. 7-8).
- C9: KL-regularized fine-tuning substantially improves on plain fine-tuning: at least a 2× speedup on Block Push, smaller gains on the Maze tasks [A1, A2, A3, A4, A5, A6, A7, A10, A11, A12, A13, A14, A15, A16] (Fig. 6, Fig. 7, Sec. 4.6, pp. 7-8).
- C10: Learning without hierarchy ("Full") failed on the complex navigation agents within the compared sample budget, especially on the sparse maze, and eventually succeeded on all manipulation tasks [A7, A12, A13, A14, A15] (Sec. 4.6, p. 7, Fig. 7, p. 8).
- C11: When the complex agent has extra abilities (a step maze the Ant can climb and the PointMass can't), for the limited number of Ant samples used, training the Ant's high level from scratch scores highest: success 0.59 ± 0.05 from scratch, 0.53 ± 0.09 fine-tuned and 0.52 ± 0.13 KL fine-tuned (intervals overlapping), while zero-shot (0.32 ± 0.16) is clearly worse; the zero-shot Ant avoids the barriers like the PointMass would, so high-level imitation helps only when the optimal solution is similar for both agents [A1, A2, A3, A4, A5, A6, A7, A10, A12, A13, A14, A15] (Table 3, Fig. 5, Sec. 4.7, pp. 7-8).
- C12: For training the discriminator, offline data (A's low level on random goals) was as good as or better than online goal-matched data in most cases (the authors' wording), e.g. PM→Ant Waypoint 546 ± 15 offline vs 467 ± 21 online vs 483 ± 39 zero-shot; by Table 11's labels offline wins all 3 navigation comparisons but loses to online in 3 of 8 manipulation pairs (BP1 3L→4L 0.15 vs 0.23, BP2 2L→4L 0.41 vs 0.46, BP2 3L→4L 0.41 vs 0.44); the authors guess online data made the discriminator's task too easy [A1, A2, A3, A4, A5, A6, A7, A8, A12, A13, A14, A15] (App. F, Table 10, Table 11, pp. 17-18).
- C13: The authors' summary of the ablations: KL-regularized high-level fine-tuning is better for gaining performance on one task, and low-level discriminative imitation for boosting performance across a suite of tasks; how the two interact in sequence was not studied [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A13, A14, A15] (App. B, p. 14).
- C14: The 2-Leg Ant also gains from a PointMass high level on Maze End zero-shot: 0.74 ± 0.17 vs 0.14 ± 0.13 with the Ant's high level [A1, A2, A3, A4, A5, A6, A7, A11, A12, A13, A14, A15, A16] (Table 8, App. E, pp. 17-18).
- C15: The conclusion claims the method transfers long-horizon behaviour from simple to complex agents in a fraction of the sample complexity of standard RL, in simulation only [A1, A2, A3, A4, A5, A6, A7, A12, A13, A14, A15, A16] (Sec. 6, p. 9).

## Implementation notes

- Base optimizer SAC on the Stable Baselines codebase (source: Sec. 4.2, p. 5); code at github.com/jhejna/hierarchical_morphology_transfer (source: App. H, p. 18; not checked by the reader).
- High-level action every 32 low-level steps (64 for the Quadruped) in navigation, every 35 in manipulation; goal space is a change in x, y of at most 4 m (navigation) or 0.07 m (manipulation) (source: App. C.1, p. 14, App. C.2, p. 16, Table 4, p. 17).
- Navigation low-level reward: displacement of s^hi projected on the unit vector to the goal, minus λ‖a‖²; manipulation: −‖g − s^hi‖ − λ‖a‖² plus a sparse bonus within ε of the goal; λ, ε and the bonus weight aren't given (source: App. C.1, p. 14, App. C.2, p. 16).
- Low-level hyperparameters (timesteps, lr, batch, layers, horizon, reset probability, buffer, goal delta max) per agent: Table 4; high level per task: Table 5, e.g. 64-64 MLPs, lr 3e-4 (source: Table 4, Table 5, p. 17).
- Discriminator: 42-42 MLP, lr 2e-4 to 5e-4, batch 64, updated every 8 steps from a class-balanced circular buffer holding transitions of both agents; input (g − s^hi_t, g − s^hi_{t+1}), i.e. a 1-step window (source: Sec. 4.2, p. 5, App. A, p. 14, Table 6, p. 18).
- Discriminator lr annealed linearly to 0 over the first "stop" fraction (0.5) of training; its reward weight annealed from 0.3 or 0.4 to 0.1 over the first 90% (source: App. D, p. 16, Table 6, p. 18). How this maps onto λ0, λ1 in Eq. 3 is not stated (source: reader's reading).
- KL fine-tuning: closed-form KL between diagonal Gaussians, added to the policy loss at every step; weight 1, lr 0.01, annealed to 0 over the first 50% (Waypoint, manipulation); Maze uses lr 0.001 and weight 0.01 (source: Sec. 4.2, p. 5, App. D, p. 16).
- Steps task: the Ant gets 16 rangefinders and a low level pre-trained on random steps (source: App. C.1, p. 15). Network activations and the SAC entropy settings aren't given (source: assumption that the Stable Baselines defaults were used).

## Limitations / open questions

- Fails when the target agent's extra abilities change the optimal high-level plan (C11); the
  authors call for better transfer algorithms for harder morphologies (Sec. 4.3, p. 6).
- Simulation only (Sec. 6, p. 9); only one source agent per transfer, and the shared goal space
  and the s^hi / s^lo split are designed by hand per task family (Sec. 2.2, p. 2, App. C, p. 14).
- The combined pipeline (both parts in sequence) is not evaluated (App. B, p. 14).
- Reader's observation, unresolved: the appendix tables don't match Table 2 (p. 7) consistently.
  Table 10 labels the PM→Ant Maze columns the other way round (zero-shot 0.30 under "Maze Sampled"),
  and its other zero-shot value is 0.65 where Table 2 has 0.62; Table 2's Block Push discriminator
  rows equal Table 11's "Discrim On" (online) row exactly, not "Discrim Off", although App. D and
  App. F say the main results are offline only (pp. 16-18); so either Table 11 is mislabelled or
  Table 2's manipulation results are online. Table 9's row also reads "Ant2 Low" for the 2-Link arm
  (p. 18). The note takes the main-text tables' numbers as printed.
- Open question (survey lens): the transferable part here is the high level over a hand-chosen
  shared state and goal space; for many games, could a learned cross-game state embedding play the
  role of s^hi, with per-game low-level action heads aligned by a discriminator like Eq. 3?
- Open question: the alignment is pairwise (A vs B, binary q_φ); with many games, would a
  multi-class discriminator over all sources keep the objective of Eq. 2 useful, or collapse?
- Open question: the step-maze failure (C11) suggests that a shared policy over many games would
  need fine-tuning, not just KL anchoring, wherever a game affords strategies the others don't.

## Related

- related [[serrano2024crossdomain]]: the survey files it as [74] under reward shaping (Sec. V-A3, p. 11 there) and cites it for decoupling high- and low-level behaviour (Sec. VI-C, p. 15 there).
- related [[devin2016modular]]: an earlier split of policies into shared task and per-robot modules, cited here as transfer-oriented architecture work (Sec. 5.1, p. 8).
- related [[zhang2020correspondence]]: also moves a policy across mismatched domains, but by learning observation and action maps from unpaired transitions, where this paper keeps the high level and retrains only the low level.
- related [[zakka2021xirl]]: also transfers across morphologies with a learned reward, an embodiment-invariant embedding distance from videos, where this paper's reward is a discriminator on goal reaching.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
