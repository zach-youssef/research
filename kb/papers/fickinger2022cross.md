---
id: fickinger2022cross
title: Cross-Domain Imitation Learning via Optimal Transport
authors: [Arnaud Fickinger, Samuel Cohen, Stuart Russell, Brandon Amos]
year: 2022
venue: ICLR 2022
url: https://arxiv.org/abs/2110.03684
arxiv: 2110.03684
doi:
pdf_sha256: 6a6afd2d98f9b33c2c64bc9c329b90e1ce4e76476edda54a55805e994fa76dce
tags: [imitation-learning, cross-domain, cross-embodiment, optimal-transport, gromov-wasserstein, unpaired-alignment, reward-learning, theory, continuous-control]
status: read
read: 2026-11-22
---

# Cross-Domain Imitation Learning via Optimal Transport

## TL;DR

GWIL rewards an agent by the Gromov-Wasserstein distance from its trajectory to one expert
trajectory of another domain, using only within-space distances: no shared space, proxy tasks or
pairs. Optimal up to isometry (theorem); qualitative on maze, pendulum→cartpole, cheetah→walker.

## Problem

Imitation learning usually assumes the expert and agent share embodiment, dynamics and
state-action spaces, which moves the effort from designing rewards to collecting in-domain
demonstrations (Sec. 1, p. 1). Cross-domain imitation methods before this paper learn a map
between domains from proxy tasks that both agents perform optimally, which rules out imitating an
expert never seen before or transferring to a new robot (p. 1, Sec. 2, p. 3). The authors also
note that existing cross-domain benchmarks assume demonstrations from both agents (Sec. 2, p. 3).

## Method

- **Metric MDP.** An MDP with a distance d on S × A (Sec. 3, p. 3). The Gromov-Wasserstein (GW)
  distance between two metric measure spaces minimizes, over couplings u, the sum of
  |d_X(x, x′) − d_Y(y, y′)|² u_{x,y} u_{x′,y′}: it compares the pairwise-distance structure of two
  spaces to find the best isometry between them (Eq. 1, p. 3).
- **GW between policies.** GW(π_E, π_A) is the GW distance between (S_E × A_E, d_E, ρ_{π_E}) and
  (S_A × A_A, d_A, ρ_{π_A}), ρ the occupancy measures (Def. 1, Eq. 2, p. 4). Two policies are
  isomorphic when a bijection between their occupancy supports is an isometry and pushes one
  occupancy onto the other (Def. 2, Eq. 3-4, p. 4).
- **Pseudo-reward.** Differentiating GW through the dynamics isn't possible, so the agent gets a
  per-state-action reward r_GW(z_A) = −(1/ρ(z_A)) Σ |d_E(z_E, z_E′) − d_A(z_A, z_A′)|² u*u*, with
  u* the optimal coupling (Def. 3, p. 6). With one expert and one agent trajectory, occupancies
  are uniform over their steps, GW is Eq. 8 and the reward Eq. 9 (the factor T_A is dropped)
  (p. 7). The authors present this as generalizing OT imitation rewards that assume uniform
  occupancy (PWIL, SIL) (Remark 2, p. 7).
- **Algorithm.** Per episode: collect an agent trajectory, compute GW to the expert trajectory,
  set the Eq. 9 rewards, update a SAC policy and value (Alg. 1, p. 6, p. 7). GW and its gradient
  follow Peyré et al. 2016, the coupling a conditional gradient method (Ferradans et al. 2013)
  (p. 7).
- **Benchmark.** Three tasks: a PointMass maze and its reflection (rigid transform), pendulum →
  cartpole (slightly different spaces), DMC cheetah → walker (significantly different), all with
  Euclidean d_E, d_A (Sec. 5, pp. 7-9). Appendices: proxy-reward curves, sparse-reward transfer
  and wall-clock time (App. A-C, pp. 13-15).

## Assumptions

- A1: The expert demonstration includes actions: GW compares state-action occupancy measures, and the pseudo-reward is defined on agent state-action pairs (Eq. 2, p. 4, Eq. 8-9, p. 7).
- A2: Only expert demonstrations are available: no proxy tasks, no demonstrations from the agent's domain, no paired or aligned data (Sec. 1, pp. 1-2, Sec. 2, p. 3).
- A3: The optimality guarantee needs isometries φ: S_E → S_A and ψ: A_E → A_A under which reward, transition probabilities and initial-state distribution coincide (Eq. 5-7, p. 4).
- A4: The metric on each state-action space is the sum of a state distance and an action distance, as Theorem 1 takes it; in all experiments both are Euclidean on the raw state and action vectors (Sec. 4.1, p. 4, p. 8).
- A5: The expert policy is optimal for its task; experts are trained on pendulum swingup and cheetah run (Sec. 4.1, p. 4, Sec. 5.2-5.3, pp. 8-9).
- A6: The occupancy measure is approximated by the uniform distribution over a single trajectory's steps, assuming all its state-action pairs differ (p. 7).
- A7: The agent can interact with its own environment and learns online with SAC from the pseudo-reward; it receives no external reward except in the sparse-reward appendix (Sec. 4.2, p. 7, App. B, p. 13).
- A8: State and action spaces are continuous (p. 7).
- A9: The theorem's proof is written for finite state-action spaces; the authors say it extends directly to infinite ones (Sec. 4.1, p. 4).
- A10: States are low-dimensional vectors (PointMass maze positions, MuJoCo and DeepMind Control Suite proprioception), not pixels (reader's inference, Sec. 5, pp. 7-9).

## Claims

- C1: GW defines a metric on isomorphic classes of policies (not on policies), so GW(π_E, π_A) = 0 if and only if the two policies are isomorphic (Proposition 1, for any metrics d_E, d_A) [] (Sec. 4.1, p. 4).
- C2: Theorem 1: if isometries φ, ψ preserve reward, dynamics and initial distribution, a policy minimizing GW to an optimal expert policy, with d = d^S + d^A, is isomorphic to an optimal policy of the agent's MDP; the isometries need not be known [A3, A4, A5, A9] (Sec. 4.1, pp. 4-6, Eq. 5-7, p. 4).
- C3: GWIL recovers optimal policies only up to isometry: the isometry class of an optimal policy can contain suboptimal policies, and in practice running different seeds finds the optimal one (Remark 1) [A3, A4] (p. 6).
- C4: Proposition 2: a policy minimizing GW(π_E, π_A) is optimal for the pseudo-reward r_GW of Def. 3, so GW minimization can be turned into RL without derivatives through the dynamics [A7] (Sec. 4.2, p. 6).
- C5: In the PointMass maze with the agent's maze a reflection of the expert's (a case satisfying Theorem 1), GWIL recovers an optimal policy from a single expert trajectory without external reward (qualitative, one trajectory shown) [A1, A2, A3, A4, A5, A6, A7, A8, A10] (Sec. 5.1, p. 8, Fig. 3, p. 8).
- C6: From one pendulum swingup trajectory, GWIL recovers the optimal behaviour on cartpole without external reward, although the transformation isn't rigid and the state-action dimensions differ (3 vs 5) (qualitative) [A1, A2, A4, A5, A6, A7, A8, A10] (Sec. 5.2, pp. 8-9, Fig. 4, p. 8).
- C7: From one cheetah run trajectory, GWIL produces, depending on the seed, a cheetah-like walker moving forward (optimal) or backward (suboptimal), the two members of the optimal policy's isometry class the authors expect (qualitative) [A1, A2, A4, A5, A6, A7, A8, A10] (Sec. 5.3, p. 9, Fig. 5, p. 9).
- C8: SAC converges quickly and consistently on the GW proxy reward across 5 seeds in all three environments (episodic GW return plateaus near −0.02 to −0.05 within about 0.5-1M samples, read from the figure) [A1, A2, A4, A6, A7, A8] (App. A, p. 13, Fig. 6, p. 13).
- C9: With a sparse reward added to each agent's OT reward, GWIL with one cross-domain demonstration is competitive with a same-domain Wasserstein (PWIL-style) learner in a sparse maze (both near 930-950 at 2M samples) and somewhat below it in sparse cartpole (about 600 vs 720 at 500k samples, read from the figure); SAC alone learns nothing meaningful in either [A1, A2, A4, A5, A6, A7, A8, A10] (App. B, pp. 13-14, Fig. 7, p. 14).
- C10: The cost of GWIL is dominated by computing the optimal coupling, which depends on the number of time steps in the trajectories and not on the state-action dimension; no entropy regularizer was used [A6] (App. C, p. 14).
- C11: In wall-clock time, GWIL in the sparse maze needs the same order of time as the same-domain Wasserstein learner, and a GWIL walker imitating a cheetah reaches walking speed (horizontal velocity 1) faster than a SAC walker trained on the run reward (about 1.4k vs 2.3k time units, read from the figure; units not stated) [A1, A2, A4, A5, A6, A7, A8, A10] (App. C, pp. 14-15, Fig. 8, Fig. 9, p. 15).

## Implementation notes

- GW distance via Peyré et al. 2016 Proposition 1, its gradient via Proposition 2, the coupling for Eq. 8 by conditional gradient as in Ferradans et al. 2013 (source: p. 7).
- Policy optimization: SAC (Haarnoja et al. 2018); its hyperparameters, network sizes and training lengths aren't in the paper beyond the sample axes of Figs. 6-7 (up to about 2M samples) (source: p. 7, App. A-B, pp. 13-14; the authors' code, linked from the project site, was not checked).
- Pseudo-reward: Eq. 9 without the constant factor T_A, recomputed for every collected episode against the single expert trajectory (source: p. 7, Alg. 1, p. 6).
- Identical state-action pairs within a trajectory can be told apart by adding the time step to the state (source: footnote 7, p. 7).
- Distances: Euclidean on states and actions; whether state dimensions are normalized before computing distances isn't stated (source: p. 8; normalization unknown, assumption).
- How the experts were trained (algorithm, steps) isn't stated; a SAC expert on the true reward is the likely reading (source: assumption).
- Seeds: 5 in App. A; the main-text figures show single rollouts and don't state seed counts (source: App. A, p. 13).
- Code and videos: https://arnaudfickinger.github.io/gwil/ (source: p. 1).
- Version: arXiv v4 (2026-01-13) was read; its header says ICLR 2022, and its title matches the API's (source: p. 1).

## Limitations / open questions

- Optimality only up to isometries; the cheetah→walker run can go backward (Remark 1, p. 6, Sec. 5.3, p. 9).
- GW ignores temporal ordering; the authors list metrics aware of MDP time structure, other OT distances with more flexible invariances, scaling to complex environments and sparse-reward exploration as future work (Sec. 6, p. 9).
- Results in the main text are qualitative (rollout images); there is no table of returns against baselines (Sec. 5, pp. 8-9).
- [[franzmeyer2022matters#C5]] later reports GWIL's MuJoCo results as highly stochastic across hyperparameters.
- Open question (reader): picking the right member of the isometry class "by seed" needs some signal about the agent's task, which the setting says the agent doesn't have; how is the seed chosen without the reward?
- Open question (reader): Euclidean distances on raw state vectors make GW sensitive to unit scales and to which dimensions are present; for games, would GW over object-centric or anchor-relative states (as in [[derstroff2026relative]]) give isometric structure across two games?
- Open question (reader): a cheap experiment is to compute GW couplings between trajectories of two Atari games with matching mechanics (object states) and check whether the coupling pairs semantically matching objects, before any RL.

## Related

- uses [[hejna2020morphological]]: the PointMass maze environment of Sec. 5.1 comes from Hejna et al. 2020.
- related [[franzmeyer2022matters]]: UDIL uses GWIL as its only like-for-like baseline and reports beating it on six MuJoCo pairs.
- related [[gupta2017invariant]]: learns a shared space from proxy-task pairs, the requirement GWIL removes.
- related [[zakka2021xirl]]: cited as a goal-driven cross-embodiment alternative that rewards task progress instead of matching structure.
- related [[zhang2020correspondence]]: also aligns two domains without paired data, by learning explicit maps with dynamics cycle-consistency.
- related [[serrano2024crossdomain]]: survey of cross-domain transfer; GWIL is a reward-level transfer without a learned mapping.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
