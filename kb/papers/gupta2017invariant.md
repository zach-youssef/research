---
id: gupta2017invariant
title: Learning Invariant Feature Spaces to Transfer Skills with Reinforcement Learning
authors: [Abhishek Gupta, Coline Devin, YuXuan Liu, Pieter Abbeel, Sergey Levine]
year: 2017
venue: ICLR 2017
url: https://arxiv.org/abs/1703.02949
arxiv: 1703.02949
doi:
pdf_sha256: 0f62837653d762ee7fdea6c136d67e10511cd858f1e6ba18549b8e34ae82eb77
tags: [reinforcement-learning, transfer-learning, cross-embodiment, morphology-transfer, representation-learning, reward-shaping, cross-domain]
status: read
read: 2026-11-21
---

# Learning Invariant Feature Spaces to Transfer Skills with Reinforcement Learning

## TL;DR

Two robots that both know a proxy skill learn nets f, g mapping their states to a shared space
(paired states by time step or DTW, contrastive plus autoencoder losses); a new skill then transfers
as a reward to track f(source states). Sparse-reward arm tasks the target can't learn from scratch.

## Problem

Transfer between morphologically different agents: earlier methods build state-to-state
isomorphisms by hand-designed mappings, exhaustive pairing search or hand-designed features, or
work on low-dimensional or linear systems, and a full correspondence may not exist between
different robots (Sec. 1, pp. 1-2, Sec. 2, p. 2). Parameter-reuse policy transfer hasn't handled
different state dimensionalities (pp. 2-3). The goal is to transfer the shared part of a skill
and learn the rest from scratch (p. 2).

## Method

- **Setting** (Sec. 3, p. 3): source and target MDPs with different states, actions and
  dynamics, assumed to share reward structure: optimal-policy state distributions match in some
  common feature space, p(f(s_S)) = p(g(s_T)) (Sec. 3.1, p. 3). Trivial if f = g = 0, so f and g
  should keep as much information as possible.
- **Proxy skills** (Sec. 3.2, pp. 3-4): states split into agent-specific s_r and task-specific
  s_env; both agents already know a proxy skill (D_Sp, D_Tp), only the source knows the test
  skill (D_S). Action and agent-state spaces are the same across an agent's proxy and test
  domains. f and g are learned on s_r only.
- **Pairing P** (Sec. 3.3, p. 4): pairs of corresponding states from the proxy skill. Either
  time-based alignment (same time step, assuming similar pace, Sec. 3.3.1), or an EM-style
  alternation: learn the space from current pairs, re-align by dynamic time warping in that
  space, repeat, starting from time alignment (Sec. 3.3.2). An unsupervised alignment is
  mentioned as possible but not used.
- **Embedding loss** (Sec. 4.1, p. 5, Fig. 1): untied nets f, g with an L2 similarity loss
  ||f(s_Sp,r) − g(s_Tp,r)||_2 on pairs (after Chopra et al. 2005), plus decoders Dec_S, Dec_T with
  reconstruction losses ||s − Dec(f(s))||_2 (printed as plain, not squared, L2 norms) so f and g stay as invertible as possible; summed and trained end to end.
- **Transfer** (Sec. 4.1.1, pp. 5-6): no direct policy transfer; the target runs RL on its own
  sparse task reward plus r_transfer = α||f(s_S,r^(t)) − g(s_T,r^(t))||_2, comparing the source's
  optimal state and the target's current state at the same time step t, as reward shaping.
- **Baselines** (Sec. 5.1, pp. 6-7): no transfer; random projections, CCA and unsupervised
  manifold alignment (UMA) as linear f, g; kernel CCA (quadratic, RBF, linear kernels); and
  "direct mapping" that regresses target states from source states on the same pairs and
  penalizes deviation from the mapped source trajectories.
- **Experiments** (Sec. 5, pp. 6-10): MuJoCo arms. (1) 3-link to 4-link button pressing; the
  reward only measures how far the button is pressed toward the goal; proxies: reach, peg
  insertion, block moving; states are joint angles and velocities (Sec. 5.2, pp. 7-8). (2)
  3-link torque-driven to 3-link tendon-driven arm (tendon lengths and velocities as state),
  block pulling, reach proxy (Sec. 5.3, pp. 8-9). (3) 3-link to 4-link block pushing from raw
  images with g = f, reach proxy (Sec. 5.4, pp. 9-10).

## Assumptions

- A1: Source and target MDPs may differ in state space, action space and dynamics, but the state distributions of their optimal policies coincide under some learnable maps f and g into a common feature space (Sec. 3, Sec. 3.1, p. 3).
- A2: The task reward is structurally similar across the two agents: it depends on quantities both can reach, such as end-effector or object positions (Sec. 3, p. 3).
- A3: Both agents have already learned at least one proxy skill, and only the source has learned the test skill (Sec. 3.2, pp. 3-4).
- A4: Each agent's state splits into an agent-specific part s_r and a task-specific part s_env, and f and g see only s_r (Sec. 3.2, p. 4, Sec. 4.1, p. 5).
- A5: For each agent, the proxy and test domains differ only in reward and task-specific state; the agent-specific state space and the action space are the same (Sec. 3.2, p. 4).
- A6: Skills are episodic with a fixed horizon, so proxy states can be paired by time step; time-based pairing assumes the two agents do the task at roughly the same rate (Sec. 3.3, Sec. 3.3.1, p. 4).
- A7: The pairs come from successful (optimal) proxy trajectories of both agents (Sec. 4, p. 5, Sec. 5.2, p. 7).
- A8: The source has optimal trajectories of the test skill, and r_transfer compares the target's state with the source's at the same time step, so the target is pushed to follow the source's timing (Sec. 4.1.1, p. 6).
- A9: The target's own task reward is sparse (only the button, block or object distance to the goal), so learning from scratch fails within the budget (Sec. 5.2, p. 7, Sec. 5.3, p. 9, Sec. 5.4, p. 10, App. 7.1, p. 14).
- A10: The source and proxy skills were learned with hand-designed shaped costs (App. 7.1, p. 14).
- A11: All RL uses trajectory-centric time-varying linear-Gaussian policies with fitted local linear dynamics and iLQG, over a few initial conditions (App. 7.1, p. 14, Fig. 2, p. 6).
- A12: The agents are simulated MuJoCo arms of similar size doing the same manipulation tasks: 3-link vs 4-link, or 3-link torque-driven vs 3-link tendon-driven (Sec. 5, p. 6, Sec. 5.2, p. 7, Sec. 5.3, p. 8).
- A13: In the state experiments each agent observes its full low-dimensional proprioceptive state: joint angles and velocities, or tendon lengths and velocities (Sec. 5.2, p. 8, Sec. 5.3, p. 9).
- A14: In the image experiment both domains give images of the same size and type, so a single encoder serves as f = g (Sec. 5.4, p. 9).
- A15: Transfer is between exactly two agents (Sec. 2, p. 3, Sec. 6, p. 10).
- A16: The agent-specific states the target visits in the test skill lie within those the proxy skills cover, since f and g are learned on proxy-skill states only (reader's inference, Sec. 3.2, p. 4, Fig. 5, p. 8).
- A17: The source's optimal test-skill trajectories are for the same task conditions (e.g. initial object positions) the target is trained on, so the time-step comparison in r_transfer is meaningful (reader's inference, Sec. 4.1.1, p. 6, Fig. 2, p. 6).

## Claims

- C1: Without transfer the sparse-reward target tasks are not learned in a tractable amount of data: best success over 75 iterations is 0.0% for button pressing, 4.2% for block pulling and 7.1% for block pushing; the text says the tasks aren't learned even with 3-4 times more experience [A9, A11, A12] (Table 1, p. 8, Sec. 5.1, p. 6).
- C2: On 3-link to 4-link button pressing, the 4-link arm trained with the transfer reward explores in a directed way and reaches almost perfect success in 7 iterations [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A15, A16, A17] (Sec. 5.2, p. 8, Fig. 5, p. 8).
- C3: On button pressing, CCA needs about 4 times more experience than the method to reach 60% success, kernel CCA doesn't perform as well (its best, quadratic, levels off near 0.3-0.4 in Fig. 5), and UMA and random projections perform much worse [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A15, A16, A17] (Sec. 5.2, p. 8, Fig. 5, p. 8).
- C4: Linear embeddings limit expressiveness: CCA allows transfer but with less gain than neural-network f and g [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A15, A16, A17] (Sec. 5.1, p. 6).
- C5: The EM-style alignment with DTW also reaches perfect performance on button pressing; on the tendon task it performs very similarly to time-based alignment, likely because the data are already well time-aligned [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A15, A16, A17] (Sec. 5.2, p. 8, Sec. 5.3, p. 9, Fig. 8, p. 10).
- C6: Direct state-to-state mapping gives only limited transfer (Fig. 5 shows it near 0.2 success on button pressing), which the authors attribute to there being no complete correspondence between different robots, whereas a common space need only keep the transferable parts [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A15, A16, A17] (Sec. 5.1, p. 7, Sec. 5.2, p. 8, Fig. 5, p. 8).
- C7: Learning the space from all three proxy tasks (reach, push, peg) beats any single proxy: both all-proxy embeddings (time-aligned and EM-optimized P) reach full success on button pressing, while single-proxy embeddings plateau at roughly 0.35-0.7 [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A15, A16, A17] (Fig. 4, p. 7, Fig. 5, p. 8).
- C8: Torque-driven to tendon-driven block pulling with one reach proxy: the transfer reward lets the tendon arm do the task, learning from scratch fails completely, CCA and direct mapping beat scratch but learn slower, random projections and UMA perform quite poorly; in Fig. 8 direct mapping levels off near 0.35 and CCA climbs only late, still below the method at the last iteration with a wide band, and linear kernel CCA approaches full success linear kernel CCA a few iterations later than the method [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A15, A16, A17] (Sec. 5.3, p. 9, Fig. 8, p. 10).
- C9: Kernel CCA isn't consistent: the quadratic kernel is competitive on block pulling (Fig. 8 shows it at full success no later than the method) but per the text doesn't work at all on button pressing, though Fig. 5 shows quadratic KCCA at about 0.3-0.4 there, below the method's 1.0, and the best kernel varies between experiments [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A15, A16, A17] (Sec. 5.1, p. 6, Sec. 5.3, p. 9, Fig. 5, p. 8, Fig. 8, p. 10).
- C10: From raw images (shared conv encoder with spatial-softmax feature points, g = f), the 4-link arm learns sparse-reward block pushing with transfer, reaching roughly 0.8-0.9 success with a wide band in Fig. 9b; random projections and linear kernel CCA have some success and no transfer fails [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A14, A15, A16, A17] (Sec. 5.4, p. 10, Fig. 9b, p. 10).
- C11: With an imperfect pairing the transfer reward can interfere once the target policy is already good, though it usually helps early in learning, so α may be decayed over training; this is done for the tendon experiment [A6, A8, A17] (Sec. 4.1.1, p. 6).

## Implementation notes

- f and g (state experiments): 3 layers, 60 hidden units each, ReLU, trained end to end with Adam; learning rate, batch size, embedding dimension and decoder architecture are not given (source: Sec. 5, p. 6; decoders mirroring the encoders is an assumption).
- Image encoder: 3 conv layers with 5×5 filters, no pooling, spatial softmax giving normalized pixel coordinates of feature points (source: Sec. 5.4, p. 9, after Levine et al. 2016); channel counts and image size not given.
- r_transfer is printed as +α||f − g||_2 (an L2 norm, not squared); as a shaping term it must penalize the distance, and App. 7.1 runs iLQG on shaped costs, so it acts as a cost term (source: Sec. 4.1.1, p. 6, App. 7.1, p. 14; the cost reading is an assumption). α and its decay schedule for the tendon task are not given (source: Sec. 4.1.1, p. 6).
- RL in all domains: trajectory-centric RL with time-varying linear-Gaussian policies p(u|x) = N(K_t x + k_t, C_t), fitted local linear dynamics and iLQG updates (source: App. 7.1, p. 14, after Levine & Abbeel 2014); an "iteration" is one such update, samples per iteration not given.
- Source and proxy skills were learned with hand-designed shaped costs; the 3-link source for button pressing had a cost for nearing the button plus button-to-goal distance (source: App. 7.1, p. 14, Sec. 5.2, p. 8).
- Pairs come from successful (optimal) proxy trajectories of both agents; the number of trajectories and how many conditions per task is not given beyond "multiple conditions where the objects start in different locations" (source: Fig. 2 caption, p. 6, Sec. 5.2, p. 7).
- Seeds and the meaning of the shaded bands in Fig. 5, 8 and 9b are not stated (source: Fig. 5, p. 8).

## Limitations / open questions

- Needs a proxy skill both agents already solve and, for the time-based pairing, similar pace
  (Sec. 3.2-3.3, pp. 3-4); only two agents, simulated arms, and optimal source trajectories.
- r_transfer compares states at the same time step, so the target must follow the source's
  timing (Sec. 4.1.1, p. 6).
- Future work the authors name: many agents and skills, deciding which skills to share (a
  wheeled and a legged robot might share manipulation but not locomotion), and growing the space
  by transferring skills one by one (Sec. 6, p. 11); an adversarial objective instead of pairs
  (Sec. 2, p. 3).
- Open question (survey lens): the supervision is weak but paired (same task, aligned in time).
  Across many games there is rarely a shared "proxy skill"; would TCC-style learned alignment or
  unpaired dynamics-consistency objectives replace the pairing?
- Open question: with g = f on images (Sec. 5.4) the method becomes a shared encoder plus a
  pairing; does it still need the reconstruction decoders?

## Related

- related [[devin2016modular]]: same group; uses a similar agent/task state split but, in contrast to reusing network modules, moves to learning a shared state embedding for two agents (Sec. 2, p. 3, Sec. 3.2, p. 4).
- related [[huang2020modular]]: transfers across morphologies with one shared per-limb policy instead of a learned state embedding and per-target RL.
- related [[zhang2020correspondence]]: learns cross-domain state and action maps from unpaired transitions, removing the paired proxy-skill data needed here.
- related [[dwibedi2019cycle]]: learned temporal alignment of videos, a possible replacement for the time-step or DTW pairing.
- related [[zakka2021xirl]]: also turns distance in an embedding learned across embodiments into a reward for the target agent.
- related [[serrano2024crossdomain]]: the survey lists this paper as [72] among learned mappings that need paired or aligned data.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
