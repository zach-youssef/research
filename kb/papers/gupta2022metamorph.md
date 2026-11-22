---
id: gupta2022metamorph
title: "MetaMorph: Learning Universal Controllers with Transformers"
authors: [Agrim Gupta, Linxi Fan, Surya Ganguli, Li Fei-Fei]
year: 2022
venue: ICLR 2022
url: https://arxiv.org/abs/2203.11931
arxiv: 2203.11931
doi:
pdf_sha256: 5700972c72ae851638e20304c02f591baabf921d5bba5140c19869fff7c8aaf5
tags: [reinforcement-learning, transformer, morphology, multi-task, zero-shot, transfer-learning, ppo, continuous-control, tokenization]
status: read
read: 2026-11-21
---

# MetaMorph: Learning Universal Controllers with Transformers

<!-- Read from arXiv v1 (2022-03-22, ICLR 2022), 18 PDF pages, all read: main text pp. 1-9,
acknowledgement and reproducibility pp. 10, references pp. 10-13 (skimmed), App. A-C pp. 14-16,
Alg. 1 p. 17, Tables 1-2 p. 18. Code: https://github.com/agrimgupta92/metamorph (not checked). -->

## TL;DR

One PPO-trained transformer policy over 100 UNIMAL robots (15-20 DoF): one token per limb (proprioception
plus morphology descriptors, DFS order, learned position embeddings), per-limb action outputs. Matches
per-robot MLPs on 2 of 3 tasks, beats a GNN 2-3x, zero-shots to unseen dynamics/kinematics, fine-tunes 2-3x faster.

## Problem

Prior cross-morphology policies covered hand-made variations of one or a few base morphologies,
low-complexity control (at most 7 DoF), and assumed with GNNs that the kinematic graph is the right
inductive bias (Sec. 1, p. 1; Sec. 2, p. 2). The UNIMAL design space holds more than 10^18 robots, so
a policy per robot is infeasible (Sec. 4, p. 3). The goal is one controller for a modular design
space that (a) generalises zero-shot to unseen dynamics, kinematics, morphologies and tasks and (b)
transfers sample-efficiently (Sec. 1, pp. 1-2; Fig. 1, p. 2). Two difficulties: robots differ in
action space, sensors, morphology and dynamics, and some robots learn a task faster than others, so
joint training is unbalanced (Sec. 1, p. 2).

## Method

- **Design space** (Sec. 3.1, p. 3): a robot is a kinematic tree (multigraph) of modules: a sphere
  head/root (radius, density) and cylinder limbs (length, radius, density); two modules are joined by
  at most two motor-actuated hinge joints (axis, limits, gear ratio).
- **Objective** (Sec. 3.2, p. 3): multi-task RL over K robots, maximise the average discounted
  return R = (1/K) Σ_k Σ_t γ^t r_t^k with PPO.
- **Observation** s^k = (s_m, s_p, s_g) (Sec. 4.1, p. 4; App. A.1, p. 14). Per module: morphology
  s_m = module parameters (shape, density, orientation and position relative to parent) and joint
  parameters (type, range, axis, actuator gear); proprioception s_p = 3D position, 4D quaternion, 3D
  linear and angular velocity, plus joint position and velocity. Joint information is given to the
  child node. The paper stresses that prior work (Amorpheus, SMP) used only limb connectivity as
  "morphology". Global s_g: a terrain heightmap on a non-uniform grid, 1 m behind to 4 m ahead and
  4 m to each side, relative to the ground under the root.
- **Encode** (Sec. 4.2, p. 4; Eq. 1; Fig. 2): local s_l = (s_m, s_p) per module, zero-padded to M
  dims (joint counts vary); tokens in depth-first order from the torso; a single-layer MLP per token
  to D dims, plus learned 1D position embeddings W_pos ∈ R^{N×D}. The sequence is zero-padded to N
  for batching across robots.
- **Process** (Eqs. 2-3, p. 4): L pre-LN transformer encoder layers (MSA and MLP, residuals).
- **Decode** (Eq. 4, p. 5): s_g goes through a 2-layer MLP to g, concatenated to every output
  token (not fed to the encoder, which would dilute the low-dim local inputs and enlarge D), then a
  per-token decoder gives each joint's action mean; Gaussian with state-dependent mean and fixed
  diagonal covariance. The critic predicts a value per limb and averages them (p. 5).
- **Dynamic replay buffer balancing** (Sec. 4.3, p. 5; Eq. 5; Alg. 1, p. 17): robots that fall early
  produce short episodes and less data ("rich gets richer"). Each iteration samples robot k with
  P_k ∝ E_k^β, E_k an EMA (factor α) of a performance metric; with episode length as metric, E_i is
  replaced by 1000/E_i (1000 = max episode length), so short-episode robots get sampled more.
  Uniform sampling for the first N_warmup iterations.
- **Setup** (Sec. 5.1, p. 6; App. A.2, p. 14): MuJoCo; 100 training UNIMALs chosen from 300
  evolved in Gupta et al. 2021 (no two with the same kinematic tree or parameters), 100 separate
  test UNIMALs. Tasks: flat terrain (FT), variable terrain (VT: hills, steps, rubble, new each
  episode), obstacles (cuboids); 1000-step episodes; one dense morphology-independent reward
  (forward motion minus a small energy penalty); early termination when torso height drops below
  50% (FT, Obstacles) or 30% (VT).
- **Baselines** (Sec. 5.2, p. 6; App. B.1, p. 15): GNN (NerveNet with MetaMorph's encode/decode,
  observations and balancing; only the process step differs), per-robot 2-layer MLPs (approximate
  upper bound), multi-task MLP, MetaMorph-AO (Amorpheus's observations). Ablations: -NPE (no
  position embeddings), -NM (no morphology inputs), -NMT (no morphology, binary robot-id context
  token), -HPE (hand-designed per-limb unique id instead of learned positions).

## Assumptions

- A1: Robots come from the UNIMAL design space: 3D kinematic trees of a sphere head (the root) and cylinder limbs, two modules joined by at most two motor-actuated hinge joints (Sec. 3.1, p. 3).
- A2: The 100 training robots are drawn from robots already evolved for locomotion tasks (Gupta et al. 2021), no two with the same kinematic tree or kinematic parameters (Sec. 5.1, p. 6, App. A.2, p. 14, App. C, p. 16).
- A3: Each module's input holds its morphology (shape, density, pose relative to the parent, joint type, range, axis and actuator gear, as found in URDF or MuJoCo XML) and its proprioception (position, quaternion, linear and angular velocity, joint position and velocity), zero-padded to a common size (Sec. 4.1, p. 4, Sec. 4.2, p. 4, App. A.1, p. 14).
- A4: The kinematic tree is known and serialised into a token sequence by depth-first traversal from the root, same-depth nodes in MuJoCo XML order, padded to a maximum module count N for which position embeddings are learned (Sec. 4.2, p. 4, Eq. 1, p. 4, App. B.3, p. 16).
- A5: Each joint's action is decoded from its module's output token; the policy is Gaussian with a state-dependent mean and a fixed diagonal covariance (Eq. 4, Sec. 4.2, p. 5).
- A6: Exteroception is a terrain heightmap on a fixed non-uniform grid around the root, relative to the ground under it, fed to the decoder rather than the encoder (Sec. 4.2, pp. 4-5, App. A.1, p. 14).
- A7: All tasks are locomotion with one dense, morphology-independent reward (forward progress, or distance from the start in Escape, minus a small energy penalty), 1000-step episodes and early termination on a fall (Sec. 5.1, p. 6, Sec. 5.4, p. 9, App. C, p. 16).
- A8: One policy is trained on all training robots with on-policy PPO and dynamic replay buffer balancing (episode length as metric, alpha 0.1, beta 1), for 10^8 iterations (Sec. 3.2, p. 3, Sec. 4.3, p. 5, App. A.3, p. 14, Sec. 5.2, p. 7).
- A9: The training robots all share the same armature and damping; there is no dynamics randomisation (Sec. 5.3, p. 8).
- A10: All environments are MuJoCo simulations (Sec. 5.1, p. 6, App. A.2, p. 14).
- A11: The zero-shot dynamics and kinematics variants are perturbations of the training robots within Table 2's ranges, new joint ranges overlapping the original by at least 50% (Sec. 5.3, pp. 7-8, App. A.2, p. 14).
- A12: The 100 unseen test robots come from the same UNIMAL design space as the training robots (Sec. 5.4, p. 8, App. A.2, p. 14).
- A13: Training curves average 3 seeds; zero-shot results use the model of a single seed, 10 trials per robot (App. A.4, p. 15, Fig. 6, p. 7).
- A14: The full simulator state of every module is observed at every step; there is no perception from pixels (reader's inference, App. A.1, p. 14).
- A15: Baselines and ablations use the same hyperparameters and about the same parameter count, about 3.3 million (App. A.3, p. 14).

## Claims

- C1: Trained jointly on 100 robots, MetaMorph matches the average reward of per-morphology MLPs on flat terrain and obstacles; on variable terrain it is slightly below them but had not saturated when training stopped at 10^8 iterations [A1, A2, A3, A4, A5, A6, A7, A8, A10, A13, A14] (Fig. 4, Sec. 5.2, pp. 6-7).
- C2: MetaMorph is about 5x more sample-efficient than training independent MLPs, which use 5×10^6 iterations per robot [A1, A2, A3, A4, A5, A6, A7, A8, A10, A13, A14] (Sec. 5.2, p. 7).
- C3: The GNN baseline, identical except for the process step, saturates at a level the authors put at 2 to 3 times below MetaMorph (read from Fig. 4: about 2.3x on flat terrain, about 3.7x on variable terrain, about 3.3x on obstacles) [A1, A2, A3, A4, A5, A6, A7, A8, A10, A13, A14, A15] (Fig. 4, Sec. 5.2, pp. 6-7).
- C4: Learned position embeddings significantly improve performance: MetaMorph-NPE is below MetaMorph in all three environments [A1, A2, A3, A4, A5, A6, A7, A8, A10, A13, A14, A15] (Fig. 4, Sec. 5.2, pp. 6-7).
- C5: Without morphology information, MetaMorph-NM fails to learn a policy that controls the diverse robots; the authors conclude morphology inputs are necessary while the kinematic graph need not be built into the architecture [A1, A2, A3, A4, A5, A6, A7, A8, A10, A13, A14, A15] (Fig. 4, Sec. 5.2, pp. 6-7).
- C6: Dynamic replay buffer balancing raises the share of robots whose final performance exceeds 75% of the MLP baseline by about 10-15 points in all three environments (read from Fig. 5: about 84 vs 69% flat terrain, 77 vs 67% variable terrain, 95 vs 84% obstacles); the authors say that without it MetaMorph fails to learn a good policy for 10-15% of robots, and even with it about 5-23% of robots stay below the 75% mark [A1, A2, A3, A4, A5, A6, A7, A8, A10, A13, A14] (Fig. 5, Sec. 5.2, pp. 6-7).
- C7: Zero-shot to dynamics variants (armature, density, damping, gear; 400 robots per type, 4 per training robot, 10 trials each), MetaMorph is significantly better than MetaMorph-NM and GNN in every variation type and environment, with larger gaps on VT and obstacles, although training used no dynamics randomisation [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A13, A14, A15] (Fig. 6, Table 2, Sec. 5.3, pp. 7-8).
- C8: Zero-shot to kinematics variants (limb radius and height, joint angle ranges), MetaMorph beats MetaMorph-NM and GNN in all three environments, but performance drops for joint-angle variations [A1, A2, A3, A4, A5, A6, A7, A8, A10, A11, A13, A14, A15] (Fig. 6, Sec. 5.3, p. 8).
- C9: On 100 unseen test morphologies in the training environment, the pre-trained policy has strong zero-shot performance, and fine-tuning reaches the from-scratch final reward with 2-3x fewer iterations (2x FT, 3x VT, 2x obstacles) [A1, A2, A3, A4, A5, A6, A7, A8, A10, A12, A13, A14] (Fig. 7, Sec. 5.4, pp. 8-9).
- C10: On unseen morphologies and new tasks (VT to Escape, obstacles to cylinder obstacles), zero-shot performance is strong and fine-tuning is 2-3x more sample-efficient than from scratch (3x Escape, 2x cylinders) [A1, A2, A3, A4, A5, A6, A7, A8, A10, A12, A13, A14] (Fig. 8, Sec. 5.4, pp. 8-9).
- C11: For two agents on flat terrain, the stable rank of the attention matrix is small and oscillates between two values, matching attention maps in which groups of limbs are attended together; the authors read this as emergent motor synergies that partly explain how one policy controls about 1600 DoF across robots [A1, A3, A4, A5, A7, A8, A10] (Fig. 9, Sec. 5.5, p. 9).
- C12: A direct Amorpheus comparison was infeasible: Amorpheus handles neither 3D robots with variable joints between limbs nor exteroception, and its sequential per-morphology collection and updates would need about 30 GPU days for 100M iterations on an RTX 2080 against 1.5 GPU days for MetaMorph (about 20x) [A1, A6, A8] (App. B.1, p. 15).
- C13: On flat terrain, MetaMorph-AO (Amorpheus's observations: one-hot limb id, pose, velocities, joint position and range) beats MetaMorph-NM but is significantly worse than MetaMorph; the 6-layer multi-task MLP and MetaMorph-NMT both learn poorly [A1, A2, A3, A4, A5, A6, A7, A8, A10, A13, A14, A15] (Fig. 10, App. B.1, pp. 15-16).
- C14: MetaMorph-HPE (hand-designed unique limb ids) is slightly better than -NPE but far below MetaMorph, which the authors take to mean learned position embeddings carry more than a position identifier [A1, A2, A3, A4, A5, A6, A7, A8, A10, A13, A14, A15] (Fig. 10, App. B.1, App. B.2, pp. 15-16).
- C15: Learned position embeddings show a diagonal cosine-similarity pattern, which the authors report for all seeds and environments though only flat terrain is shown [A4, A8] (Fig. 11, App. B.2, p. 16).
- C16: Zero-shot transfer is not robust to the opposite ordering of same-depth nodes in the DFS sequence (about 75% performance drop); randomising the visiting order of same-depth nodes during training makes it robust [A1, A3, A4, A8, A10] (App. B.3, p. 16).
- C17: Training curves are robust to the seed (small standard deviation over 3 seeds); zero-shot results use one seed's model, fine-tuning uses each of the 3 pre-training seeds [A8, A13] (App. A.4, p. 15).

## Implementation notes

- PPO: γ 0.99, GAE λ 0.95, clip 0.2, 8 epochs, batch 5120, entropy 0.01, reward and observation normalisation and clipping to [−10, 10], 2560 timesteps per rollout, 16 workers, 32 environments, 1×10^8 total timesteps, Adam lr 3e-4 with linear warmup (5 iterations) and cosine decay, gradient-norm clip 0.5, clipped value loss with coefficient 0.5 (source: Table 1, p. 18).
- Transformer: 5 layers, 1 head, embedding 128, feedforward 1024, ReLU, dropout 0.1 (source: Table 1, p. 18).
- Exteroception MLP and per-robot MLP baseline: 2 layers of 64; GNN baseline: 5-layer MLP of width 512; all models about 3.3M parameters (source: App. A.3, p. 14).
- Balancing: α = 0.1, β = 1.0, metric = 1000 / episode length (source: App. A.3, p. 14; Sec. 4.3, p. 5); N_warmup for the balancing is not given (source: Alg. 1, p. 17, names it only).
- DFS ties broken by the order nodes appear in the MuJoCo XML, as MuJoCo parses it (source: App. B.3, p. 16).
- Joint-angle variants: a random range from Table 2 with at least 50% overlap with the original (source: App. A.2, p. 14).
- "Iterations" (Figs. 4, 6-8) and "total timesteps" (Table 1) are both 10^8; read here as environment steps (source: assumption).
- Whether padded tokens are masked out of attention, and the per-token action head width (up to 2 joints per module, padded) are not stated; use a key-padding mask and a 2-wide masked head (source: assumption; check the authors' code).
- Maximum sequence length N: Fig. 11 shows 12 positions, so likely N = 12 (source: assumption from Fig. 11, p. 16).
- The fixed diagonal covariance Σ and the decoder φ's depth are not stated (source: assumption; see authors' code).

## Limitations / open questions

- Needs a pre-existing collection of task-optimised robots (two stages: evolve robots, then
  pre-train); combining them is future work (App. C, p. 16).
- Zero-shot performance drops on new morphologies, so sample-efficient transfer is still needed
  (App. C, p. 16); joint-angle changes hurt most (Sec. 5.3, p. 8).
- Locomotion only; multi-skill controllers are future work (App. C, p. 16).
- Sensitive to token order (C16) unless trained with order randomisation (App. B.3, p. 16).
- The authors could not explain what learned position embeddings capture beyond position (App. B.2,
  p. 16).
- Open question (reader): the training set is 100 robots from one design space with one shared
  reward; how far the zero-shot results depend on that shared reward and module vocabulary is not
  tested.
- Open question (survey lens): the recipe maps to games as one token per entity or action group,
  with a fixed-size feature vector (state plus a descriptor of what the entity is and what its
  actions do) and per-token action outputs; MetaMorph's descriptors come from the URDF/XML, so a
  game version would need such descriptors learned from frames, which this paper does not address.
- Open question (survey lens): C5 and C13 suggest that knowing what each token *is* matters more
  than graph structure; for games without a given schema, what plays the role of s_m is open.

## Related

- uses [[schulman2017proximal]]: PPO trains the policy and the averaged per-limb critic.
- uses [[vaswani2017attention]]: pre-LN transformer encoder over limb tokens, learned position embeddings.
- related [[reed2022generalist]]: Gato also controls many embodiments with one transformer, by imitation and flat tokenisation rather than per-limb tokens with PPO.
- related [[zhang2026balanced]]: also reweights task sampling in multi-task on-policy RL so that hard tasks are not starved.
- related [[serrano2024crossdomain]]: survey of cross-domain transfer in RL, the frame for this survey batch.
- extends [[kurin2021cage]]: builds on the AMORPHEUS per-limb transformer, adding morphology descriptors, learned position embeddings, exteroception and parallel PPO for 3D robots (Sec. 2, App. B.1).

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
