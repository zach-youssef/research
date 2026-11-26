---
id: kujanpaa2023hierarchical
title: Hierarchical Imitation Learning with Vector Quantized Models
authors: [Kalle Kujanpää, Joni Pajarinen, Alexander Ilin]
year: 2023
venue: ICML 2023 (read the arXiv v2 version)
url: https://arxiv.org/abs/2301.12962
arxiv: 2301.12962
doi:
pdf_sha256: 9cfffbc63677e35e637b0415ef448cf70adced772f2e6cf571b63896922bb80c
tags: [hierarchical-imitation-learning, subgoals, vq-vae, discrete-codes, planning, search, reinforcement-learning, offline]
status: read
read: 2026-11-25
---

# Hierarchical Imitation Learning with Vector Quantized Models

## TL;DR

HIPS segments expert trajectories into variable-length subgoal segments with a REINFORCE detector (action
log-likelihood minus a per-subgoal penalty), fits a VQ-VAE over the subgoals, and plans over its codes with
PHS*/A*/MCTS. On 4 discrete puzzle tasks (no Atari) it beats BC, CQL, DT and Option-GAIL, e.g. 97.5% on Sokoban.

## Problem

Learning both low- and high-level planning models from demonstrations is hard, especially from
high-dimensional inputs; prior work needs a fixed number of subgoals, fixed-length subtasks, or one
model per subtask length (Sec. 1, p. 1; Sec. 2, p. 2-3). The paper wants one model that
handles variable-length subtasks, in a fully observed, discrete-state, sparse-reward (1 on solving)
setting, offline, with no environment access before evaluation (Sec. 3, p. 3).

## Method

The agent (HIPS) has three trained parts plus a detector used only in training (Fig. 1, p. 3):
- **Detector d_xi(s_g_{k+1} | s_g_k)** (Sec. 3.1, p. 4): an RL segmentation policy. Each episode is one
  demonstration; at each step it picks the next subgoal among the next H states of the trajectory.
  Reward R_k = sum of log pi_theta(a_i | s_i, s_g_k) over the segment minus a penalty alpha per
  segment (Eq. 1, p. 4). Trained with REINFORCE and a learned value baseline (Alg. 1, p. 4).
- **Low-level policy pi_theta(a | s, s_g)**: goal-conditioned behaviour cloning on the segments the
  detector produces (Eq. 2, p. 4), trained concurrently with the detector. A goal state is reached by
  rolling the policy; there is no learned termination head: a subgoal ends when it is reached, with an
  exact match required for discrete states (Sec. 3.3, p. 5).
- **VQ-VAE over subgoals** (Sec. 3.2, Eq. 3, p. 4-5; Alg. 2, App. I, p. 20): encoder gets (s_g_{k+1}, s_g_k),
  a code from a finite codebook (K = 32-64) is picked by nearest neighbour, the decoder g(e_k, s) rebuilds
  the subgoal state from the code and the current state. Two stages: pre-train without quantisation on random
  state pairs j in (i, i+H]; then train with quantisation on detector-produced consecutive subgoal pairs,
  with the codebook initialised by KMeans++ on encoder outputs. A prior p(e_{k+1} | s_g_k) over codes is learned.
- **Planning** (Sec. 3.3, p. 5): each search node is a subgoal; its K children are the K decoded codes.
  Unreachable children (low-level policy, rolled with the true or a learned dynamics model, doesn't hit the
  subgoal within a step limit) and duplicates are dropped. A value function V(s) predicts steps to the
  terminal state. Searches used: GBFS, PHS*, A*, MCTS (App. H, p. 18-19). The low-level policy then executes the plan.
- HIPS-env uses the real simulator for reachability; HIPS-k replaces the detector with fixed-interval
  subgoals (App. G, p. 17).

## Assumptions

- A1: Environment states are fully observed, Markovian and discrete (Sec. 3, p. 3).
- A2: The setting is offline imitation: only expert trajectories with a terminal reward of 1 are available, no environment interaction before evaluation (Sec. 3, p. 3).
- A3: The task is goal-oriented: reaching a terminal state, sparse reward (Sec. 3, p. 3).
- A4: Subgoals are themselves states of the trajectory, so a subgoal is a full state (Sec. 3.1, p. 4).
- A5: A single exact-match test decides that a discrete subgoal was reached; a threshold would be needed for continuous states (Sec. 3.3, p. 5).
- A6: A dynamics model (true or learned) is available at planning time to simulate the low-level policy for reachability checks (Sec. 3.3, p. 5).
- A7: Evaluation uses open-loop planning: one plan is generated, executed and scored (Sec. 4.3, p. 7).

## Claims

- C1: On Sokoban, sliding tile puzzle (STP), Box-World (BW) and grid TSP, HIPS reaches 97.5, 94.7, 55.7 and 100.0 percent success, above BC (18.7, 82.5, 41.1, 28.8), CQL, DT, RIS, Option-GAIL and IQ-Learn on all four [A1, A2, A3, A4, A5, A6, A7] (Table 1, p. 6).
- C2: With the true simulator for planning, HIPS-env reaches 98.1, 94.6, 99.6, 100.0 percent, above AdaSubS (91.4, 0.0, -, 22.4) and kSubS (90.5, 93.3, -, 87.9) on Sokoban, STP and TSP [A1, A2, A3, A4, A5, A6, A7] (Table 1, p. 6, Table 2, p. 7).
- C3: HIPS's detector beats fixed-interval subgoals (HIPS-k) in Sokoban, STP and TSP at small search budgets (after 100 expansions in Sokoban: 88.5 vs 77.7); Box-World is the extreme case at 55.7 vs 20.1 [A1, A2, A3, A4, A5, A6, A7] (Table 2, p. 7, Table 1, p. 6).
- C4: Variable-length segments matter: with fixed k no value outperformed the detector variant (HIPS-GBFS), and too large a k hurts the final solve rate while speeding up early solving (STP, Table 10) [A1, A2, A3, A4, A5, A6, A7] (App. G, Table 10, p. 17).
- C5: In TSP the learned subgoals improve on the data: HIPS-A* needs 168.2 average steps vs the teacher's 336.5, with 100 percent success, versus Christofides at 139.0 and an oracle MCTS with procedural subgoals at 199.9; BC, CQL and AdaSubS stay near 339 [A1, A2, A3, A4, A5, A6, A7] (Table 3, p. 8).
- C6: Trained on 4-box Sokoban levels, HIPS solves 97.1 percent of 5-box levels without fine-tuning (97.5 at 4 boxes), though it needs more expansions (91.5 vs 94.7 at 500) [A1, A2, A3, A4, A5, A6, A7] (App. E, Table 5, p. 14).
- C7: Both the value function and the VQ-VAE prior help in Sokoban, but one of them suffices at a larger budget: after 100 expansions HIPS-PHS* 88.5, GBFS (no prior) 78.4, LevinTS (no value) 78.6; at 1000, 95.9, 89.7, 95.6 [A1, A2, A3, A4, A5, A6, A7] (App. D, Table 4, p. 14).
- C8: Subgoal lengths found by the detector are spread over short distances: histograms of the steps the policy needs to reach a proposed subgoal run from about 1 up to about 25 (Sokoban), 20 (STP), 30 (BW) and 50 (TSP), with most mass below 10 in the first three [A1, A2, A3, A4, A5] (Fig. 5, p. 17).
- C9: HIPS is limited by the generator's coverage and by the learned dynamics model: in Box-World HIPS (55.7) is far below HIPS-env (99.6), which the authors attribute to search exploiting dynamics-model inaccuracies [A1, A2, A3, A4, A5, A6, A7] (Sec. 4.3, p. 7, Table 1, p. 6).

## Implementation notes

- Hyperparameters: learning rates 2e-4 (dynamics, VQ-VAE), 1e-3 (policy, detector, value); REINFORCE discount 0.99 (source: Table 11, p. 20).
- Per environment (Sokoban, STP, BW / TSP): subgoal penalty alpha 0.1 / 0.05; VQ beta 0.1 / 0; codebook dim 128 / 64; horizon H 10 / 50; codebook size K 64 / 32; fixed-k ablation 5 / 4 (source: Table 12, p. 20).
- Networks are ResNet-based CNNs; Box-World adds Deep Recurrent Convolutions; the decoder and policy use FiLM layers (source: Sec. 4.2, p. 6).
- Training data: Sokoban 10,340 trajectories (10,240 solved by Curry, 100 random); STP 5,100; BW 22,100 (source: Sec. 4.1, p. 6). Only the 100 random trajectories train the dynamics model.
- Seeds: 10 for HIPS, at least 5 per ablation, at least 3 per baseline (source: Sec. 4.3, p. 7).
- Hardware: one V100 and 6 CPU workers per run; feasible on one GTX 1080 Ti with fewer workers (source: App. B, p. 13).
- Authors' code: not linked in the PDF; the paper cites open-source baselines (d3rlpy for CQL) (source: Sec. 4.2, p. 6-7).
- Mapping to frames-in/gamepad-out, reader's guess: a subgoal would be a learned latent, not a decoded frame; the exact-match reachability test would need a threshold, and 'lasting until reached' is not reactive (source: assumption).

## Limitations / open questions

- Stated future work: model uncertainty, discrete high-level with continuous low-level control, more abstract goals than observation-space states, ignoring task-irrelevant input, curricula (Sec. 5, p. 9).
- The authors tried and dropped: straight-through and I-MLE detector training, a non-goal-conditioned low-level policy, an autoregressive or GAN generator, and skipping reachability checks (App. C, p. 13).
- Reachability must be checked during planning because the decoder emits K candidates in every state (App. C, p. 13).
- Open question (reader): the skill lens. A code encodes a *next subgoal state* (conditioned on the current state), not a behaviour; duration is the detector's variable segment (mostly under 10-50 steps), and termination is "goal reached". No Atari, no transfer across tasks beyond 4 to 5 boxes.
- Open question: with pixels and real-time play, there is no exact state match, no search over a simulator, and no sparse terminal reward; which of these survive a shmup?

## Related

- related [[chen2021decision]]: Decision Transformer baseline (DT scores 36.7 on Sokoban, 0 on STP/TSP, Table 1, p. 6).
- related [[schmidt2024actions]]: LAPO, a VQ latent-action bottleneck; codes there stand for actions, here for subgoal states.
- related [[ye2025latent]]: LAPA, VQ latent actions used for policy pretraining.
- related [[gao2025adaworld]]: another discrete/latent abstraction learned from data, but for actions rather than subgoals.
- related [[hafner2022director]]: Director picks subgoals at a slower timescale.
- related [[vezhnevets2017feudal]]: Feudal networks; the manager sets goal directions at a slower timescale.
- related [[lynch2019play]]: Play-LMP picks latent plans at a slower timescale.
- related [[correia2023hierarchical]]: HDT, a Decision Transformer conditioned on sub-goal states; HIPS also conditions its low-level policy on subgoals.
- related [[bacon2017option]]: Option-Critic; the paper names options learned end-to-end as the alternative it does not use (Sec. 2, p. 2).

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
