---
id: hubert2021learning
title: Learning and Planning in Complex Action Spaces
authors: [Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Mohammadamin Barekatain, Simon Schmitt, David Silver]
year: 2021
venue: ICML 2021 (arXiv v1 read)
url: https://arxiv.org/abs/2104.06303
arxiv: 2104.06303
doi:
pdf_sha256: 220bff7dbc71e157f56f38627f4567b5221425ee5c3541a41ad337a06f7efc63
tags: [model-based-rl, muzero, sampled-muzero, mcts, planning, continuous-control, large-action-space, policy-iteration, dm-control]
status: read
read: 2026-11-26
---

# Learning and Planning in Complex Action Spaces

## TL;DR

Sampled MuZero runs MuZero's tree search over K actions sampled from the policy instead of all
actions, correcting for the sampling with a beta-hat/beta prior in PUCT. It matches full-action
MuZero in Go with K=50 of 362 moves, in Ms. Pac-Man with K=3 of 18, and learns DM Control tasks
up to 56-D actions, from states or pixels, beating DMPO/D4PG on most tasks and Dreamer on all 20.

## Problem

MuZero's search enumerates every action at each node, so it only works for small action sets
(Sec. 1, p. 1). Sampling a subset of actions is the obvious fix, but the paper argues one has to
understand how the sampling interacts with policy improvement and evaluation (Sec. 1, p. 1).
Earlier AlphaZero/MuZero extensions to continuous actions were limited to very low dimensions
(Sec. 2, p. 2).

## Method

- Policy iteration as operators (Sec. 4.1-4.2, p. 4): improvement = an improvement operator I
  followed by projection P. An operator is "action-independent" if I-pi(a|s) = f(a, Z(s)) with a
  state-dependent normaliser Z; PPO, MPO and AWR operators are of this form.
- Sample-based operator (Sec. 4.3, p. 4): with K actions drawn from a proposal beta, the empirical
  distribution is beta-hat; the sample-based improved policy is (beta-hat/beta)(a|s) f(a, s, Z-hat_beta(s)).
- Theorem (Sec. 4.4, p. 4): expectations under I-pi are the K to infinity limit of expectations under
  the sample-based policy, approximately Normal with variance sigma^2/K (proof App. E, p. 17).
- Sampled MuZero (Sec. 5, pp. 5-6): in PUCT, replace the prior pi(s,a) by (beta-hat/beta)*pi (s,a)
  (full formula App. D, p. 17); the visit counts then approximate the sample-based improved policy
  (Theorem in Sec. 5.1, pp. 5-6; justification via Grill et al. in App. F, pp. 17-18). The rest of
  MuZero is unchanged: n-step value targets, actor acts by visit-count sampling, policy network
  trained by KL to visit counts (Sec. 5.2-5.3, p. 6).
- Node expansion samples K << N actions from beta and returns their pi and beta (Sec. 5.4, p. 6);
  beta = pi, optionally with a temperature, plus Dirichlet noise at the root (Sec. 5.5, p. 6).
- Control networks (App. A, p. 12): fully-connected residual tower in place of convolutions (10
  blocks, 512 units), factored categorical policy with 7 bins per action dimension (a Gaussian head
  also works, Fig. 9, p. 14), 51-bin value and reward heads, n=5, discount 0.99, K=20, 50 simulations.
  RWRL adds an LSTM in the representation function.

## Assumptions

- A1: Actions are sampled from a proposal beta with beta = pi (or a tempered pi) at every expanded node (Sec. 5.5, p. 6).
- A2: The transition model used in search is deterministic (Sec. 2, p. 2).
- A3: The policy improvement operator is action-independent (Sec. 4.2, p. 4).
- A4: A simulator exists to generate training experience; the learned model is used only in search (Sec. 5, p. 5; Sec. 6, p. 6).
- A5: Control experiments use 50 simulations per move and K=20 samples unless a K sweep is stated (App. A, p. 12).
- A6: The theory is asymptotic in the number of samples K (Sec. 4.4, p. 4).

## Claims

- C1: As K grows, the sample-based improved policy converges in distribution to the true improvement operator, with variance of order sigma^2/K [A3, A6] (Sec. 4.4, p. 4, App. E, p. 17).
- C2: Searching with prior (beta-hat/beta)*pi gives visit counts approximately equal to the sample-based improved policy, so only the PUCT prior needs changing in MuZero [A1, A3] (Sec. 5.1, pp. 5-6, App. F, pp. 17-18).
- C3: In Go (362 actions), K=50 samples already closely approach the baseline that searches all actions; K=15 is clearly worse; 1 seed per setting [A1, A4] (Fig. 2, p. 7, Sec. 6.1, p. 7).
- C4: In Ms. Pac-Man (18 actions), K=2 is not sufficient for efficient learning but K=3 rapidly approaches the all-actions baseline (200M frames; 1 seed) [A1, A4] (Fig. 3, p. 7, Sec. 6.2, p. 7).
- C5: On DM Control Suite hard and manipulator tasks Sampled MuZero from states (3 seeds, 25M frames) beats DMPO and D4PG on most tasks, with large gaps on manipulator.bring_ball, bring_peg, insert_ball and insert_peg; on humanoid.walk and stand all three are about equal [A1, A4, A5] (Fig. 4, p. 8).
- C6: On the 20-task suite compared with Dreamer at 5M frames, from pixels and with no action repeat, no observation reconstruction and no hyperparameter re-tuning, Sampled MuZero equals or beats Dreamer on all tasks (all 20 table entries are numerically higher for Sampled MuZero, some by under 1%) [A1, A4, A5] (Table 2, p. 15, App. A.2, p. 15).
- C7: Learning from raw pixels works on the hard and manipulator tasks with the same budget, but is slower than from state on most (e.g. acrobot.swingup_sparse, humanoid.run, humanoid.walk) and about equal at 25M frames on bring_ball and insert_peg [A1, A4, A5] (Fig. 5, p. 8).
- C8: On 56-D CMU humanoid locomotion tasks (forage, go-to-target, run-gaps, run-walls) it exceeds previously reported results using more than an order of magnitude fewer environment interactions (1 seed) [A1, A4] (Fig. 6, p. 8).
- C9: On Real-World RL (cartpole, walker, quadruped, humanoid; easy/medium/hard, LSTM added) it beats DMPO, D4PG and STACX in most cells, e.g. humanoid easy 289.36 vs 102.92 (D4PG), but on humanoid hard all methods stay near 1.2-1.4 and STACX wins quadruped hard (351.56 vs 348.09) [A1, A4] (Table 1, p. 9).
- C10: On humanoid.run (21-D) K=3 already learns a good policy and K > 10 gives no visible gain (6 seeds); K=3 is clearly slower than K>=5 [A1, A4, A5] (Fig. 10, p. 15, App. A.3, p. 15).
- C11: Using (beta-hat/beta)*pi instead of plain pi in PUCT is much more robust at K=5 and K=20 on humanoid.run, and evaluating the Q of sampled actions at the root before search reduces but does not remove the gap [A1, A4] (Fig. 13, p. 16, App. A.5, p. 15).
- C12: Seed-to-seed variation is minimal from state and pixel inputs on the control tasks [A1, A4] (Fig. 11, p. 16, Fig. 12, p. 16).

## Implementation notes

- Adam with decoupled weight decay 2e-5, batch 1024, lr 1e-4 cosine-decayed to 0 over 1M batches (source: App. A, p. 12).
- Replay keeps the last 2000 sequences, split into length-500 subsequences, prioritised replay as in MuZero (source: App. A, p. 12).
- LSTM for RWRL trained with truncated BPTT of 8 steps from stored states, 4 stacked observations (effective unroll 32) (source: App. A, p. 12).
- Before the search, the Q of all sampled root actions is evaluated to initialise Q(s,a) in PUCT (source: App. A, p. 12).
- Acme baselines were run with bigger networks (policy 512,512,256,128; critic 1024,1024,512,256) and batch 1024 (source: App. A, p. 12).
- Go uses n=25-step bootstrapping with averaged target values of 4 consecutive states, 400 simulations; Atari uses MuZero's settings (source: App. B-C, pp. 16-17).
- c1 = 1.25, c2 = 19652 as in MuZero (source: App. D, p. 17).
- The temperature tau for beta = pi^(1/tau) is not given for the experiments (source: assumption: tau = 1).

## Limitations / open questions

- Many results use 1 seed (Go, Ms. Pac-Man, CMU humanoid); the control suites use 3 (Figs. 2, 3, 6 captions, pp. 7-8).
- Sampling is of actions only; transition models are deterministic and state sampling is not combined (Sec. 2, p. 2).
- Progressive widening is mentioned as a possible way to sample more actions at hot nodes but not tried (Sec. 5.4, p. 6).
- The PDF read is arXiv v1, marked "Under submission, do not distribute" (p. 1); the ICML camera-ready may differ.
- Open question: a shmup action is a small joystick plus buttons (a few hundred combinations at most), so full enumeration is feasible and K-sampling is not needed for the discrete case; its value would be a factored policy over buttons, which this paper's factored head (App. A, p. 12) partly addresses.
- Open question: beta = pi is the policy network. A behaviour-cloned prior could serve as beta (and as pi), but the paper never tests a supervised prior or a mismatch between beta and the acted policy beyond the temperature remark (Sec. 5.5, p. 6).

## Related

- extends [[schrittwieser2020mastering]]: MuZero with search over sampled actions.
- related [[hafner2024mastering]]: DreamerV3; Sampled MuZero is compared with the earlier Dreamer on 20 control tasks (Table 2, p. 15).
- related [[schulman2017proximal]]: PPO's improvement operator is expressed as an action-independent operator (Sec. 4.1, p. 4).

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
