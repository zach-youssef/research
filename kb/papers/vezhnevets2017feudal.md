---
id: vezhnevets2017feudal
title: FeUdal Networks for Hierarchical Reinforcement Learning
authors: [Alexander Sasha Vezhnevets, Simon Osindero, Tom Schaul, Nicolas Heess, Max Jaderberg, David Silver, Koray Kavukcuoglu]
year: 2017
venue: ICML 2017 (arXiv v2 read)
url: https://arxiv.org/abs/1703.01161
arxiv: 1703.01161
doi:
pdf_sha256: edba899be9381aaba69f71467e83fe64aac5bc4a61a779a8ecbaafa270c60e48
tags: [hierarchical-rl, manager-worker, feudal, goal-conditioned, dilated-lstm, atari, a3c, montezumas-revenge, memory, long-horizon]
status: read
read: 2026-11-25
---

# FeUdal Networks for Hierarchical Reinforcement Learning

## TL;DR

A Manager LSTM emits a unit-norm goal direction in a learned latent space every step (dilated, so it
updates slowly), trained by a transition policy gradient; a Worker is rewarded for moving the latent
state that way. End-to-end A3C beats an LSTM baseline on Montezuma, several Atari games and Labyrinth.

## Problem

Long-term credit assignment and memory remain hard for end-to-end RL, especially under sparse
rewards (Montezuma's Revenge) and in non-Markovian tasks (Sec. 1, p. 1). Option-style
hierarchies tend to collapse to one option or to switching every step, and need hand-given
sub-goals (Sec. 2, p. 2). FuN wants a differentiable two-level hierarchy where sub-goals emerge.

## Method

- Shared perception z_t = f_percept(x_t) (CNN plus 256-unit FC); Manager state space s_t = f_Mspace(z_t)
  (another FC layer, size not stated); Manager recurrent net outputs a goal g_t, normalized to unit length (Eq. 1-3,
  Sec. 3, p. 3; Sec. 4, p. 4).
- Worker: its own LSTM outputs an embedding matrix U_t (one row per action, k=16 dims); the
  Manager's last c goals are summed and projected by a bias-free linear phi to w_t in R^k;
  policy = softmax(U_t w_t) (Eq. 4-6, Sec. 3.1, p. 3). With no bias, phi can never output a
  constant non-zero vector, so the goal always influences the policy (Sec. 3.1, p. 3).
- Manager update, "transition policy gradient": grad g_t = A^M_t grad d_cos(s_{t+c} - s_t, g_t),
  with A^M the Manager's advantage; goals are directions in latent space over horizon c, with no
  gradient from the Worker to the Manager (Eq. 7, Sec. 3.2, p. 3). Justified by assuming the
  transitions follow a von Mises-Fisher distribution around g_t (Sec. 3.3, Eq. 10, p. 4).
- Worker intrinsic reward: r^I_t = (1/c) sum_{i=1..c} d_cos(s_t - s_{t-i}, g_{t-i}); the Worker
  maximizes R + alpha R^I with advantage actor-critic (Eq. 8-9, Sec. 3.2, pp. 3-4). The paper
  keeps the extrinsic reward for the Worker, a softer variant of Dayan and Hinton's hidden-reward
  idea (Sec. 3.2, p. 4).
- Dilated LSTM (Manager): the state is split into r groups ("cores"); at time t only group
  t mod r is updated, and the output is pooled over the last c outputs, so the Manager sees every
  input but has long memory and low temporal resolution. r = c = 10 (Sec. 4.1, p. 5).
- Separate discounts: Manager and Worker may differ. Atari runs fix the Worker at 0.95 and use 0.99 or 0.95 for the Manager (Sec. 5.2, p. 7); Montezuma uses 0.99 for the Worker and 0.999 for the Manager (Sec. 5.1, p. 6).
- Exploration: with small probability epsilon the Manager's goal is replaced by a random
  Gaussian sample (Sec. 4, p. 4).
- Training: A3C, 32 asynchronous threads, RMSProp, BPTT unroll K=400 for FuN versus 40 for the
  LSTM baseline (100 in one variant), action repeat 4, rewards clipped to [-1, 1] (Sec. 5, p. 5, p. 6).

## Assumptions

- A1: Observations are pixels with a fixed action repeat of 4 in Atari; the "time" of both levels is counted in agent steps after repeat (Sec. 5, p. 5).
- A2: Manager goals are directions in a learned latent space, and transitions in it are approximately von Mises-Fisher around the goal, which the Worker's reward encourages (Sec. 3.3, p. 4).
- A3: The Worker is trained on intrinsic plus extrinsic reward; the Manager only on extrinsic reward (Sec. 3.2, p. 4).
- A4: Manager horizon c and dilation r are fixed at 10 for all games (Sec. 4.1, p. 5).
- A5: Results are the average episode score of the top 5 of 100 randomly sampled hyperparameter settings, training epochs of one million observations (Sec. 5, pp. 5-6).
- A6: Atari rewards are clipped to [-1, 1] (Sec. 5, p. 6).
- A7: The action set is finite and discrete, since the Worker's policy is a softmax over actions (Sec. 3, Eq. 6, p. 3).

## Claims

- C1: On Montezuma's Revenge FuN begins learning earlier and scores far higher than the LSTM baseline: LSTM needs over 300 epochs for score 400 and stagnates until about 900; FuN solves the first room in under 200 epochs and reaches up to 2600 [A1, A3, A4, A5, A6, A7] (Sec. 5.1, Fig. 2a, pp. 5-7).
- C2: Latent goals correspond to interpretable waypoints in Montezuma's first room (key and ladder), found without hand-crafting [A2, A4, A7] (Sec. 5.1, Fig. 2b, p. 5, p. 6).
- C3: With the Manager discount at 0.99, FuN clearly beats LSTM on Ms. Pacman, Amidar, Gravitar and Enduro (where LSTMs fail), and with Manager discount 0.95 beats the rest by a factor of 7 on Frostbite [A1, A4, A5, A6, A7] (Sec. 5.2, Fig. 4, p. 6, p. 7).
- C4: On Seaquest, Breakout, Alien, Hero and Space Invaders FuN is not better: LSTM with discount 0.95 is best on Seaquest and Breakout, on Alien LSTM beats FuN, and on Hero and Space Invaders all agents are about equal [A1, A4, A5, A6, A7] (Sec. 5.2, Fig. 4, p. 6, p. 7).
- C5: An LSTM baseline with BPTT unroll 100 is much worse than with 40, while FuN trains well at BPTT 400 thanks to the dilated LSTM [A1, A5, A7] (Sec. 5.2, Fig. 4, p. 7).
- C6: In 3D Labyrinth memory tasks (water maze, T-maze, T-maze+, non-match) FuN learns faster and reaches higher reward than LSTM; in the water maze the LSTM keeps circling at a fixed radius [A4, A5, A7] (Sec. 5.3, Fig. 5, Fig. 6, pp. 7-8).
- C7: After 200 epochs FuN matches the published Option-Critic scores on Seaquest and beats them on the other shared games: doubling on Ms. Pacman, more than tripling on Zaxxon and over 20x on Asterix [A1, A4, A5, A7] (Sec. 5.2, Fig. 7, p. 7, p. 8).
- C8: Ablations: the non-feudal FuN (Manager trained by Worker gradients, no intrinsic reward), Manager-via-policy-gradient, absolute goals, and pure feudal (Worker without extrinsic reward) are each worse than full FuN on Ms. Pacman, Amidar, Gravitar and Enduro [A1, A4, A5, A7] (Sec. 5.4, Fig. 8, p. 9).
- C9: A regular (non-dilated) LSTM in the Manager fails catastrophically, and a prediction horizon c=1 hurts less but still hurts [A1, A4, A5, A7] (Sec. 5.4, Fig. 10, p. 9, p. 10).
- C10: Score correlates with high alpha (the intrinsic weight) on Gravitar and Amidar, while on other games the best alpha is below 1; learning rate and entropy are fixed at 1e-3.5 and training is 200 epochs [A1, A5, A7] (Sec. 5.4, Fig. 11, p. 9, p. 10).
- C11: A dilated-LSTM agent without the rest of FuN generally underperforms both LSTM and FuN [A1, A5, A7] (Sec. 5.4, Fig. 12, p. 10, p. 11).
- C12: Initializing from a model trained with action repeat 4 and then removing the repeat (discounts adjusted, dilation and c multiplied by 4) beats FuN and LSTM trained from scratch, with positive transfer in each of Frostbite, Ms. Pacman, Amidar and Space Invaders, while the transferred LSTM only helps on Ms. Pacman [A1, A4, A7] (Sec. 5.5, Fig. 9, p. 9, p. 10).
- C13: Sub-policies emerge: fixing a random goal makes the Seaquest agent concentrate on distinct screen regions, one swimming up for oxygen [A2, A4, A7] (Sec. 5.2, Fig. 3, p. 6, p. 7).

## Implementation notes

- Dimensions: CNN 16 8x8 stride 4 then 32 4x4 stride 2 filters, FC 256, all rectified; Worker and Manager LSTMs 256 units, baseline LSTM 316 units, k=16 (source: Sec. 4, p. 4; Sec. 5, p. 5).
- Hyperparameter search: 100 runs with random learning rate, entropy penalty and alpha; learning rate LogUniform(1e-4.5, 1e-3.5) for FuN, (1e-4, 1e-3) for LSTM; alpha Uniform(0, 1) (source: Sec. 5, p. 6; Sec. 5.2, p. 7).
- The random-goal exploration probability epsilon is not given (source: assumption, none stated in Sec. 4, p. 4).
- Value critics: separate value estimates for Manager and Worker, with possibly different gamma (source: Sec. 3.2, p. 4).
- The goal gradient treats s_t as a constant when computing grad d_cos (source: Eq. 7 note, Sec. 3.2, p. 3).

## Limitations / open questions

- Deeper hierarchies (goals at multiple time scales) and transfer/multitask use are left as future work (Sec. 6, pp. 10-11).
- FuN is not better on several games, and the best discount differs per game (Sec. 5.2, p. 7).
- Results are the best 5 of 100 random hyperparameter runs per method, a selection that favors high-variance methods (reader's reading of Sec. 5, p. 5).
- Open question (reader): the Manager's goals are fixed-clock (c=10 agent steps, 40 frames); a shmup strategist (weapon, shop, rank) acts over minutes, 3 to 4 orders of magnitude longer, so a cascade of levels or event-triggered goals would be needed.
- Open question (reader): the dilated LSTM is a ready-made low-rate memory for the high level. Could the memory-token approaches ([[bulatov2022recurrent]], [[cherepanov2026recurrent]]) replace it in a transformer?
- Open question (reader): the Manager's directions are in a self-learned latent space without any grounding in game semantics; the shop or rank would have to appear in that latent, unlike Director's decodable goals ([[hafner2022director]]).

## Related

- related [[hafner2022director]]: a later two-level design that cites FuN, with goals decoded to images and a worker trained on goal reward alone.
- related [[chen2021decision]]: Decision Transformer conditions on return-to-go; FuN conditions on a learned goal direction.
- related [[lee2022multigame]]: Atari multi-game sequence model without any hierarchy.
- related [[bulatov2022recurrent]]: recurrent memory for long sequences, an alternative to the dilated LSTM.
- related [[parisotto2020stabilizing]]: a gated transformer as RL memory, a candidate replacement for the dilated LSTM (see the open question above).
- related [[ni2023decoupling]]: separates memory length from credit-assignment length, the two problems the Manager's slow clock targets.
- related [[magne2026nitrogen]]: flat frames-to-gamepad policy that a Manager could condition.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
