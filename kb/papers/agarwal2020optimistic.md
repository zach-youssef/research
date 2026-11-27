---
id: agarwal2020optimistic
title: An Optimistic Perspective on Offline Reinforcement Learning
authors: [Rishabh Agarwal, Dale Schuurmans, Mohammad Norouzi]
year: 2020
venue: ICML 2020 (arXiv v4 read)
url: https://arxiv.org/abs/1907.04543
arxiv: 1907.04543
doi:
pdf_sha256: 7723884091707391a1f3aa15439c4941ecfb19acd61583b20885395a88b5d49d
tags: [offline-rl, atari, dqn-replay, dataset, rem, benchmark]
status: read
read: 2026-11-27
---

# An Optimistic Perspective on Offline Reinforcement Learning

## TL;DR

Releases the DQN Replay Dataset (5 DQN runs on each of 60 Atari games, 200M frames each) and
shows offline QR-DQN and the new REM (random convex mixture of Q-heads) beat the fully trained
online DQN on it. Table 2 fixes the Atari settings: sticky 0.25, skip 4, 84x84 grey, 108K cap.

## Problem

Recent work says standard off-policy deep RL diverges or does poorly when trained only on a
fixed dataset (Sec. 1, p. 1). The paper tests this on a large, diverse logged dataset and
provides a standardised offline benchmark that avoids replay-buffer and exploration design
choices (Sec. 1, Sec. 3, pp. 1-3).

## Method

- DQN Replay Dataset: 5 DQN (Nature) agents with random initialisation per game, 200M frames
  each, every (observation, action, reward, next observation) tuple stored; 5 datasets per
  game, 300 in total (Sec. 5, pp. 4-5).
- Offline agents train with no environment interaction on the fixed replay, with Dopamine
  hyperparameters; evaluation is online, once per training iteration (Sec. 5, p. 5, App. A.4,
  p. 13).
- Ensemble-DQN: K Q-heads, each with its own target (Eq. 6, Sec. 4.1, p. 3).
- REM: per mini-batch draw alpha from a categorical distribution on the (K-1)-simplex (K i.i.d.
  Uniform(0,1) values, normalised), train the alpha-mixture of heads against its own mixture
  target; act on the mean of the K heads (Eq. 7, Sec. 4.2, p. 4). K = 200 heads in the offline
  Atari experiments (Sec. 5.2, p. 6, Table 2, p. 14).
- Proposition 1: under assumptions (a)-(d), all heads converge to the same Q-function and it is
  Q* (Sec. 4.2, p. 4, App. A.1, p. 12).
- Score normalisation relative to online DQN and random (Eq. 10, App. A.3, p. 13).

## Assumptions

- A1: The environments are ALE Atari 2600 games with sticky actions (probability 0.25),
  frame skip 4 and 60 games (Sec. 5, p. 4, App. A.4, p. 13).
- A2: Offline data are the full training trajectories of 5 DQN (Nature) agents per game, so they
  mix all intermediate policies (Sec. 5, p. 5).
- A3: Offline agents are scored by online evaluation during training, taking the best
  evaluation score averaged over 5 runs (App. A.4, p. 13).
- A4: Agents use the Dopamine baseline hyperparameters and Table 2 settings (Sec. 5, p. 5,
  Table 2, p. 14).
- A5: Scores are normalised between online DQN (Nature) and the random agent (App. A.3, p. 13).
- A6: Proposition 1 needs P_Delta with full support, finitely many global minimisers, Q* defined
  on the data-induced MDP, and Q* in the function class (Prop. 1, p. 4).

## Claims

- C1: The DQN Replay Dataset has 5 independent DQN (Nature) runs per game, 60 games, 200M frames (50M agent-environment steps, because of frame skip 4) each, giving 300 datasets and about 15 billion tuples; each of the 300 datasets (one run, about 50M tuples) is about 3.5 times ImageNet [A1, A2] (Sec. 5, pp. 4-5, App. A.4, p. 13).
- C2: Tuples are stored in the order encountered during training, so the first k million frames are a low-quality dataset and the last k near-expert data; subsets can be made by taking trajectories at random [A2] (Sec. 8, p. 8, Sec. 6, p. 7).
- C3: The Atari settings for both online and offline agents are: sticky actions on with probability 0.25, grey-scaling, 84x84 down-sampling, 4 stacked frames, frame skip 4, reward clipping to [-1, 1], terminal condition Game Over, 108K maximum frames per episode, discount 0.99 [A1, A4] (Table 2, p. 14, Sec. 5, p. 4).
- C4: Evaluation is online every training iteration with epsilon-greedy epsilon = 0.001, 125K evaluation steps per iteration; an iteration is 250K training steps (1M frames) with an update every 4 steps; online agents train 200 iterations, offline 200 or 1000 [A3, A4] (Table 2, p. 14, App. A.4, p. 13, Fig. 1, p. 2).
- C5: The reported score per agent is the best evaluation score reached during training, averaged over 5 runs, with 5 seeds for data collection, simulator and network initialisation [A3] (App. A.4, p. 13).
- C6: Normalised score is 100 x (Score_normalized - 1) improvement over DQN, with Score_normalized = (Agent - min)/(max - min), min = min(DQN, Random), max = max(DQN, Random), all means over 5 runs; the max/min is needed since DQN is worse than random on Solaris and Skiing [A5] (App. A.3, p. 13, Eq. 10).
- C7: Offline DQN (Nature) trained for the same number of gradient steps as the online run underperforms fully trained online DQN on most games; offline QR-DQN outperforms offline DQN and online DQN on most games, and offline QR-DQN beats offline C51 [A1, A2, A3, A4, A5] (Sec. 5.1, p. 5, Fig. 3, Fig. 4, p. 5).
- C8: With 5x the gradient steps, median normalised scores (60 games, 5 runs) and games better than online DQN are: DQN (Nature) 83.4%/17, DQN (Adam) 111.9%/41, Ensemble-DQN 111.0%/39, Averaged Ensemble-DQN 112.1%/43, QR-DQN 118.9%/45, REM 123.8%/49 [A1, A2, A3, A4, A5] (Table 1, p. 6, Sec. 5.2, p. 6).
- C9: At the same gradient steps as online DQN (1x), the same table reads DQN (Nature) 74.4%/10, DQN (Adam) 104.6%/39, Ensemble-DQN 92.5%/26, Averaged Ensemble-DQN 88.6%/24, QR-DQN 115.0%/44, REM 103.7%/35 [A1, A2, A3, A4, A5] (Table 3, p. 14).
- C10: Offline agents run about 3 times faster per gradient step than online DQN on a P100, so 5x offline steps are 1.67x slower than a 200M-frame online run [] (App. A.4, p. 13).
- C11: Offline REM and QR-DQN trained on random subsets of the data recover online DQN on most of the 5 test games with at least 10% of the data, and degrade sharply at 1% [A1, A2, A4] (Sec. 6, Fig. 6, p. 7).
- C12: Offline REM and QR-DQN trained on only the first 20M frames of the dataset still exceed the best policy in that subset on most games [A1, A2, A4] (Sec. 6, Fig. 7, p. 7, Fig. A.7, p. 20).
- C13: Behaviour cloning performs poorly on the DQN Replay Dataset because of its diverse composition [A2] (Sec. 6, p. 7).
- C14: Online REM (4 separate Q-networks) performs on par with online QR-DQN and above Bootstrapped-DQN on 60 games over 200M frames [A1, A4] (Sec. 5.3, Fig. 5, pp. 6-7).
- C15: REM with separate Q-networks does better than the multi-head REM for K = 4 and K = 16 on 6 games over 500 offline iterations, which the paper reads as more diverse Q-estimates helping [A1, A4] (Sec. 5.2, p. 6, Fig. A.3, p. 16).
- C16: Offline TD3, trained on 1M DDPG transitions for 3 MuJoCo tasks, significantly outperforms both the DDPG behaviour policy and offline DDPG and performs comparably to BCQ; the paper uses this to reconcile with the final-buffer failures of Fujimoto et al. [] (Sec. 6, Fig. 8, pp. 7-8, App. A.2, p. 12).
- C17: Under assumptions (a)-(d), the REM loss at its global minimum gives identical Q-heads and the common solution is Q* [A6] (Prop. 1, p. 4, App. A.1, p. 12).
- C18: The dataset and code are released: dataset at offline-rl.github.io, code at github.com/google-research/batch_rl [] (Abstract, p. 1, Sec. 1, footnote 1, p. 1).

## Implementation notes

- Not stated in the paper: max-pooling over the last two frames, no-op starts, the action set
  (full 18 or minimal), whether life loss ends a training episode (Table 2 says terminal is
  Game Over), and the list of 60 games (it appears only as the per-game plot labels). These
  follow Dopamine's Atari defaults (source: assumption; Castro et al. 2018, not read).
- Dataset file layout, compression, licence and the per-game checkpoint structure are not
  described; a user must read offline-rl.github.io (source: assumption).
- Offline training uses Dopamine hyperparameters: mini-batch 32, target update every 2000
  updates, discount 0.99, Q-network conv 32/64/64 with filters 8x8, 4x4, 3x3 and strides 4, 2, 1,
  512 hidden units, Adam with lr 0.00005 and epsilon 0.01/32 for DQN (Adam) and the multi-head variants
  (source: Table 2, p. 14, App. A.4, p. 13).
- Online agents: replay 1M steps, minimum replay 20,000, training epsilon 0.01 with decay over
  250K steps, uniform replay; offline replay is fixed at 50M steps (source: Table 2, p. 14).
- Online REM tuned epsilon_Adam over {0.01/32, 0.005/32, 0.001/32} on 5 games; the best (lr
  0.00005, 0.001/32) was used on all 60 (source: App. A.4, p. 13).
- Evaluating the best iteration over training is a maximum-based protocol; for a held-out
  comparison use final-iteration scores or note the difference (source: assumption, see
  [[agarwal2021deep#C5]]).
- Evaluation uses epsilon 0.001, not a greedy policy (source: Table 2, p. 14).

## Limitations / open questions

- Offline agents overfit: after some number of gradient steps their performance deteriorates, so
  the paper uses online-evaluation early stopping, which "true" offline RL cannot (Sec. 8, p. 8).
- Divergence of the Huber loss appears with 1% of the data (Sec. 8, p. 8).
- The REM-over-ensemble gain is a hypothesis (noise-based robustness, like dropout) and
  randomised ensembles against minimum Q-targets need further study (Sec. 5.2, p. 6).
- Open question: scores use the best iteration averaged over 5 runs, so reported numbers may be
  optimistically biased relative to final-iteration scores.
- Open question: DQN Replay holds one algorithm's trajectories; does a model trained on it
  transfer to evaluations with different sticky/no-op settings (as in [[lee2022multigame]])?

## Related

- uses [[machado2018revisiting]]: sticky actions (p = 0.25) for dataset collection and evaluation (Sec. 5, p. 4).
- related [[agarwal2021deep]]: same first author; its normalisation and statistics are the later standard.
- related [[lee2022multigame]]: trains on this dataset (41 games, 5 held out).
- related [[cheng2025jointly]]: pretrains on 15 games of this dataset and fine-tunes on 5 held-out ones.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->

