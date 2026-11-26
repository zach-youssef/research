---
id: ye2021mastering
title: Mastering Atari Games with Limited Data
authors: [Weirui Ye, Shaohuai Liu, Thanard Kurutach, Pieter Abbeel, Yang Gao]
year: 2021
venue: NeurIPS 2021 (arXiv v2 read)
url: https://arxiv.org/abs/2111.00210
arxiv: 2111.00210
doi:
pdf_sha256: 057c3466e3aabec6d9a7dc6ac597b43caf546d73ea50241263f7113fdc416a69
tags: [model-based-rl, muzero, efficientzero, mcts, sample-efficiency, atari, atari-100k, self-supervised, consistency-loss, value-prefix]
status: read
read: 2026-11-26
---

# Mastering Atari Games with Limited Data

## TL;DR

EfficientZero adds three changes to MuZero Reanalyze: a SimSiam-style consistency loss between
predicted and encoded next latent states, an LSTM "value prefix" in place of per-step rewards, and
model-based off-policy value correction. On Atari 100k (2 h of play) it gets 194.3% mean and 109.0%
median human-normalized score, with a 4-GPU, 7 h open-source implementation.

## Problem

MuZero-type search needs a lot of data; on the 100k-step Atari benchmark prior methods stayed below
human level (Sec. 1, p. 2). The authors name three obstacles under limited data (Sec. 4, pp. 4-5):
the learned model gets only scalar, sparse or noisy supervision (reward, value, policy); reward
prediction suffers from aleatoric uncertainty and state aliasing over long rollouts; and multi-step
value targets from old trajectories are off-policy.

## Method

- Base: MuZero Reanalyze (Sec. 3.1, p. 3; losses Eq. 1, p. 4; UCT Eq. 2, p. 4). The learned
  model has representation H, dynamics G, reward R, value V and policy P networks; it is trained only
  through reward, value and policy unless changed below.
- Consistency loss (Sec. 4.1, Fig. 2, p. 5; Eq. 6, p. 15): the predicted next state G(s_t, a_t) is
  pulled to the encoder's H(o_(t+1)) with a SimSiam-like negative cosine loss (projector P1, predictor
  P2, stop-gradient on the target branch), unrolled 5 steps, weight lambda3 = 2; observations get
  random 0-4 pixel shifts and intensity changes (App. A.2, p. 17).
- Value prefix (Sec. 4.2, Eq. 3, p. 6): instead of summing per-step predicted rewards in Q =
  sum gamma^i r + gamma^k v, an LSTM over the unrolled states predicts the discounted reward sum
  directly (supervised at every step); the LSTM state is reset after 5 recurrent steps in search
  (App. A.1, p. 14).
- Off-policy correction (Sec. 4.3, Eq. 4, p. 6; Eq. 12, p. 19): value target z = sum_(i<l) gamma^i u +
  gamma^l * (root value of a fresh MCTS from s_(t+l)) with a horizon l <= k that shrinks for older
  data: l = clip(k - floor((T_current - T_data) / (tau * T_total)), 1, k), tau = 0.3, k = 5.
- Search details (App. A.3, pp. 17-18): 50 simulations; mean-Q initialisation for unvisited nodes
  (Eq. 8); soft min-max with epsilon 0.01 (Eq. 10); Dirichlet noise rho=0.25, xi=0.3 at the root
  in self-play and reanalyze; temperature 0.5 and 0.25 at 50% and 75% of training.
- Setup (Sec. 5.1, p. 7; Table 6, p. 15): Atari 100k = 100k agent steps = 400k frames (frameskip 4),
  26 games; 4 stacked 96x96 frames; 120k training steps with data collected in the first 100k;
  batch 256, SGD lr 0.2 to 0.02. Networks are much smaller than MuZero's (1 residual block in
  dynamics, 64 planes; App. A.1, pp. 14-16).

## Assumptions

- A1: The environment is a discrete-action, pixel-observation MDP with a simulator to collect the limited budget of interaction (Sec. 5.1, p. 7).
- A2: The budget is 100k agent steps (400k frames, about 2 hours of play) for Atari and 100k steps for DMControl (Sec. 5.1, p. 7).
- A3: The dynamics model is deterministic and the learned latent has to match the encoder's next-state encoding (Sec. 4.1, p. 5).
- A4: Continuous-control tasks are handled by discretising each action dimension into 5 slots, limited to low-dimensional tasks (Sec. 5.1, p. 7).
- A5: The replay data is stored with its collection time so the correction horizon can depend on age (Eq. 12, p. 19).
- A6: Training uses 50 MCTS simulations per move and reanalyzes 99% of policies and 100% of values (Table 6, p. 15, App. A.4, p. 19).

## Claims

- C1: On Atari 100k (26 games, 3 runs, 32 evaluation seeds each) EfficientZero reaches 194.3% mean and 109.0% median human-normalized score, above SPR (70.4%, 41.5%), MuZero as the authors reimplemented it (56.2%, 22.7%) and the other baselines [A1, A2, A6] (Table 1, p. 8, Fig. 1, p. 1).
- C2: It beats the human score in 14 of 26 games with 2 hours of game experience, while DQN needs 200M frames (500x more) for 220% mean and 95.9% median [A1, A2] (Sec. 5.2, p. 7, Sec. 1, p. 2).
- C3: With 95% CIs from rliable, EfficientZero is significantly better than the other methods on median, IQM, mean and optimality gap [A1, A2] (Fig. 8, p. 20).
- C4: On DMControl 100k with action dimensions discretised, EfficientZero scores 813 on cartpole swingup, 952 on reacher easy and 942 on ball-in-cup catch (10 seeds), above CURL, Dreamer and SAC from pixels, and, against State SAC (835, 746, 746), slightly below on cartpole swingup and above on the other two; MuZero on the same tasks scores 218.5, 493, 542 [A2, A4] (Table 2, p. 8).
- C5: Ablating one component at a time: without consistency 88.1% mean and 34.0% median, without value prefix 148.2% and 55.2%, without off-policy correction 147.5% and 83.6%, against 194.3% and 109.0% full [A1, A2, A6] (Table 3, p. 9, Table 7, p. 16).
- C6: Latent states from a model trained without the consistency loss decode to garbage when unrolled 1-5 steps, while with it the decoder (trained on the frozen model) recovers coarse observations, which the authors read as reduced shift between encoded and predicted latent states [A3] (Fig. 4, p. 9, Sec. 5.3, p. 9).
- C7: On a fixed half-trained Pong dataset the direct-reward model fits the training set better but has a higher 5-step validation error than value prefix, which the authors read as evidence value prefix avoids overfitting to aliased reward timing [A1] (Fig. 5, p. 10, Sec. 5.3, pp. 9-10).
- C8: In UpNDown, off-policy correction lowers the L1 error of value targets against Monte-Carlo ground truth (0.765 to 0.533 at the current state; 0.657 to 0.569 over all states), and errors fall with fresher data [A1] (Table 4, Table 5, p. 10).
- C9: The self-supervised loss, not the image augmentation, accounts for the consistency gain: without augmentation scores are higher in 2 of 8 tested games and lower, often modestly, in 6, while without consistency they are lower in all 8 [A1, A2] (Table 8, p. 17).
- C10: In the off-policy correction, removing the dynamic horizon hurts more than removing the MCTS root value (8 games, 100k steps), the no-horizon variant is below Full in 7 of 8 games (higher only in Breakout, 427 vs 388.8); the no-root-value variant is close to Full in most games but clearly lower in Pong (14.5) and Seaquest (876), where it is also below the no-horizon variant [A1, A2] (Table 9, p. 17).
- C11: The implementation trains one Atari agent for 100k steps on 4 GPUs in 7 hours, versus 64 TPUs for 12 hours for MuZero [A1, A2] (App. A.6, p. 21).
- C12: Seed variance is large on some games: Asterix 25557.8 +- 12565.7, Crazy Climber 83940.2 +- 13774.6, Frostbite 296.3 +- 79.4; Freeway stays at 21.8 and Private Eye near 100 across runs [A1, A2] (Table 10, p. 21).

## Implementation notes

- Reward clipping on, terminal on loss of life, discount 0.997^4, max 108K frames per episode, min replay size 2000, self-play network update every 100 steps, target update every 200 (source: Table 6, p. 15).
- Priority exponent 0.6, beta 0.4 to 1, sequences of 400 moves collected; priorities of new transitions set to the buffer max (source: App. A.4, p. 19).
- Projector P1 is a 3-layer MLP, P2 a 2-layer MLP, hidden 512, output 1024, batch norm between layers (source: App. A.1, pp. 15-16).
- Value and reward heads use 601-bin categorical outputs; last layer weights and bias zero-initialised (source: App. A.1, pp. 14-15).
- The reward/value network listings in App. A.1 put the LSTM under the value network heading while the text says the LSTM predicts the value prefix (reward) (source: App. A.1, p. 14; reader's reading: the LSTM is in the reward/value-prefix head).
- Reference code: github.com/YeWR/EfficientZero; the commit used is not stated (source: App. A.6, p. 21).
- The pipeline is Ray actors: self-play, CPU context preparation, GPU batch reanalysis (C++ batched MCTS via Cython), learner (source: Fig. 6, p. 19, App. A.6, p. 21).

## Limitations / open questions

- The authors list as future work a better design for continuous action spaces, faster MCTS and combining with lifelong learning (Sec. 6, p. 10).
- DMControl was limited to 3 low-dimensional tasks because of the discretisation (Sec. 5.1, p. 7).
- Sec. 5.2 gives mean 1.904, median 1.160 and "170% and 180% better" than SPR (p. 7), likely from an earlier version; the abstract, Table 1 and the captions of Fig. 1 and Table 1 agree on 1.943/1.090 and 176%/163% (p. 1, p. 8). The note uses Table 1.
- 100k steps are with the environment's own frameskip and a simulator, so data efficiency here counts environment steps, not compute: each step uses 50-simulation search and reanalyze of the whole buffer.
- On Freeway (21.8, in every run, vs SPR 24.4 and CURL 26.7) and Private Eye (96.7 vs CURL 218.4, SPR 124.0) EfficientZero is below several baselines (Table 1, p. 8); the paper does not analyse why (reader's observation).
- Open question: whether the consistency loss's latent is trustworthy for several-step bullet prediction; only a qualitative decoder figure (Fig. 4) tests it.
- The 64-TPU/12 h figure for MuZero (C11) doesn't match MuZero's own App. G (8 TPUs for training plus 32 for self-play per Atari game, [[schrittwieser2020mastering#C12]]), and the two runs differ by about 50,000x in frames (20B vs 400k), so C11 is not a like-for-like compute comparison (reader's observation).
- Not read: the code repository.

## Related

- extends [[schrittwieser2020mastering]]: MuZero Reanalyze plus consistency loss, value prefix and off-policy correction for 100k-step data.
- related [[hubert2021learning]]: Sampled MuZero, cited as the continuous-action extension that EfficientZero's authors plan to build on (Sec. 2.2, p. 3).
- related [[hafner2024mastering]]: DreamerV3, the model-based alternative without tree search; Dreamer (V1) is a DMControl baseline here.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
