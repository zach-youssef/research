---
id: srinivas2020curl
title: "CURL: Contrastive Unsupervised Representations for Reinforcement Learning"
authors: [Aravind Srinivas, Michael Laskin, Pieter Abbeel]
year: 2020
venue: ICML 2020 conference paper
url: https://arxiv.org/abs/2004.04136
arxiv: 2004.04136
doi:
pdf_sha256: 30600b6955c79d3912e5fc41c2828cfedac5f6feac12865469c9bf7dd6e4930a
tags: [reinforcement-learning, contrastive-learning, self-supervised, representation-learning, sample-efficiency, data-augmentation, atari, dm-control, state-representation]
status: read
read: 2026-11-20
---

# CURL: Contrastive Unsupervised Representations for Reinforcement Learning

## TL;DR

Adds a MoCo-style instance-discrimination loss (InfoNCE, bilinear similarity, momentum key encoder,
random crops of frame stacks) to the pixel encoder of SAC or Efficient Rainbow, trained jointly.
1.9x Dreamer's median at DMControl100k, near state-based SAC at 500k, and 1.2x SimPLe on Atari100k.

## Problem

RL from pixels is much less sample-efficient than RL from physical state (Sec. 1, p. 1). Prior
fixes are auxiliary tasks (often reconstruction) or learned world models; contrastive auxiliary
learning (CPC) had given only marginal gains on a few DMLab tasks (Sec. 1, p. 2). The hypothesis:
if an agent learns a useful semantic representation from pixels, control built on it should be
much more data-efficient (Sec. 1, p. 2). Unlike vision pretraining there is no large static
unlabelled dataset: the data come online from the agent's experience, and representation and RL
are learned at the same time (Sec. 1, p. 2; App. A, p. 14).

## Method

- **Observation = frame stack.** The instance is a stack of consecutive frames (3 RGB frames in
  DMControl, 4 grayscale in Atari), so the contrastive task is over stacks, letting it pick up
  temporal as well as spatial features (Sec. 4.1, Sec. 4.3, pp. 5-6; App. B, p. 15).
- **Query-key pairs.** A minibatch sampled from the replay buffer for the RL update is augmented
  twice; the two views of one observation are anchor and positive, the other batch elements are
  negatives (no memory bank, unlike MoCo) (Sec. 4.3, p. 5; App. A, p. 14). Augmentation is random
  crop, 84×84 out of a 100×100 render, with the same crop box on every frame of a stack
  (Fig. 3, p. 6; App. A, p. 14). Evaluation uses a centre crop (App. A, p. 14).
- **Encoders.** Query encoder f_q is the RL encoder; key encoder f_k is its EMA,
  θ_k = mθ_k + (1-m)θ_q, with stop-gradient on keys (Sec. 4.5, p. 6). The EMA of the conv encoder
  is shared with SAC's target critic (App. A, p. 14).
- **Loss.** InfoNCE with bilinear similarity q^T W k, W learned (Eq. 4, p. 5; Sec. 4.4, p. 6),
  computed as cross-entropy over the B×B logit matrix with diagonal labels (Sec. 4.7, p. 6).
- **Joint training.** The encoder gets gradients from the contrastive loss and the RL loss
  (critic for SAC), weighted equally with no extra balancing hyperparameter (Sec. 4.1, p. 5;
  App. A, pp. 13-14). Base learners: SAC (Eqs. 1-3, p. 4) on DMControl, data-efficient Rainbow on
  Atari (Sec. 3.2, p. 4; App. B, p. 15).
- **Encoder (DMControl).** 4 conv layers, 32 filters, 3×3 kernels, stride 2 then 1, then an MLP
  to a 50-d latent with LayerNorm and tanh; actor and critic share it (App. A, p. 13, Table 3).
  Atari uses Efficient Rainbow's encoder unchanged (5×5 kernels, stride 5) (App. B, p. 15, Table 4).
- **Benchmarks.** DMControl100k/500k (environment steps) and Atari100k (100k interaction steps =
  400k frames, about 2 hours of play) (Sec. 5.1, p. 6; Sec. 5.2, Sec. 5.3, p. 7; App. A, p. 14).
  Sample efficiency = steps a baseline needs to reach CURL's score; performance = return ratio at
  a fixed step count (Sec. 5.1, p. 7).

## Assumptions

- A1: Observations are rendered pixel frames, fed to the agent as a stack of consecutive frames (3 RGB frames in DMControl, 4 grayscale in Atari) (Sec. 4.3, pp. 5-6, Table 3, p. 13, Table 4, p. 15)
- A2: The contrastive instance is the whole frame stack, not a single frame (Sec. 4.1, p. 5, Sec. 4.3, p. 6)
- A3: The augmentation is random crop with the same box on every frame of a stack, keeping most of the frame (84 of 100 pixels per side in DMControl), with a centre crop at evaluation (Fig. 3, p. 6, App. A, p. 14)
- A4: Small translations of the frame leave the task state unchanged and the task-relevant objects stay inside the crop, so two crops of one stack are a valid positive pair (reader's inference, App. A, p. 14)
- A5: Positives are two augmentations of one observation and the negatives are the other observations in the same replay minibatch, with no memory bank (Sec. 4.3, p. 5, Sec. 4.7, p. 6, App. A, p. 14)
- A6: Keys come from a momentum (EMA) copy of the query encoder with stop-gradient, and similarity is a learned bilinear product q^T W k (Sec. 4.4-4.5, p. 6)
- A7: The contrastive loss is trained jointly with the RL loss on the same replay minibatches, without a tuned balancing weight (down-weighted only on 4 of 26 Atari games), and the conv encoder receives gradients from both (Sec. 4.1, p. 5, App. A, pp. 13-14, App. B, p. 15)
- A8: The base learner is an off-policy, replay-based algorithm: SAC for continuous control and data-efficient Rainbow for discrete Atari actions; on-policy learners are said to be possible but are not tested (Sec. 3, p. 3, Sec. 4, p. 5)
- A9: Data come online from one agent's own experience in one environment: every agent is trained from scratch on a single DMControl task or Atari game, with no multi-task training, pretraining or transfer across environments (reader's inference, Sec. 5.2, p. 7, App. A, p. 14)
- A10: The encoder is a small conv net learned with the agent (in DMControl 4 conv layers and a 50-d latent shared by actor and critic; in Atari, Efficient Rainbow's unchanged encoder), not pretrained features (App. A, p. 13, App. B, p. 15)
- A11: The regime is low-data: DMControl at 100k or 500k environment steps and Atari at 100k interaction steps (400k frames) (Sec. 5.1, p. 6, Sec. 5.3, p. 7)
- A12: The RL loss uses the environment's task reward; only the contrastive objective is reward-independent (Table 4, p. 15, App. G, pp. 18-19)
- A13: Baseline scores are taken from prior work or data shared by their authors rather than re-run under one protocol; for SimPLe and human, the best score reported for each game in prior work is used (p. 8, p. 9, Fig. 6, p. 16)
- A14: A 3-frame pixel stack carries enough information about the underlying physical state for control (App. E.5, p. 18)
- A15: Each environment renders from a fixed simulator view with the same visual appearance in training and evaluation, so no shift in visuals has to be bridged (reader's inference, App. A, p. 14)
- A16: Tasks needing balance or coordination (walker, cheetah) require dynamics information from the frame stack, while visually solvable tasks (reacher, ball-in-cup) do not (App. E.1, pp. 15-16, Fig. 8, p. 17)
- A17: The state-based SAC oracle observes the simulator's low-level proprioceptive state and is taken as an approximate upper bound for pixel agents (Sec. 5.3, p. 7)
- A18: The state-regression probe uses simulator state (joint positions and velocities, or joint angles and angular velocities) as supervised labels (App. E.5, p. 18, Fig. 11, p. 18)

## Claims

- C1: On DMControl500k CURL has the best score among pixel methods on 5 of 6 environments (all but cheetah-run, where SLACv1 is best), over 10 seeds [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15] (Table 1, p. 8, Sec. 6.1, p. 9).
- C2: At DMControl100k CURL's median score over the 6 environments is 1.9x Dreamer's (560 vs. 296) [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15] (Fig. 4, p. 9, Sec. 6.1, p. 9).
- C3: At DMControl500k CURL's median (914) about matches state-based SAC's (923); at 100k it trails (560 vs. 779) [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A17] (Fig. 4, p. 9).
- C4: Dreamer needs on average 4.5x as many steps as CURL to reach CURL's 100k-step score, over 16 environments (clipped at 1M steps) [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15] (Fig. 6, p. 16, Sec. 6.1, p. 9).
- C5: CURL nearly matches, and sometimes beats, state-based SAC's sample efficiency on most of 16 DMControl environments, which the authors call a first for pixel-based RL; it lags on harder ones [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A14, A15, A17] (Fig. 7, p. 16, Sec. 6.1, p. 9, App. D, p. 15).
- C6: On Atari100k CURL on Efficient Rainbow reaches median human-normalised score 17.5% vs. 14.4% (SimPLe) and 16.1% (Eff. Rainbow); the means are 38.1%, 44.3% and 28.5% [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A15] (Sec. 6.2, p. 9).
- C7: On Atari100k CURL improves on Efficient Rainbow in 19 of 26 games, by 1.3x on average, with median gains of 1.2x over SimPLe and 1.1x over Efficient Rainbow; it is the best method on 7 of 26 games, over 20 seeds [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A15] (Table 2, p. 8, Sec. 6.2, p. 9).
- C8: CURL beats human scores on JamesBond (1.6 HNS) and Krull (2.5 HNS) at Atari100k [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A15] (Sec. 6.2, p. 9, Table 2, p. 8).
- C9: On cheetah-run, a momentum (EMA) key encoder clearly beats using the query encoder for keys, and bilinear similarity clearly beats cosine similarity [A1, A2, A3, A4, A5, A7, A8, A9, A10, A11, A12, A15] (Fig. 5, p. 13).
- C10: Contrasting whole frame stacks beats contrasting single frames on most environments; the exceptions are reacher and ball-in-cup, where visual signal suffices, unlike walker and cheetah, which need balance or coordination [A1, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A14, A15, A16] (App. E.1, pp. 15-16, Fig. 8, p. 17).
- C11: With the conv encoder trained only by the contrastive loss (critic gradients stopped), policies are near-optimal on most of 6 environments, cheetah-run being the major exception [A1, A2, A3, A4, A5, A6, A8, A9, A10, A11, A12, A15] (App. E.3, p. 17, Fig. 9, p. 17).
- C12: Augmenting only the contrastive input and feeding the policy unaugmented frames still improves pixel SAC by 2.0x mean, 1.7x median, at DMControl500k (batch 128); augmenting the SAC input too improves further on most environments [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A15] (App. E.4, pp. 17-18, Fig. 10, p. 17).
- C13: Error in regressing proprioceptive state from 3-frame pixel stacks tracks performance: low where CURL matches state SAC, higher for cheetah-run and walker-walk, highest for humanoid and swimmer, where pixel methods make no progress [A1, A14, A15, A17, A18] (App. E.5, p. 18, Fig. 11, p. 18).
- C14: With 3 gradient updates per agent step CURL beats SLACv2 on 3 of 4 DMControl500k environments (all but cheetah-run), at a significant compute and wall-clock cost [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15] (Table 5, p. 16, App. E.2, p. 16).

## Implementation notes

- DMControl hyperparameters: replay buffer 100k, 1000 initial random steps, batch 512, Adam, lr 2e-4 (cheetah) or 1e-3, α lr 1e-4, critic EMA τ 0.01, encoder EMA τ 0.05, target update every 2, γ 0.99, initial temperature 0.1, MLP hidden 1024; action repeat 2 (finger, walker), 8 (cartpole), 4 otherwise (source: Table 3, p. 13).
- Atari hyperparameters: Efficient Rainbow's exactly; batch 32, lr 1e-4, multi-step 20, target update every 2000 updates, momentum τ 0.001, noisy nets, CPU hardware (source: Table 4, p. 15). Contrastive loss weight 0.05 on Pong, Freeway, Boxing and PrivateEye, 1 elsewhere (source: App. B, p. 15).
- The Atari crop's source size (what 84×84 is cropped from) isn't stated (source: Table 4, p. 15); check the authors' code https://github.com/MishaLaskin/curl (commit not pinned) or Kaixhin/Rainbow.
- DMControl builds on SAC+AE's released code (source: App. A, p. 13); Atari on https://github.com/Kaixhin/Rainbow (source: App. B, p. 15).
- Random crop must be implemented efficiently (vectorised windows), which the authors call extremely important (source: App. A, p. 14).
- The Fig. 10 caption compares its batch 128 with "the 512 batch size used for results in Table 4"; Table 4 is Atari (batch 32), so the reader takes Table 3 to be meant (source: assumption).
- App. A says the crop is shared across "all four frames", while DMControl stacks 3; the reader takes it as "all frames" (source: assumption, from Table 3, p. 13).
- Results changed across arXiv versions (walker action repeat 4 to 2, Atari moved to Kaixhin codebase with 20 seeds); this note describes v4 (source: App. F, p. 18).

## Limitations / open questions

- Gap to state-based SAC remains on harder environments, possibly because the pixels lack state
  information (partial observability) (App. D, p. 15; App. E.5, p. 18).
- Atari100k variance across seeds is high, hence 20 seeds (Table 2, p. 8; App. E.6, p. 18).
- RAD and DrQ (augmentation without a contrastive loss) beat CURL on both benchmarks; the
  authors say they are likely better with rich rewards, and CURL's reward-agnostic objective is
  the case for task-agnostic latents and reward-free pretraining (App. G, pp. 18-20).
- Task-agnostic representations are left to future work; only the detached-encoder ablation
  tests it (App. E.3, p. 17).
- Open question (lens): every agent is trained per environment; there is no multi-game training
  or transfer to an unseen game, so whether a CURL encoder shared across games yields a common
  state space is untested.
- Open question (lens): the "temporal" context is only a 3-4 frame stack fed to a conv net; how
  instance discrimination scales to N-frame sequences tokenised for a transformer is not studied.
- Open question (lens): the encoder is tiny (4 conv layers, 50-d latent); does the gain persist,
  or vanish as with RAD/DrQ, for a large pretrained encoder (e.g. MAE or SAM features)?
- Open question (lens): CURL is tested only with off-policy learners (SAC, Rainbow), though the
  paper says any RL algorithm works (Sec. 3, p. 3); PPO is untested here.

## Related

- related [[he2021masked]]: CURL uses contrastive instance discrimination where MAE uses masked reconstruction; both are candidate self-supervised encoders.
- related [[schulman2017proximal]]: CURL is framed as pluggable into any RL learner but tested only off-policy; PPO is the survey's on-policy candidate.
- related [[serrano2024crossdomain]]: reward-agnostic visual encoder that could form a shared latent state across domains; transfer not tested here.
- related [[kirillov2023segment]]: an alternative, supervised source of visual state (segments) to CURL's self-supervised latent.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
