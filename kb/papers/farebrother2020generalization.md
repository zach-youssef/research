---
id: farebrother2020generalization
title: "Generalization and Regularization in DQN"
authors: [Jesse Farebrother, Marlos C. Machado, Michael Bowling]
year: 2020
venue: arXiv preprint (read as v3, 2020; earlier versions at the NeurIPS 2018 Deep RL Workshop and RLDM 2019)
url: https://arxiv.org/abs/1810.00123
arxiv: 1810.00123
doi:
pdf_sha256: 384211551bec7f11bf9d4b340154975ea514c0194b2462f6036043598bad9bb4
tags: [reinforcement-learning, generalization, overfitting, regularization, dropout, fine-tuning, transfer-learning, dqn, atari, ale-modes, robustness]
status: read
read: 2026-11-22
---

# Generalization and Regularization in DQN

## TL;DR

Uses the ALE's native game modes and difficulties ("flavours") as held-out tests: DQN trained 50M
frames on the default flavour of Freeway, HERO, Breakout or Space Invaders scores far below scratch
training on 8 of 9 variants. Dropout + ℓ2 helps zero-shot a little; its real gain is fine-tuning.

## Problem

Deep RL policies are trained and evaluated on the same environment and rarely transfer even to
near-identical tasks; the paper asks whether DQN's representations generalise, or can at least be
reused and refined, on small variations of a game, and whether supervised-learning regularisation
(dropout, ℓ2) and fine-tuning help (Sec. 1, pp. 1-2).

## Method

- **Testbed.** ALE flavours (mode/difficulty pairs, Machado et al. 2018): native Atari 2600 variants
  that change sprites, velocities or object observability, designed for humans, not for AI agents
  (Sec. 3, pp. 5-6). 13 flavours of 4 games: Freeway m0d0, m1d0, m1d1, m4d0; HERO m0d0, m1d0, m2d0;
  Breakout m0d0, m12d0; Space Invaders m0d0, m1d0, m1d1, m9d0 (Sec. 3, p. 7, Fig. 2, p. 7, pp. 22-23).
  - Freeway: mode 1 adds buses, more vehicles and higher speed; mode 4 is mode 1 with vehicles
    oscillating between two speeds; difficulty 1 sends the chicken back to the start on a collision
    instead of one lane down (Fig. 2, p. 7). (Sec. 4, pp. 8-9, also says m4d0 has the same sprites as m0d0.)
  - HERO: mode k starts at level 5k (Fig. 2, p. 7). Breakout m12d0 and Space Invaders m9d0 hide the
    bricks or aliens until hit; Space Invaders mode 1 oscillates the shields, difficulty 1 widens the
    player sprite (Fig. 2, p. 7).
- **Zero-shot evaluation.** Train DQN 50M frames on m0d0, evaluate the ε-greedy policy for 100
  episodes on each other flavour, 5 runs; compare with DQN trained from scratch 50M frames on that
  flavour ("Learn Scratch") (Sec. 4, p. 8, Table 1, p. 8). Checkpoints every 500k frames give
  transfer curves (Sec. 4, p. 9, Fig. 3, p. 9).
- **Regularisation.** Dropout on the three conv layers and first FC layer, ℓ2 on all weights; grid
  search on Freeway only over p_conv, p_fc ∈ {(0.05,0.1) … (0.25,0.5)} and λ ∈ {1e-2 … 1e-6},
  5 runs each; chosen λ = 1e-4, p_conv, p_fc = 0.05, 0.1 for all games (Sec. 5, p. 10, pp. 25-27).
- **Fine-tuning.** Initialise from the m0d0 network (regularised or not) and train 50M frames on the
  target flavour; compare with scratch at 50M and 100M frames (Sec. 6.1, pp. 12-13, Table 3, p. 14).
  Variant: keep only the 3 conv layers, or conv + first FC, and re-initialise the rest (Sec. 6.2, p. 13, Table 4, p. 15).

## Assumptions

- A1: Flavours share the underlying dynamics and differ in few latent variables, so generalisation here means invariance to small changes in the observation space (Sec. 3, p. 6).
- A2: DQN trained from scratch on a flavour is treated as an (informal) upper bound on what a transferred policy can reach there (Sec. 4, p. 8).
- A3: Evaluation follows Machado et al. 2018: sticky actions (0.25), all 18 actions, no loss-of-life termination, mean of 100 episodes averaged over 5 runs, no maximum over runs or over the learning curve (p. 24).
- A4: Regularisation hyperparameters are tuned on Freeway (m0d0 training, m1d0/m4d0 evaluation) and reused unchanged for the other three games (Sec. 5, p. 10, p. 27).
- A5: The evidence covers 4 games and 13 flavours, chosen to allow exhaustive multi-seed runs (Sec. 3, p. 7).

## Claims

- C1: Zero-shot from m0d0 (50M frames) vs scratch (50M) on the target flavour: Freeway m1d0 0.2 vs 4.8, m1d1 0.1 vs 0.0, m4d0 15.8 vs 29.9; HERO m1d0 82.1 vs 1425.2, m2d0 33.9 vs 326.1; Breakout m12d0 43.4 vs 67.6; Space Invaders m1d0 258.9 vs 753.6, m1d1 140.4 vs 698.5, m9d0 179.0 vs 518.0 (5 runs; std up to 1755) [A1, A2, A3, A5] (Table 1, p. 8).
- C2: Freeway m4d0 uses the same sprites as m0d0 and differs only in cars that accelerate and decelerate, yet the near-optimal m0d0 policy scores 15.8, about half of scratch (29.9); flavours with new sprites or more cars do worse [A1] (Sec. 4, pp. 8-9, Table 1, p. 8).
- C3: In HERO the default-mode agent reaches level 3, sometimes 4; mode 1 starts at level 5, which adds few new mechanics, yet performance collapses, which the authors read as memorised trajectories rather than a robust level-solving policy [A1] (Sec. 4, p. 9).
- C4: In Freeway, transfer to m1d0 and m1d1 rises and then falls during m0d0 training (bell shape, peak near 20M frames), while m4d0 flattens out; the authors call this overfitting to the training flavour and don't see the bell shape in the other games (5 seeds) [A3] (Sec. 4, pp. 9-10, Fig. 3, p. 9).
- C5: With dropout + ℓ2, zero-shot scores (vs unregularised) are Freeway 5.8/4.4/20.6 (vs 0.2/0.1/15.8), HERO m1d0 116.8 (82.1) and m2d0 30.0 (33.9), Breakout m12d0 31.0 (43.4), Space Invaders 456.0/146.0/290.0 (258.9/140.4/179.0); in Freeway m1d0 and m1d1 this beats scratch training [A2, A3, A4] (Table 2, p. 11, Sec. 5, p. 11).
- C6: With regularisation, Freeway transfer performance keeps increasing over most of training instead of the bell shape of C4 [A4] (Fig. 4, p. 12, Sec. 5, p. 11).
- C7: Regularisation alone doesn't make policies generalise: learning from scratch on the new flavour is still better than reusing the regularised policy [A2, A4] (Sec. 5, p. 11).
- C8: Regularisation costs default-flavour performance at 50M frames: Freeway 25.9 vs 31.4, HERO 6505.9 vs 9034.4, Breakout 34.1 vs 72.5, Space Invaders 623.1 vs 698.8 (Table 6 vs Table 5); the authors say dropout generally needs more training to reach the same level [A3, A4] (Table 5-6, p. 17, Sec. 5, p. 10).
- C9: Fine-tuning the whole regularised network 50M frames on the new flavour reaches Freeway 25.4/25.4/32.2, HERO 4104.6/211.0, Breakout m12d0 96.1, Space Invaders 1033.5/920.0/583.0; this beats scratch at 50M frames on every flavour but HERO m2d0 (326.1), and scratch at 100M (7.5/2.5/32.8, 5026.8/323.5, 55.2, 979.7/906.9/567.7) on all but HERO and Freeway m4d0 [A2, A3, A4] (Table 3, p. 14, Sec. 6.1, p. 13).
- C10: Fine-tuning the unregularised network is mixed (e.g. HERO m1d0 496.7, Freeway m1d0 22.5); the Freeway gain is attributed to optimistic initialisation, a bias to go up under the sparse mode-1 rewards, not to general features [A3] (Table 3, p. 14, Sec. 6.1, p. 13).
- C11: Counting frames, regularised fine-tuning over two flavours uses 150M frames (50M default + 50M each) against 200M for scratch on both, at 100M frames per flavour seen; the gap grows with the number of tasks [A2] (Sec. 6.1, p. 13).
- C12: Re-initialising the later layers and fine-tuning only the conv layers (or conv + first FC) is generally worse than fine-tuning the whole network, e.g. Freeway m1d0 0.7 or 4.9 vs 25.4; at 50M frames three cells are exceptions: HERO m2d0 3Conv 455.2 and 3Conv+1FC 307.6 vs 211.0, and Space Invaders m9d0 3Conv+1FC 586.7 vs 583.0 [A3, A4] (Table 4, p. 15, Sec. 6.2, p. 13).
- C13: Meta-RL methods (MAML, REPTILE) gave no positive results on these flavours [A5] (Sec. 3, p. 6, Sec. 7, p. 16).
- C14: Ablation on Freeway: large λ hurts training, small λ lets the agent overfit m0d0; ℓ2 helps m1d0 but not m4d0; large dropout rates flatline training and evaluation (a state-independent action bias); dropout + ℓ2 stabilises training (p_conv, p_fc = 0.1, 0.2 scores under 15 with dropout alone, about 20 with moderate λ) [A4] (pp. 25-27, Fig. 6-10, pp. 25-28).

## Implementation notes

- Network: Mnih et al. 2015 DQN, conv 32 8×8/4, 64 4×4/2, 64 3×3/1, then FC and 18 outputs; Fig. 5 labels the hidden FC 1024 (source: App. Fig. 5, p. 24).
- Hyperparameters: lr 0.00025, minibatch 32, update every 4 actions, frame skip 5, sticky-action prob 0.25, replay 1M, ε 1.0 → 0.01 over 1M frames, γ 0.99 (source: Table 7, p. 24).
- 50M frames = 10M agent steps = 2.5M gradient updates (source: App. p. 24).
- ℓ2 loss written as λ‖θ‖² in the appendix (p. 25) but λ/2‖θ‖² in the background (p. 3); which one the code uses isn't stated (source: paper, pp. 3 and 25).
- Optimiser, frame preprocessing (resize, grayscale, stacking) and target-network period aren't stated; presumably Machado et al. 2018 / Mnih et al. 2015 defaults (source: assumption, not checked); the code is at https://github.com/jessefarebro/dqn-ale (source: App. p. 24).
- Compute and wall-clock aren't reported (source: paper, none found).

## Limitations / open questions

- Four games, five seeds, large variances (HERO std up to 2192.8) and per-game tuning only on Freeway (A4, A5).
- HERO flavours change the start level, so transfer there mixes generalisation with curriculum (Sec. 7, p. 16).
- Breakout is the one game where regularisation lowers zero-shot (C5) and its default score halves (C8); the paper doesn't discuss it.
- The paper says whole-network fine-tuning is "consistently better" than re-learning co-adaptations (p. 13), which Table 4's three exceptions (C12) overstate.
- Freeway m4d0 is described two ways: "almost identical to Mode 1" (buses, more vehicles; Fig. 2, p. 7, p. 22) and "exactly the same sprites" as m0d0 (Sec. 4, pp. 8-9); the p. 22 m4d0 screenshot shows no bus and looks like m0d0, supporting Sec. 4. C2 follows Sec. 4; check the ALE mode before building on it.
- Open question (reader): flavours change sprites, speeds and visibility natively, like HackAtari's RAM edits in [[delfosse2025intelligence]]; whether object-centric inputs (OCAtari-style extraction) close C1's gap, e.g. on Freeway m4d0 (same sprites, C2), is a cheap test this paper doesn't run.
- Open question (reader): C4's bell-shaped transfer curve on Freeway m1d0 is a small, checkable experiment (one game, 50M frames, checkpoint evaluation).

## Related

- related [[delfosse2025intelligence]]: HackAtari tests the same within-game generalisation with RAM-edited variants instead of native modes, and finds the same sharp drops.
- related [[derstroff2026relative]]: relative object representations aimed at robustness on HackAtari variants.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
