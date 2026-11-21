---
id: castro2020scalable
title: Scalable methods for computing state similarity in deterministic Markov Decision Processes
authors: [Pablo Samuel Castro]
year: 2020
venue: AAAI 2020 conference paper
url: https://arxiv.org/abs/1911.09291
arxiv: 1911.09291
doi:
pdf_sha256: a7f1546c8046cea49a927f686c0810986c357aa47ffac17943bc74a21eff7429
tags: [reinforcement-learning, bisimulation, state-similarity, metric-learning, representation-learning, atari, theory]
status: read
read: 2026-11-21
---

# Scalable methods for computing state similarity in deterministic Markov Decision Processes

<!-- Read from arXiv v1 (2019-11-21, the only version), whose comment says it is the AAAI-20 paper (p. 1 carries the AAAI 2020 copyright). 18 PDF pages: pp. 1-7 main text, p. 8 references, pp. 9-18 supplement (proofs, mini-batch loss, GridWorld distances and config, clustering code, Atari plots and sweep, lax bisimulation, training curves). Sections are unnumbered, so references give pages. -->

## TL;DR

Defines on-policy π-bisimulation, a metric bounding |V^π(s) − V^π(t)|, and for deterministic MDPs
a sampled update provably converging to the bisimulation metric plus a TD-style neural loss
ψ(φ(s), φ(t)); it fits a 31-state GridWorld closely and gives plausible distances on 3 Atari games.

## Problem

Bisimulation metrics (Ferns et al. 2004) measure behavioural similarity between states and
bound differences in optimal value, but computing them needs a full enumeration of states,
which rules out large and continuous state spaces (pp. 1-2). They are also pessimistic: they
take the worst case over all actions, even ones a policy never uses, and match actions by
label (p. 2). The author knows of no prior method approximating them with neural networks (p. 2).

## Method

- Background (p. 2): the bisimulation metric d_∼ is the unique fixed point of
  F(d)(s, t) = max_a (|R(s, a) − R(t, a)| + γ W1(d)(P(s, a), P(t, a))) (Eq. 2), with W1 the
  1-Wasserstein distance as an LP (Eq. 1); iterating to accuracy δ costs
  O(|A||S|^4 log|S| ln δ / ln γ).
- On-policy π-bisimulation (Definition 3, Theorem 2, p. 3): policy-averaged rewards R^π_s and
  transitions P^π_s replace the per-action terms, so actions need not be matched by label and
  unused actions don't count.
- Deterministic MDPs (Definition 4, Lemma 1, p. 3): W1 reduces to d of the two next states, so
  F(d)(s, t) = max_a (|R(s, a) − R(t, a)| + γ d(N(s, a), N(t, a))), and for a deterministic
  policy the π-version uses π(s) and π(t).
- Sampled exact computation (Eq. 3, p. 4): from d_0 ≡ 0, for each sampled pair of transitions
  with the same action, replace d(s, t) by the larger of its old value and the one-step backup.
- Learned approximation (pp. 4-5, Fig. 3): ψ_θ takes the concatenation [φ(s), φ(t)] of fixed
  k-dimensional state representations; target T = max(|ΔR| + γ ψ_θ⁻(next states), ψ_θ⁻(s, t))
  for d_∼, and |R(s, π(s)) − R(t, π(t))| + γ ψ_θ⁻(next states) for d^π, zero when s = t; squared
  loss, with a target network θ⁻ updated every C steps as in DQN.
- Mini-batch form (p. 5; derivation pp. 11-12): all b² pairs within a batch, a mask W keeping
  pairs with equal actions (d_∼ only), the diagonal zeroed, and a factor β grown from 0 to 1 that
  lengthens the backup's effective horizon.
- Experiments (pp. 5-6): a 31-state deterministic GridWorld with (x, y) inputs, also with
  Gaussian noise for a continuous variant; d^π for Dopamine's trained Rainbow agents on Space
  Invaders, Pong and Asterix (no sticky actions), with φ the agent's penultimate layer.

## Claims

- C1: The π-bisimulation operator F^π has a least fixed point d^π_∼, which is a π-bisimulation metric (p. 3, p. 9).
- C2: For any states s, t, |V^π(s) − V^π(t)| ≤ d^π_∼(s, t) (p. 3, p. 9).
- C3: The original metric can be uninformative about optimal behaviour: in the 3-state example with γ = 0.9, V*(s) = V*(t) = 10K while d_∼(s, t) = 10K; lax bisimulation gives 0 there (Fig. 2, p. 3).
- C4: In a deterministic MDP, W1(d)(P(s, a), P(t, a)) = d(N(s, a), N(t, a)) for any pseudometric d, so no Wasserstein LP is needed (p. 3, pp. 9-10).
- C5: If every pair of transitions with a shared action has positive probability under the sampling distribution, the sampled update of Eq. 3 from d_0 ≡ 0 converges to d_∼ almost surely; for d^π the condition is that all states of the Markov chain induced by π are updated infinitely often (Eq. 3, p. 4, p. 10).
- C6: The learned approximant ψ is not in general a proper pseudometric: it can violate identity of indiscernibles, symmetry and subadditivity (p. 5, p. 11).
- C7: On the 31-state GridWorld, over 10 runs, the absolute error ||d_∼ − ψ||∞ falls from about 150 to about 30-45 within roughly 2,400 steps, and the error of the normalized distances falls to the low 10^-3 range (Fig. 5, pp. 5-6).
- C8: Absolute errors are not guaranteed to be bounded because there is no fixed learning target, and the max in the target makes the networks tend to overshoot; training was stopped at 2,500 steps, before overshooting (p. 5).
- C9: Adding clipped Gaussian noise (σ = 0.1, clipped to ±0.3) to the GridWorld inputs, making the state space continuous, makes little difference to the learned metric and, per the author, seems to help, hypothesized as a regularizing effect (untested); in the plots it helps the normalized error (about 2e-3 vs 3e-3, Fig. 5), while the absolute error ends higher with noise (about 42 vs 32) with a wider 95% CI (Fig. 5, p. 5, p. 6).
- C10: d_∼ on the GridWorld, computed by the sampled update (Eq. 3), captures similarity beyond spatial proximity: the bottom row of the top room, next to the hallway cell, is furthest from it (Fig. 7, p. 12).
- C11: Clustering continuous GridWorld samples (100 per cell) by the learned distance recovers many of the environment's symmetries, with the hallway cell its own cluster (Fig. 8, p. 13).
- C12: On Space Invaders, the learned d^π from the first state is low at game or level starts and peaks when an enemy alien is destroyed, capturing differences beyond pixel differences, and the frames furthest from the start are usually ones where the agent is about to be killed; Pong peaks when the agent scores and Asterix plateaus when the agent doesn't move (Fig. 6, pp. 6-7, Fig. 9, Fig. 10, Fig. 11, pp. 15-17).
- C13: On Atari, training ran for about 600K steps, though about half that was needed to reach a stable approximant (p. 6, Fig. 12, Fig. 13, Fig. 14, pp. 17-18).
- C14: The author describes this as, to his knowledge, the first mechanism for approximating bisimulation metrics with neural networks, shown valid quantitatively on GridWorld and qualitatively on Atari (p. 2, p. 7).

## Implementation notes

- GridWorld net: input 4 (two normalized (x, y) states in [−1, 1]), one hidden layer of 729 units, output 1; γ = 0.99, C = 500, b = 256, Adam lr 0.01 (source: p. 5; p. 13 config).
- GridWorld β: "increased from 0 to 1 by a factor of 0.9" per target update (p. 5); the config calls it bisim_horizon_discount = 0.9, read here as β = 1 − 0.9^k after k target updates (source: assumption).
- GridWorld length: the text says 2,500 steps (p. 5) but the config lists num_iterations = 240000 (p. 13); unresolved, check the code (source: assumption).
- Atari: d^π net with two hidden layers of 16 units, γ = 0.99, C = 500, b = 128, β factor 0.99, Adam lr 7.5e-5 (Pong 0.001), about 600K steps; replay buffer filled by the trained agent in evaluation mode, sampling starts after 10,000 transitions and the buffer keeps filling (source: p. 6).
- Atari sweep, 3 runs per setting: lr 5e-5 to 1e-3, batch 4-128, target period 10-2000, 1-3 hidden layers, 16-128 units; final values in the code's rainbow.gin (source: p. 15).
- φ for Atari is the trained Rainbow's penultimate layer on Dopamine's stack of the last 4 frames (source: p. 6); the paper doesn't say whether gradients reach φ, presumably frozen (source: assumption).
- Batch pairs are all b² combinations inside one sampled batch, the diagonal zeroed (source: pp. 11-12).
- Code: https://github.com/google-research/google-research/tree/master/bisimulation_aaai2020 (source: p. 5 footnote); not read for this note.
- Hardware: one Tesla P100 GPU (source: p. 5).

## Limitations / open questions

- Only deterministic MDPs: stochastic ones need Wasserstein distances from replay samples,
  which is hard without a generative model; a learned transition model is suggested (p. 7).
  Atari was run without sticky actions for this reason (p. 5).
- Architecture and hyperparameters for d^π are "by no means optimal" (p. 7); training
  overshoots and absolute errors aren't bounded (p. 5).
- Atari results are qualitative, on 3 games, using a fixed trained agent's policy and features
  (pp. 6-7); using the losses to learn representations is left to future work (p. 7).
- d_∼ matches actions by label and is pessimistic (pp. 2-3); d^π depends on one policy π.
- Open question: both metrics compare states within one MDP; for cross-game transfer, states of
  different games (different action sets and rewards) would need a between-MDP version, which
  isn't defined or tested here.
- Open question: d^π changes as the PPO policy changes; whether a moving target stays stable as
  an auxiliary embedding loss for a large transformer policy is untested here.

## Related

- related [[serrano2024crossdomain]]: the survey cites this paper as [131], an MDP-similarity measure based on transition dynamics, for judging when transfer helps (its p. 14).

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
