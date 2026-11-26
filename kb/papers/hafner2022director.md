---
id: hafner2022director
title: Deep Hierarchical Planning from Pixels
authors: [Danijar Hafner, Kuang-Huei Lee, Ian Fischer, Pieter Abbeel]
year: 2022
venue: NeurIPS 2022 (arXiv v1 read)
url: https://arxiv.org/abs/2206.04114
arxiv: 2206.04114
doi:
pdf_sha256: 5d228526c900f323249fec95abca2c63ea6a75cd83154560cad7d4b9db990762
tags: [hierarchical-rl, world-model, goal-conditioned, latent-goals, manager-worker, atari, sparse-rewards, exploration, dreamer, long-horizon]
status: read
read: 2026-11-25
---

# Deep Hierarchical Planning from Pixels

<!-- Version read: arXiv v1 (8 Jun 2022, "Preprint. Under review"); published at NeurIPS 2022. -->

## TL;DR

A manager picks a discrete-code latent goal every K=8 steps inside a Dreamer world model; a worker is
rewarded for matching it, all trained in imagination. Beats flat Dreamer and Plan2Explore on egocentric
Ant Mazes and Pin Pads; matches Dreamer on Atari/DMC only when the worker also sees task reward.

## Problem

Flat RL reasons at the clock rate of primitive actions, so tasks needing thousands of
decisions with sparse rewards are out of reach (Sec. 1, p. 2). Earlier hierarchical methods
needed hand-specified subtasks or goal spaces (global XY, robot poses), pre-training tasks,
or privileged inputs; FuN (Vezhnevets et al., 2017) learned from pixels but showed no clear
benefit over an LSTM baseline (Sec. 1, Sec. 4, pp. 2, 9). Director wants a general two-level
method learning from pixels with no domain knowledge.

## Method

- World model: the RSSM of PlaNet/DreamerV2 (representation, dynamics, decoder, reward predictor),
  trained on replay-buffer sequences with the variational loss of Eq. 2; state s_t is a
  1024-dimensional continuous vector (Sec. 2.1, Eq. 1-2, Sec. 2.2, p. 3).
- Goal autoencoder: encodes s_t into a vector of 8 categoricals with 8 classes each (a sparse
  one-hot vector, 8 of 64 dims set), decoded back to a 1024-d state; trained with MSE plus KL to
  a uniform prior, straight-through gradients, no gradients into the world model (Sec. 2.2,
  Eq. 3-4, Fig. G.1, pp. 3-4, p. 17).
- Manager: every K=8 steps it samples a code z from mgr(z | s_t); the goal is g = dec(z).
  It maximizes task reward plus an exploration reward, the goal autoencoder's squared
  reconstruction error of s_{t+1} (novel states reconstruct badly), weighted 1.0 and 0.1 after
  normalizing each by an exponential moving standard deviation (Sec. 2.3, Eq. 5-6, p. 4).
- Worker: wkr(a_t | s_t, g) gets only the goal reward, the "max-cosine" similarity
  (g/m)^T (s_{t+1}/m) with m = max(||g||, ||s_{t+1}||) (Sec. 2.4, Eq. 7-8, pp. 4-5). A variant
  also feeds the worker task reward (weights 1.0 goal, 0.5 task) (Sec. 3.2, p. 7).
- Both policies are Dreamer actor-critics trained on the same imagined rollouts; the manager's
  trajectory is abstracted to every K-th state with rewards summed per block of K, the worker's
  is cut into length-K pieces with the goal fixed (Sec. 2.3-2.4, App. H, Eq. 9-12, p. 4, p. 17).
- Training loop: one gradient step per 16 policy steps, 4 parallel envs; goals change on a fixed
  clock (Alg. 1, p. 16).
- Hyperparameters: K=8, imagination horizon H=16, gamma 0.99, MLP 4x512, learning rate 1e-4,
  same across all tasks (Table F.1, p. 16).

## Assumptions

- A1: Pixel observations are enough to learn a useful latent state through the world model's reconstruction loss (Sec. 2.1, p. 3).
- A2: The manager changes goals on a fixed clock, every K=8 steps (Sec. 2.3, Sec. 5, p. 4, p. 9).
- A3: Rewards are either sparse task rewards or a dense per-step reward; the worker's reward is the goal similarity, independent of task reward in the standard variant (Sec. 2.4, p. 5).
- A4: Pin Pad displays the history of activated pads on screen so no long-term memory is needed (Sec. 3.1, p. 7).
- A5: Results use 4 environment instances, one GPU-day budget (V100, under 24 h), and the DreamerV2 default hyperparameters (Sec. 3, p. 5).
- A6: Atari and Control Suite runs use no action repeat (App. J, App. K, p. 19, p. 20).

## Claims

- C1: In egocentric Ant Maze S/M/L/XL (sparse reward for touching the goal, first-person camera plus proprioception, 3000-step episodes at 50 Hz), Director learns to reach the goal on all four mazes, while flat Dreamer succeeds only on the smallest and Plan2Explore on none reliably [A1, A2, A3, A5] (Sec. 3.1, Fig. 4, p. 6).
- C2: On Visual Pin Pad (3-6 pads, one sparse reward for activating all in order, 2000-step episodes), Director solves all four sizes; Plan2Explore gets some reward with four pads, little with five and none with six; Dreamer fails beyond three [A1, A2, A3, A4, A5] (Sec. 3.1, Fig. 5, pp. 6-7).
- C3: Without any task reward to the worker, Director learns across 20 DMC tasks, Atari, Crafter and DMLab, but, read off Fig. A.1/K.1, is below flat Dreamer on about half of the tasks (the paper's text: worker task reward "is not as important as expected"); giving the worker task reward (weight 0.5) closes the gap to DreamerV2 [A1, A2, A3, A5, A6] (Sec. 3.2, Fig. A.1, Fig. J.1, Fig. K.1, p. 7, p. 14, p. 19, p. 20).
- C4: Director with worker task reward reaches a higher human-normalized median than Dreamer over the 55 Atari games [A1, A2, A5, A6] (Sec. 3.2, Fig. K.1, p. 7, p. 20).
- C5: Removing the goal autoencoder (manager picks 1024-d continuous goals) works in some environments but fails on sparse-reward ones such as Ant Maze L, Humanoid Walk and Acrobot Swingup [A1, A3] (App. B, Fig. B.1, p. 14).
- C6: The max-cosine goal reward is robust: inner-product variants work about as well on most tasks, while an L2 goal reward is clearly worse on several (Breakout, Crafter, Acrobot, Ant Maze L) [A1, A3] (App. C, Fig. C.1, p. 15).
- C7: An exploration bonus on the manager is needed for Ant Maze, and a bonus on the worker hurts (chaotic leg movements); on short-horizon standard tasks it is not needed [A1, A2, A3] (App. D, Fig. D.1, p. 15).
- C8: Decoded goals are interpretable: in Ant Maze M the manager walks the agent through maze sections, in Crafter it requests inventory counts via the inventory display, in Pong it asks for a higher score via the score display, and in DMLab it goals the teleport animation [A1, A3] (Sec. 3.3, Fig. 1, Fig. 6, Fig. L.1, pp. 8-9, p. 21).
- C9: Implementing Director costs about 250 lines on top of DreamerV2 and 20% more compute; each run took under 24 h on one V100 [A5] (Sec. 3, p. 5).
- C10: Early in training the manager picks more intermediate subgoals; later, as the worker reaches farther, it picks fewer [A1, A3] (Sec. 3.3, p. 8).

## Implementation notes

- Policies are trained only on imagined rollouts; the actor-critic is Dreamer's, with separate critics and EMA-std-normalized returns per reward stream (source: App. H, p. 17).
- Manager reward blocks: with H=16 and K=8, an imagined trajectory contains only two manager steps (source: assumption, derived from Table F.1, p. 16 and Sec. 2.3, p. 4).
- Entropy coefficient eta, lambda for lambda-returns and the replay/batch sizes are not stated here (source: DreamerV2 defaults, Hafner et al. 2020a, as the paper says it reused them, Sec. 3, p. 5; values not in this paper).
- Code was to be released with the paper (source: Sec. 3, Implementation, p. 5).

## Limitations / open questions

- The manager treats its action space as a black box and goals change on a fixed clock; adapting timing, distributional or masked goals, and more than two levels are left as future work (Sec. 5, p. 9).
- The paper does not explain why Director works beyond the ablations (Sec. 5, p. 9).
- Director without task reward is worse than flat Dreamer on many tasks, for example Freeway where the plot shows it near zero (Fig. K.1, p. 20).
- Open question (reader): for a shmup, the manager's goal is a state in the world model's latent, and Fig. L.1c shows it can cheat by asking for the score display; a score HUD makes a latent goal cheaply specifiable but also hackable.
- Open question (reader): Director's high level sees only the Markov latent s_t, not a memory; Pin Pad avoids the memory problem by showing history on screen (A4). A shmup's shop, weapon and rank state are partly on screen too, but long-term strategy would need memory tokens (see [[bulatov2022recurrent]], [[cherepanov2026recurrent]]) feeding the manager.
- Open question (reader): K=8 world-model steps are 8 frames, about 0.13 s at 60 Hz (no action repeat); a strategist deciding per shop visit would need a much longer or event-triggered interval (the paper suggests switching on goal reached, Sec. 5, p. 9).
- Open question (reader): Director is online RL in a learned model; training a manager from offline gameplay data (as NitroGen does for behaviour cloning) is not tested here.

## Related

- related [[chen2021decision]]: the Decision Transformer conditions on return-to-go; Director's goals are latent states, an alternative conditioning signal.
- related [[lee2022multigame]]: multi-game Atari sequence model without hierarchy.
- related [[magne2026nitrogen]]: flat gamepad policy that a manager-over-worker split could sit on top of.
- related [[hafner2025scalable]]: later Dreamer 4 from the same first author, imagination training in a larger world model.
- related [[bulatov2022recurrent]]: memory tokens as a possible high-level state.
- related [[samsami2024mastering]]: memory inside a Dreamer-style world model, relevant to the memory-less manager.
- related [[ni2023decoupling]]: separates memory length from credit-assignment length, the two things a long-horizon manager needs.
- related [[cherepanov2026recurrent]]: memory tokens for a Decision Transformer, the sort of recurrence Director's manager lacks.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
