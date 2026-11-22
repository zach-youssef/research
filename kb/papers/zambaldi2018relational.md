---
id: zambaldi2018relational
title: Relational Deep Reinforcement Learning
authors: [Vinicius Zambaldi, David Raposo, Adam Santoro, Victor Bapst, Yujia Li, Igor Babuschkin, Karl Tuyls, David Reichert, Timothy Lillicrap, Edward Lockhart, Murray Shanahan, Victoria Langston, Razvan Pascanu, Matthew Botvinick, Oriol Vinyals, Peter Battaglia]
year: 2018
venue: arXiv preprint
url: https://arxiv.org/abs/1806.01830
arxiv: 1806.01830
doi:
pdf_sha256: 4b24fcd89d518fea13f9a2709eb86ecfede8ba832920453fbfe7a22c15f43443
tags: [reinforcement-learning, relational-reasoning, self-attention, transformer, entity-representation, generalization, starcraft, box-world, impala]
status: read
read: 2026-11-22
---

# Relational Deep Reinforcement Learning

## TL;DR

Treats CNN feature-map cells (plus x, y coordinates) as entities and runs shared multi-head
self-attention blocks over them inside an IMPALA agent; on Box-World it solves >98% of levels and
generalises zero-shot to longer box chains and unseen key-lock pairs; strong SC2 mini-game scores.

## Problem

Deep RL agents are sample-inefficient and fail to generalise to small changes in a task (Sec. 1,
p. 1). Relational RL (RRL) gets generalisation from first-order state representations and
background knowledge, but needs them hand-specified (Sec. 2, p. 2). The paper puts relational
inductive biases into the architecture instead: non-local pairwise computation with a shared
function, applied iteratively (Sec. 3, p. 2).

## Method

- **Entities from pixels.** A CNN gives k feature maps of size n×n; x and y coordinates are
  appended to each cell's feature vector, and the n² vectors form the entity matrix E; no
  object segmentation, so entities are agnostic to what an object is (Sec. 3, pp. 3-4).
- **Attention block.** Multi-head dot-product attention over E (Eq. 1, p. 3), queries, keys and
  values layer-normalised; heads concatenated, passed through a 2-layer MLP, added residually
  and layer-normalised. Blocks are stacked with shared (recurrent) or unshared (deep) weights
  to reach higher-order relations (p. 3, Fig. 2).
- **Box-World agent.** Two conv layers (12 and 24 kernels, 2×2, stride 1), coordinate channels in
  [−1, 1], a variable number of shared-weight attention blocks, feature-wise max-pooling, a small
  MLP to policy logits and value; IMPALA with 100 actors and one GPU learner, mini-batches of
  32 trajectories (Sec. 3, p. 4). Control agent: residual conv blocks in place of attention
  (p. 4, App. A, p. 12).
- **StarCraft II agent.** Two residual blocks (3 conv layers, 3×3, 32 channels each), a
  ConvLSTM after them because action consequences aren't always visible later, attention
  blocks, and heads for action, value, non-spatial arguments and spatial arguments (upsampled
  from the relational module) (Sec. 3, p. 4, App. B, pp. 12-14).
- **Box-World task.** 12×12 room; the agent (one dark grey pixel) moves in 4 directions; loose
  keys are single pixels; a box is a lock pixel (right, its colour names the key) and a content
  pixel (left); the held key shows in the top-left corner; the gem is white (Sec. 4.1, p. 5).
  Each level is a random tree: a solution path and distractor branches that dead-end; keys are
  used once. Difficulty knobs: solution length, number of distractor branches and their length
  (p. 5). 20 keys and 20 locks; reward +10 for the gem, +1 for a solution-path box, −1 for a
  distractor box, which ends the level (App. A, p. 12).
- **Evaluation.** Box-World: fraction of levels solved, training on solution lengths 1-4 with 0-4
  distractor branches of length 1 or 3, and zero-shot tests on held-out level types (pp. 6-7).
  SC2: full action set, mean score over 30 episodes per mini-game (Sec. 4.2, p. 8).

## Assumptions

- A1: Entities are cells of a CNN feature map tagged with their spatial coordinates; no object segmentation or symbolic state is given (Sec. 3, pp. 3-4).
- A2: Box-World levels are procedurally generated from a random tree, with keys and boxes at random positions, so no level is likely to repeat and held-out level types can be made by construction (App. A, p. 12, Sec. 4.1, p. 5).
- A3: Box-World is fully observed: a single small pixel grid shows every key, lock, box and the held key (reader's inference, Sec. 4.1, p. 5).
- A4: Agents are trained with IMPALA actor-critic at large scale (up to about 1.4e9 Box-World environment steps, 10 billion steps per SC2 mini-game) (Sec. 3, p. 4, Fig. 3, p. 5, App. B, p. 12).
- A5: Train and test levels share the object vocabulary (key and lock colours, the agent, the gem) and the game's rules; generalisation tests change only the composition (chain length, key-lock pairing, number of units) (reader's inference, Sec. 4.1, p. 7, Sec. 4.2, p. 9).
- A6: SC2 Table 1 scores are the best configuration of a sweep of about 100 experiments per mini-game (over heads, blocks, entropy cost and learning rate, 3 seeds each); the paper gives the sweep but does not say how Table 1's numbers were selected (reader's inference, App. B, p. 13, Table 4, p. 15).

## Claims

- C1: Relational agents solve more than 98% of Box-World levels in both training variants (distractor branches of length 1 and 3); one attention block suffices for length-1 branches while length-3 branches need more blocks, and control agents solve under 75% across both variants [A1, A2, A3, A4] (Sec. 4.1, p. 6, Fig. 3, p. 5).
- C2: The relational agent also learns faster on Box-World than the control agents, which plateau near 0.7 fraction solved over 1.4e9 steps (branch-length-1 variant; the length-3 panel has no baseline curves; read from the figure) [A1, A2, A3, A4] (Fig. 3, p. 5, Sec. 5, p. 9).
- C3: With backward branching (any key opens one box, so a reactive policy works) the control agent reaches about 0.95 solved, but with forward branching, which needs planning, about 0.5; the relational agent solves both (read from the figure) [A1, A2, A3, A4] (Sec. 4.1, p. 6, Fig. 6, p. 12).
- C4: Attention heads show interpretable relations: in one head keys attend to the locks they open, in another to the agent; with one-hot inputs (where key-lock colour matching is arbitrary) keys attend to their locks, locks to their keys, all objects to the agent, and agent and gem to each other [A1, A2] (Sec. 4.1, pp. 6-7, Fig. 4, p. 6).
- C5: Zero-shot on levels needing longer box chains (6, 8, 10; training used up to 4), the relational agent solves more than 88% at every length while the control agent falls to 5% at 6 and 0% at 8 and 10 [A1, A2, A3, A4, A5] (Sec. 4.1, p. 7, Fig. 5a, p. 7).
- C6: Zero-shot on levels needing a key-lock combination seen during training only on distractor paths, the relational agent solves 97% and the control agent 13% [A1, A2, A3, A4, A5] (Sec. 4.1, p. 7, Fig. 5b, p. 7).
- C7: A random policy solves 2.3% of levels with solution length 1 and 0.0% with length 4 [A2] (p. 6).
- C8: On the 7 SC2LE mini-games the relational agent scores 27, 196, 62, 303, 736, 4906 and 123, the best of the compared RL agents on six (tied with the control agent on Move To Beacon; humans score higher on Move To Beacon, Build Marines and Collect Minerals And Gas) and above the StarCraft grandmaster on four (Collect Mineral Shards, Find And Defeat Zerglings, Defeat Roaches, Defeat Zerglings And Banelings) [A1, A4, A6] (Sec. 4.2, p. 8, Table 1, p. 8).
- C9: The text says the relational agent is equal or better than the control agent on all mini-games, but Table 1 gives the control agent 5055 against 4906 on Collect Minerals And Gas; elsewhere the gaps are 0-134 points [A1, A4, A6] (Sec. 4.2, p. 8, Table 1, p. 8).
- C10: Both agents beat the previous best (FullyConv LSTM, PBT-A3C) by a wide margin, which the authors attribute to a better RL algorithm, hyperparameter tuning, longer training, architecture and action selection, not only the relational module [A4, A6] (Sec. 4.2, p. 8).
- C11: Trained on Collect Mineral Shards with 2 marines and tested without training on 1, 3, 4, 5 or 10, the relational agent's best seeds score higher than the control's at 3-10 marines (mean of the ten best seeds), but results vary widely and the effect shrinks with larger models [A1, A4, A5, A6] (Sec. 4.2, p. 9, Fig. 7, p. 13).

## Implementation notes

- Box-World: entropy cost 0.005, γ 0.99, unroll 40, 2-4 heads, d = 64, 4 FC layers of 256 after max-pooling, RMSprop (momentum 0, ε 0.1, decay 0.99), lr tuned in 1e-5 to 2e-4; A3C reproduces the results more slowly (source: App. A, p. 12).
- Box-World control: 3-6 residual blocks, each two 3×3 conv layers, stride 1, 26 channels (source: App. A, p. 12).
- SC2: Adam (β 0.9, 0.999, ε 1e-8), batch 32, unroll 80, γ 0.99, baseline loss scale 0.1, gradient-norm clip 100, linear decay of lr and entropy; ConvLSTM 96 channels, 3×3; Conv2DTranspose 16 channels, 4×4, stride 2 (source: App. B, pp. 12-14, Table 2, p. 14).
- SC2 attention: 2-layer MLP of 384 units, embedding size 32, MLP and embeddings shared across blocks; swept heads {1, 3}, blocks {1, 3, 5}, entropy {1e-1, 1e-2, 1e-3}, lr {1e-4, 1e-5} (source: Table 3, p. 14, Table 4, p. 15).
- SC2 inputs: minimap, screen, player and last-action; numerical features log-rescaled, categorical ones embedded in 10 dimensions; spatial encoders 4×4 stride-2 conv then a residual block, giving 8×8 maps (source: App. B, p. 13).
- SC2 control: a 12-layer residual network (4 blocks of 3 conv layers, 32 channels) in place of attention (source: App. B, pp. 13-14).
- Box-World is said to be released later; no code link in this version (source: p. 5, footnote 2).
- Number of heads and blocks used for the Box-World generalisation runs is not stated (source: assumption; guess: the best training configuration, 2 or 4 blocks per Fig. 3).

## Limitations / open questions

- SC2 generalisation results are highly variable and weaker for larger models, which may overfit the training set; more work is needed (Sec. 4.2, p. 9).
- Future work: better structured perception (beyond feature-map cells) and hierarchical RL or planning (Sec. 5, p. 9).
- The ICLR 2019 version of this work is retitled "Deep reinforcement learning with relational inductive biases"; this note describes arXiv v2 (reader's note).
- Open question (reader): entities are tied to one game's CNN features and coordinates (A1); whether the same attention module transfers across games with different visuals is not tested, only within-game recomposition (A5).
- Open question (reader): Box-World's generalisation tests keep all colours and rules (A5), so they measure compositional, not cross-domain, transfer.

## Related

- related [[vaswani2017attention]]: the multi-head dot-product attention used as the relational module.
- related [[shang2021agentcentric]]: also adds attention as an inductive bias, across agents' features in cooperative MARL.
- related [[delfosse2024ocatari]]: object-centric Atari states, an alternative to this paper's feature-map-cell entities.
- related [[derstroff2026relative]]: a set transformer over extracted objects with positions relative to an anchor object, instead of feature-map cells with absolute coordinates.
- related [[yoon2023investigation]]: tests pre-trained object-centric representations for RL, the structured perception this paper names as future work.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
