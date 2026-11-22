---
id: shang2021agentcentric
title: Agent-Centric Representations for Multi-Agent Reinforcement Learning
authors: [Wenling Shang, Lasse Espeholt, Anton Raichuk, Tim Salimans]
year: 2021
venue: arXiv preprint
url: https://arxiv.org/abs/2104.09402
arxiv: 2104.09402
doi:
pdf_sha256: e9276c610d0bdc980b7f4e18602c1477551e357f62f48954dee154bbef73fc62
tags: [reinforcement-learning, multi-agent, object-centric, attention, auxiliary-loss, pretraining, learning-from-observation, state-representation, games, football]
status: read
read: 2026-11-22
---

# Agent-Centric Representations for Multi-Agent Reinforcement Learning

## TL;DR

Adds agent-centric biases to a shared-weight cooperative MARL agent (V-trace): attention across
agents' features (ACNN), and each agent predicting its own next location (auxiliary loss, or
pretraining on action-free replays). In Google Research Football, only ACNN beats the self-play AI.

## Problem

Object-centric inductive biases had helped relational reasoning in manipulation and VQA, but
were under-explored in multi-agent RL (Sec. 1, p. 1). Earlier attention-based MARL put
self-attention on a single network or, as MAAC, gave each agent its own critic (p. 1).
Observations of others playing, without action labels, are often available (football
broadcasts, FPS videos) and unused (Sec. 2.3, p. 4).

## Method

- **Setting.** Fully cooperative MAMDP with a shared reward; each agent i gets s^i, the global
  state plus a specification of its own location, and one model (shared weights, separate
  forward passes) outputs π^i and V^i; trained with V-trace in SEED RL (Sec. 2.1, pp. 2-3, Eq. 1-2).
- **ACNN.** Each agent's CNN encoding z^i yields key, query and value after LayerNorm;
  multi-head attention runs across the N agents' features, and LN(z^i + ṽ^i) feeds the policy
  and value heads (Eq. 3, Sec. 2.2, pp. 3-4, Fig. 1b, p. 2). A 2-head module replaces the FC
  layer after the CNN (Sec. 4.1, pp. 6-7).
- **Agent-centric prediction.** Each agent predicts its own next-step location as two softmax
  heatmaps over the map's height and width, trained by cross-entropy (Eq. 4, Sec. 2.3, p. 4).
  It is used as an auxiliary loss next to the policy and value heads (Sec. 2.3.1, p. 5), or for
  pretraining from action-free replays, transferred by weight initialization or as a frozen
  column of a progressive network (PrCNN, PrACNN) (p. 4, Fig. 1c, p. 2).
- **Environments.** Google Research Football (GRF) 5-vs-5, every home player controlled (and the
  goalie too against the built-in AI), 19 actions, 3000 frames per game, against the rule-based
  built-in AI or a self-play AI; Super Mini Map input (72×96, 4 channels, 4 frames stacked).
  DeepMind Lab 2D "Cleanup" with 4 agents and 72×96 RGB frames. Ablation: GRF 11-vs-11 "Hard
  Stochastic" with one active player (Sec. 3, p. 5).
- **Evaluation.** Learning curves averaged over 3 seeds without error bars (Sec. 3.3, p. 6); a
  round-robin tournament with ELO ratings and the public GRF multi-agent leaderboard (Sec. 5,
  p. 10).

## Assumptions

- A1: The task is fully cooperative: all controlled agents share one team reward (Sec. 2.1, p. 2, Sec. 3.1, p. 5).
- A2: All agents share one model's weights and each acts on its own forward pass; the attention module connects their intermediate features at execution time (Sec. 2.1, p. 3, Sec. 2.2, p. 3).
- A3: Each agent's input is the global 2D map plus its own location, i.e. the agents see the whole field (Sec. 2.1, p. 3, Sec. 3.1, p. 5).
- A4: Observations are low-resolution 2D maps (GRF Super Mini Map or 72×96 DMLab2D frames) with four stacked frames (Sec. 3.1-3.2, p. 5).
- A5: Each agent's location is available as a training target, from the map or extractable by a detector (Sec. 2.3, p. 4).
- A6: Pretraining data are about 300K action-free frames per task, recorded during baseline RL training or from self-play checkpoints, so they come from the same environment and agents (Sec. 4.2, pp. 7-8).
- A7: Rewards are shaped: ±1 per goal plus +0.1 for the ball carrier advancing; the shaping keeps the policy-gradient direction approximately, per the authors' appendix (Sec. 3.1, p. 5, Sec. 2.1, p. 3).
- A8: Opponents are fixed: GRF's rule-based built-in AI, which can be exploited by offside tricks, or a self-play-trained AI (Sec. 3.1, p. 5).
- A9: Training is large-scale (500M frames against the built-in AI, 4.5G against self-play, 100M on DMLab2D, 900M single-agent GRF), with learning rate and auxiliary weight swept and the best setting shown (Sec. 3.3, p. 6, Sec. 4.3, p. 8).
- A10: The number of controlled agents is fixed per environment (5 home players or 4 Cleanup agents) (reader's inference, Sec. 3, p. 5).

## Claims

- C1: Against the self-play AI, ACNN models pass the zero-return mark (start winning) and none of the CNN models do; the authors read this as attention providing the reasoning for complex cooperative strategies [A1, A2, A3, A4, A8, A9, A10] (Sec. 4.3.1, p. 8, Fig. 5, p. 6).
- C2: In 2 of 3 environments (GRF self-play, DMLab2D Cleanup) PrACNN is the best model, and on DMLab2D it converges faster and more stably than its CNN counterpart [A1, A2, A3, A4, A5, A6, A9, A10] (Sec. 4.3.1, p. 8, Fig. 6, p. 7).
- C3: Against the built-in AI, CNN models score more than ACNN in absolute terms, which the authors attribute to exploiting the built-in AI's offside weakness rather than to better play [A1, A3, A4, A8, A9] (Sec. 4.3.1, p. 8, Fig. 5, p. 6).
- C4: Attention weights concentrate around the ball, and two home agents attend differently, e.g. one to the ball holder and one to the player being passed to (qualitative) [A2, A3] (Sec. 4.3.1, p. 8, Fig. 2, p. 3).
- C5: The agent-centric auxiliary loss mostly improves performance and sample efficiency, especially for CNN models and on both GRF tasks; against the built-in AI CNN+aux reaches about 9.5 return at 500M frames vs about 1 for CNN (read from the figure) [A1, A2, A3, A4, A5, A7, A8, A9] (Sec. 4.3.2, pp. 8-9, Fig. 5, p. 6).
- C6: Pretraining on action-free observations, by initialization or as a progressive column, improves sample efficiency, most on DMLab2D and GRF against the built-in AI; it never hurt final performance, progressive networks sometimes beat initialization, and with the auxiliary loss the gain is smaller [A1, A2, A3, A4, A5, A6, A9] (Sec. 4.3.2, p. 9, Fig. 6, p. 7).
- C7: In single-agent GRF 11-vs-11 Hard Stochastic, attention lowers performance (cooperation matters less, attention harder to optimize), while the auxiliary loss and pretraining still help, to a lesser degree [A3, A4, A5, A6, A9] (Sec. 4.4, p. 9, Fig. 7, p. 9).
- C8: Predicting each agent's own location beats the agent-agnostic observe-all target (binary occupancy masks of all players) as auxiliary loss, initialization and progressive column against the built-in AI [A1, A2, A3, A4, A5, A6, A8, A9] (Sec. 4.5, pp. 9-10, Fig. 8, p. 9).
- C9: In a round robin of 20 matches per pair among agents trained against the self-play AI, ACNN variants rate 24 to 182 ELO and CNN variants −420 to −283; pretrained ACNNs (24 init, 35 progressive) rate below from-scratch ones (147, 182 with aux), which the authors read as pretraining possibly limiting generalization [A1, A2, A3, A4, A6, A8, A9, A10] (Sec. 5, p. 10, Table 1, p. 10).
- C10: On the public GRF multi-agent leaderboard (300 games per agent), ACNN+aux rates 1992 and ACNN 1841, above the self-play AIs CNN-v1/v2 (1659, 1630); CNN+aux trained against the built-in AI rates 1048, near the built-in AI's 1000; ACNN+aux ranked first at submission time [A1, A2, A3, A4, A8, A9, A10] (Sec. 5, p. 10, Table 1, p. 10).
- C11: In pretraining, attention gives no significant advantage in prediction NLL, which the authors conjecture is because each agent's location is predicted independently [A5, A6] (Sec. 4.2, p. 8).
- C12: The placement of LayerNorm within the attention module is crucial (stated without data) [A2] (Sec. 2.2, p. 4).
- C13: The authors call their environments relatively stable to random seeds and show all 3 runs of selected models on GRF self-play and DMLab2D; in that figure the DMLab2D CNN runs visibly diverge mid-training (reach 2500 return at about 0.2e8 vs 0.55e8 frames, read from the figure) [A9] (Sec. 3.3, p. 6, Fig. 4, p. 6).

## Implementation notes

- RL: V-trace in SEED RL; lr swept over {7e-5, 1.4e-4, 2.8e-4}, auxiliary weight over {1e-4, 5e-4}; value loss weight 0.5, unroll 32, discount 0.99, entropy 5e-4, batch 120 (160 single-agent), TensorFlow Adam defaults otherwise (source: Sec. 3.3, p. 6).
- Compute: GRF multi-agent on 16 TPU cores and 2400 CPU cores, single-agent on 32 TPU cores and 480 CPU cores, DMLab2D on 8 TPU cores and 1472 CPU cores (source: Sec. 3.3, p. 6).
- Pretraining: batch 32, Adam with a learning-rate sweep, trained to convergence on a validation set; heads output 72- and 96-way softmax heatmaps (source: Sec. 4.2, p. 8, Sec. 3.1, p. 5).
- Network sizes, attention dimensions and the progressive column's non-linearity are in supplementary material on the project page, not in the PDF (source: Sec. 4.1, p. 7; not read, use SEED RL's GRF network as a stand-in: assumption).
- Shaping proof, self-play opponent details, action lists, NLL results and ELO method are in appendices on https://sites.google.com/view/agent-centric-marl (source: p. 1, footnote; not read).

## Limitations / open questions

- Attention hurts when cooperation matters less (single-agent GRF), and pretraining may limit generalization in the tournament (Sec. 4.4, p. 9, Sec. 5, p. 10).
- Curves are 3-seed means without error bars, from the best hyperparameter setting (Sec. 3.3, p. 6, Sec. 4.3, p. 8).
- Open question (reader): "agent-centric" here means a per-agent prediction target and attention across agents, not re-expressing positions relative to the agent as in [[derstroff2026relative]]; combining the two (anchor-relative inputs plus own-location prediction) is untested.
- Open question (reader): the own-location target needs agent positions (A5); in a cross-game embedding, a cheap test is whether own-avatar next-position prediction from frames, as an auxiliary loss, transfers across games with different layouts.

## Related

- related [[derstroff2026relative]]: also anchors the representation on the controlled agent, but by relative coordinates of extracted objects rather than a prediction target.
- uses [[vaswani2017attention]]: the agent-centric module is multi-head self-attention across agents.
- related [[kurin2021cage]]: attention over a set of entities (limbs there, agents here) in a shared policy.
- related [[yoon2023investigation]]: object-centric pre-training for RL, the single-agent counterpart of an entity-anchored bias.
- related [[srinivas2020curl]]: another auxiliary representation loss added to RL for sample efficiency.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
