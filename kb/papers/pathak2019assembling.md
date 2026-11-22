---
id: pathak2019assembling
title: "Learning to Control Self-Assembling Morphologies: A Study of Generalization via Modularity"
authors: [Deepak Pathak, Chris Lu, Trevor Darrell, Phillip Isola, Alexei A. Efros]
year: 2019
venue: NeurIPS 2019 conference paper (read as arXiv v2)
url: https://arxiv.org/abs/1902.05546
arxiv: 1902.05546
doi:
pdf_sha256: 675e2c753726abe686c93d6fed4a935c1f3ce05dac903881f29645cb50216323
tags: [reinforcement-learning, modular-policy, message-passing, graph-neural-network, morphology, cross-embodiment, zero-shot, ppo, continuous-control, self-assembly]
status: read
read: 2026-11-21
---

# Learning to Control Self-Assembling Morphologies: A Study of Generalization via Modularity

## TL;DR

Six simulated limbs learn with PPO to link and unlink while one shared per-limb policy (a Dynamic
Graph Network, 32-d messages leaf-to-root) controls whatever body forms. Zero-shot it keeps 62-113%
of training reward with 0.5x or 2x the limbs, 37-96% in new terrains, water and wind.

## Problem

Sensorimotor learning usually starts from a fixed, hand-designed body and learns one controller
for it; such controllers are tied to that body and generalise poorly (Sec. 1, p. 1). The paper
asks whether learning morphology and control together, with a policy that is modular in the
same way as the body, generalises better to test-time changes in the environment and in the
number of limbs (Sec. 1, p. 2). Prior self-reconfiguring modular robots used hand-defined
controllers, and morphology search treated the body as an optimisation target between
generations, not as an action within an episode (Sec. 5, p. 9).

## Method

- **Agents.** Identical cylindrical limbs with a 3-DoF motor at the parent end and a free
  child end (Sec. 2, p. 3). A limb can choose to link: a magnet in its parent end docks the
  closest child end within about 1.33 limb lengths; nothing happens if none is in range. A child
  can unlink. Linked limbs form a single agent sharing one reward. Cycles are disallowed
  (Sec. 2, p. 4). Simulated in Unity ML-Agents with MuJoCo-like contact and control settings.
- **Actions.** Per limb, three continuous torques plus binary link and unlink actions
  (Sec. 3.1, p. 4; Alg. 2, p. 14).
- **Observations.** Local only: the limb's position, velocity, rotation and angular velocity,
  the relative position of the nearest limb it can join, a trinary touch sensor per end
  (floor, limb, nothing), and a 9x9 height grid around its ground projection (Sec. 2, p. 4).
- **Reward.** The reward of a connected component is shared by its limbs, e.g. the max height of
  the component for standing up, distance along X for locomotion (Sec. 3, p. 4; Sec. 4.1, p. 7;
  Alg. 2, p. 14).
- **Dynamic Graph Network (DGN).** One policy network with parameters shared across limbs. Each
  limb maps [own state, sum of its children's messages] to [action, message to its parent];
  messages flow leaves to root in topological order, with zeros for a limb without children
  (Sec. 3.3, pp. 5-6; Alg. 1, p. 13). Because links are actions, the graph changes as a function
  of the policy's own outputs. A no-message variant sets all messages to 0 (Alg. 3, p. 14).
- **Training.** PPO on the sum over limbs of their expected returns (Eq. 1, Sec. 3.3, p. 5),
  always with 6 limbs that start disconnected at random positions (Sec. 4, p. 6).
- **Baselines.** A monolithic policy over the concatenated states and actions of all limbs,
  either with self-assembly (dynamic graph) or with a fixed hand-designed 6-limb linear chain
  (Sec. 4, p. 6). A monolithic policy has fixed input and output sizes, so it can't be run on
  a different number of limbs (Sec. 4.2, p. 8).
- **Evaluation.** The best training checkpoint is run without fine-tuning on more or fewer
  limbs, and on new terrains (bumps, stairs, hurdles, gaps, valley), water (higher drag) and
  wind (random forces at random points). Scores are mean reward over 50 episodes of 1200
  steps, with the percentage of training reward kept (Sec. 4, p. 6; Table 1, Table 2, p. 7).

## Claims

- C1: In the training environments, with 6 limbs, DGN (with messages) beats both monolithic baselines: standing up 17518 vs 4104 (dynamic graph) and 5351 (fixed chain); standing up in wind 18423 vs 4176 and 4500; locomotion 8.71 vs 0.96 and 2.96 (Table 1, p. 7).
- C2: Zero-shot on the number of limbs (trained on 6), DGN keeps 19796 (113%) with 2x limbs and 10839 (62%) with 0.5x limbs on standing up, and 5.47 (63%) and 6.64 (76%) on locomotion; 2x limbs plus strong wind on the wind task gives 15351 (83%). The table labels the smaller count 0.5x, while Sec. 4.2 says the standing and locomotion policies were tested on 12 and 4 limbs and Sec. 1 says 3 or 12 (Table 1, p. 7, Sec. 4.2, p. 8, Sec. 1, p. 2).
- C3: A monolithic fixed-chain baseline retrained from scratch on 4 limbs reaches 7356 on standing up, below DGN's zero-shot 10839 on 4 limbs from a 6-limb policy (Sec. 4.2, p. 8).
- C4: Zero-shot on new standing-up environments, DGN keeps 16871 (96%) in water with 2x limbs, 16803 (96%) in wind and 15853 (90%) in strong wind; the monolithic baselines keep a similar share in wind and strong wind (dynamic 96% and 96%, fixed 85% and 93%), though at about a quarter of DGN's absolute reward (Table 2, p. 7).
- C5: The wind-trained DGN keeps 17384 (94%) in strong wind and 17068 (93%) in water with 2x limbs and strong wind; the baselines keep 96% and 100% of their much lower scores in strong wind (Table 2, p. 7).
- C6: Zero-shot on new locomotion terrains, DGN keeps 6.57 (75%) in water with 2x limbs, 6.39 (73%) on hurdles, 3.25 (37%) on gaps, 6.62 (76%) on bi-modal bumps, 6.6 (76%) on stairs and 5.29 (61%) in a valley; the monolithic baselines are mostly negative, with exceptions: the fixed chain keeps 2.09 (71%) on gaps, a larger share than DGN but a lower score, and the dynamic monolithic policy keeps 0.47 (48%) in the valley (Table 2, p. 7).
- C7: The authors note that in standing up the baselines' generalisation "is also more than 90%" (Table 2 shows 85% for the fixed chain in wind), so there DGN's advantage comes largely from training better, while in locomotion its train-test gap is much smaller than the baselines' (Sec. 4.3, p. 8, Table 2, p. 7).
- C8: Message passing does as well or better than no messages: read off Fig. 4, standing up ends near 14000 with messages vs near 11000 without, standing up in wind near 15500 vs near 9500, and locomotion is similar for both (about 7); the authors attribute the standing-up gain to the need for long-range coordination to build the single good morphology, a linear tower (Fig. 4, p. 6, Sec. 4.1, p. 7).
- C9: A no-message DGN on the fixed 6-limb chain trains much better than the monolithic fixed chain but worse than self-assembling DGN, which the authors read as both software and hardware modularity being needed; read off Fig. 7, in locomotion it stays near 1-1.5 vs about 7 for self-assembling DGN, while in standing up it peaks near 13000 vs near 15800 and its curve crosses DGN with messages near the end of training (both about 12000-13000 at 2500 steps); against the self-assembling DGN without messages, the same no-message policy on a dynamic body, the static one is at or above it for most of standing-up training (below it only around 900-1500 steps; peak near 13000 vs near 11800, about 12000 vs 9300 at 2500 steps), so there the gap is only to DGN with messages, and the authors' ordering holds clearly only in locomotion (Fig. 7, p. 13, Sec. 4.1, p. 8, App. A.4, p. 13).
- C10: The monolithic fixed-chain baseline trains with few limbs but not with 6, in the authors' words it "can perform well on up to three limbs but does not reach the optimum on four to six" (the Fig. 5 caption says 1-3), which they take as evidence that the fixed graph, not the implementation, causes its failure; read off Fig. 5, in standing up 3 and 4 limbs both reach about 7300-8500 while 2, 5 and 6 limbs stay near 4000-5500 and 1 limb near 2500, and in locomotion only 1 and 2 limbs get clearly positive reward (Fig. 5, p. 12, App. A.3, p. 12, Sec. 4, p. 6).
- C11: Read off Fig. 6, the 6-limb standing-up DGN's max reward rises with the number of limbs to about 20000 near 6-7 on the axis and stays around 16000-17500 beyond; the axis starts at 0 limbs and its exact limb counts aren't explained (Fig. 6, p. 13, App. A.3, p. 12).

## Implementation notes

- Each limb's policy is a 4-layer MLP with ReLU; messages are 32-d floats (source: App. A.1, p. 12).
- PPO with RMSProp, learning rate 3e-4, discount 0.995, entropy coefficient 0.01, GAE (advantage) parameter 0.95, batch size 2048; training episodes are 5000 steps, test episodes 1200 (source: App. A.1, p. 12).
- The standing-up reward is max height, so the 4-limb optimum is presumably higher than the
  3-limb one, and their similar Fig. 5 levels (C10) may mean 4 limbs is also short of its optimum
  (source: assumption, from App. A.3, p. 12 and Fig. 5).
- The pseudo-code splits θ into θ1 and θ2 without saying which part each covers; presumably action and message heads or two networks (source: assumption, from Alg. 1, p. 13).
- The PPO objective in Alg. 2 is written with a KL penalty (β KL) over the joint action of all limbs and an advantage from the summed limb rewards, not the clipped surrogate (source: Alg. 2, p. 14).
- Hidden-layer width, the number of PPO epochs, the wind force magnitudes, terrain heights, and the link-action threshold are not given in the paper; the released code on the project page would be the source (source: assumption).
- Messages are summed over children; when an edge is missing the input message is 0 and the output is ignored (source: Sec. 3.3, p. 6, Sec. 4, p. 6).

## Limitations / open questions

- Cycles are disallowed for computational reasons; the authors say the setup extends to general graphs with loopy message passing, but don't test it (Sec. 2, p. 4, Sec. 3.3, p. 5).
- Messages only flow leaves to root; a root-to-leaves pass is mentioned as an alternative but not tested (Sec. 3.3, p. 6).
- Training always uses 6 limbs, and the authors call the results initial steps (Sec. 4, p. 6, Sec. 6, p. 9).
- The authors concede that in standing up the gain may come from better trainability rather than better generalisation (Sec. 4.3, p. 8).
- Open question: Table 1 and Table 2 give no seeds or error bars, only means over 50 test episodes of the best checkpoint; how much would the retained percentages vary across training runs?
- Open question: the table scores (e.g. 17518 for standing up) are above the plotted training curves (peak near 15800 in Fig. 4a), presumably because the table uses the best checkpoint; the paper doesn't explain the gap.
- Open question: is the 0.5x-limb test 3 or 4 limbs (see C2)?
- Open question (survey lens): the analogy to games is a policy decomposed into shared per-entity modules whose composition differs per game; here every module has the same observation and action space, which games' entities wouldn't.

## Related

- related [[huang2020modular]]: SMP (with Pathak as an author) carries the shared per-limb module with messages over to fixed, given morphologies, passing both leaves-to-root and root-to-leaves.
- related [[kurin2021cage]]: argues graph message passing over the morphology adds little to such modular policies; a counterpoint to C8 and C9 here, though tested on SMP and NerveNet, not DGN.
- related [[gupta2022metamorph]]: replaces per-limb message passing with a transformer over limb tokens, for zero-shot control of unseen morphologies.
- related [[devin2016modular]]: earlier modular transfer, with robot-specific and task-specific modules instead of one shared limb module.
- related [[hejna2020morphological]]: morphology transfer via a shared high-level policy and a retrained low level, against zero-shot transfer here.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
