---
id: yoon2023investigation
title: "An Investigation into Pre-Training Object-Centric Representations for Reinforcement Learning"
authors: [Jaesik Yoon, Yi-Fu Wu, Heechul Bae, Sungjin Ahn]
year: 2023
venue: ICML 2023 conference paper (read as arXiv v3)
url: https://arxiv.org/abs/2302.04419
arxiv: 2302.04419
doi:
pdf_sha256: adaa3fdd4c539e72113e0363d210b7d53cc1dc065bbf660e50e45ccf2f8817bc
tags: [reinforcement-learning, object-centric, slot-attention, unsupervised-pretraining, state-representation, benchmark, relational-reasoning, generalization, ppo, transformer]
status: read
read: 2026-11-22
---

# An Investigation into Pre-Training Object-Centric Representations for Reinforcement Learning

## TL;DR

On 4 simple Spriteworld tasks and a CausalWorld reach, PPO over task-agnostic pre-trained slots
(SLATE best) with transformer pooling matches ground-truth states on odd-one-out tasks, where CNN,
VAE and MAE features fail; it doesn't help every task, and segmentation scores don't predict RL.

## Problem

Unsupervised object-centric representations (OCR: slots from IODINE, Slot Attention, SLATE) are
said to help RL with sample efficiency, generalization and reasoning, but were judged mostly by
reconstruction, segmentation or property prediction, not by RL from images. The paper builds a
benchmark to test, task by task, when OCR pre-training helps image-based RL (Sec. 1, pp. 1-2).

## Method

- **Representations** (Table 1, p. 5): single vector (end-to-end CNN; pretrained VAE, MAE-CLS,
  SLATE-CNN), fixed region (end-to-end CNN feature map, CNNFeat; pretrained MAE patches) and
  object-centric (end-to-end MultiCNNs; pretrained SLATE, Slot Attention, Slot-Attention-Large,
  IODINE). Ground-truth states (GT: per object color, shape and size indices plus x, y) are the
  reference (Sec. 3.1, p. 4, App. A.1.1, p. 13).
- **Pooling and policy.** Single vectors go through an MLP; region and slot sets go through a
  transformer encoder (region embeddings get positions; slots are order-invariant). PPO, 2M
  steps, 3 seeds (Sec. 3.1, p. 4, Sec. 4, pp. 5-6, App. A.2, p. 14).
- **Pre-training data.** For the 2D tasks, one task-agnostic dataset of 1M random scenes
  (100K validation) of 5 objects from 4 colors, 4 shapes and 2 sizes, shared by all 2D tasks; for
  the 3D task, 1M observations from a random policy (Sec. 3.2, p. 4, App. B.1-B.2, pp. 14-15).
- **Tasks** (Sec. 3.2, pp. 4-5, Fig. 3, p. 3): the agent is a red ball moving in 4 directions in
  64x64 Spriteworld scenes. Object Goal: reach the blue square without touching distractors.
  Object Interaction: push the blue square to a goal area. Object Comparison: reach the odd
  object out. Property Comparison: reach the object whose color or shape is unique. Object
  Reaching: a CausalWorld tri-finger arm touches the blue cube first, from images only.
- **Tests.** Success rate after 2M steps; sample efficiency; wall-clock; unseen numbers of
  objects, unseen colors and shapes; pooling ablations; correlation of OCR metrics with RL
  (Sec. 4, pp. 6-8, App. D, pp. 16-21).

## Assumptions

- A1: Scenes are visually simple, so that OCR models segment them well and RL results aren't confounded by bad segmentation; 2D scenes are 64x64, with no occlusion between the agent and objects (Sec. 3.2, p. 4, Sec. 5, p. 9, App. B.1, p. 14).
- A2: Encoders are pre-trained on a task-agnostic dataset gathered by random scenes or a random policy, which assumes a random policy sees diverse enough observations (Sec. 3.2, p. 4, App. E, p. 21).
- A3: The policy is PPO with an MLP or transformer pooling layer over the encoder's output, trained 2M steps and averaged over 3 seeds (Sec. 3.1, p. 4, Sec. 4, pp. 5-6).
- A4: Ground-truth states encode color and shape as integer indices from fixed sets (App. A.1.1, p. 13).
- A5: Out-of-distribution tests change only object count, color or shape within the same task and scene type (Sec. 4, p. 7, App. C.1, p. 15, App. D.1, p. 16).

## Claims

- C1: On Object Goal every model succeeds more than 80% of the time except VAE (0.674) and MAE-CLS (0.448), so explicit objects aren't needed when the task needs no interaction modeling [A1, A3] (Sec. 4, p. 6, Fig. 4, p. 5, Table 12, p. 22).
- C2: On Object Interaction only pretrained SLATE (0.787) and the end-to-end CNN (0.784) approach GT (0.795); Slot Attention (0.000), Slot-Attention-Large (0.047) and IODINE (0.003) fail, which the authors ascribe to SLATE's transformer decoder suiting a transformer policy [A1, A3] (Sec. 4, p. 6, Table 12, p. 22).
- C3: On the two comparison tasks, OCR pre-training performs like GT (SLATE 0.979 and 0.980, Slot Attention 0.975 and 0.947, GT 0.973 and 0.968; IODINE 0.698 and 0.624); end-to-end MultiCNNs reach about 50%, and the fixed-region CNNFeat and MAE-Patch fail (0.19-0.22) despite transformer pooling [A1, A3] (Sec. 4, p. 6, Table 12, p. 22).
- C4: OCR pre-training is more sample-efficient than the other pre-trained representations, and than end-to-end training on the comparison tasks, but less sample-efficient than end-to-end training on Object Interaction [A1, A3] (Sec. 4, pp. 6-7, Fig. 5, p. 7).
- C5: With unseen numbers of objects, OCR pre-training generally keeps its performance, though Object Interaction success is low (SLATE generalizes comparably there); GT and CNN generalize comparably on Object Goal, CNN also on Interaction (GT, with MLP pooling, isn't evaluated there), and on the comparison tasks OCR and GT hold up, e.g. SLATE 0.923 and 0.913 with 6 objects [A3, A5] (Sec. 4, p. 7, Fig. 6, p. 8, Table 13, p. 22, Table 14, p. 27).
- C6: With unseen colors on the comparison tasks, every model above chance drops; GT falls from 0.933 and 0.943 to 0.183 and 0.293 with two unseen colors (attributed to the unseen integer index), while SLATE falls to 0.647 and 0.590; OCR is more robust than GT except on Property Comparison with one unseen color (GT 0.663, SLATE 0.593) [A4, A5] (Sec. 4, p. 7, Fig. 6, p. 8, Table 16, p. 27).
- C7: With unseen shapes, OCR also holds up better than GT: Object Comparison GT 0.933 to 0.213 with two unseen shapes, SLATE 0.950 to 0.447, Slot Attention 0.937 to 0.573 [A4, A5] (App. D.1, p. 16, Table 3, p. 16).
- C8: On the CausalWorld Object Reaching task, where SLATE's segmentation splits objects across slots and misses the finger, SLATE still has better sample efficiency and final success than GT, CNN and VAE [A2, A3] (Sec. 4, pp. 7-8, Fig. 7, p. 8, Fig. 8, p. 16).
- C9: Pooling matters: replacing SLATE's transformer pooling by an MLP gives 0.9, 0.03, 0.238 and 0.229 on Goal, Interaction, Object and Property Comparison, against 0.985, 0.787, 0.979 and 0.98; Deep Sets and Relational Network pooling are also far worse for SLATE (0.103 and 0.093 on Interaction, 0.513 and 0.150 on Property Comparison) [A3] (Sec. 4, p. 8, Table 2, p. 8, Table 7, p. 18, Table 8, p. 18).
- C10: Fewer or more objects show a binding problem in single vectors: Object Goal with 6 objects gives SLATE 0.95 vs CNN 0.746, and with one object VAE matches SLATE (0.999 vs 0.997), but with 3 objects VAE drops to 0.686 (Interaction 0.345 vs SLATE 0.787) [A1, A3] (App. D.4, p. 18, Table 5, p. 17, Table 6, p. 17).
- C11: SLATE pre-trained without the test colors and shapes (cyan, pink; star_5, spoke_4) still succeeds 0.963 on Object and 0.9 on Property Comparison, as long as it segments the objects [A2, A5] (App. D.8, p. 20, Table 9, p. 19, Fig. 11, p. 20).
- C12: Standard OCR metrics don't predict RL performance: FG-ARI is negatively correlated with RL success on all four tasks, MSE and property accuracy are inconsistent, and SLATE has the worst FG-ARI and MSE but similar or better RL [A1, A3] (App. D.9, p. 21, Fig. 12, p. 21, Table 10, p. 20, Table 11, p. 20).
- C13: In wall-clock time OCR pre-training is efficient on the comparison tasks and comparable on Object Goal, but SLATE's compute widens its gap to the end-to-end CNN on Object Interaction; IODINE is too slow to plot with the others (its x-axes run to 112-188 relative hours) [A3] (App. D.3, p. 17, Fig. 9, p. 17, Fig. 16, p. 26).
- C14: On harder comparison tasks (3 colors or 3 shapes, up to 4 colors, 3 shapes and 2 sizes), learning slows and no model solves Property Comparison with two sizes within 2M steps, but SLATE stays comparable to GT [A1, A3] (App. D.7, p. 19, Fig. 10, p. 19).

## Implementation notes

- Transformer pooling: hidden 128, 8 heads, 1 or 3 layers tuned per task and encoder; MLP pooling: 2 layers of 128, ReLU (source: App. A.2, p. 14).
- PPO: stable-baselines3, lr 3e-4, 4 environments, rollout 2048 or 8192 steps, entropy coefficient from {0, 0.01, 0.03, 0.05, 0.1}, other settings tuned per task and model (source: App. A.2, p. 14).
- SLATE: DVAE vocabulary 4096, slot dim 192, 3 iterations, 4-layer 4-head decoder, batch 24; Slot Attention: slot dim 64, 7 iterations; the large one: slot dim 192, 3 or 7 iterations (source: Tables 17-18, p. 28). The number of slots varies per task and isn't listed (source: App. A.1.6, p. 14).
- MAE: ViT-Base, masking ratio 0.5, patch size 16; VAE latent 256, KL weight 5 (source: App. A.1.3-A.1.4, p. 13).
- Whether the pre-trained encoder is frozen during RL isn't stated (source: assumption that it is, since "pre-training" is contrasted with end-to-end learning, Table 1, p. 5).
- Benchmark and code: sites.google.com/view/ocrl/home (source: p. 1).

## Limitations / open questions

- The text attributes the sub-80% Object Goal results to "pre-trained single-vector" models (p. 6), but SLATE-CNN, also single-vector (Table 1, p. 5), reaches 0.905 (Table 12, p. 22).
- The authors call the benchmark very synthetic; segmentation quality was kept high on purpose, and only one 3D task checks imperfect segmentation (Sec. 5, p. 9).
- Pre-training relies on a random policy seeing enough diverse observations, which doesn't hold in many tasks (App. E, p. 21).
- Reader's observation: GT encodes color and shape as integer indices (A4), so its collapse on unseen colors and shapes is partly an encoding choice, as the authors note (Sec. 4, p. 7).
- For the synthesis (reader): slots come from pixels with no per-game engineering, unlike RAM extraction ([[delfosse2024ocatari]]), but this paper doesn't test Atari or game frames.
- Testable cheaply (reader): train a SLATE or Slot Attention encoder on random-policy Atari frames and compare its RL with OCAtari object lists on one game.

## Related

- uses [[schulman2017proximal]]: every policy is trained with PPO.
- uses [[vaswani2017attention]]: transformer pooling over slots is the key policy component.
- uses [[he2021masked]]: MAE-CLS and MAE-Patch are pre-trained baselines.
- related [[delfosse2024ocatari]]: hand-built RAM object extraction, the alternative to unsupervised slots.
- related [[kirillov2023segment]]: a promptable segmenter as another pixel-to-object front end.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
