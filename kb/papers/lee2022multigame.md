---
id: lee2022multigame
title: Multi-Game Decision Transformers
authors: [Kuang-Huei Lee, Ofir Nachum, Mengjiao Yang, Lisa Lee, Daniel Freeman, Winnie Xu, Sergio Guadarrama, Ian Fischer, Eric Jang, Henryk Michalewski, Igor Mordatch]
year: 2022
venue: NeurIPS 2022 conference paper
url: https://arxiv.org/abs/2205.15241
arxiv: 2205.15241
doi:
pdf_sha256: 38a8e65a4bfaab6f42ab588c0ee681af8d3d36d36f83b0c2e3d6335dfca60edb
tags: [reinforcement-learning, offline-rl, decision-transformer, multi-game, atari, scaling, transfer-learning, sequence-modeling]
status: read
read: 2026-11-21
---

# Multi-Game Decision Transformers

<!-- Read from arXiv v2 (2022-10-15), 24 PDF pages: main text pp. 1-10, references pp. 11-15, NeurIPS checklist pp. 15-16, appendices A-I pp. 17-24 (all read). -->

## TL;DR

One decoder-only Decision Transformer (200M params) trained offline on mixed expert and non-expert
DQN trajectories of 41 Atari games plays all of them at 126% human-normalized IQM (data: 101%),
scales with model size, fine-tunes well to 5 held-out games, and beats BC, CQL and online C51.

## Problem

Generalist models trained at scale on diverse data drive language and vision, but RL mostly trains
small models on one task or on several tasks of one environment; training one policy across
environments with very different dynamics, rewards, visuals and embodiments was little studied
(Sec. 1, p. 1). Earlier multi-game Atari agents learned online or by distillation (Sec. 2, p. 3);
the concurrent Gato trains only on near-optimal data and needs expert-trajectory prompts at
inference (Sec. 2, p. 3).

## Method

- **Sequence** (Sec. 3.1, p. 3, Fig. 3, p. 4): per timestep the tokens are ⟨o¹…o^M, R̂, a, r⟩:
  M image-patch tokens, then the return-to-go R̂ (undiscounted sum of future rewards), the action
  and the reward. Next-token cross-entropy on return, action and reward tokens; future
  observations are not predicted (pp. 3-4). Putting R̂ after the observation (unlike Chen et
  al.'s ⟨R̂, o, a⟩) lets the model predict a return distribution and sample from it.
- **Tokenization** (Sec. 3.2, p. 4, App. B.1, p. 17): 84×84 grayscale frames cut into 6×6 = 36
  patches of 14×14 px, each linearly projected and given a learned patch-position embedding
  (ViT-style). Rewards clipped to [−1, 1] become ternary {−1, 0, +1}; returns are uniformly
  quantized into one range shared by all games, {−20, …, 100} with bin size 1 (footnote 2, p. 4;
  App. B.1 says 120 buckets, Alg. 1 says 121 logits). Atari's shared 18-action set is used for all
  games (App. B.4, p. 18), so no per-game action head is needed.
- **Attention mask** (App. B.1, p. 17): causal, except patch tokens of one timestep may attend to
  each other; a learned positional embedding is added at every sequence position.
- **Expert action inference** (Sec. 3.4, p. 5, Fig. 4, Alg. 1, p. 19): with a binary "expert"
  classifier P(expert|R) ∝ exp(κR), Bayes gives log P(R|expert, …) = log P_θ(R|…) +
  κ(R − R_low)/(R_high − R_low); sample R̂ from it (κ = 10, R_low = −20, R_high = 100), then sample
  a ~ P_θ(a|R̂, …). Training is unchanged; only inference differs. Target returns are regenerated
  autoregressively each step, so no state is kept outside the model (App. B.3, p. 18).
- **Data** (Sec. 3.3, pp. 4-5): Agarwal et al.'s DQN-Replay dataset; 46 games where DQN clearly
  beats random, 41 for training and 5 held out (Alien, MsPacman, Pong, SpaceInvaders,
  StarGunner, chosen as different game types). Per game 2 DQN training runs × 50 checkpoints ×
  1M steps: 4.1B steps, almost 160B tokens. No filtering, curation or balancing between games.
- **Models** (Sec. 4.1, p. 6, Table 1, p. 18): GPT-2-style decoders DT-10M (4 layers, d 512,
  8 heads), DT-40M (6, 768, 12), DT-200M (10, 1280, 20); context 4 frames = 156 tokens.
- **Baselines** (Sec. 4.2, pp. 6-7, App. C, pp. 19-20): BC = the same transformer without return
  tokens; online multi-game C51 DQN on an Impala CNN; offline CQL (C51 + BC loss) on Impala at
  5M-120M params; CPC, BERT and ACL representation pretraining with a BC head for fine-tuning;
  UDRL with an Impala CNN and 4 stacked frames (App. D, pp. 20-21).
- **Metric** (Sec. 4.1, p. 6): human-normalized score (HNS) aggregated by interquartile mean
  (IQM); DQN-normalized scores use the best DQN score in the dataset; one training seed.

## Claims

- C1: A single DT-200M trained offline on 41 games reaches 126% human-normalized IQM, above the 101% IQM of its training dataset (Fig. 1, p. 2, Sec. 1, p. 2)
- C2: Among multi-game generalists DT is best (126%), ahead of multi-game BC 68%, online multi-game C51 DQN (Impala) 66% and offline CQL (Impala) 38%; single-game specialists remain stronger (online DQN 144%, offline BCQ 135%) (Fig. 1, p. 2, Sec. 4.3, p. 7)
- C3: DT's training-game IQM increases reliably with size (about 56%, 93%, 126% for 10M, 40M, 200M), BC rises more slowly (46%, 54%, 68%), and CQL stays flat or falls (23%, 38%, 38%, 35% at 5M, 30M, 60M, 120M) (Fig. 5a, Sec. 4.4, pp. 7-8)
- C4: On the 5 held-out games after fine-tuning, DT's IQM grows with model size (72% at 10M to about 91-94% at 200M; a grid line hides one digit of the label), while CQL's is inconsistent and drops sharply for the largest model (46%, 26%, 50%, 6%) (Fig. 5b, p. 7, Sec. 4.4, p. 8)
- C5: The authors attribute CQL's "inverse" scaling to TD methods' instability with larger models; offline C51, DQN and DDQN objectives did even worse and are not reported (Sec. 4.4, p. 8, App. C, p. 19)
- C6: Fine-tuning for 100k steps on 1% of each held-out game's data, DT pretraining gives the best result of all methods, and every pretraining method beats CQL trained from scratch on that 1% (Fig. 6, Sec. 4.5, p. 8)
- C7: CPC and BERT pretraining fine-tune worse than DT, which the authors read as state representations alone not sufficing for transfer; ACL's added action prediction barely helps over BERT (Sec. 4.5, p. 8, Fig. 6)
- C8: The top-3 DT rollouts beat the best episode in the training data by a large margin in a number of games, though not in all (Fig. 7, Sec. 4.6, pp. 8-9)
- C9: Expert action inference beats BC (same transformer, no return conditioning) on 31 of 41 games, by mean DQN-normalized score over 16 trials (Fig. 8, Sec. 4.7, p. 9)
- C10: Training on all data helps DT but hurts BC: with 40M models, DT full data 93% vs top-10%-expert data 89% IQM, BC full 54% vs expert 79%, so DT on full data beats BC on expert data (Fig. 9, Sec. 4.8, pp. 9-10)
- C11: Larger models train faster: at the same number of training steps (tokens seen), 200M reaches higher scores than 40M and 10M (shown for Asterix and Seaquest) (Fig. 15, App. G, pp. 22-23)
- C12: Both the transformer and the return-conditioned (UDRL) formulation contribute, per the authors: UDRL on an Impala CNN scales worse than DT (44%, 67%, 63% and about 71-74% from 5M to 120M; a grid line hides one digit of the last label) but still beats CQL on the same Impala network at every size (Fig. 10, App. D, pp. 20-21)
- C13: By median HNS the ranking is unchanged: single-game DQN 110%, BCQ 97%, multi-game DT 85%, multi-game DQN 68%, BC 55%, CQL 30% (Fig. 11, App. E, p. 21)
- C14: Online multi-game C51 DQN's 68% median is close to the 70% median of multi-game Impala (IMPALA paper) on this suite, computed by the authors from that paper (Sec. 4.3, p. 7)
- C15: Seed variance is small: 5 seeds of DT-40M give median 0.79 ± 0.06 and IQM 0.95 ± 0.05 (App. B.1, p. 18)
- C16: Tokenizing frames through a convolutional network gave no significant benefit over linear patch projection, and rotary position embedding none over learned embeddings (Sec. 3.2, p. 4, App. B.1, p. 17)
- C17: Hyperparameters searched on DT-40M and applied unchanged to DT-200M still give a clear scaling trend in multi-game and transfer performance (App. B.1, p. 18)
- C18: Picking the best of N = 128 sampled returns (inverse temperature 0.75, then top-50% action logits) works about as well as κ-weighted return sampling and needs no κ (App. B.3, p. 18)
- C19: Of the augmentations tried, random cropping (pad 4, crop 84×84) and random rotations work best (App. B.5, pp. 18-19)
- C20: Attention heads consistently attend to patches holding meaningful entities: the player, free movement space, non-player objects and environment features (Fig. 16, App. H, p. 23)

## Implementation notes

- Optimizer: LAMB, lr 3e-4 with 4000 linear warm-up steps, no weight decay, gradient clip 1.0, β = (0.9, 0.999), batch 2048, 10M steps; fine-tuning 100k steps, lr 1e-4, weight decay 1e-2, batch 256; augmentation in both (source: Sec. 4.1, p. 6).
- Compute: 64 TPUv4 per model; 1, 2 and 8 days for DT-10M, 40M, 200M (source: Table 1, p. 18).
- Sampling at evaluation: temperature 1, only the top 85th-percentile logits for actions and returns (nucleus-style), for DT and BC; the last predicted action of the context is executed (source: App. B.3, p. 18).
- Environment: Atari 2600 Gym, preprocessing as Agarwal et al., frames stored and fed as JPEG, no sticky actions, 16 episodes per game score (source: App. B.4, p. 18).
- Fine-tuning data size is stated two ways: "1% (1M steps)" uniformly sampled from the 50M-step dataset (Sec. 4.5, p. 8) and "roughly 500 000 transitions" (App. B.2, p. 18); which is right is not resolved (source: the paper, inconsistent).
- One single-channel frame per timestep (C = 1 is the channel count; no frame stacking for DT), with the context's 4 timesteps standing in for the usual 4-frame stack (source: App. B.1, p. 17; App. D, pp. 20-21 for UDRL's stacking).
- Return, action and reward tokens are presumably looked up from learned embedding tables of size 121, 18 and 3 and read out by linear heads; the paper doesn't describe the embeddings or heads (source: assumption).
- Return-to-go at inference is in the clipped-reward scale shared by all games; the bin range {−20…100} was chosen because it covers most dataset returns, so games with larger clipped returns saturate at 100 (source: footnote 2, p. 4; saturation is the reader's inference).
- Models and code were released via sites.google.com/view/multi-game-transformers (p. 1, footnote 1); not read here (source: the paper).

## Limitations / open questions

- Atari games share aligned observation and action spaces, so conclusions may not hold where they differ; it is unclear whether offline Atari data has the scale and diversity for NLP-like scaling, and zero-shot adaptation was not tested (Sec. 5, p. 10).
- Best generalists still fall short of single-game specialists at times (Sec. 5, p. 10, Fig. 1).
- One training seed per configuration because of training cost (Sec. 4.1, p. 6).
- Varying action spaces across embodiments could be tokenized by other methods, and VQ-VAE could replace uniform quantization; neither is tried (Sec. 3.2, p. 4).
- Future-observation prediction is left out as a possible representation objective (Sec. 3.1, p. 4).
- Open question: for cross-domain game-state embedding, the paper's transfer evidence is Atari-to-Atari with a shared 18-action space and identical 84×84 frames; how much of C4/C6 survives when observation or action formats differ is untested.
- Open question: the dataset is unbalanced across games (each has its own DQN runs); the effect of per-game sampling weights is not studied.

## Related

- uses [[vaswani2017attention]]: the model is a GPT-2-style decoder-only Transformer.
- related [[serrano2024crossdomain]]: survey of cross-domain RL transfer; this paper's transfer is between games with shared state and action spaces.
- related [[schulman2017proximal]]: online policy-gradient alternative; the paper argues offline sequence modeling is more "data-absorbent" than online RL in the multi-game setting (Sec. 5, p. 10).

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
