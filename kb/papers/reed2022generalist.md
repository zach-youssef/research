---
id: reed2022generalist
title: "A Generalist Agent"
authors: [Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, Tom Eccles, Jake Bruce, Ali Razavi, Ashley Edwards, Nicolas Heess, Yutian Chen, Raia Hadsell, Oriol Vinyals, Mahyar Bordbar, Nando de Freitas]
year: 2022
venue: TMLR 2022 journal article (read as arXiv v3)
url: https://arxiv.org/abs/2205.06175
arxiv: 2205.06175
doi:
pdf_sha256: 1d9ce8dfcac0f01296ac1dd6f1c0fbc94f7e180af2956adb5d27689407cf5332
tags: [generalist-agent, transformer, imitation-learning, multi-task, multi-embodiment, tokenization, atari, robotics, transfer, scaling]
status: read
read: 2026-11-20
---

# A Generalist Agent

<!-- Read from arXiv v3 (TMLR 11/2022), 42 PDF pages; pp. 20-27 are acknowledgements and references, pp. 28-42 are Apps. A-L (model card, tokenisation, architecture, pre-training/fine-tuning setup, data, robotics, per-domain results). -->

## TL;DR

Gato: one 1.18B decoder-only transformer, one set of weights, trained by masked next-token
prediction on tokenised text, images and expert-agent data from 596 control tasks; over 50% of expert
on 450 of 604 tasks, but no transfer to held-out Atari Boxing and a context too short for in-context use.

## Problem

Can a single generic sequence model, with the same weights, act across hundreds of tasks with different
modalities, observation formats and embodiments (Atari, 3D navigation, continuous control, real robot,
captioning, chat), rather than one policy per task (Sec. 1, p. 2; Sec. 6, p. 16)? A secondary question is
whether such a pretrained model adapts to new tasks with few demonstrations (Sec. 5.2, p. 11).

## Method

- **Tokenisation** (Sec. 2.1, p. 3; App. B, p. 31): text by SentencePiece into [0, 32000); images as
  non-overlapping 16×16 patches in raster order (ViT style), pixels normalised to [−1, 1] and divided by
  √16 = 4; discrete values (e.g. Atari buttons) flattened row-major into integers in [0, 1024); continuous
  values flattened row-major, mu-law companded to [−1, 1] if not already there (Eq. 3, μ = 100, M = 256;
  actions are already in [−1, 1] and are not companded), clipped, discretised into 1024 uniform bins and
  shifted to [32000, 33024).
- **Sequencing** (Sec. 2.1, p. 3; App. B, p. 31): per timestep, observation tokens (text, then image patches,
  then tensors; nested keys in lexicographic order), a separator token '|', then action tokens; timesteps in
  time order. L = T(k + m + n + 1 + A) tokens for T timesteps.
- **Embedding** (Sec. 2.2, p. 4; App. C.2-C.3, pp. 33-35; Figs. 16-18): text, discrete and continuous tokens
  use a learned lookup table; each image patch goes through a single pre-activation (v2) ResNet block with
  GroupNorm (32 groups) and GELU. Patch tokens get learnable row and column position embeddings from the
  patch's normalised interval quantised into 128 bins (a random index within the interval in training, the
  rounded mean at test). Observation tokens get a learnable *local* position embedding by index within the
  timestep (table size 512); per the App. C.3 text all action tokens share one local position
  embedding (the Fig. 18 caption says none is added; see Limitations).
- **Objective** (Sec. 2.3, p. 4; Eqs. 1-2): autoregressive log-likelihood with a mask: only text tokens and
  logged action tokens contribute to the loss; image and other non-text observation tokens are not
  predicted. This is behaviour cloning of the data-generating agents, not RL: no reward, return or value
  input (model card: "multi-task, multi-modal behavior cloning", Table 4, App. A, p. 28).
- **Task conditioning by prompt** (Sec. 2.3, p. 4): no task id. For 25% of the sequences in a batch a prompt
  from an episode of the same source agent and task is prepended, half taken from the episode's end (goal
  conditioning) and half sampled uniformly; at evaluation a successful demonstration is the prompt.
- **Model and training** (Sec. 2.3, p. 4; App. C.1, p. 33; App. D, p. 35): decoder-only transformer, 24
  blocks, width 2048, 16 heads, sequence length L = 1024, batch 512, 1M steps on a 16×16 TPU v3 slice
  (about 4 days); subsequences of L tokens are sampled from episodes, batches mix roughly uniformly over
  domains with some manual upweighting of larger and higher-quality datasets.
- **Deployment** (Sec. 2.4, p. 5; Fig. 3): the first 1024 tokens of a demonstration are the initial prompt;
  each observation is tokenised and appended, and the action is sampled token by token (count set by the
  environment's action spec) and de-tokenised. Transformer-XL memory was used at deployment though not in
  training.
- **Data** (Sec. 3, pp. 5-7; Table 1, p. 6; App. F, pp. 36-38): control data come from specialist SoTA or
  near-SoTA RL agents (Muesli for ALE Atari and Sokoban, IMPALA for DM Lab, R2D2 for Procgen, D4PG / MPO for
  DMC, MPO for Meta-World, CRR for DM Manipulation Playground, the BabyAI bot); episodes are filtered to
  returns of at least 80% of the expert return. Control is 596 tasks, 63M episodes, about 1.5T tokens and
  85.3% of the sample weight; vision and language (MassiveText, M3W, ALIGN, LTIP, captions, VQA) is the
  other 14.7%.

## Claims

- C1: A single pretrained model with one set of weights performs over 50% of the expert score on over 450 of 604 simulated control tasks, scores normalised so that 100% is the per-task expert and 0% a random policy, 50 rollouts per task (Fig. 5, Sec. 4.1, p. 7)
- C2: On ALE Atari Gato reaches at least average human score on 23 games and over twice human on 11; the single-task online RL agents that generated the data still outperform it, which the authors suggest capacity or offline RL rather than purely supervised training may overcome (Sec. 4.1, p. 8)
- C3: Normalised per-domain scores vary widely: DM Lab 91.4%, BabyAI 93.2%, Meta-World 87.0%, DM Manipulation Playground 83.8%, Sokoban 68.0%, DMC 63.6%, Modular RL 62.9%, Procgen 60.8%, RGB Stacking sim 58.0%, ALE Atari Extended 57.8%, ALE Atari 30.9% and DMC Pixels 26.3%, 50 evaluations per task (Table 8, App. L, p. 42)
- C4: On BabyAI Gato reaches over 80% of expert score on nearly all levels, BossLevel 75%, against 77% and 90% for BabyAI 1.0 and 1.1 baselines trained on that single task with a million demonstrations (Sec. 4.1, p. 8)
- C5: On Meta-World Gato achieves more than 50% on 44 of the 45 trained tasks, over 80% on 35 and over 90% on 3; on DMC from state it exceeds 50% of expert on 21 of 30 tasks and 80% on 18 (Sec. 4.1, p. 8)
- C6: On the real-robot RGB Stacking Skill Generalization benchmark, unseen object shapes and 200 episodes per test triplet, Gato averages 50.2% success against 49% for the single-task BC-IMP baseline (Table 2, Sec. 4.2, p. 10)
- C7: Across 79M, 364M and 1.18B models, in-distribution normalised return, averaged within and then across domains, improves consistently with model size for a fixed number of training tokens (Fig. 8, Sec. 5.1, p. 10)
- C8: Adapting to a new task by prompting with demonstrations was not possible, because accelerator memory and the length of tokenised demonstrations keep the usable context from being informative enough; held-out tasks are therefore adapted by fine-tuning on a few demonstrations (Sec. 5.2, p. 11)
- C9: For held-out cartpole.swingup and assembly-v2, using 364M models, pretraining on all data gives the best fine-tuning results, then same-domain-only data; pretraining on only non-control (image and text) data gives no benefit on cartpole and negative transfer on assembly (Fig. 9, Sec. 5.2, p. 11)
- C10: For held-out DM Lab order_of_apples_forage_simple, DM Lab-only pretraining already approaches the maximum reward of 19, and non-control pretraining helps, which the authors attribute to DM Lab's natural-looking images allowing transfer from captioning and VQA (Fig. 9, Sec. 5.2, pp. 11-12)
- C11: For held-out Atari Boxing no pretraining helps and a randomly initialised model works better than every pretrained variant; the authors hypothesise the game's images are visually very distinct from the other data, and cite prior work that Atari transfer is hard because of differing visuals, controls and strategy (Fig. 9, Sec. 5.2, pp. 11-12, Sec. 6, p. 17)
- C12: Fine-tuned on Skill Generalization test-object data, Gato recovers the sim-to-real expert's success with only 10 episodes, peaks above the expert at 100 or 1000 episodes and degrades slightly at 5000, in both real and simulation; for this experiment training conditions on the episode's normalised return instead of using only successful stacks, with Lee et al.'s 5k dataset split into subsets; CRR trained on 35k episodes is the upper bound (Fig. 10, Sec. 5.3, p. 12)
- C13: In simulation, larger models adapt better from fewer fine-tuning episodes: the 79M model is clearly worse, and the 364M model overfits on one episode (Fig. 10, Sec. 5.3, p. 13)
- C14: On a new real-robot task, stacking blue on green, which is absent from pretraining, fine-tuning on 500 demonstrations (2 h 45 min) plus 10% simulated demonstrations reaches 60% success, against 0.5% (1/200) for BC trained from scratch on the same data (Sec. 5.3, p. 13)
- C15: In simulated Skill Generalization ablations, 364M models, a DM Control-only pretrained model beats base Gato zero-shot and with much fine-tuning data, and a same-domain-only model is best overall; the authors read this as Gato perhaps not using its text-based representations for robotics and trading generalisation for few-shot adaptation at current scale (Fig. 19, App. I, p. 39)
- C16: On real-robot Skill Mastery, in-distribution shapes and no fine-tuning, an earlier Gato architecture, with a local transformer patch embedder and no local or patch position embeddings, averages 75.6% against BC-IMP's 74.6% (Table 3, Sec. 5.4, p. 14, App. H, p. 38)
- C17: Single-domain specialists with the same architecture are stronger: a 79M Meta-World agent distilled from MPO experts reaches 96.6% average success over MT-50, 500 evaluations per task, and a 1.18B agent trained on the 51 ALE Atari games beats average human on 44 games, while the data-generating experts were themselves below human on the other 7 (Sec. 5.5, p. 14, Table 7, App. K, p. 41)
- C18: Attention over image patches tracks task-relevant objects and regions; certain first-layer heads clearly track task-specific entities in Atari, Procgen, RGB Stacking and DMC, and the first layer was the most interpretable (Fig. 12, Sec. 5.6, p. 15, Fig. 20, App. J, pp. 39-40)
- C19: In a t-SNE of averaged layer-12 embeddings of 128-token subsequences from 11 tasks, tasks cluster together, clusters from the same domain and modality lie close, and held-out cartpole.swingup lies next to a DMC Pixels task (Fig. 13, Sec. 5.7, pp. 15-16)
- C20: Because 1024 tokens cover only a few timesteps when an image observation alone can exceed 100 tokens, different prompt structures performed very similarly, and early prompt-based in-context learning on new environments showed no significant gain over prompt-less evaluation; quadratic self-attention is named as the main cause (Sec. 8.2, p. 19)
- C21: Validation loss on held-out data did not correlate well with episode return on held-out tasks, so the authors report episode returns instead (Table 4, App. A, p. 28)
- C22: To meet the 20 Hz real-robot control rate, inference used a context of length 1 and parallel sampling of all action tokens, with action tokens zeroed in the training inputs; the 1.18B model ran on an RTX 3090 but still overran the 20 Hz budget by about 0.01 s (App. G, p. 38)
- C23: Training only on episodes with returns of at least 80% of the expert return, a windowed maximum average return, was found effective (Sec. 3.1, p. 6)
- C24: Results come from a single model per configuration, without model-uncertainty estimates; the authors say they saw no statistically significant variation between models on benchmark subsets, and fine-tuning results average 3 independent demonstration subsets (Table 4, App. A, p. 29, App. E, p. 36)

## Implementation notes

- Gato 1.18B: 24 blocks, 16 heads, width 2048, feedforward 8192, key/value size 128; 364M: 12 blocks, 12 heads, width 1536, feedforward 6144, kv 128; 79M: 8 blocks, 24 heads, width 768, feedforward 3072, kv 32; shared embedding, pre-norm, GEGLU (source: Table 5, App. C.1, p. 33).
- Pre-training: AdamW with β = (0.9, 0.95), ε = 1e-8 and weight decay 0.1; linear warmup over 15k steps from 1e-7 to the peak, then cosine decay by 10× over 1M steps; peak and final lr 1e-4 / 1e-5 for 1.18B and 79M, 2e-4 / 2e-5 for 364M; stochastic depth 0.1 per sub-layer; batch 512, sequence 1024 (source: App. D, Table 6, p. 35).
- Fine-tuning: Adam, constant lr 1e-5, batch 64, sequence 1024, 10k steps, dropout 0.1; evaluation every 100 steps as a mean of 10 runs, moving average over 5, and the maximum of the smoothed scores is reported; data are the top 2000 episodes, nested subsets of 1000/100/10/5/3/1, repeated 3 times (source: App. E, p. 36).
- Expert return for filtering: max over j of the mean return of W consecutive episodes, W = min(1000, 0.1 N) (source: Sec. 3.1, p. 6; the printed formula sums from j to j+L−1, and reading it as j+W−1 is an assumption).
- Atari data: ALE Atari, 51 canonical games, and ALE Atari Extended, 13 alternative games (28 tasks in Table 1, p. 6) with game mode and difficulty randomised per episode, from Muesli agents trained for 200M environment steps, about 20,000 episodes recorded per environment during training (source: App. F.1, p. 36).
- The Atari action appears as one discrete token, e.g. `atari_controller [13]`, embedded by the lookup table (source: Fig. 15, p. 32). The paper does not state the Atari action-set size per game or whether the full 18-action set is used (source: assumption, the paper is silent).
- Atari observation resolution, frame stacking, grayscale vs RGB and frame skip are not stated; the 80×64 image in Fig. 17 is an illustration of patch encoding, not a stated input size (source: Fig. 17, p. 34; that the rest is unstated: assumption, from a full read).
- Discrete tokens use [0, 1024), which overlaps the text-token range [0, 32000), whereas continuous values are shifted past it; with a shared embedding table this suggests discrete values reuse text-token embeddings, but the paper does not say so (source: Sec. 2.1, p. 3, Table 5, p. 33; the shared-embedding reading is an assumption).
- The ResNet patch embedder: x → GroupNorm → GELU → weight → GroupNorm → GELU → weight, plus a residual (source: Fig. 16, App. C.2, p. 33).
- No code or weights are linked in the paper (source: assumption, from a full read finding no code link; the model card gives only a contact email, Table 4, p. 28).

## Limitations / open questions

- Learning is imitation only and needs action-labelled data from strong RL experts; the authors flag observation-only video data (Baker et al. 2022) as a future direction (Sec. 8.1, pp. 18-19).
- The 1024-token context, a few image timesteps, rules out in-context adaptation; held-out tasks need fine-tuning (Sec. 5.2, p. 11; Sec. 8.2, p. 19).
- The paper counts tasks inconsistently: Table 1 totals 596 control tasks (p. 6), while Sec. 4.1 evaluates 604 (p. 7). Sec. 2.3 gives feedforward size 8196 (p. 4) and Table 5 gives 8192 (p. 33). On action-token positions, App. C.3 says all action tokens share one local position encoding, but the Fig. 18 caption says none is added to action tokens (p. 35).
- No error bars: each configuration is one model (Table 4, App. A, p. 29).
- Open question (survey lens): held-out *game* transfer is tested on one Atari game, Boxing, and fails (C11); cross-game transfer within a genre isn't measured.
- Open question (survey lens): the image tokens are raw patches through a per-patch ResNet block trained end to end with the policy, not a separately pretrained or unsupervised embedding; how a frozen self-supervised encoder would compare is not tested.
- Open question (survey lens): action spaces are unified only by tokenisation (discrete ints or binned continuous values with per-environment decoding); no shared semantic action space across games is attempted.
- Open question (survey lens): whether PPO / actor-critic fine-tuning of such a supervised pretrained transformer improves on it is not studied; the authors only point to offline RL as a possible way to beat the experts (Sec. 4.1, p. 8).

## Related

- related [[chen2021decision]]: cited as the closest architecture (Sec. 6, p. 15); Gato drops return conditioning and per-task models for one network over many tasks and modalities.
- uses [[vaswani2017attention]]: decoder-only Transformer as the sequence model (Sec. 2.3, p. 4).
- related [[he2021masked]]: Gato's image tokens are the same ViT-style 16×16 patches (Sec. 2.1, p. 3) that MAE pretrains an encoder on; Gato trains its patch embedder end to end.
- related [[schulman2017proximal]]: Gato is trained by supervised imitation, not PPO; the survey's plan to train with PPO is a different design choice.
- related [[serrano2024crossdomain]]: survey of cross-domain RL transfer; Gato's held-out results (Sec. 5.2) are a data point for it.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
