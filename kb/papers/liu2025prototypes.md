---
id: liu2025prototypes
title: Cross-domain Random Pre-training with Prototypes for Reinforcement Learning
authors: [Xin Liu, Yaran Chen, Haoran Li, Boyu Li, Dongbin Zhao]
year: 2025
venue: arXiv preprint
url: https://arxiv.org/abs/2302.05614
arxiv: 2302.05614
doi:
pdf_sha256: 9f5c16c8a4522114fbfe8a9cb2b02ee332e02f782b01f278a11906ed6bdb6a46
tags: [reinforcement-learning, self-supervised, representation-learning, pretraining, cross-domain, prototypes, random-policy, dmcontrol]
status: read
read: 2026-11-21
---

# Cross-domain Random Pre-training with Prototypes for Reinforcement Learning

<!-- Read from arXiv v5 (2025-03-24; first version 2023-02-11), 13 PDF pages in the IEEE TSMC:
Systems template but marked "submitted to the IEEE"; arXiv lists no journal ref. p. 12 holds
Table VI (hyperparameters); pp. 12-13 are references. No appendix or supplementary material. -->

## TL;DR

CRPTpro pre-trains one conv encoder on uniform-random-action frames from 3 DMControl domains with
a SwAV-style next-frame prototype loss plus a prototype-spreading loss; frozen, it gets RAD-SAC to
1.956 mean expert-normalized score on 12 tasks (Proto-RL(C): 1.096) in 54% of its pre-training time.

## Problem

Active unsupervised pre-training (APT, Proto-RL) trains exploration agents with an intrinsic
reward and the encoder at the same time; this works within one domain but, moved to several
domains, the encoder degenerates because of a chicken-and-egg problem (good exploration needs a
good encoder and vice versa, once per domain), and the extra RL for exploration agents costs more
than downstream RL and multiplies with the number of domains (Sec. I, p. 2; Sec. III-A, p. 3).
The goal is one encoder shared across tasks *and* domains (Fig. 1c, p. 1).

## Method

- **Setting** (Sec. III, p. 3): pixel POMDPs made Markov by stacking 3 consecutive frames;
  84×84×3 renders, so the state is 84×84×9; action repeat 2 (Sec. IV-A, p. 6).
- **Decoupled random collection** (Sec. III-A, pp. 3-4; Alg. 1, p. 5): before any training, an
  off-the-shelf policy sampling actions uniformly interacts with each of N reward-free domains and
  stores observations into one buffer per domain (capacity b); the dataset is "collected once and
  used permanently". Argued advantages: no chicken-and-egg conflict, wider (not deeper)
  exploration of motor-control domains, and a static dataset, so a stable training process.
- **Efficient prototypical learning** (Sec. III-B, pp. 4-6; Fig. 3, p. 4): each update uses one
  domain buffer, cycling o = index mod N, so the encoder does not favour the latest domain. A batch
  of M frames x_t and their next frames x_{t+1}: x_t → random shift → conv encoder f_θ → MLP
  projector g_θ → z_t → MLP predictor v_ψ → u_t; softmax over dot products of l2-normalised u_t
  with M l2-normalised prototypes c_j, temperature τ (Eq. 1). Targets: x_{t+1} through EMA copies
  f_θ̄, g_θ̄ (Eq. 2), then Sinkhorn-Knopp on the M×M matrix C_ij = ẑ_{t+1,i}·ĉ_j, three rounds of
  row then column normalisation (Eq. 3-6); row i of the result is the target q_{t+1}.
  L_comp = −(1/M) Σ q^T log p (Eq. 7), i.e. predict the next frame's cluster assignment.
- **Intrinsic loss** L_intr (Eq. 8-9, p. 5): sum over prototype pairs of det(ĉ_j)·ĉ_k /
  (det(ĉ_j·ĉ_k) + w), det = stop-gradient, w > 1; pushes prototypes apart, with remote prototypes
  getting a larger gradient. L_SSL = L_comp + α L_intr (Eq. 10, p. 6). θ, ψ and the prototypes
  are trained by backprop.
- **Downstream** (Sec. III-C, p. 6; Alg. 1): encoder and prototypes frozen; RAD-SAC with
  random-shift augmentation learns on encoder outputs; the reward is r + β r̂ with Proto-RL's
  exploration bonus r̂ = distance from z_{t+1} to its k-th nearest neighbour in a set Q of latents
  picked from the replay buffer by each prototype (Eq. 11). Optional single-domain fine-tuning
  (for an unseen domain, collect a random buffer first) trades cross-domain versatility for
  performance; it is not used in the cross-domain results.
- **Experiments** (Sec. IV, pp. 6-11): 8 DMControl tasks in 8 domains (Cheetah-Run, Walker-Run,
  Quadruped-Run, Cartpole-Swingup sparse, Pendulum-Swingup, Manipulation-Reach duplo, Hopper-Hop,
  Finger-Spin), split into 4 groups of 3 pre-training domains (Table I, p. 6). Baselines
  (Sec. IV-A, p. 7): cross-domain ATC (given CRPTpro's random data), APT(C), Proto-RL(C), BeCL;
  single-domain APT(S), Proto-RL(S); end-to-end DrQ (1M steps, used as the "expert" for
  normalising). Cross-domain methods: 50k encoder updates over 3 domains (active ones 200k
  task-agnostic RL steps), then 500k steps downstream RL per task.

## Claims

- C1: With encoders pre-trained on a group's 3 domains and frozen, CRPTpro reaches a mean score of 495 and a mean expert-normalized (DrQ = 1) score of 1.956 over the 12 group-task pairs, against 400 and 1.096 for the next best, Proto-RL(C); ATC, APT(C) and BeCL score 225/0.440, 191/0.412 and 166/0.993 (Table II, p. 7)
- C2: CRPTpro beats Proto-RL(C) on 11 of 12 downstream tasks and is similar on the remaining one (Pendulum-Swingup in Group B, 865±26 vs 868±35); it beats each of APT(C), ATC and BeCL on 11 of 12 (Sec. IV-B, p. 7, Table II, p. 7)
- C3: CRPTpro's mean score and mean expert-normalized score are 23.8% and 78.5% above the next best cross-domain baseline (Sec. IV-B, pp. 7-8, Table II, p. 7)
- C4: Frozen encoders transferred to unseen domains without fine-tuning: pre-trained on Group A (robots only) and tested on 5 tasks in 5 unseen domains, CRPTpro is best on 3 (Pendulum 671, Hopper 187, Cartpole 93), with ATC best on Finger-Spin (879 vs 809) and BeCL on Manipulation-Reach (64 vs 62); pre-trained on Group C and tested on 5 unseen domains, it is best on 4, with BeCL best on Quadruped-Run (322 vs 270) (Table III, p. 8)
- C5: On some unseen-domain tasks the frozen CRPTpro encoder gives better downstream policies than DrQ trained end to end for 1M steps on that task (e.g. Pendulum-Swingup from Group A 671 vs 635, Walker-Run from Group C 570 vs 485) (Sec. IV-C, p. 8, Table III, p. 8, Table II, p. 7)
- C6: With an unseen background colour (RGB channels reversed), CRPTpro pre-trained on Group C scores 163/567/485 on Manipulation-Reach, Pendulum-Swingup and Cheetah-Run, and is the only method whose Cheetah-Run score does not drop (448 with normal colours) (Sec. IV-C, p. 8, Table III, p. 8)
- C7: Pre-training on the more diverse Group C (3 domain types) generalises better than on Group A (robots only): Group C solves all its unseen tasks including Cartpole-Swingup sparse (682 vs 93 from Group A), and even on Walker-Run, a Group A domain, the Group C encoder scores higher (570 vs 513); the authors conclude data diversity is key (Sec. IV-C, p. 8, Table II, p. 7, Table III, p. 8)
- C8: Fine-tuning the Group A encoder in one domain (2k, 2k, 3k, 1k and 0.5k updates in Pendulum, Manipulation, Cartpole, Cheetah and Walker; 58.5k encoder updates in total for 5 domains) gives performance comparable to Proto-RL(S), whose and APT(S)'s total pre-training for 5 domains is 625k updates (Sec. IV-D, p. 9, Fig. 4, p. 9)
- C9: On an NVIDIA Tesla P100, pre-training wall clock is 4.8 h for CRPTpro whether 1, 2 or 3 domains (50k updates), vs 8.8 h Proto-RL(C), 9.4 h BeCL, 12.2 h APT(C), and 18.8/37.6/56.4 h Proto-RL(S) and 20.6/41.2/61.8 h APT(S) for 1/2/3 domains (Table IV, p. 9)
- C10: CRPTpro needs 54% (abstract: 54.5%) of Proto-RL(C)'s pre-training wall clock while beating it by 78.5% on mean expert-normalized score; the saving comes mainly from not training exploration agents (Sec. IV-E, p. 9, Table IV, p. 9, p. 1)
- C11: The k-NN exploration reward costs about 1‰ of downstream RL compute on a P100, owing to the small k-NN buffer and low-dimensional particles (Sec. IV-E, p. 10)
- C12: Removing L_intr from CRPTpro lowers downstream return, and adding L_intr to Proto-RL(C) or Proto-RL(S) raises it (Cheetah-, Walker-, Quadruped-Run; Hopper-Hop and Cheetah-Run curves); since Proto-RL(C) with L_intr equals CRPTpro without random collection, this also shows random collection helps (Sec. IV-F, p. 10, Fig. 5, p. 9)
- C13: L_intr helps and is robust to its weight α when α is not too large; at α = 0.2 the Cheetah-Run return drops; a larger random buffer b (5×10³ to 10⁵) is better, though even small buffers beat all cross-domain baselines (Sec. IV-F, p. 10, Fig. 6, p. 10)
- C14: With L_intr, prototypes end pre-training more spread out: KNE/ANE (×10⁻⁴, lower is wider) are −26.5/8.0 for CRPTpro vs −10.9/23.8 without it, −38.6/21.6 vs −7.7/51.3 for Proto-RL(C), and −80.1/20.6 vs −44.0/48.8 for Proto-RL(S) (Sec. IV-G, p. 10, Table V, p. 10)
- C15: L_intr decreases even in vanilla Proto-RL(S), which does not optimise it, which the authors read as reducing it being "instinctual" in prototypical learning (Sec. IV-G, p. 11, Fig. 7, p. 10)
- C16: In the cross-domain setting (Group C, Manipulation), the intrinsic-reward collection agents of Proto-RL(C) explore less than the uniform random policy (k-NN particle coverage) until near the end of 200k reward-free steps (Sec. IV-H, p. 11, Fig. 8, p. 11)
- C17: Training Proto-RL(C)'s self-supervised task on random-collection data instead of its agents' data gives a lower and steadier L_comp over 50k updates (Sec. IV-H, p. 11, Fig. 8, p. 11)
- C18: In PCA of encoder features, the uniform policy within one domain is less exploratory than Proto-RL(S)'s agent "but not a lot worse", reaches some observations that agent does not, and the cross-domain random dataset covers more than a single-domain exploration dataset (Sec. IV-H, p. 11, Fig. 9, p. 11)

## Implementation notes

- Encoder: 4 conv layers, 32 channels, strides (2,1,1,1), 3×3 filters, 39200-d output; features 50-d; latent (prototype) space 128-d; 512 prototypes; MLP and predictor hidden 1024 (source: Table VI, p. 12; architecture "the same" as Proto-RL, Sec. IV-A, p. 6).
- Pre-training: batch 512, Adam 1e-4, buffer b = 100000 per domain, α = 5×10⁻³ (1e-3 for Cheetah-Run and 5e-4 for Hopper-Hop in the Proto-RL(S) ablation), encoder target EMA 0.05 every update, τ = 0.1, w = 1.5, 50k updates across 3 domains (source: Table VI, p. 12; Sec. IV-A, p. 6; Sec. IV-F, p. 10).
- Downstream RAD-SAC: random shift ±4, initial temperature 0.1, γ = 0.99, replay 40k (Proto-RL's 100k reduced), actor update every 2, critic target every 2 with EMA 0.01, log std in [−10, 2], k = 3, β = 0.2, exploration-reward buffer 2048; 500k steps; results over at least 6 evaluations (60 episodes) (source: Table VI, p. 12; Sec. IV-A, pp. 6-7).
- Whether the 500k downstream "steps" are environment steps or agent steps under action repeat 2 is not stated (source: assumption from Sec. IV-A, p. 6).
- The size of Q and how each prototype picks its member from the replay buffer are deferred to Proto-RL (source: Sec. III-C, p. 6, citing [14]).
- Number of seeds, network for the predictor v_ψ (depth) and the uniform policy's action range are not given; take the action range as each task's full box (source: assumption).
- No code: the footnote says the implementation "will be arranged and released after publication" (source: p. 1).
- For the user's design (reader's inference): the recipe maps onto unsupervised game-state embedding from N stacked frames, with random-action gameplay as the data and a frozen encoder feeding any policy; the downstream learner is learner-agnostic, so a PPO transformer could replace RAD-SAC, but the paper never tests that (source: assumption from Sec. III, pp. 3-6).

## Limitations / open questions

- The uniform policy is worse at reaching far-off states than active exploration (e.g. the end of a maze); the authors argue this matters less in motor control and name "the upper limits of random policies" as a reason for future work (Sec. III-A, pp. 3-4; Sec. V, p. 11).
- Fine-tuning improves one domain but reduces cross-domain versatility (Sec. III-C, p. 6).
- Evaluation is DMControl only, 8 domains, 3 pre-training domains at a time, with groups chosen by the authors (Sec. IV-A, p. 6).
- Reader's observation: the mean expert-normalized score is dominated by tasks where DrQ is weak (e.g. Manipulation-Reach 163 vs DrQ 27, about 6×), so 1.956 reflects a few tasks more than a uniform gain (Table II, p. 7).
- Reader's observation: pre-training uses at most 3 domains and a small conv net; whether random-data prototypes keep working with many more, visually more varied domains (video games) or with a large ViT encoder is untested (Sec. IV-A, p. 6, Table VI, p. 12).
- Open question: in games where random play dies early or never leaves the first screen, does uniform-random data cover the states a policy will need, or does it need the active exploration the paper drops?
- Open question: next-frame cluster prediction rewards features that change with actions (motion); does it keep score, inventory or other slowly changing game state a policy needs?

## Related

- related [[serrano2024crossdomain]]: the survey's [15]; it summarises this method and groups it among parameter-transfer methods suitable for on-policy learners ([[serrano2024crossdomain#C3]]).
- related [[he2021masked]]: another self-supervised visual pre-training objective (masked reconstruction) that could stand in for the prototype loss.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
