---
id: schmidt2024actions
title: Learning to Act without Actions
authors: [Dominik Schmidt, Minqi Jiang]
year: 2024
venue: ICLR 2024 conference paper (read as arXiv v2)
url: https://arxiv.org/abs/2312.10812
arxiv: 2312.10812
doi:
pdf_sha256: a7ddbc946a2d7570565b32dea2a47db3213c4364577329b3c5df203d60175080
tags: [latent-actions, inverse-dynamics, forward-dynamics, vector-quantization, imitation-from-observation, behavioral-cloning, procgen, pretraining, lapo]
status: read
read: 2026-11-25
---

# Learning to Act without Actions

## TL;DR

LAPO trains an inverse and a forward dynamics model jointly on action-free frames with a
vector-quantized latent action between them; behaviour cloning on the latents gives a latent policy.
On 16 Procgen games the latents cluster by true action, and online decoding recovers expert return in 4M frames.

## Problem

Web videos have no action labels, so imitation learning and offline RL cannot use them (Sec. 1,
p. 1). Earlier fixes (VPT, BCO, ACO) train an IDM on a labelled action dataset first (Sec. 2,
p. 3); ILPO learns discrete latent policies but collapses on diverse visual levels (Sec. 2, p. 3).
LAPO wants latent actions, latent policies and dynamics models learned with no action labels at
all (Sec. 1, p. 2).

## Method

- **Latent IDM** (Sec. 4.1, p. 5, Fig. 2): the IDM sees observations o_{t-k}..o_{t+1} and outputs a
  continuous latent action z_t; the FDM sees only o_{t-k}..o_t plus z_t and predicts o_{t+1}. Both
  are trained jointly on the squared next-observation error.
- **Bottleneck** (Sec. 4.1, p. 5): since only the IDM sees o_{t+1}, z_t must carry the change
  between o_t and o_{t+1}. A vector-quantization step is applied to z_t before the FDM, to force
  re-use of few discrete latents across states and so a simpler, more state-invariant code.
- **Latent policy** (Sec. 4.2, p. 5): a policy π(o_t) is trained by behaviour cloning, loss
  ||π(o_t) − z_t||², on the same observation-only dataset, with z_t from the IDM.
- **Decoding to real actions** (Sec. 4.3, p. 5-6): with a small labelled set, a decoder d: Z → A is
  trained on IDM latents of labelled transitions and composed with the frozen latent policy. With
  an online environment, two decoder heads on the latent policy (supervised on (a_t, z_t), and
  trained by PPO) have their per-action logits averaged before softmax; the whole composed policy
  is then fine-tuned with RL (Sec. 4.3, p. 6; Sec. 5, p. 6).
- Setup (Sec. 5, p. 6): ~8M frames from a PPO expert (trained on 50M frames per Sec. 5; Sec. 6.1, p. 6 says 25M for the same expert, a discrepancy in the paper) per Procgen game;
  IMPALA-CNN for policy and IDM, a U-Net for the FDM (~8M parameters).

## Assumptions

- A1: The environment's transitions are close to deterministic, so the difference between o_t and
  o_{t+1} is mostly explained by the agent's action (Sec. 4.1, p. 5; Sec. 6.4, p. 9).
- A2: Actions change the observation promptly; a delayed effect is attributed to the later
  transition (Sec. 6.4, p. 9).
- A3: The observation-only dataset is expert-level and of one environment per model (Sec. 5, p. 6).
- A4: The action space is discrete (Procgen's discrete button set; continuous control only in an appendix)
  (reader's inference, Sec. 5, p. 6, App. A.3, p. 16).
- A5: Decoding needs either a small labelled set or online interaction, and for the online route a
  reward signal (Sec. 4.3, p. 5-6).

## Claims

- C1: With no action labels, the IDM's latent actions cluster by true action in most of the 16
  Procgen games (UMAP, e.g. Miner, BigFish, Chaser, Maze) [A1, A3] (Fig. 1, p. 2, Fig. 5, p. 8, Sec. 6.3, p. 8).
- C2: In games with partial observability (Climber, CoinRun, FruitBot, Jumper, Ninja) the latent
  space is more fragmented, which the authors hypothesise comes from the IDM encoding off-screen
  information [A1, A3] (Sec. 6.3, pp. 8-9).
- C3: Decoding the latent policy online (supervised head plus PPO head) recovers expert-level
  return within 4M frames, while PPO from scratch reaches 44% of expert in 4M; in 9 of 16 games it
  exceeds the expert (trained on 25M frames per Sec. 6.1; Sec. 5 says 50M) [A3, A5] (Sec. 6.1, p. 6, Fig. 3, p. 7).
- C4: A decoder trained offline on fewer than 256 labelled transitions matches PPO from scratch
  at 4M steps; performance plateaus below online decoding as data grows (intro says ~200
  transitions), plausibly since latent actions are not state-invariant [A3, A5] (Sec. 6.2, p. 7, Fig. 4, p. 7, Fig. 6, p. 14).
- C5: Removing vector quantization lowers online performance (ablation "RL, no-SL, no-VQ" in Fig.
  3) and fragments latent clusters, even though its FDM loss is lower; VQ acts as a useful
  bottleneck [A1, A3] (Fig. 3, p. 7, Sec. 6.3, p. 9, Fig. 7, p. 15).
- C6: The ILPO baseline collapses on several Procgen tasks and, where it does not, is no better
  than PPO from scratch; the authors reproduce ILPO's positive CoinRun result on a single level
  only, and attribute the collapse to ILPO's few discrete latents modelling the full level
  distribution poorly [A3] (Sec. 6.1, p. 7, Fig. 9, p. 17, Sec. 2, p. 3).
- C7: On two DeepMind Control tasks (cheetah_run, walker_walk), 2 seeds, online decoding of the
  latent policy beats PPO from scratch over 1M steps, with Procgen's hyperparameters [A5] (App. A.3, Fig. 8, p. 16).
- C8: Named limits: delayed action effects are modelled as occurring with the same delay,
  stochasticity degrades the latent code, and scaling to web video needs larger architectures
  [A1, A2] (Sec. 6.4, p. 9).

## Implementation notes

- Latent action is a 128-d continuous vector, split and quantized into 8 discrete latents with
  16-d embeddings: 2 codebooks x 4 latents, 64 embeddings, commitment 0.05, EMA decay 0.999
  (source: Table 1, p. 17).
- IDM training: lr 2e-4, batch 128, 70,000 updates; latent BC: lr 2e-4, batch 128, 60,000 updates;
  online: 4M steps, PPO as in Cobbe et al. 2020; Procgen mode `easy`; context k = 1 (source: Table 1, p. 17).
- Action decoder: MLP with hidden sizes (128, 128); for online decoding conv layers stay frozen
  and only the last layers train, at lr 0.01 (source: Sec. 5, p. 6).
- Authors' code: github.com/schmidtdominik/LAPO (source: abstract, p. 1; not inspected).
- For a shmup (reader's guess, source: assumption): the chunked-action policy would need the
  latent a_t to cover a window, not one transition; k and the frame gap are untested here.

## Limitations / open questions

- Admitted: delayed actions, stochasticity, scaling (C8); latent actions are not strictly
  state-invariant, so offline decoding plateaus below online (Sec. 6.2, p. 7).
- Tested only on Procgen with synthetic expert data and 2 DMC tasks; no real web video (Sec. 5, p. 6).
- Open question (user's lens): shmups have continuous scrolling and many off-screen-moving
  objects with stochastic spawns; LAPO's IDM would see most change as non-agent. Does the latent
  capture the ship's motion or the background? C2 suggests partial observability fragments it.
- Open question: grounding needs ~200 labelled transitions; for shmup superplay video one could
  use gamepad logs of a few runs (as NitroGen-style data) as that set. Untested.
- Open question: as compact history tokens, latent actions are per-transition and compress
  controls, not strategy (weapons, economy, rank); nothing here says they carry long-term state.

## Related

- related [[baker2022video]]: VPT's IDM needs labelled data; LAPO removes that need (Sec. 2, p. 3).
- related [[gao2025adaworld]]: AdaWorld extracts latent actions from frame pairs with a VAE instead of a VQ bottleneck.
- related [[jiang2026olaf]]: later latent-action alignment work built on AdaWorld-style latents.
- related [[magne2026nitrogen]]: the frames-in, gamepad-out policy a latent labeller could feed.
- related [[reed2022generalist]]: cited as a source of the pretraining paradigm (Sec. 1, p. 1).

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
