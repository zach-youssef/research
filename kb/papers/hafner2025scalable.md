---
id: hafner2025scalable
title: "Training Agents Inside of Scalable World Models"
authors: [Danijar Hafner, Wilson Yan, Timothy Lillicrap]
year: 2025
venue: arXiv preprint
url: https://arxiv.org/abs/2509.24527
arxiv: 2509.24527
doi:
pdf_sha256: 8655cce4bf12ce6210f6694f83c1a723c7acd7579214ca3ebc57c4394d0b1aeb
tags: [world-model, model-based-rl, offline-rl, imagination-training, video-pretraining, diffusion, shortcut-models, tokenizer, transformer, minecraft, keyboard-mouse, action-conditioning, behavioral-cloning]
status: read
read: 2026-11-21
---

# Training Agents Inside of Scalable World Models

## TL;DR

Dreamer 4: a causal MAE tokenizer and a 2B block-causal transformer trained by shortcut forcing
(21 FPS, one H100); BC heads plus PMPO RL in imagination reach diamonds in 0.7% of episodes from the
2.5K-h VPT contractor data alone (first offline); 100 h of action labels recover most conditioning.

## Problem

- Dreamer 3-style world models are fast and accurate in narrow environments but lack the capacity
  for complex distributions; scalable controllable video models (Genie 3, Oasis and others) still
  get object interactions and game mechanics wrong and often need many GPUs for one real-time
  scene, so they are poor simulators for training agents (Sec. 1, p. 2).
- Offline learning matters where online interaction with a partly trained agent is unsafe or slow
  (robots); earlier Minecraft diamond agents all used online interaction (Sec. 4.1, p. 10).
- Action labels are missing from most video, so a world model should learn mostly from unlabelled
  video and ground actions from a small labelled subset (Sec. 1, p. 2; Sec. 4.3, p. 14).

## Method

Three phases (Alg. 1, p. 4): pretrain tokenizer and dynamics on video (optionally with actions),
finetune policy and reward heads into the world model, then train the policy in imagination.

- **Causal tokenizer** (Sec. 3.1, p. 5, Fig. 2, p. 4): encoder and decoder are block-causal
  transformers. Each frame is patch tokens plus learned latent tokens; the latents are read out by a
  linear projection to a small channel count and a tanh (the bottleneck), projected back up for the
  decoder. Loss is MSE + 0.2·LPIPS, with RMS loss normalisation (Eq. 5, p. 5). Input patches are
  replaced by a learned embedding with probability p ~ U(0, 0.9) per image (MAE training), which
  improved the spatial consistency of generated video (p. 5). Minecraft: 360×640 frames zero-padded
  to 384×640, 16×16 patches (960 tokens), bottleneck 512×16 reshaped to 256×32 for the dynamics
  (App. A, p. 25).
- **Interactive dynamics** (Sec. 3.2, pp. 5–6): on the frozen tokenizer's representations,
  interleaved per time step with actions, S_r register tokens and one token for signal level and step
  size. Each action component (mouse, keyboard) becomes S_a tokens, summed with a learned embedding;
  continuous parts are projected linearly, categorical or binary ones looked up. For unlabelled
  video only the learned embedding is used (p. 5).
- **Shortcut forcing** (Eqs. 6–8, p. 6): diffusion forcing (a signal level per time step) combined
  with shortcut models (conditioning on step size d, bootstrap targets for d > d_min; Eqs. 3–4,
  p. 3). The network predicts clean latents (x-prediction) instead of velocity, since v-prediction's
  high-frequency outputs accumulate errors frame by frame; the bootstrap loss is computed in v-space
  and scaled by (1−τ)² back to x-space (Eq. 7 and footnote, p. 6). A ramp weight w(τ) = 0.9τ + 0.1
  favours high signal levels (Eq. 8). Inference: autoregressive in time, K = 4 sampling steps per
  frame, past context corrupted to τ_ctx = 0.1 (p. 6).
- **Agent finetuning** (Sec. 3.3, pp. 6–7): agent tokens fed task embeddings are interleaved into
  the dynamics transformer; they attend to everything, but nothing attends back to them, so the
  world model's predictions can't depend on the task (avoiding causal confusion, p. 6). MLP policy
  and reward heads on them are trained with multi-token prediction over L = 8 future steps (Eq. 9,
  p. 7) while the video loss continues. Reward head: symexp twohot; policy: categorical or
  vectorised binary (p. 7).
- **Imagination RL** (pp. 7–8): transformer frozen, only policy and value heads updated (full
  finetuning gives small extra gains at more compute, footnote p. 7). One rollout per dataset context
  (unlike earlier Dreamers). Value head: symexp twohot, TD on λ-returns with γ = 0.997 (Eq. 10,
  p. 7). Policy: PMPO, which uses only the sign of the advantage, averaging a likelihood loss
  separately over positive and negative states (α = 0.5) plus a reverse KL to a frozen BC prior
  (β = 0.3) (Eq. 11, p. 8).
- **Efficient transformer** (Sec. 3.4, p. 8): 2D (time × space) transformer, pre-RMSNorm, RoPE,
  SwiGLU, QKNorm, attention logit soft-capping; separate space-only and time-only attention, time
  attention only every 4 layers, GQA in the dynamics. Training alternates many short and occasional
  long batches, then long batches only; batch length exceeds context length so the model doesn't
  overfit to a start frame at context position 0 (p. 8).
- **Setup** (Sec. 4, p. 9; Sec. 4.1, p. 10): 2B parameters (400M tokenizer, 1.6B dynamics) on 256 to
  1024 TPU-v5p, batch 1 per device, FSDP. 30% of batch videos are treated as single images so the
  model can generate start frames. Minecraft: 256 spatial tokens, context 192 frames, batch length
  256 (T1 = 64 and T2 = 256 in App. A). 20 one-hot tasks with sparse binary rewards annotated from
  VPT event logs (Table 4, p. 27); evaluation follows a fixed prompt sequence of tasks (Table 6,
  p. 27). Data mix 50% uniform / 50% task-relevant sequences, BC loss on the relevant half, dynamics
  loss on the uniform half only, to avoid optimistic generations. Keyboard: 23 binary
  distributions; mouse: 121-class categorical (μ-law, 11 bins per axis) (p. 10; App. A, p. 25).

## Assumptions

- A1: Training and evaluation are in one game, Minecraft (Sec. 4.1, p. 10)
- A2: The training data are the VPT contractor dataset, 2541 h of human play with keyboard and mouse actions, with no web video (Sec. 4.1, p. 10, Sec. 4.3, p. 14)
- A3: Learning is purely offline: the agent never interacts with the environment during training (Sec. 4.1, p. 10)
- A4: The agent is multi-task over 20 tasks with sparse binary rewards annotated from the VPT event logs, conditioned on one-hot task ids (Sec. 4.1, p. 10, Table 4)
- A5: Evaluation follows the VPT protocol: raw pixels, low-level actions (23 binary keys, 121-class foveated mouse), 60-minute episodes from an empty inventory in random worlds (Sec. 4.1, p. 10)
- A6: The action-labelled hours are the first in dataset order, so they cover fewer worlds and players than a random subset (Sec. 4.3, p. 14)
- A7: Action-conditioning quality is video fidelity (PSNR, SSIM of 16-step generations against holdout ground truth, normalised between no-action and all-action models), not agent performance (Sec. 4.3, p. 14, Fig. 7)
- A8: Inference speeds are measured on a single H100 GPU (Table 1, p. 12, Sec. 4.4, p. 15)
- A9: Action-free video of the target domain (Nether, End) is in the training data (Sec. 4.3, pp. 14–15)

## Claims

- C1: Dreamer 4 is the first agent to obtain diamonds in Minecraft purely from offline data, using only the 2.5K-hour VPT contractor dataset, raw 360×640 pixels and low-level mouse and keyboard actions, in 60-minute episodes from random worlds and an empty inventory [A1, A2, A3, A4, A5] (p. 1, Sec. 4.1, p. 10, Fig. 3, p. 9)
- C2: Over 1000 episodes Dreamer 4 reaches over 90% success up to the stone pickaxe (90.1%), 29.0% for the iron pickaxe and 0.7% for diamonds; VLA (Gemma 3) reaches 11.2% iron pickaxe, BC 0.6%, VPT (finetuned) 0.0%, and no other method gets a diamond [A1, A2, A3, A4, A5] (p. 12, Fig. 3, p. 9, Table 7, p. 28)
- C3: Dreamer 4 needs 100× less data than previous keyboard-and-mouse Minecraft agents: 2.5K offline hours, against VPT's 2.5K plus 270K web hours (plus 194K online hours for VPT RL) [A1, A2, A3] (p. 2, Table 3, p. 16)
- C4: VPT (finetuned) reaches sticks 53% of the time and gets stone, iron ore and ingots only rarely through edge cases (exploding creepers, loot chests); the paper's BC on contractor actions beats it [A1, A2, A3, A5] (p. 12, Fig. 3, p. 9)
- C5: BC on world-model representations (WM+BC) outperforms BC from Gemma 3 (VLA) and from scratch, e.g. 89.4% vs 76.7% vs 53.8% stone pickaxe and 16.9% vs 11.2% vs 0.6% iron pickaxe, which the authors read as video prediction learning a world understanding useful for decisions [A1, A2, A3, A4, A5] (p. 12, Fig. 4, p. 10, Table 7, p. 28)
- C6: Imagination RL's gain over WM+BC grows with milestone difficulty (iron ingot 39.5% vs 27.8%, iron pickaxe 29.0% vs 16.9%, diamond 0.7% vs 0.0%), though Dreamer 4 is slightly below WM+BC on the six earliest items (e.g. cobblestone 95.9% vs 97.2%) [A1, A2, A3, A4, A5] (p. 12, Table 7, p. 28)
- C7: Imagination training makes the policy faster: Dreamer 4 reaches every milestone sooner than WM+BC in successful episodes, e.g. stone pickaxe 6.7 vs 8.9 min and iron pickaxe 13.3 vs 17.0 min, and its diamonds take 20.7 min [A1, A2, A3, A4, A5] (Fig. 4, p. 10, Table 8, p. 28)
- C8: In human play inside the world models on 16 tasks, Dreamer 4 allows 14/16 tasks, Oasis (large) 5/16, Oasis (small) 0/16 and Lucid-v1 0/16; MineWorld was not evaluated because it isn't interactive [A1, A2] (Table 1, p. 12, Sec. 4.2, p. 13)
- C9: Dreamer 4's failed tasks are crafting a wooden pickaxe and turning 360° to enter a house; inventory items are sometimes unclear or change over time, and temporal consistency is limited to the 9.6 s context [A1, A2] (p. 13, Fig. 14, p. 32)
- C10: Oasis (large) fails building tasks by hallucinating large structures after a few placed blocks ("autocompletion"), and Lucid-v1's generations diverge or ignore interactions [A1] (p. 13)
- C11: Dreamer 4 runs interactively at 21 FPS on a single H100 with a 9.6 s context, 6× longer than prior Minecraft world models (0.8–1.6 s); Lucid-v1 reaches 44 FPS at 1.0 s, Oasis (small) 20 FPS, Oasis (large) an estimated ~5 FPS, MineWorld 2 FPS [A1, A8] (Table 1, p. 12, p. 13)
- C12: With all 2541 h of video but actions for only 10 h, action-conditioned generation reaches 53% of the PSNR and 75% of the SSIM gap between a no-action and an all-action model; with 100 h, 85% PSNR and 100% SSIM (16-step generations from 320 context frames on holdout data; labelled hours are the first in dataset order, so fewer worlds and players) [A1, A2, A6, A7] (Sec. 4.3, p. 14, Fig. 7)
- C13: Trained on Overworld and Nether/End video with actions only for the Overworld, the model's action-conditioned generation on the Nether and End reaches 76% of the PSNR and 80% of the SSIM of the all-actions model, normalised as in C12 [A1, A2, A7, A9] (Sec. 4.3, pp. 14–15, Fig. 7, p. 14, App. A, p. 25)
- C14: Training only on Overworld video without any Nether video gives poor generation scores for Nether start frames (a prior experiment; no numbers given) [A1, A2, A7] (p. 15)
- C15: In the design cascade (48 h training each, FVD on 1024 context-free 384-frame videos with actions from a fixed BC policy), FVD goes from 306 for a diffusion forcing transformer (64 steps, 0.8 FPS) to 57 for the full model at 21.4 FPS, and train step time from 9.8 s to 1.7 s [A1, A2, A8] (Sec. 4.4, p. 15, Table 2)
- C16: Cutting to K = 4 sampling steps alone raises FVD to 875; shortcut models bring it back to 329; x-prediction, x-loss and the ramp weight lower it to 326, 151 and 102 [A1, A2] (Table 2, p. 15, p. 16)
- C17: The complete architecture with v-space prediction and losses reaches FVD 124, against 57 with x-space [A1, A2] (p. 16)
- C18: Shortcut forcing with 4 sampling steps approaches the quality of diffusion forcing with 64 steps, i.e. 16× faster generation; read off the figure, about 55 vs 50 FVD, while diffusion forcing at 4 steps is about 530 [A1, A2] (Fig. 8, p. 16)
- C19: Temporal attention every 4 layers speeds up training and inference (18.9 vs 9.1 FPS) and improves FVD (70 vs 80); GQA speeds inference (23.2 FPS) with FVD 71 vs 70; time-factorised long context reaches 30.1 FPS at a mild quality cost (FVD 91); register tokens don't change FVD (91) but qualitatively improve temporal consistency [A1, A2, A8] (Table 2, p. 15, p. 16)
- C20: Alternating short and long batch lengths cuts the train step from 9.8 s to 1.5 s and FVD from 102 to 80 [A1, A2] (Table 2, p. 15)
- C21: Agent tokens that no other modality can attend to are crucial to avoid causal confusion, so the world model's future predictions are influenced only by actions, not the task [A4] (Sec. 3.3, p. 6)
- C22: MAE-style patch dropout in the tokenizer improves the spatial consistency of the dynamics model's generations [A1, A2] (Sec. 3.1, p. 5)
- C23: Trained on the SOAR robotics dataset (180 h), the world model shows accurate physics and counterfactual interactions with real objects under human-controlled actions (qualitative only) [] (p. 13, Fig. 6, App. A, p. 25)

## Implementation notes

- Action labels for unlabelled video: none are inferred; the dynamics model is fed only the learned action embedding, i.e. a null action, with no inverse dynamics model or latent actions (source: Sec. 3.2, p. 5)
- Agent tokens get one-hot task ids; the paper says text embeddings could be used instead (source: Sec. 4.1, p. 10)
- Values not stated: S_r, S_a, the tokenizer's latent-token count per frame beyond the 512×16 bottleneck, layer counts and widths, λ, the imagination horizon, optimiser, learning rates, total training steps and wall-clock, and how many TPUs each run used within 256–1024 (source: assumption that they are unstated after a full read; the website named on p. 1 may have more)
- Evaluation statistics: success rates over 1000 episodes; no seeds or error bars are reported (source: Fig. 3, p. 9, Table 7, p. 28)
- Fig. 7 caption says "over 80%" with 100 h; the text gives 85% PSNR and 100% SSIM, so the caption is the weaker bound (source: Fig. 7, p. 14)
- C12 reports 320 context frames while the Minecraft dynamics context length is C = 192; how the evaluation reaches 320 frames isn't said (source: assumption, App. A, p. 25 vs Sec. 4.3, p. 14)
- For the survey's cross-game embedding, the reusable parts are the causal tokenizer as a frame encoder, the null-action embedding for action-free video, and task or agent tokens read out of a frozen sequence model with one-way attention (source: assumption)

## Limitations / open questions

- The world model is far from a full clone of Minecraft: short memory and imprecise inventory predictions (Sec. 6, p. 18)
- Future directions the authors list: pretraining on general internet video, long-term memory, language, small amounts of corrective online data, automatic goal discovery (Sec. 6, p. 18)
- Action-label and extrapolation results are within one game (Minecraft dimensions), not across games; robotics and kitchen results are qualitative (Sec. 4.3, p. 14, App. B, p. 26)
- Genie 3 couldn't be compared, since it supports only camera actions and one "interact" button (p. 12)
- Open question: the action-label experiments measure video-prediction fidelity (PSNR, SSIM), not agent performance; does a policy trained in a world model grounded with 100 h of actions keep the C2 results?
- Open question: tasks and rewards come from VPT event annotations; how much of the offline result depends on that dense task supervision, which unlabelled web video lacks?

## Related

- extends [[hafner2024mastering]]: Dreamer 3's imagination-training agent with a scalable tokenizer and diffusion-transformer world model in place of the RSSM, PMPO with a BC-prior KL in place of return normalisation and entropy (App. F, p. 29)
- uses [[baker2022video]]: the VPT contractor dataset, its mouse discretisation and evaluation protocol, and VPT as the main baseline (Sec. 4.1, p. 10, p. 11)
- uses [[he2021masked]]: masked-patch training of the tokenizer (Sec. 3.1, p. 5)
- related [[magne2026nitrogen]]: another route to action labels for gameplay video (overlay parsing) for BC, where Dreamer 4 needs few labels for a world model
- related [[wang2025gametars]]: a VLM-based generalist game agent with keyboard and mouse actions, the other recipe Dreamer 4's VLA baseline represents
- related [[simateam2024instructable]]: keyboard-and-mouse BC agents across many games, which Dreamer 4's single-game world model doesn't attempt

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
