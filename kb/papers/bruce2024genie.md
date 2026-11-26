---
id: bruce2024genie
title: "Genie: Generative Interactive Environments"
authors: [Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, Tim Rocktäschel]
year: 2024
venue: ICML 2024 conference paper (read as arXiv v1)
url: https://arxiv.org/abs/2402.15391
arxiv: 2402.15391
doi:
pdf_sha256: 94c63a5ced89355326706c88236591c2972ced12f3dece4f8982f020110c81f3
tags: [latent-actions, world-model, video-generation, foundation-model, vq-vae, spatiotemporal-transformer, maskgit, platformers, imitation-from-observation, scaling]
status: read
read: 2026-11-25
---

# Genie: Generative Interactive Environments

## TL;DR

An 11B world model trained on 30k hours of 2D-platformer video without action labels: a VQ-VAE latent
action model (8 codes) infers actions between frames and conditions a MaskGIT dynamics model, so
users play prompted images frame by frame. A frozen LAM also labels expert video for behaviour cloning.

## Problem

World models need action-labelled video, and video models are controllable only through text
(Table 1, p. 3); labelled action data is rare on the Internet (Sec. 2.1, p. 3). Genie wants frame-level
controllability from video alone, learned at scale (Sec. 1, p. 2).

## Method

- Three parts (Sec. 2.1, p. 3, Fig. 3, p. 4): a latent action model (LAM), a video tokenizer and an
  autoregressive dynamics model; all use a memory-efficient spatiotemporal (ST) transformer whose
  spatial attention is within a frame and temporal attention is causal across frames, with one FFW
  per block (Sec. 2.1, p. 3, Fig. 4, p. 4).
- **LAM** (Sec. 2.1, pp. 3-4, Fig. 5, p. 4): an encoder reads all previous frames x_{1:t} and the next
  frame x_{t+1} (raw pixels) and emits a continuous latent action; a decoder sees the history and
  that action and predicts x_{t+1}. A VQ-VAE objective limits actions to |A| = 8 codes, chosen
  small to keep the model playable by a human or agent. The decoder only provides training signal:
  at inference the whole LAM apart from the codebook is discarded and the user picks a code.
- **Video tokenizer** (Sec. 2.1, p. 4, Fig. 6): a VQ-VAE with ST-transformer encoder and decoder
  (ST-ViViT), giving per-frame discrete tokens that carry causal temporal context.
- **Dynamics model** (Sec. 2.1, p. 5, Fig. 7): a MaskGIT decoder over tokens z_{1:t-1} and stop-gradient
  latent actions, with token masking at a Bernoulli rate drawn uniformly from [0.5, 1]; latent
  actions enter as additive embeddings rather than concatenation, which improved controllability.
- **Inference** (Sec. 2.2, p. 5, Fig. 8): prompt image tokenized, user picks an integer action
  in [0, |A|), the dynamics model predicts the next frame's tokens iteratively, tokens are decoded to
  pixels. A code's meaning stays consistent across inputs, "akin to learning the buttons on a new
  controller" (footnote 2, p. 5).
- **Agents** (Sec. 3.3, p. 8; App. E, pp. 24-25): a frozen LAM labels expert CoinRun video
  with latent actions, a policy is trained to predict them from observations (transformer with
  ST-ViViT encoder), and a small action-labelled set builds a dictionary D from latent to real action.

## Assumptions

- A1: The training videos are 2D platformer gameplay (30k filtered hours), robot videos for the
  smaller model; no action or text annotations are used (Sec. 3, p. 5).
- A2: The latent action vocabulary is small and discrete (8 codes) (Sec. 2.1, p. 3, Table 5, p. 22).
- A3: Context is 16 frames at 10 FPS for all components (Sec. 3, p. 6).
- A4: For imitation, the target environment has discrete latent actions and a small labelled
  set maps latents to real actions (Sec. 3.3, p. 9, App. E.1, p. 24).
- A5: A latent action's meaning does not depend on the observation (the mapping to real actions
  "contains no information about the current observation") (Sec. 3.3, p. 9).

## Claims

- C1: A video-only trained model gives frame-level controllability: users prompt with a text-to-image
  picture, hand-drawn sketch or photo and step it with discrete latent actions; from several
  starting frames of the Platformers dataset (not out-of-distribution prompts) the same latent action
  gives the same motion (e.g. left, right, jump, no-op) (qualitative) [A1, A2] (Fig. 10, p. 7, Fig. 17, p. 21).
- C2: The architecture scales: final training loss falls with dynamics-model size from 41M to 2.7B
  parameters and with batch size 128, 256, 448 (2.3B model); the final model is a 10.1B dynamics
  model (10.7B in total), batch 512, 125k steps, 942B tokens, 256 TPUv5p [A1, A3] (Sec. 3.1, pp. 6-7, Fig. 9, p. 6, Table 10, p. 24).
- C3: Feeding the LAM raw pixels rather than video tokens gives better controllability: ΔtPSNR 1.91
  vs 1.33 (Platformers, 2.5B vs 2.3B models) and 2.07 vs 1.65 (Robotics, 1B). FVD is mixed: token
  input is slightly better on Platformers (38.8 vs 40.1) and worse on Robotics (257.8 vs 136.4) [A1, A2] (Sec. 3.4, p. 9, Table 2, p. 9).
- C4: The ST-ViViT tokenizer beats ViT and C-ViViT: FVD 81.4 vs 114.5 and 272.7, ΔtPSNR 1.66 vs 1.39
  and 1.37, at 0.9 GB memory against 1.6 GB for C-ViViT [A1] (Sec. 3.4, pp. 9-10, Table 3, p. 9).
- C5: A model trained on a curated 6.8M-video set (30k h, from a learned quality classifier) beats
  one on the original 55M-video set (244k h): FVD 54.8 vs 61.4 at 580M parameters [A1] (App. B.1, p. 22, Table 4, p. 22).
- C6: On CoinRun (easy and hard), a policy trained on LAM-labelled expert videos matches the
  oracle (true-action) BC with about 200 expert samples to map latents to actions, though the LAM
  was almost certainly not trained on CoinRun [A4, A5] (Sec. 3.3, p. 9, Fig. 15, p. 9, App. E.1, p. 25).
- C7: A 2.5B model trained on action-free robot videos (RT1 data, ~130k demos plus simulation and 209k
  real episodes) gets FVD 82.7 on test and learns consistent actions (down, up, left) and
  deformable-object behaviour [A1, A2] (Sec. 3.2, p. 8, Fig. 13, p. 8).
- C8: Admitted limits: 16 frames of memory limits long-horizon consistency, hallucination as in
  other autoregressive transformers, and about 1 FPS inference [] (Sec. 5, p. 11).

## Implementation notes

- LAM: 20-layer encoder and decoder, d_model 1024, 16 heads, 8 codes, patch 16, latent dim 32 (source: Table 5, p. 22); 300M parameters (source: Sec. 3, p. 6).
- Tokenizer: 200M parameters, patch 4, 1024 codes, dim 32, 300k steps, AdamW lr 3e-4 (source: Sec. 3, p. 6, Tables 7-8, p. 23).
- Dynamics: lr 3e-5 to 3e-6, bfloat16, QK norm, 25 MaskGIT steps, temperature 2 at inference
  (source: Sec. 3, p. 6, Table 9, p. 24). The final 10.1B model: 48 layers, 36 heads, d_model 5120
  (source: Table 12, p. 25).
- BC for agents: transformer policy, sequence length 4, batch 16, ST-ViViT 12 layers; oracle agent
  samples random actions 10% of the time (source: App. E.2, pp. 25-26, Tables 13-14).
- Reproducible CoinRun case study: 10M random-policy transitions, a 6-code action model
  (source: App. F, pp. 26-27, Table 16).
- No code or weights released (source: Sec. 5, p. 11).

## Limitations / open questions

- Admitted: C8; checkpoints, data and examples are not released (Sec. 5, p. 11).
- The agent result is on one environment (CoinRun) and is measured as a share of levels solved
  (Fig. 15, p. 9); it uses a frozen Platformers-trained LAM.
- Open question (user's lens): the LAM is the same idea as [[schmidt2024actions]] but with 8
  codes read from raw pixels over full history; for shmups, would 8 codes separate ship motion
  from scroll? The paper only shows platformer and robot cases.
- Open question: Genie's 16-frame memory is the same limit the user wants to avoid; the paper
  has no compressed history, and a latent action covers one frame step, not strategy.
- Open question: does the same codebook index mean the same thing across games? The paper shows
  it across prompts of one trained model, not across separately trained ones (compare
  [[jiang2026olaf#C1]] on latents drifting across contexts).

## Related

- related [[schmidt2024actions]]: contemporaneous work; Genie cites LAPO as prior work on latent actions for pre-training RL agents (Sec. 4, p. 10).
- related [[baker2022video]]: VPT labels video with an IDM trained on labelled data; Genie labels with a latent model and keeps the quality classifier idea (App. B.1, p. 22).
- related [[reed2022generalist]]: authors overlap; cited as motivation for generalist agents (Sec. 1, pp. 2-3).
- related [[gao2025adaworld]]: AdaWorld compares its continuous latent actions to Genie-style discrete ones.
- related [[jiang2026olaf]]: later work on aligning latent actions across contexts.
- related [[magne2026nitrogen]]: a policy from video that could label superplay via latent actions.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
