---
id: saito2025soundreactor
title: "SoundReactor: Frame-level Online Video-to-Audio Generation"
authors: [Koichi Saito, Julian Tanke, Christian Simon, Masato Ishii, Kazuki Shimada, Zachary Novack, Zhi Zhong, Akio Hayakawa, Takashi Shibuya, Yuki Mitsufuji]
year: 2025
venue: arXiv preprint (v1)
url: https://arxiv.org/abs/2510.02110
arxiv: 2510.02110
doi:
pdf_sha256: 8e669c9ccf5feb3ab975bb39b3421536f0e1905d04ac3379f76df6ac247ff925
tags: [audio-generation, video-to-audio, game-audio, autoregressive, diffusion-head, consistency-model, streaming, latency, stereo, dinov2]
status: read
read: 2026-11-26
---

# SoundReactor: Frame-level Online Video-to-Audio Generation

## TL;DR

Causal transformer with a diffusion head generates 48 kHz stereo audio, one 30 Hz latent per frame,
from DINOv2 grid features of past and current frames, trained on AAA-game video. Consistency tuning
gives 26.3 ms (NFE=1) to 31.5 ms (NFE=4) waveform latency per frame (causal decoder) on one H100, 30 FPS, 480p.

## Problem

Video-to-audio (V2A) generators are offline: they need the whole clip or chunks. Interactive use and
world models need audio from frames as they arrive, with causality end to end (vision encoder
included) and low per-frame latency; even AR V2A systems like V-AURA use non-causal video encoders
(Sec. 1, p. 2).

## Method

- Task: p(x_{1:n} | v_{1:n}) = prod_i p(x_i | x_{<i}, v_{<=i}), frame-aligned continuous audio tokens
  x_i and video tokens v_i (Eq. 5, Sec. 3.1, p. 4).
- Video tokens: frozen DINOv2 ViT-S/14 with registers (21M) gives grid (patch) features per frame; each
  frame's features are concatenated with the difference to the previous frame's, passed through a
  projection and 2D conv downsample, flattened, and a one-layer non-causal transformer with a learnable
  aggregation token reduces them to one token v_i per frame. Appendix: bilinear halving of the grid,
  PCA to 59 dims (70% explained variance), 512-d aggregation token; features are precomputed
  (Sec. 3.2, p. 5, App. B.1, pp. 22-23). [CLS] tokens are not used because adjacent frames give cosine
  similarity 0.99 (0.98 every other frame), i.e. little temporal signal (App. B.1, pp. 22-24; Fig. 5, p. 24).
- Audio tokens: a VAE (Stable Audio style, trained from scratch, 157M parameters, 48 kHz stereo,
  downsampling 1600) gives 64-dim continuous latents at 30 Hz, one per frame (Sec. 3.2, p. 5, App. B.1,
  p. 23).
- Backbone: LLaMA-style decoder (18 layers, width 1024, 16 heads, RMSNorm, SwiGLU, RoPE; about 250M)
  over interleaved [v_1, x_1, v_2, x_2, ...]; the output at each audio position conditions a 70M
  MAR-style diffusion head (8 blocks, width 1280) sampling one latent (Fig. 2, Sec. 3.2, pp. 4-5;
  App. B.1, p. 23).
- Stage 1: diffusion pretraining with the EDM2 loss with an uncertainty term (Eq. 6, Sec. 3.3, p. 5).
  Stage 2: Easy Consistency Tuning (ECT) of the whole network, annealing delta t to 0 (Eq. 7-8,
  Sec. 3.3, p. 6).
- Sampling: classifier-free guidance applied on the transformer output (video tokens replaced by a null
  embedding with probability 10% in training), scale w = 3.0, KV-cache, Heun solver for the stage 1 model
  (30 steps, NFE=59) and 1-4 steps for the ECT model (Sec. 3.4, p. 6, Sec. 4.2, p. 7, App. B.3, p. 25, App. C.2, p. 28); the 10% null-embedding probability is in App. B.2, pp. 24-25.
- Data: OGameData250K, 230K clips collected, clips with only background music filtered out manually;
  training split 94K samples (about 400 h), test 3,830; 30 FPS, 480p, 48 kHz stereo, random 8 s
  training clips (Sec. 4.1, p. 6).

## Assumptions

- A1: Frame-aligned pairing: each video frame has exactly one audio latent (30 Hz audio latents with 30 FPS video) (Sec. 3.1, p. 4, App. B.1, p. 23).
- A2: Training and test data are AAA gameplay videos from OGameData250K with their original audio, clips with only background music filtered out (Sec. 4.1, p. 6).
- A3: The model sees past and current frames only, but the output audio is for the same frame, so frame i's audio depends on frame i's content (Eq. 5, p. 4).
- A4: Latency figures are on a single H100, batch size 1, one CUDA stream, FlashAttention-2, torch.compile, CUDA graph for the VAE decoder, and a causal VAE decoder fine-tuned from the default one; video features come from a frozen DINOv2 ViT-S (App. B.5, p. 26).
- A5: Metrics are computed with feature extractors (OpenL3, LAION-CLAP, PaSST) on 8-s clips; the baseline V-AURA is retrained on this data and is an offline-designed model (Sec. 4.2, Sec. 4.3, pp. 6-7).
- A6: The audio is that of modern 3D AAA games (sound effects, ambience, music), not retro or 2D games (reader's inference, Sec. 4.1, p. 6).

## Claims

- C1: Per-frame latency on one H100 (30 FPS, 480p video, 16 s clip, mean over 50 clips): token-level 24.3 / 26.1 / 28.6 ms and waveform-level 26.3 / 28.2 / 31.5 ms for NFE = 1 / 2 / 4, all below the 33.3 ms real-time budget; waveform-level latency includes encoding the raw video frame and incremental waveform decoding [A1, A3, A4] (Table 6, p. 9, Sec. 4.4, p. 9, App. B.5, p. 26).
- C2: With the causal VAE decoder needed for streaming, reconstruction and generation quality are worse than with the non-causal one: rMMD-OpenL3 62.7 vs 60.4, rMMD-CLAP 0.538 vs 0.301; for generation (ECT, NFE=4) MMD-OpenL3 65.5 and MMD-CLAP 0.682 vs 64.6 and 0.451 [A1, A4, A6] (App. B.5, p. 26).
- C3: ECT needs 1-4 diffusion-head NFEs vs 59 for the stage 1 model: 14.8x fewer at NFE=4 and 59x fewer at NFE=1, with a small quality loss; ECT at NFE=1 is comparable to the diffusion model in most metrics and beats it at NFE=4 [A1, A2, A5, A6] (Sec. 4.4, p. 8, Table 1, p. 7).
- C4: Against the retrained V-AURA (an AR model whose video encoder is not causal), SoundReactor-ECT (NFE=4) is better on FAD, MMD, KL, IB-Score and FSAD, e.g. FAD-OpenL3 center 33.9 vs 68.3, KL-PaSST 1.57 vs 1.96, IB-Score 0.285 vs 0.223, FSAD 0.0180 vs 0.113 (VAE reconstruction: FAD 33.0); V-AURA is better on DeSync (0.982 vs 1.04 for ECT NFE=4) [A1, A2, A5, A6] (Table 1, Table 2, p. 7).
- C5: Listening test (50 clips, 17 raters, 0-100 scale, MUSHRA-style, ground-truth hidden as reference): overall quality GT 83.6, V-AURA 42.2, ECT NFE=1 61.9, NFE=4 64.9; semantic alignment 82.2, 42.5, 64.5, 65.2; temporal alignment 83.4, 50.5, 58.4, 64.3; stereo panning 81.2, 43.2, 60.5, 65.3 [A2, A5, A6] (Table 3, p. 7, Sec. 4.3, pp. 7-8).
- C6: Zero-shot extension to 16 s clips (twice the 8 s training window): NTK-aware RoPE interpolation shows no significant degradation on first vs last 8 s (FAD-CLAP 0.116/0.111), Position Interpolation degrades throughout and slows periodic sounds, and sliding-window attention degrades in the second half; overall FAD-CLAP 0.0992 (NTK), 0.108 (SWA), 0.138 (PI) [A1, A2, A5, A6] (Table 4, Table 5, p. 8, App. B.4, p. 25).
- C7: Vision conditioning ablations (stage 1 model): removing the adjacent-frame difference worsens MMD-OpenL3 at every PCA size (35 dims 71.2 vs 62.8; 59 dims 74.7 vs 63.9; 99 dims 72.1 vs 67.4) and most other metrics, though at 35 dims the no-subtraction variant is better on KL (1.60 vs 1.79), IB-Score and DeSync; fewer PCA dimensions (35: MMD-OpenL3 62.8) do as well or better than more (99: 67.4), and the authors say compact representations "might be feasible". These runs used half the default batch size [A1, A2, A5, A6] (Table 9, p. 28).
- C8: Generator ablations: a 10M diffusion head failed to give good audio (MMD-OpenL3 76.4) while 30M-70M heads work; fine-tuning only the diffusion head in ECT still yields good samples but fine-tuning the whole network is better (NFE=4 MMD-OpenL3 59.79 vs 62.1); the dropout rate matters, and only the CF-0.2 setting works at NFE=1 [A1, A2, A5, A6] (Table 7, Table 8, p. 27, Fig. 4, p. 9).
- C9: Supplementary VGGSound (real-world YouTube video, general sound) results: ECT NFE=4 gets FAD-PaSST 113, KL 2.37, IB 0.24, DeSync 1.03, better FAD than V-AURA (219) but behind the offline MMAudio-L (60.6 FAD, IB 0.33, DeSync 0.44); the authors note it lags offline models on some metrics [A1, A3, A5] (Table 10, p. 30, App. C.4, p. 29).
- C10: Guidance scale w in 2-3 is the optimal range for both models; at least 30 Heun steps (NFE=59) are needed for the diffusion model, since 10 steps fail and 20 are marginal [A1, A2, A5, A6] (Fig. 6, Fig. 7, p. 28, App. C.2, p. 28).

## Implementation notes

- Audio latent rate: 48 kHz / 1600 = 30 latents per second, 64 dims; the transformer sequence has 2 tokens per frame (one video, one audio), so about 60 tokens per second of video (source: App. B.1, p. 23; Sec. 3.2, p. 5, derivation by the reader).
- Parameter count: App. B.1 states 320M in total (250M transformer + 70M head); Table 10 lists 336M for the SoundReactor models, which a footnote there says counts only the generative part; the difference is not otherwise explained (source: App. B.1, p. 23, Table 10, p. 30).
- Training cost: stage 1 about 36 h on 8 H100 at batch 128, lr 1e-4, AdamW, EMA 0.9999, bf16; stage 2 200K iterations about 20 h on 8 H100 (source: App. B.2, pp. 24-25).
- The video encoder features are precomputed before training (source: App. B.1, p. 22); the PCA is presumably fitted on the training features beforehand (source: assumption).
- Stage 2 sampling schedule: intermediate time t = 2.5 for NFE=2 and t = [5.0, 1.1, 0.08] for NFE=4 (source: App. B.3, p. 25).
- Latency numbers use a 3-clip warm-up and 50 clips; stage-1 diffusion model latency is not given (source: App. B.5, p. 26).
- For long-sequence generation the NTK base scaling is xi' = xi (n'/n)^(c/(c-2)) (source: App. B.4, p. 26).
- Code: not stated in the paper; demo page only (source: abstract, p. 1).

## Limitations / open questions

- The causal VAE decoder lowers quality; improving it while keeping low overhead is future work (App. B.5, p. 26).
- A large gap remains to the best offline V2A model on VGGSound (App. C.4, p. 29).
- Minute- to hour-scale generation is left to future work (Sec. 4.4, p. 9).
- DeSync appears to favour models using Synchformer features, a possible methodological bias (App. C.4, footnote 13, p. 29).
- Open question: audio is generated from video, and no action or policy is involved; the paper does not test audio as an input to an agent.
- Open question: modern 3D AAA games with rich music; whether 8-bit/16-bit game sound (chip synthesis, tiny sample sets) is easier or harder is untested (reader's inference).
- Open question: the 26-32 ms latency covers the generator and the DINOv2 ViT-S, but excludes the game emulator and, in a policy loop, a downstream policy; a policy that only consumes game audio would use an encoder, not this generator (reader's inference).

## Related

- related [[bruce2024genie]]: the intro cites Genie as the silent frame-level world models this task would extend to sound.
- related [[chen2022soundspaces]]: cited as audio-visual simulation for embodied learning.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
