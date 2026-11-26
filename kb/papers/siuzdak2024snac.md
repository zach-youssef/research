---
id: siuzdak2024snac
title: "SNAC: Multi-Scale Neural Audio Codec"
authors: [Hubert Siuzdak, Florian Grötschla, Luca A. Lanzendörfer]
year: 2024
venue: NeurIPS 2024 workshop on AI-Driven Speech, Music, and Sound Generation (workshop paper)
url: https://arxiv.org/abs/2410.14411
arxiv: 2410.14411
doi:
pdf_sha256: 086d07029fb434ce865e7d85006a7775755c91874c48cc5a08f18cb4955f81d0
tags: [audio, neural-codec, rvq, multi-scale, speech, music, sound-effects, tokenizer, open-weights, low-token-rate]
status: read
read: 2026-11-26
---

# SNAC: Multi-Scale Neural Audio Codec

## TL;DR

Multi-scale RVQ: each of 3-4 codebooks quantizes the residual at a different frame rate (about
10-115 Hz), so coarse levels are sparse in time. General-audio variants (44.1 and 32 kHz) trained on
80% music, 10% sound effects, 10% speech; a speech variant gets 0.98 kbps. Open weights.

## Problem

Existing codecs emit tokens at a high, fixed rate, which limits long-range modelling with
transformers, and acoustic tokens at high frequency carry noise (Sec. 1, p. 1). SNAC wants a
hierarchy of tokens over multiple timescales for more efficient compression (Sec. 1, p. 1).

## Method

- Builds on the RVQGAN codec: residual VQ in the bottleneck; at iteration i the residual is
  average-pooled by factor W_i, quantised, then upsampled by W_i with nearest-neighbour
  interpolation (Sec. 3, pp. 2-3).
- Noise Block after each decoder upsampling layer (x plus a learned linear map of x times Gaussian noise);
  depthwise convolutions (stabilise training, allow lr 6e-4); one local windowed attention layer at the
  lowest temporal resolution in encoder and decoder (Sec. 3, p. 3).
- General-audio models: 44.1 kHz with encoder rates (2, 3, 8, 8) and RVQ strides (8, 4, 2, 1), giving 4
  token streams at 14, 29, 57 and 115 Hz, codebooks of 4096 (12 bit), 2.6 kbps, 54.5M parameters (16M encoder,
  38.3M decoder). A 32 kHz version gives 10, 21, 42 and 83 Hz at 1.9 kbps (Sec. 4.1, p. 3).
- Speech model: 24 kHz, strides (4, 2, 1), token streams at 12, 23 and 47 Hz, about 984 bits/s,
  19.8M parameters, purely convolutional (no attention) (Sec. 4.1, p. 3).
- Training: multi-period and complex multi-scale STFT discriminators, AdamW lr 6e-4 decay 0.999994 per
  iteration, no gradient clipping; 800k iterations, batches of 16 clips of about 0.8 s (2,730 h of
  unique audio from an internal dataset); general model sampling weights 80% music, 10% sound effects,
  10% speech; the 24 kHz model on a speech dataset that may still contain podcast intro music, jingles
  and background effects (Sec. 4.2, pp. 3-4).
- Evaluation: MUSHRA-like test (webMUSHRA, Opus 6 kbps anchor), DAPS (10 speech samples) and MUSDB18-HQ
  (10 music samples of 5 s), plus ViSQOL, SI-SDR, mel and STFT distances (Sec. 4.4, p. 4).

## Assumptions

- A1: The training data is an internal dataset; general-audio models include 10% sound effects but the evaluation covers only speech (DAPS) and music (MUSDB18-HQ), not sound effects (Sec. 4.2, Sec. 4.4, p. 4).
- A2: The listening study is small: 10 speech and 10 music samples (Sec. 4.4, p. 4).
- A3: Ablations use a 44.1 kHz model, batch size 8 and 250k iterations (Sec. 4.3, p. 4).
- A4: Bitrate counts one 12-bit code per token, with no entropy coding (reader's inference, Sec. 4.1, p. 3).
- A5: The 24 kHz speech model is trained on a speech dataset that the authors say may still contain non-speech sounds such as podcast intro music, jingles and background effects (Sec. 4.2, p. 4).

## Claims

- C1: The 44.1 kHz general model emits four token streams at 14, 29, 57 and 115 Hz (about 215 tokens/s in total) with 12-bit codes, 2.6 kbps; the 32 kHz model uses 10, 21, 42 and 83 Hz (about 156 tokens/s) at 1.9 kbps [A4] (Sec. 4.1, p. 3).
- C2: The speech model uses three streams at 12, 23 and 47 Hz (about 82 tokens/s) for 0.98 kbps [A4, A5] (Sec. 4.1, p. 3, Table 2, p. 8).
- C3: In the ablation at 44.1 kHz the multi-scale model (2.6 kbps) beats a single-scale RVQ with 3 levels at a higher 3.1 kbps on all objective metrics (ViSQOL 4.00 vs 3.89; SI-SDR 3.95 vs 3.73; mel 1.38 vs 1.44) [A3] (Sec. 4.3, Table 1, p. 4).
- C4: Removing the noise block lowers SI-SDR (3.32 vs 3.95); removing local windowed attention changes metrics only marginally (ViSQOL 3.99); without depthwise convolutions training was unstable and collapsed [A3] (Sec. 4.3, Table 1, p. 4).
- C5: On speech (DAPS), SNAC at 0.98 kbps gets MUSHRA 88.4 +/- 2.6, above EnCodec at 6 kbps (75.9), DAC at 2.5 kbps (85.0) and Opus at 6 kbps (39.3); its SI-SDR (0.82) is far below those of DAC at 2.5 kbps (6.43) and EnCodec at 6 kbps (6.83), its ViSQOL (4.14) is below both (4.28, 4.36), and its mel distance (1.50) is worse than DAC at 2.5 kbps (1.37) but better than EnCodec at 6 kbps (1.60) [A1, A2, A5] (Sec. 4.4, Fig. 2a, p. 5, Table 2, p. 8).
- C6: On music (MUSDB18-HQ), SNAC 32 kHz at 1.9 kbps gets MUSHRA 77.9 and SNAC 44 kHz at 2.6 kbps gets 76.8, above EnCodec 32 kHz at 2.2 kbps (64.4) and DAC at 2.5 kbps (54.0) but below DAC at 5.16 kbps (83.0) and 7.74 kbps (90.4) [A1, A2] (Sec. 4.4, Fig. 2b, p. 5, Table 3, p. 8).
- C7: The authors note the difference in perceived quality between the 32 kHz and 44 kHz general models is marginal (MUSHRA 77.9 vs 76.8), so the 32 kHz model is enough for most tasks [A1, A2] (Sec. 4.4, Fig. 2b, p. 5, Table 3, p. 8).

## Implementation notes

- Finest stream rate = sample rate over the encoder's total stride: 44100 / (2 x 3 x 8 x 8 = 384) = 115 Hz; the coarser streams divide it by the RVQ strides 2, 4, 8 (14.4 Hz coarsest). Speech: 24000 / 512 = 47 Hz (reader's arithmetic) (source: assumption, reader's arithmetic from Sec. 4.1, p. 3).
- For a 60 fps game: the 44.1 kHz model's 215 tokens/s total is about 3.6 tokens per video frame, but spread over four rates (about 0.24, 0.48, 0.95 and 1.9 per frame), so a frame-synced policy would need to group or interleave the streams (reader's arithmetic) (source: assumption).
- Code and weights: github.com/hubertsiuzdak/snac (source: p. 1); licence not stated.

## Limitations / open questions

- No latency or streaming evaluation; the authors mention that deeper attention would increase computational cost and latency (Sec. 4.3, p. 4).
- Training data and dataset are internal, so they can't be reproduced (Sec. 4.2, p. 4).
- The paper only reports MUSHRA on speech and music; sound effects, 10% of the training mix, are not evaluated (Sec. 4.4, p. 4).
- SNAC speech has a low SI-SDR (0.82), i.e. its waveforms are not phase-aligned with the input although perceptual scores are high (Table 2, p. 8; reader's reading).
- Causality, latency, real-time factor and codebook-usage figures are not reported; the local attention layer at the coarsest rate suggests non-causal use (reader's inference, Sec. 3, p. 3).
- The local attention window size is not given (reader's observation).
- Open question: whether the multi-rate token streams help or hinder a transformer policy that wants one token group per video frame.

## Related

- extends [[kumar2023rvqgan]]: builds on the RVQGAN codec and compares against DAC at 3, 6 and 9 codebooks.
- related [[defossez2023encodec]]: baseline codec (24 kHz and 32 kHz checkpoints).
- related [[zeghidour2022soundstream]]: origin of RVQ for neural codecs.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
