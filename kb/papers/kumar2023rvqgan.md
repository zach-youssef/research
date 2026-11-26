---
id: kumar2023rvqgan
title: High-Fidelity Audio Compression with Improved RVQGAN
authors: [Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, Kundan Kumar]
year: 2023
venue: NeurIPS 2023 conference paper
url: https://arxiv.org/abs/2306.06546
arxiv: 2306.06546
doi:
pdf_sha256: dbec2d2cd55f62dce77f3401f7c10309b9ab0025af5b25bf2db2f8b73686708e
tags: [audio, neural-codec, rvq, speech, music, general-audio, environmental-sound, tokenizer, gan, open-weights, codebook-collapse]
status: read
read: 2026-11-26
---

# High-Fidelity Audio Compression with Improved RVQGAN

## TL;DR

Descript Audio Codec (DAC): an RVQGAN with Snake activations, factorised L2-normalised codebooks,
quantizer dropout at p = 0.5 and a multi-band STFT discriminator. One model compresses 44.1 kHz
speech, music and environmental sound to 8 kbps (86 Hz, 9 codebooks), beating EnCodec; weights open.

## Problem

SoundStream and EnCodec show GAN artifacts (tonal, periodicity, poor high frequencies), are tuned
to speech or music, and under-use their codebooks (Sec. 1, p. 2). The paper wants one universal
high-compression codec across audio types and sampling rates that works as a token source for
language models (Sec. 1, p. 2; Sec. 2, p. 3).

## Method

- Convolutional encoder (strides 2, 4, 8, 8, so 512x downsampling) and decoder with Snake
  activations; decoder dim 1536, 76M parameters (22M encoder, 54M decoder) (Sec. 3.1, Sec. 4.3, pp. 3-6).
- Frame rate S = fs / M: 44.1 kHz / 512 gives 86 Hz; 9 codebooks of 10 bits give about 8 kbps and
  90x compression. The paper gives no causality or streaming latency figure (Table 1, p. 3).
- Improved RVQ: code lookup in a low-dimensional space (8-d) with L2-normalised codes (cosine
  similarity), plus the usual VQ-VAE codebook and commitment losses, no k-means init or restarts
  (Sec. 3.2, p. 4; App. A, p. 14).
- Quantizer dropout applied with probability p; p = 0.5 is the final setting (Sec. 3.3, p. 4).
- Discriminators: multi-period waveform plus a complex multi-band multi-scale STFT discriminator
  (windows 2048, 1024, 512; band limits 0-1 split in 5) (Sec. 3.4, p. 5; Sec. 4.3, p. 6).
- Losses: multi-scale mel loss (windows 32-2048, mel bins 5-320, hop = window/4), hinge adversarial loss,
  L1 feature matching, codebook and commitment losses; weights 15, 2, 1, 1.0 and 0.25; no balancer (Sec. 3.5, p. 5).
- Data (all resampled to 44.1 kHz): speech (DAPS, DNS Challenge 4 clean, Common Voice, VCTK), music
  (MUSDB, Jamendo), environmental sound (AudioSet balanced and unbalanced); balanced sampling over the
  three domains and between full-band and band-limited sources (Sec. 4.1, 4.2, pp. 5-6).
- Test set: 3000 10 s segments, 1000 per domain (Sec. 4.1, p. 6). Metrics: ViSQOL, mel distance,
  STFT distance, SI-SDR, bitrate efficiency, and a MUSHRA-style test with 10 expert listeners (Sec. 4.4, p. 7).

## Assumptions

- A1: Training and test audio cover speech, music and environmental sound (AudioSet) at 44.1 kHz; no game audio or synthesiser-chip audio is evaluated (Sec. 4.1, pp. 5-6).
- A2: Quality is measured by reconstruction metrics and a small expert MUSHRA test (10 listeners, 12 samples), not by downstream-model use (Sec. 4.4, p. 7).
- A3: Reported bitrates are upper bounds; variable-bitrate use selects fewer codebooks (Sec. 3, p. 3).

## Claims

- C1: The final 44.1 kHz model gives about 90x compression at 8 kbps (frame rate 86 Hz, 9 codebooks); EnCodec's 24 kHz settings use 75 Hz with 32 codebooks, and 48 kHz uses 150 Hz with 16 [A3] (p. 1, Table 1, p. 3).
- C2: In listening tests at 44 kHz the proposed codec at 2.67, 5.33 and 8 kbps scores far above EnCodec at 3, 6 and 12 kbps (MUSHRA about 44, 64 and 69 vs about 23, 33 and 40), although below the reference [A1, A2] (Fig. 3, p. 9).
- C3: Objectively at 8 kbps: mel distance 0.93, ViSQOL 4.18, SI-SDR 10.75; EnCodec at 12 kbps has 1.70, 3.13, 8.36 and at 24 kbps 1.61, 3.16, 9.59; Opus at 24 kbps has ViSQOL 4.15 with only a 16 kHz bandwidth, and beats the 8 kbps model on mel distance (0.88 vs 0.93) and SI-SDR (11.65 vs 10.75) [A1] (Table 3, p. 9).
- C4: Trained with EnCodec's configuration (24 kHz, 24 kbps, 32 codebooks) the model still beats EnCodec in MUSHRA at all bitrates 3-12 kbps and in metrics (e.g. ViSQOL 4.38 vs 4.30 at 6 kbps) [A1, A2] (Fig. 4, Table 4, p. 9).
- C5: Factorised L2-normalised codes reach near-full codebook use (bitrate efficiency 99%), while EMA-learned codebooks, as in EnCodec, leave later codebooks under-used (efficiency 97%, SI-SDR 8.33 vs 9.12) [A1] (Sec. 3.2, Fig. 1, p. 4, Table 2, p. 7).
- C6: Always applying quantizer dropout (p = 1.0, the baseline row) degrades full-bandwidth quality, and p = 0.0 reconstructs poorly with few codebooks; p = 0.5 closes the full-rate gap while keeping low-bitrate quality (at 8 kbps ViSQOL 4.03 for p = 0.5 vs 3.96 for p = 1.0 and 4.09 for p = 0.0; mel loss curves in Fig. 2) [A1] (Sec. 3.3, Fig. 2, p. 4, Table 2, p. 7).
- C7: Snake activation versus ReLU is the biggest single architectural change (SI-SDR 9.12 vs 6.92; ViSQOL 3.96 vs 3.83) [A1] (Sec. 4.5, Table 2, p. 7).
- C8: Adversarial losses are critical: with reconstruction loss only, bitrate efficiency falls from 99% to 62% and SI-SDR from 9.12 to 1.07, with audible buzzing [A1] (Sec. 4.5, Table 2, pp. 7-8).
- C9: Replacing the multi-scale low-hop mel loss with a single-scale high-hop one (80 mels, window 512) lowers SI-SDR from 9.12 to 7.68, and the multi-scale loss is kept in the final recipe [A1] (Sec. 4.5, Table 2, pp. 7-8).
- C10: The codebook latent dimension matters: 8 is best; 2 or 256 give efficiency 84% and 59% and worse metrics [A1] (Sec. 4.5, Table 2, pp. 7-8).
- C11: Without balanced data sampling the output bandwidth is capped near 18 kHz [A1] (Sec. 4.2, Sec. 4.5, pp. 6-8).
- C12: By domain, the model performs best on speech and has more trouble with environmental sounds; it doesn't model glockenspiel or synthesiser sounds perfectly [A1, A2] (Sec. 5, Fig. 5, p. 10).
- C13: Training an 8 kbps ablation model with max 24 kbps gives the best metrics (ViSQOL 4.16, SI-SDR 13.83 at 24 kbps in Table 2's setting) but the final model is trained at max 8 kbps to push compression [A1] (Sec. 4.5, Table 2, pp. 7-8).

## Implementation notes

- Tokens per second: 86 frames/s times n_q (9 at 8 kbps); each codebook is 10 bits (1024 entries) so about 0.86 kbps per codebook (derived) (source: assumption, reader's arithmetic from Table 1, p. 3).
- Training budget: ablations use batch size 12 for 250k iterations; the final model batch size 72 for 400k iterations on 0.38 s excerpts (source: Sec. 4.3, p. 6).
- For a 60 fps game: 86 Hz is 1.43 audio frames per video frame; 9 codebooks give about 13 tokens per game frame, 2 codebooks about 3 (reader's arithmetic) (source: assumption).
- Frame rate follows from strides (2, 4, 8, 8) (source: Sec. 4.3, p. 6).
- Learning rate 1e-4, AdamW with beta1 0.8, beta2 0.9, lr decay gamma 0.999996 per step, for generator and discriminator; loudness normalised to -24 dB LUFS; only phase-shift augmentation (source: Sec. 4.3, Sec. 4.1, pp. 6-7).
- Code and weights: github.com/descriptinc/descript-audio-codec (source: footnote 1, p. 1); licence not stated in the paper.
- The "Proj." quantizer in Table 2 uses an 8-d codebook latent (source: Table 2, p. 7).

## Limitations / open questions

- The authors say the model has difficulty with some challenging audio; slicing by domain shows weaker results on environmental sound (Sec. 5, p. 10).
- Even at 8 kbps it falls short of the reference in MUSHRA (Sec. 4.6, p. 9).
- A misuse risk is noted (deepfakes) with watermarking suggested (Sec. 5, p. 10).
- The paper states no causality, latency or real-time factor (reader's observation).
- Ambiguity in Sec. 4.5 (p. 8): the high-hop single-scale model is printed as doing "much better at capturing certain sorts of sounds, such as cymbal crashes, beeping and alarms, and singing vocals", yet the multi-scale loss is kept and the paragraph says low-hop reconstruction is critical for fast transients; the reader reads "much better" as a slip for "worse", but the paper prints "much better".
- Open question: causal/streaming operation and latency, which a frame-synced agent would need, are not addressed.
- Open question: 86 Hz with 9 codebooks is about 13 tokens per game frame at 60 fps; fewer codebooks (coarse tokens) may be enough, as the paper says early codebooks carry the most significant information (Sec. 3.3, p. 5).

## Related

- extends [[defossez2023encodec]]: same RVQ-GAN family; adds Snake, factorised codes, a different discriminator and compares directly.
- extends [[zeghidour2022soundstream]]: quantizer dropout is introduced there and modified here to p = 0.5.
- related [[he2021masked]]: both are self-supervised modality tokenisers/encoders; MAE targets images.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
