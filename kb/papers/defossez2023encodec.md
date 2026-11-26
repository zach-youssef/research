---
id: defossez2023encodec
title: High Fidelity Neural Audio Compression
authors: [Alexandre Défossez, Jade Copet, Gabriel Synnaeve, Yossi Adi]
year: 2023
venue: TMLR 2023 journal (arXiv v1 read, venue not printed in the PDF)
url: https://arxiv.org/abs/2210.13438
arxiv: 2210.13438
doi:
pdf_sha256: f307ade111fdaad081ac8092f4ab7543b9c925192159de5a1816dcf26a5129f7
tags: [audio, neural-codec, rvq, streaming, speech, music, general-audio, tokenizer, gan, open-weights]
status: read
read: 2026-11-26
---

# High Fidelity Neural Audio Compression

## TL;DR

EnCodec: a streaming conv+LSTM encoder/decoder with residual VQ (1024-entry codebooks, 75 frames/s
at 24 kHz), trained on speech, music and general audio with an MS-STFT discriminator and a
loss balancer. Beats Opus, EVS and Lyra-v2 in MUSHRA at 1.5-12 kbps; 13 ms latency; open weights.

## Problem

Neural codecs must cover diverse signals without overfitting and run fast on one CPU core
(Sec. 1, pp. 1-2). SoundStream was the closest related work (Sec. 2, p. 3), but its original implementation is not open sourced, so the authors re-implemented it (App. A.1, p. 16). The paper uses a single multi-scale spectrogram discriminator, a loss balancer for stable training, and a small language model to compress codes further (Abstract, p. 1).

## Method

- Encoder: 1D conv (C = 32 channels, kernel 7), B = 4 blocks of residual unit + strided conv with
  strides (2, 4, 5, 8), a two-layer LSTM, and a final conv to D dims; ELU. 24 kHz audio gives 75
  latent steps/s, 48 kHz gives 150. The decoder mirrors it (Sec. 3.1, pp. 3-4).
- Streamable variant: all padding before the first step, weight norm instead of layer norm, output
  320 samples (13 ms) once the first 320 arrive. Non-streamable variant: layer norm, 1 s chunks with
  10 ms overlap (Sec. 3.1, p. 4).
- RVQ: up to 32 codebooks (16 at 48 kHz) of 1024 entries (10 bits each); EMA codebook updates with
  decay 0.99, unused entries re-sampled, straight-through estimator, commitment loss (Sec. 3.2, p. 4;
  Eq. 3, p. 6). Training samples the number of codebooks per batch to cover 1.5, 3, 6, 12, 24 kbps
  (24 kHz) or 3, 6, 12, 24 kbps (48 kHz), with one discriminator per bandwidth (Sec. 3.2, p. 4; Sec. 3.4, p. 6).
- Optional entropy coding: a 5-layer, 8-head Transformer (200 channels, 800 feed-forward) predicts
  each codebook's distribution at step t from step t-1 (codebooks of one step treated independently),
  with a range coder; trained on 5 s sequences, 3.5 s receptive field (Sec. 3.3, pp. 4-5).
- Loss: L1 time loss, multi-scale mel L1+L2 (window 2^i, i = 5..11), hinge adversarial loss on a
  multi-scale STFT discriminator (5 scales, windows 2048-128), relative feature matching, commitment
  loss; the balancer rescales each loss's gradient so weight λ_i is the fraction of the gradient
  (Eq. 1-5, pp. 5-6). Weights: λ_t 0.1, λ_f 1, λ_g 3, λ_feat 3 at 24 kHz (Sec. 4.4, p. 8).
- Data: 24 kHz mono models on speech (DNS Challenge 4, Common Voice), general audio (AudioSet, FSD50K)
  and music (Jamendo); mixing of 2-3 sources and random reverberation; the 48 kHz stereo model uses
  music only (Sec. 4.1, pp. 6-7; Table A.1, p. 16).

## Assumptions

- A1: Training includes speech, general audio (AudioSet, FSD50K) and music; the four test categories are DNS clean speech, speech mixed with FSD50K samples, Jamendo music and a proprietary music set, so general sound is only tested as noise mixed into speech, and no game audio is used (Sec. 4.1, p. 7, Table 1, p. 8).
- A2: Quality is judged by MUSHRA (50 5-s samples per category, at least 10 ratings each), ViSQOL and SI-SNR, not by downstream-model usefulness (Sec. 4.3, p. 8).
- A3: Timings are on a single thread of a 2019 MacBook Pro CPU at 6 kbps (Sec. 4.6, Table 5, p. 10).
- A4: The bandwidth quoted is the nominal RVQ rate; entropy-coded rates come from the optional Transformer (Sec. 3.3, Table 1, pp. 4-8).

## Claims

- C1: In the streamable setup at 1.5, 3, 6 and 12 kbps, EnCodec beats Opus, EVS and Lyra-v2 in MUSHRA at the bandwidths compared; at 3 kbps (average 67.0 clean speech, 89.6 Music Set-1) it beats Lyra-v2 at 6 kbps and Opus at 12 kbps on average [A1, A2] (Sec. 4.5, Fig. 3, Table 1, pp. 7-8).
- C2: The entropy-coding Transformer cuts bandwidth by about 25-40% (3 kbps to 1.9 kbps; 6 to 4.1; 12 to 8.9), with a lower ratio at higher bandwidth, which the authors tie to the small Transformer [A1, A4] (Sec. 4.5, Table 1, p. 8).
- C3: An MS-STFT discriminator alone is enough for high quality (MUSHRA 77.5, ViSQOL 4.35, SI-SNR 6.67); adding MPD gives MUSHRA 79.0, and MSD+Mono-STFT scores 62.9 [A1, A2] (Sec. 4.5.1, Table 2, p. 9).
- C4: Streamable vs non-streamable at 6 kbps on a speech/music mix: SI-SNR 6.67 vs 7.46, ViSQOL 4.35 vs 4.39, both far above Opus (2.45, 2.60) and EVS (1.89, 2.74) [A1, A2] (Table 3, p. 9).
- C5: On music-only (Jamendo) training with the DiffQ quantizer, runs with the loss balancer give ViSQOL 3.62-4.17 across 14 weightings, while without it some weightings fail (SI-SNR -35.8, ViSQOL 2.82 at lambda_g = 100); the authors conclude it stabilises training, makes the weights interpretable and shows "no degradation compared to an identified combination of coefficients"; that holds for SI-SNR, but in three of the five λ_t = 10 pairs (those with λ_g ≤ 4) ViSQOL is higher without the balancer (4.00 vs 3.65, 3.74 vs 3.62, 3.88 vs 3.73) [A1] (Sec. 3.4, Eq. 5, p. 6, App. A.3, p. 18, Table A.4, p. 19).
- C6: Stereo 48 kHz music (the only 48 kHz training data): EnCodec at 6 kbps (MUSHRA 82.9) matches Opus at 24 kbps (82.9) and is comparable to MP3 at 64 kbps (82.7), while 12 kbps scores 88.0; entropy coding saves 20-30% [A1, A2] (Sec. 4.5.2, Table 4, pp. 9-10).
- C7: Initial latency of the 24 kHz streaming model is 13.3 ms; the 48 kHz non-streaming model's is 1 s; entropy coding adds about 13 ms more, since frame t needs t+1 partially received [A3] (Sec. 4.6, p. 10).
- C8: At 6 kbps the 24 kHz model encodes at RTF 9.8 and decodes at 10.4 (about 10x faster than real time); with entropy coding RTF is 1.6; 48 kHz is 6.8/5.1, and 0.68/0.66 with entropy coding (slower than real time) [A3] (Sec. 4.6, Table 5, p. 10).
- C9: With RVQ, EnCodec at 3 kbps (MUSHRA 76.8) beats the authors' re-implementation of SoundStream (71.8) and EnCodec with a DiffQ quantizer (72.3) [A1, A2] (App. A.3, p. 17, Table A.2, p. 18).
- C10: Capacity trade-off: C = 16 gives RTF 26/25.7 but ViSQOL 4.32; C = 64 gives RTF 1.3/3.1 and ViSQOL 4.38; removing the LSTM keeps ViSQOL 4.35 with RTF 15.0/14.6 but lowers SI-SNR (6.40 vs 6.67) [A3] (App. A.3, Table A.3, p. 18).

## Implementation notes

- Tokens per second: 75 frames/s at 24 kHz times n_q; each codebook costs 0.75 kbps, so 1.5, 3, 6, 12, 24 kbps are n_q = 2, 4, 8, 16, 32 (the text says a multiple of 4, which doesn't give 1.5 kbps; derived by reader) (source: Sec. 3.2, p. 4, assumption for the arithmetic).
- For a 60 fps game, 75 audio frames/s is 1.25 per video frame; at 6 kbps (8 codebooks) about 10 tokens per game frame, at 1.5 kbps (2 codebooks) about 2.5 (reader's arithmetic) (source: assumption).
- Training: 300 epochs of 2,000 updates, Adam, batch 64 of 1 s, lr 3e-4, betas (0.5, 0.9), 8 A100 GPUs; discriminator updated with probability 2/3 (24 kHz) or 0.5 (48 kHz); balancer R = 1, β = 0.999 (source: Sec. 4.4, p. 8, Sec. 3.4, p. 6).
- Entropy-coding probabilities are rounded to 1e-6, range width 2^24, minimum range 2, since encoder and decoder floating point may differ (source: Sec. 3.3, p. 5).
- Weights and code: github.com/facebookresearch/encodec (source: p. 1). Licence is not stated in the paper.
- Streaming batch-vs-stream numerical mismatch above 1e-8 motivated the rounding (source: Sec. 3.3, p. 5).

## Limitations / open questions

- The 48 kHz model is not streamable at low latency (1 s) and slower than real time with entropy coding (Sec. 4.6, p. 10).
- The Transformer entropy coder models codebooks of one step independently and compresses less at high bandwidth (Sec. 3.3, Sec. 4.5, pp. 4, 8).
- General audio is in training data but results by content type are only speech and music (Table 1, p. 8).
- Open question: how well EnCodec codes synthesised console sound (chip tunes, noise-channel effects), which are far from the training distribution.
- Open question: whether the first codebooks alone (n_q = 2, 1.5 kbps) retain enough event information for a policy.

## Related

- extends [[zeghidour2022soundstream]]: same RVQ-GAN design; replaces the discriminators with MS-STFT, adds a loss balancer, a Transformer entropy coder and open weights.
- related [[he2021masked]]: both learn modality tokens/features by reconstruction; MAE is image only.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
