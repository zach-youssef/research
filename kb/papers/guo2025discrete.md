---
id: guo2025discrete
title: Recent Advances in Discrete Speech Tokens: A Review
authors: [Yiwei Guo, Zhihan Li, Hankun Wang, Bohan Li, Chongtian Shao, Hanglei Zhang, Chenpeng Du, Xie Chen, Shujie Liu, Kai Yu]
year: 2025
venue: arXiv preprint
url: https://arxiv.org/abs/2502.06490
arxiv: 2502.06490
doi:
pdf_sha256: afd8c05a6bf56afd5b7560c716081cca6023953f1d89706543c463d52ba6e4a8
tags: [survey, speech-tokenizer, neural-codec, acoustic-tokens, semantic-tokens, rvq, fsq, token-rate, benchmark, speech-only]
status: read
read: 2026-11-26
---

# Recent Advances in Discrete Speech Tokens: A Review

## TL;DR

Review of discrete speech tokens split into acoustic (codec) and semantic (SSL, supervised) tokens, with
a unified LibriTTS benchmark of reconstruction, voice conversion and ASR/intent probing at 0.45-6 kbps.
Speech only: no music or sound-effect results. Open problems: low bitrate, streaming, variable rate.

## Problem

Existing reviews cover spoken LMs or neural codecs but skip the taxonomy of semantic tokens and a
common comparison across token types (Sec. I, pp. 1-2).

## Method

A survey (pp. 1-18) plus one experiment (Sec. VI, pp. 12-14, App. II, pp. 24-26). Taxonomy (Fig. 2, p. 3):
- Quantisation basics (Sec. II, pp. 2-4): k-means, VQ with STE and commitment loss, Gumbel VQ, FSQ,
  GVQ and RVQ.
- Acoustic tokens (Sec. III, pp. 4-8): VQ-GAN codecs (CNN, Transformer or U-Net generators) vs
  diffusion codecs; general-purpose designs (single codebook, global encoders, multi-resolution,
  variable bitrate); semantic distillation (guide RVQ layer 1 with HuBERT/WavLM/Whisper features, fixed
  semantic codebook, semantic features as input/output); disentanglement (GRL, perturbation, source separation).
- Semantic tokens (Sec. IV, pp. 8-10): SSL tokens (contrastive, predictive, perturbation-invariant)
  by external clustering or an inner quantizer; supervised ASR tokens (S3 Tokenizer); token vocoders.
- Length reduction (Sec. V, pp. 10-12): deduplication, acoustic BPE (Fig. 8, p. 11), variable-frame-rate tokens.
- Experiment (Sec. VI, pp. 12-14): LibriTTS testset-B, 500 utterances, 16 kHz; semantic tokens are
  resynthesised with one trained CTX-vec2wav vocoder (Table I, p. 13); ASR and intent classification
  (SLURP) use an LSTM probe on token indices (DASB setup, App. II-B, pp. 24-26).
- Applications (Sec. VII, pp. 14-16) and challenges (Sec. VIII, pp. 16-17). Table II (p. 25) lists
  ~55 acoustic tokenizers with Q, frame rate F, vocabulary V and bitrate (1/1000 sum F_i ceil(log2 V_i) kbps).

## Assumptions

- A1: All evaluated tokens are on 16 kHz speech from LibriTTS (reconstruction, VC) and LibriSpeech/SLURP (probing); no music, sound effects or game audio (Sec. VI-C, Sec. VI-D, Sec. VI-E, pp. 13-14, App. II, pp. 24-26).
- A2: Semantic tokens are scored after resynthesis by a vocoder trained for them on 585 h of LibriTTS and given the original utterance's timbre prompt, so their reconstruction numbers are not on equal footing with codecs that need no prompt (App. II-A, p. 24).
- A3: The probing tasks use a small LSTM on token indices following DASB; tokens with large vocabularies sometimes fail to converge (NC) (App. II-B, p. 26).
- A4: One configuration per model is reported, and different frame rates or quantizer counts of the same model change the results significantly (Sec. VI-C, p. 13).
- A5: Bitrates in Table II are computed without entropy coding (Table II, p. 25).

## Claims

- C1: Acoustic tokens (codec codes) reach decent speech quality but remain far below spectrogram-vocoder quality: PESQ 2.00-3.52 at 0.9-6 kbps for EnCodec (Q=8, 6 kbps: PESQ 2.83, WER 1.53), DAC (6 kbps: PESQ 3.52, WER 1.34), TiCodec (1.5 kbps: 2.00, 3.31), SNAC (0.98 kbps: 2.23, 2.25), WavTokenizer (0.90 kbps: 2.47, 2.45), against 4.30 PESQ and 1.18 WER for mel + BigVGAN [A1, A4] (Table I, Sec. VI-C, p. 13).
- C2: Retaining low WER becomes particularly hard when the frame rate is low; low-bitrate codecs lose intelligibility: Stable-Codec (0.70 kbps) WER 4.94, SemantiCodec (0.70 kbps) 3.44, LLM-Codec (0.85 kbps) 6.25 [A1, A4] (Table I, Sec. VI-C, p. 13).
- C3: Semantic distillation can give strong reconstruction: X-Codec (4 kbps) WER 1.27, SpeechTokenizer 1.47; Mimi (1.10 kbps) WER 2.44, PESQ 2.27 [A1, A4] (Table I, Sec. VI-C, p. 13).
- C4: Semantic tokens at 0.55-0.60 kbps (HuBERT, WavLM, ContentVec, wav2vec 2.0 with k-means, S3 Tokenizer) resynthesise with WER 1.67-2.51 (S3 2.12) but GPE 4.25-18.88 and PESQ 1.15-1.37, i.e. content is kept and pitch/acoustics are lost [A1, A2, A4] (Table I, Sec. VI-C, p. 13).
- C5: In voice conversion, semantic tokens reach much higher speaker similarity (SECS 0.81-0.88) than acoustic tokens, but lose prosody (P.Corr 0.35-0.76, HuBERT 0.375, WavLM 0.374, with the authors' caveat that P.Corr is meaninglessly high when VC similarity is low, i.e. the source timbre is barely changed); disentangled acoustic tokens FACodec (SECS 0.773) and LSCodec (0.852, 0.45 kbps, WER 4.04) and TiCodec (0.642) are the acoustic ones that can do VC [A1, A2] (Table I, Sec. VI-D, pp. 13-14).
- C6: In ASR and intent-classification probing, general-purpose acoustic tokens do worse than semantic ones (ASR WER 19.4-37.2 for EnCodec/DAC/TiCodec/SNAC/WavTokenizer vs 5.5-6.1 for WavLM/HuBERT/ContentVec/wav2vec 2.0 with k-means; Mel baseline 17.4); X-Codec, which uses SSL features as inputs and outputs, reaches ASR WER 9.8 and intent accuracy 69.6 [A1, A3] (Table I, Sec. VI-E, p. 14).
- C7: Semantic-token downstream ASR correlates negatively with prosody preservation (vq-wav2vec, wav2vec 2.0 inner quantizer, S3 Tokenizer), a trade-off between emphasising semantics and preserving everything [A1, A3] (Sec. VI-E, p. 14).
- C8: Acoustic BPE shortens token sequences more for semantic tokens than for acoustic ones, and more for speaker-decoupled LSCodec than for WavTokenizer or BigCodec, so the reduction is negatively correlated with information density; deduplication discards duration, BPE keeps it [A1] (Sec. V-A, Fig. 8, p. 11).
- C9: Typical speech-token frame rates are 25-100 Hz; text is far shorter, which makes LM training harder and less stable; recent work reaches about 5 Hz for semantic units (SyllableLM, Sylber) [A1] (Sec. V, p. 10, Sec. VII-C, p. 16).
- C10: Multi-resolution coding (SNAC with three RVQ streams at 12, 23 and 47 Hz, 0.98 kbps, 24 kHz) and variable-bitrate RVQ (VRVQ, quantizer count per frame) lower bitrate; variable-frame-rate acoustic tokens (TFC, CodecSlime) are only just starting [A1] (Sec. III-B, p. 6, Sec. V-B, pp. 11-12, Table II, p. 25).
- C11: Token rate by design: Table II lists SoundStream/EnCodec at 75 Hz, DAC 86 Hz (9 quantizers, 7.74 kbps), Mimi 12.5 Hz (8 quantizers, 1.10 kbps), WavTokenizer 40 or 75 Hz (single codebook), Single-Codec 23 Hz, TS3-Codec 40/50 Hz, TaDiCodec 6.25 Hz; many are speech-trained and "some are also applicable to general audio" without saying which [A5] (Table II, p. 25).
- C12: Challenges named: the lower bound of bitrate and frame rate is open; streaming needs causal SSL encoders and token transformers with attention masks, and the cost of going causal is unclear; larger acoustic token models do better at low bitrate but cost efficiency; disentanglement of acoustic tokens is still suboptimal; variable-frame-rate tokens need exploration [A1] (Sec. VIII-1, Sec. VIII-2, Sec. VIII-3, Sec. VIII-4, p. 17).
- C13: Noise: most acoustic tokens are designed to capture noise and channel effects, but how they behave under different noise types is unclear, and this extends beyond speech to general neural audio codecs; if noise is unwanted, robustness of semantic tokens must be examined [A1] (Sec. VIII-7, p. 17).
- C14: Semantic tokens from simple clustering likely discard speaker information and prosody, harming emotion, tones, singing and non-verbal vocalisations; mitigations (more codebooks, VQ on SSL features) cost bitrate [A1] (Sec. VIII-6, p. 17).
- C15: Discrete tokens still underperform continuous features (e.g. Whisper) in many spoken-language-understanding tasks and in speaker-related tasks, and quantisation loses information; most LLM-based SLU models use continuous inputs [A1] (Sec. VII-A4, p. 15).
- C16: There is an open debate over discrete vs continuous representations for speech generation: diffusion/flow-matching on continuous features avoids quantisation error and may have a higher quality bound, while discrete-token systems are usually less robust but support decoder-only LMs [A1] (Sec. VII-B, pp. 15-16).

## Implementation notes

- Evaluation recipe: CTX-vec2wav^alpha vocoder (~32M parameters, 1M steps on 4 GPUs, ~36 s speech per GPU batch, 585 h LibriTTS), timbre from WavLM-Large layer 6; metrics NeMo-ASR WER, YIN-based GPE, wide-band PESQ, STOI, Resemblyzer SECS (source: App. II-A, p. 24).
- Probing: DASB-style LSTM (2 layers, 1024 hidden), ASR with CTC on LibriSpeech 960 h, IC on SLURP; AdamW lr 1e-4 (ASR), Adam lr 2e-4 (IC), 20 epochs, token streams below 50 Hz repeated up to 50 Hz for ASR (source: App. II-B, pp. 25-26).
- Training hyperparameters of the surveyed tokenizers are not given; take them from each tokenizer's paper (source: assumption).
- Nothing here is for sound effects, music or chiptune: a tokenizer for game audio would need its own measurement (source: assumption).

## Limitations / open questions

- Every benchmark is speech: no music, environmental sound or game audio is evaluated; the review itself says the extension of noise questions to general audio is unclear (Sec. VI, pp. 12-14, Sec. VIII-7, p. 17).
- Semantic-token reconstruction depends on an auxiliary vocoder and the reference timbre, which the authors note could leak content information (App. II-A, p. 24).
- Open question (reader's): which of the Table II codecs hold up on non-speech audio and at what rate; the review does not separate them.
- Open question (reader's): the review focuses on tokens for generation and spoken LMs, not tokens as inputs for a control policy.

## Related

- related [[wu2024overview]]: earlier overview of codecs and codec-based LMs (this review cites it as [8]); same batch.
- related [[he2021masked]]: masked-autoencoder features are a continuous alternative to discrete tokens, which this review says retain higher fidelity or better understanding performance (Sec. I, p. 1).
- Not on main yet: notes on SNAC, WavTokenizer, Moshi/Mimi, EnCodec, DAC (branch papers-audio-codecs).

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
