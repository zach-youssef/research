---
id: ji2025wavtokenizer
title: "WavTokenizer: An Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling"
authors: [Shengpeng Ji, Ziyue Jiang, Wen Wang, Yifu Chen, Minghui Fang, Jialong Zuo, Qian Yang, Xize Cheng, Zehan Wang, Ruiqi Li, Ziang Zhang, Xiaoda Yang, Rongjie Huang, Yidi Jiang, Qian Chen, Siqi Zheng, Zhou Zhao]
year: 2025
venue: ICLR 2025 conference paper
url: https://arxiv.org/abs/2408.16532
arxiv: 2408.16532
doi:
pdf_sha256: 5fabe482a081fd234740d3ef722632fbc00bea0e3aa6864a9cdf03415d754a48
tags: [audio, neural-codec, single-codebook, tokenizer, speech, music, general-audio, open-weights, low-token-rate, vocos]
status: read
read: 2026-11-26
---

# WavTokenizer: An Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling

## TL;DR

Single-codebook (4096 entries) 24 kHz acoustic codec: 75 tokens/s (0.9 kbps) or 40 tokens/s, using
a larger VQ space, attention and an inverse-STFT decoder. On ~8k h of speech, music and audio
it gets UTMOS 4.05 vs DAC's 3.91 at 900 tokens/s; open weights.

## Problem

Standard codecs use RVQ with several quantizers and a high frame rate (DAC: 9 quantizers, 900
tokens/s at 9 kbps), which complicates and slows audio language models (Sec. 1, p. 2). The paper
asks whether one quantizer with 40-75 tokens/s can keep high reconstruction quality and richer semantics
(Sec. 1, p. 2; Abstract, p. 1).

## Method

- VQ-GAN framework as SoundStream/EnCodec: conv encoder (C = 32, B = 4, D = 512, two-layer LSTM; strides
  (2, 4, 5, 8) for 320x or (4, 5, 5, 6) for 600x downsampling of 24 kHz), a single quantizer, and a decoder
  (Sec. 3, Sec. 3.1, p. 4).
- Single quantizer with a codebook enlarged from 2^10 to 2^12 (best trade-off), k-means initialisation,
  and random re-awakening of unused codes (EMA 0.99) (Sec. 3.2, Fig. 2, pp. 4-5).
- Decoder: no mirrored upsampling; instead a conv1D, an attention block over a 3 s context window, then
  ConvNeXt blocks and an inverse Fourier transform as in Vocos (Sec. 3.3, p. 6).
- Losses: quantizer, mel L1, hinge adversarial and feature matching, with multi-period and multi-band
  multi-resolution complex-STFT discriminators (Sec. 3.4, Eq. 2-7, pp. 6-7).
- Data: about 8k h: LibriTTS, VCTK and 3,000 h of CommonVoice (speech), 2,000 h of AudioSet, Jamendo
  and MusicDB; ablations use only 585 h of LibriTTS (Sec. 4.1, p. 7; Sec. 4.3, p. 10).
- Training: up to 2M iterations (1M generator, 1M discriminator), 8 A800 GPUs, batch 40, 3 s crops, AdamW
  lr 2e-4, cosine decay (App. A, p. 17).

## Assumptions

- A1: Reconstruction on speech uses LibriTTS test-clean/test-other and LJSpeech, music uses MusicDB and audio uses AudioSet eval; no game audio is tested (Sec. 4.1, Sec. 4.2, pp. 7-8).
- A2: The baselines are the official open weights, run with the quantizer counts they were trained on; whether the one-quantizer DAC rows use a DAC trained with one quantizer or the 9-quantizer DAC evaluated with one is not stated (reader's inference that it is the latter, see Limitations) (Sec. 4.1, p. 7, Table 1, p. 8).
- A3: Ablations are run with 585 h of LibriTTS and a single quantizer at 0.9 kbps, and evaluated on LibriTTS test-clean (Sec. 4.3, p. 10).
- A4: UTMOS (a learned MOS predictor limited to 16 kHz) is the main objective metric (Sec. 4.1, p. 7).
- A5: Speed is measured on an A100 GPU; no causal or streaming operation is described (App. H, p. 17).
- A6: The 16 kHz and 48 kHz models of App. G are retrained on LibriTTS only, with default hyperparameters (App. G, p. 17).
- A7: The 24 kHz row of Table 10 is the main model trained on about 8k h, since its UTMOS (4.05) matches Table 1's 4.0486 (reader's inference, Table 1, p. 8, Table 10, p. 18).

## Claims

- C1: Compression: one quantizer with 75 tokens/s (0.9 kbps, stride 320) or 40 tokens/s (0.5 kbps, stride 600) for one second of 24 kHz audio [A1] (p. 1, Sec. 3.1, p. 4).
- C2: On LibriTTS test-clean WavTokenizer at 0.9 kbps (75 tokens/s) gets UTMOS 4.0486 vs 3.9097 for DAC at 9 kbps (900 tokens/s) and 3.0399 for EnCodec at 6 kbps; at 0.5 kbps (40 tokens/s) UTMOS 3.6016 [A1, A2, A4] (Table 1, p. 8).
- C3: DAC restricted to one quantizer (100 tokens/s) scores UTMOS 1.4940 on test-clean, far below WavTokenizer at 40 or 75 tokens/s; PESQ/STOI/V-UV F1 are also lower for DAC-1 [A1, A2] (Table 1, p. 8).
- C4: On the objective speech metrics PESQ, STOI and V/UV F1 WavTokenizer at 75 tokens/s is comparable to Vocos (4 quantizers) and SpeechTokenizer (8 quantizers) but lower than the 9-quantizer DAC (e.g. test-clean PESQ 2.37 vs 3.91, STOI 0.914 vs 0.970) [A1, A2] (Table 1, p. 8).
- C5: MUSHRA at 0.9 kbps (75 tokens/s) is 96.1 on LibriTTS test-clean, 92.9 on MusicDB and 94.4 on AudioSet, against DAC at 9 kbps (92.8, 92.6, 92.7) and EnCodec at 6 kbps (78.6, 76.9, 81.2); the reference scores 96.4, 95.3, 95.8 [A1, A2] (Table 2, p. 9).
- C6: On the ARCH benchmark (12 datasets across speech, music, acoustic events) the paper states WavTokenizer's representations substantially outperform DAC and EnCodec with one or two quantizers, and beat 9-quantizer DAC and 8-quantizer EnCodec on AM, SLURP, MTT, IRMAS, FSD50K and VIVAE [A1] (Sec. 4.2, Table 3, p. 9, App. B, p. 17).
- C7: As the token source for an autoregressive TTS model (ParlerTTS 600M config), WavTokenizer at 1 layer is the CMOS reference (CMOS-Q and CMOS-P 0.00 by construction) and DAC scores -0.35 and -0.29 against it, and WER 5.1 vs 6.9 and speaker similarity 0.61 vs 0.59 (ground truth WER 2.4) [A1, A2] (Sec. 4.2, Table 4, p. 9, Table 12, p. 18).
- C8: Codebook size 4096 (utilization 100%) gives UTMOS 4.0486 vs 3.4967 at 1024; 8192 gives 68% utilization and 16384 gives 27%, with a lower UTMOS than 4096 [A3, A4] (Sec. 4.3, Table 5, p. 10, Fig. 2, p. 5).
- C9: Training on 4,000 h instead of 585 h does not raise utilization at 16384 codes (26.5% vs 27%) [A3] (App. D, Table 8, p. 16).
- C10: A 3 s attention context beats a 1 s one (UTMOS 4.0486 vs 3.7448); 5 s is similar to 3 s (4.0448) [A3, A4] (Sec. 4.3, Table 6, p. 10).
- C11: Removing the attention module (UTMOS 3.6020), the multi-scale STFT discriminator (3.7806) or replacing the inverse-STFT decoder with a mirrored one (2.7782) each hurts at 0.9 kbps [A3, A4] (Sec. 4.3, Table 7, p. 10).
- C12: At 16 kHz (50 tokens/s), 24 kHz (75) and 48 kHz (150) the models stay at UTMOS 3.96, 4.05 and 3.96; objective metrics other than UTMOS rise with more tokens; the 16 and 48 kHz models were trained on LibriTTS only, so the comparison isn't like for like with the 24 kHz model [A1, A4, A6, A7] (App. G, Table 10, pp. 17-18).
- C13: Reconstruction RTF on an A100 is 0.0098 for WavTokenizer vs 0.0144 for DAC and 0.0175 for EnCodec [A5] (App. H, Table 11, p. 18).
- C14: The authors state that current codec tokens, WavTokenizer's included, lack the ASR-like understanding of semantic models, and propose richer semantics and training on far more speech data as future work [] (App. F, p. 17).
- C15: On LJSpeech (out-of-domain) the 9 kbps DAC has a higher UTMOS than WavTokenizer at 0.9 kbps (4.3007 vs 4.2580), although the text says WavTokenizer surpasses it "on all test sets"; on both LibriTTS sets WavTokenizer is higher [A1, A2, A4] (Sec. 4.2, p. 7, Table 1, p. 8).

## Implementation notes

- Tokens per second: 75 (stride 320) or 40 (stride 600) at 24 kHz, 50 at 16 kHz, 150 at 48 kHz, one token each; bitrate 0.9 kbps = 75 x 12 bits (Table 1, Table 10) (source: Table 1, p. 8, Table 10, p. 18).
- For a 60 fps game: 75 tokens/s is 1.25 tokens per video frame; 40 tokens/s is 0.67, so a frame-synced agent would add about 1 token per frame (reader's arithmetic; the rates don't divide 60) (source: assumption).
- Codebook 4096 (12 bits); k-means with 200 cluster centres is used for initialisation (source: Sec. 3.2, p. 5).
- Batch size 40, 3 s random crops from 10 s segments, 2M iterations total, A800 GPUs (source: App. A, p. 17).
- Code and checkpoints: github.com/jishengpeng/WavTokenizer (source: p. 1); licence not stated in the paper.
- Streaming/causality: the decoder's attention is trained on 3 s windows and the encoder has an LSTM; the paper says the model "achieves good reconstruction even for long audio sequences during inference" (Sec. 3.3, p. 6) but doesn't describe a causal or chunked mode or latency, so whether it can run frame by frame is the reader's open question (source: assumption).

## Limitations / open questions

- Authors state semantic content is weaker than in semantic (ASR-like) models, and note that distillation is only a temporary solution (App. F, p. 17).
- Large codebooks (16384) have low utilization even with more data (Sec. 4.3, App. D, pp. 10, 16).
- Subjective test sizes: 50 samples per category, at least 10 ratings each (App. C, p. 17).
- The single-codebook claim for DAC's baseline compares a DAC trained with 9 quantizers evaluated with one at 100 tokens/s (Table 1, p. 8, reader's reading).
- The loss weights λ_q, λ_mel, λ_adv, λ_feat are called hyper-parameters but not given (Eq. 7, p. 7).
- Open question: whether a decoder trained with a 3 s attention context (Sec. 3.3, p. 6; long inputs are fine offline per the paper) works for causal, frame-by-frame online use, which the paper does not evaluate.
- Open question: whether one token per 13 ms captures short chip-music sound effects; nothing is tested on game audio.

## Related

- extends [[kumar2023rvqgan]]: compares against DAC at 1 and 9 quantizers; adopts its complex STFT discriminator idea.
- extends [[defossez2023encodec]]: same encoder layout (C = 32, strides, LSTM); baseline.
- extends [[zeghidour2022soundstream]]: VQ-GAN codec framework.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
