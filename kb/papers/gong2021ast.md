---
id: gong2021ast
title: "AST: Audio Spectrogram Transformer"
authors: [Yuan Gong, Yu-An Chung, James Glass]
year: 2021
venue: Interspeech 2021 (arXiv v3 read)
url: https://arxiv.org/abs/2104.01778
arxiv: 2104.01778
doi:
pdf_sha256: c237e981755e9ba5047e7290ffcb7d38d8292e166730ae3cd180b7833f184303
tags: [audio, transformer, spectrogram, audio-classification, audioset, imagenet-transfer, supervised, vit]
status: read
read: 2026-11-26
---

# AST: Audio Spectrogram Transformer

## TL;DR

A convolution-free audio classifier: a log-Mel spectrogram is cut into overlapping 16x16 patches and fed
to a ViT/DeiT initialised from ImageNet (positional embeddings cut and interpolated). Supervised only:
AudioSet 0.459 mAP single (0.485 ensemble), ESC-50 95.6%, Speech Commands V2 98.1%.

## Problem

Audio classification was dominated by CNNs, or CNN plus attention hybrids; it was unclear whether the CNN
is needed, and Transformers need much data while audio sets are small (Sec. 1, p. 1; Sec. 2.2, p. 2).

## Method

- Input (Sec. 2.1, p. 2): t seconds of audio become a 128 x 100t log-Mel filterbank (25 ms Hamming window,
  10 ms shift). Split into N 16x16 patches with overlap 6 in both axes, N = 12 ceil((100t-16)/10), linearly
  projected to 768-d, plus a trainable positional embedding and a [CLS] token.
- Encoder: standard Transformer encoder, 768-d, 12 layers, 12 heads; [CLS] output goes to a linear layer
  with sigmoid (Sec. 2.1, p. 2). No decoder, no self-supervised stage: purely supervised.
- ImageNet transfer (Sec. 2.2, p. 2): the 3-channel patch embedding weights are averaged to 1 channel;
  the 24x24 ViT positional grid is cut and bilinearly interpolated to 12x100; the classifier head is
  reinitialised; the two DeiT [CLS] tokens are averaged. DeiT with distillation (87M params, 85.2% IN top-1) is used.
- Training (Sec. 3.1.1, p. 3): mixup 0.5, SpecAugment-style masking (192 frames, 48 bins), balanced
  sampling for the full set, weight averaging and ensembles; Adam, BCE, batch 12; the full set takes 5 epochs.
- Variable input length with the same architecture: 1 s (Speech Commands), 5 s (ESC-50), 10 s (AudioSet) (Sec. 1, p. 1).

## Assumptions

- A1: Pre-training is supervised ImageNet (plus supervised AudioSet for the "-P" variants), not self-supervised
  (Sec. 2.2, Sec. 3.2, pp. 2-4).
- A2: Tasks are clip-level classification on 1-10 s clips with the full spectrogram visible to the encoder
  (Sec. 3, pp. 2-4).
- A3: Content is real recordings (YouTube events, environmental sounds, spoken commands); nothing synthetic
  or game-like (Sec. 3.1.1, Sec. 3.2, pp. 2-4).
- A4: Results use sequence length of about 1200 patches for 10 s, with quadratic attention cost (Table 5, p. 4).

## Claims

- C1: On AudioSet, AST gets 0.347 mAP (balanced set) and 0.459 (full set) single model, 0.378 / 0.485 with a six-to-eleven model ensemble, above the previous best PSLA (0.319 single / 0.444; ensemble 0.362 / 0.474) [A1, A2, A3] (Sec. 3.1.2, Table 1, p. 3)
- C2: On ESC-50 AST gets 88.7% with ImageNet pre-training only and 95.6% with additional AudioSet pre-training (previous best 86.5 / 94.7) [A1, A2, A3] (Sec. 3.2, Table 7, p. 4)
- C3: On Speech Commands V2 (35 classes) AST gets 98.11% with ImageNet only and 97.88% with AudioSet pre-training (previous 97.4 / 97.7), so AudioSet pre-training was not needed for speech commands [A1, A2, A3] (Sec. 3.2, Table 7, p. 4)
- C4: ImageNet pre-training matters, most with little data: balanced AudioSet 0.148 from scratch vs 0.347 pre-trained; full set 0.366 vs 0.459 [A1, A2, A3] (Sec. 3.1.3, Table 2, p. 3)
- C5: The ImageNet model that is best on ImageNet (DeiT with distillation, 85.2% top-1) also gives the best AST on balanced AudioSet (0.347), but the ordering is not monotone: DeiT without distillation (ImageNet 82.9%) gets 0.330, tying ViT-L (85.1%, trained without patch overlap for memory reasons) and above ViT-B (84.6%, 0.320); positional embedding reuse matters (reinitialised 0.305 vs interpolated 0.347; nearest-neighbour 0.346) [A1, A2, A3] (Sec. 3.1.3, Table 3, Table 4, pp. 3-4)
- C6: Patch overlap raises accuracy but lengthens the sequence: 512 patches (no overlap) 0.336 / 0.451, 1212 patches (overlap 6) 0.347 / 0.459 on balanced / full set; no overlap still beats prior systems [A1, A2, A3, A4] (Sec. 3.1.3, Table 5, p. 4)
- C7: Trained from scratch without overlap, 128x2 temporal-order patches beat 16x16 squares (0.154 vs 0.143 balanced), but no ImageNet model exists for them, so squares are used; also from scratch, 16x16 patches beat 32x32 (0.143 vs 0.139), the only patch-size comparison [A1, A2, A3] (Sec. 3.1.3, Table 6, p. 4)
- C8: The same architecture handles 1, 5 and 10 s inputs and speech and non-speech content, which the authors read as potential for a generic audio classifier [A1, A2, A3] (Sec. 3.2, p. 4)
- C9: AST converges in 5 epochs on full AudioSet versus 30 for the CNN-attention hybrid PSLA [A1, A2, A3] (Sec. 3.1.2, p. 3)

## Implementation notes

- Patch count: the text's formula N = 12 ceil((100t-16)/10) gives 1188 for t = 10, while Table 5 lists 1212 for the used overlap-6 setting; the paper doesn't reconcile them (source: Sec. 2.1, p. 2, Table 5, p. 4).
- Time resolution: reader's arithmetic from Sec. 2.1, p. 2; stride 10 frames = 100 ms per patch column (about 6 game frames at 60 Hz), 12 frequency positions per column, so about 10 columns (120 tokens) per second at overlap 6 (source: assumption).
- Input normalisation to dataset mean 0 and std 0.5 (source: Sec. 2.2, p. 2).
- ESC-50: LR 1e-4 (AST-S) / 1e-5 (AST-P), batch 48, 20 epochs; Speech Commands: LR 2.5e-4, batch 128 (source: Sec. 3.2, p. 4).
- Code at https://github.com/YuanGongND/ast, not inspected (source: p. 1).

## Limitations / open questions

- The paper states no limitations section; the main cost it notes is quadratic overhead growth with patch overlap (Sec. 3.1.3, p. 4).
- Open question: ImageNet initialisation is the whole recipe here; whether it is useful for non-natural audio (chip music) is untested. Later Audio-MAE found ImageNet init does not help once AudioSet SSL is used ([[huang2022masked#C8]]).
- Reader's note: inference latency is not reported.
- Open question: a 1-10 s bidirectional clip classifier has no notion of per-frame output; a frame-synced policy would need to chunk audio into short windows (reader's inference).

## Related

- related [[huang2022masked]]: Audio-MAE (a year later) compares against AST and drops ImageNet supervision for AudioSet self-supervised pre-training.
- related [[chen2023beats]]: BEATs is also a spectrogram-patch ViT and reports AST as a baseline.
- related [[vaswani2017attention]]: the encoder is the unmodified original Transformer encoder.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
