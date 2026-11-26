---
id: chen2023beats
title: "BEATs: Audio Pre-Training with Acoustic Tokenizers"
authors: [Sanyuan Chen, Yu Wu, Chengyi Wang, Shujie Liu, Daniel Tompkins, Zhuo Chen, Furu Wei]
year: 2023
venue: ICML 2023 (arXiv preprint v1 read)
url: https://arxiv.org/abs/2212.09058
arxiv: 2212.09058
doi:
pdf_sha256: ca16a4100e27272cb220cf74b81b5ad30179d3a2a7cbd101157ea17ee1bc2bfc
tags: [audio, self-supervised, masked-prediction, tokenizer, audioset, general-audio, spectrogram, vit]
status: read
read: 2026-11-26
---

# BEATs: Audio Pre-Training with Acoustic Tokenizers

## TL;DR

Iterative audio SSL on AudioSet: a 90M ViT predicts tokenizer labels of masked spectrogram patches;
the tokenizer starts random, then is distilled from the last model. Self-supervised (iter2/3): AS-2M
48.0-48.1 mAP, ESC-50 95.1-95.6%; with AS labels in the tokenizer (iter3+) 48.6 mAP. Clip-level.

## Problem

State-of-the-art audio SSL (Audio-MAE and kin) reconstructs spectrogram patches, which the authors
argue spends capacity on low-level time-frequency detail (Sec. 1, pp. 1-2). Masked discrete-label
prediction works for speech (HuBERT) but needs phoneme-like tokens; general audio (events, nature,
music) has no such labels and the speech tokenizers do not transfer (Sec. 1, p. 2).

## Method

- Input: audio resampled to 16 kHz, 128-d Mel filterbank, 25 ms Povey window, 10 ms shift, normalised;
  split into 16x16 patches, flattened to a sequence (Sec. 4.2, p. 8). Clips are AudioSet 10 s clips.
- Iteration loop (Sec. 3.1, Fig. 1, p. 4): tokenizer labels the unlabeled audio, the SSL model is trained
  on masked label prediction, then the converged model (pre-trained, or fine-tuned on labels) is the
  teacher for the next tokenizer.
- Iteration 1 tokenizer: frozen random linear projection plus frozen random codebook, label = nearest
  codebook vector index (Eq. 1, Sec. 3.2.1, p. 5).
- Later tokenizers (Sec. 3.2.2, Eq. 2-3, p. 6): 12-layer Transformer encoder, l2-normalised nearest
  codebook lookup (straight-through gradient, EMA codebook), then a 3-layer Transformer estimator
  predicts the teacher's last-layer output from the quantised vectors; loss is cosine similarity plus
  VQ commitment terms. The estimator is dropped at inference. Codebook: 1024 entries of 256 dims (p. 9).
- SSL model (Sec. 3.3, pp. 6-7): ViT with convolutional relative position embedding, gated relative
  position bias and DeepNorm; 12 layers, 768-d, 8 heads, 90M params (Sec. 4.2, p. 8). Masked Audio Modeling masks 75% of
  patches, encodes only visible patches, and a label predictor outputs labels with cross entropy over
  masked positions (Eq. 4). Fine-tuning drops the predictor, adds a linear classifier on mean-pooled
  outputs, with SpecAugment and mixup.
- Variants: iter1 (random projection), iter2, iter3 (self-distilled from the previous SSL model), and
  iter3+ (tokenizer distilled from a supervised fine-tuned iter2 model, predicting classification logits,
  so it uses downstream labels in pre-training) (Sec. 4.2, p. 8).
- Pre-training: 400k steps, batch 5.6K seconds, peak LR 5e-4 (p. 9; App. A.1, Table 4, p. 16).

## Assumptions

- A1: Pre-training data is AudioSet (about 1.9M unbalanced plus 21K balanced 10 s YouTube clips, mostly
  real-world event audio, no game audio) (Sec. 4.1, p. 8).
- A2: The tasks evaluated are clip-level classification (AudioSet tagging, ESC-50, Speech Commands,
  IEMOCAP), the encoder run bidirectionally over the whole clip (Sec. 3.3.3, Sec. 4.1, pp. 7-8).
- A3: Fine-tuning uses downstream labels; no frozen-encoder or RL use is evaluated, except the SUPERB
  frozen setting for the wav2vec numbers in Table 1 (Sec. 4.3, Table 1, p. 9).

## Claims

- C1: BEATs_iter3+ (90M) sets single-model AS-2M mAP 48.6 vs the previous best 47.4 (Audio-MAE Large, 304M) and AS-20K 38.9 vs 37.6; iter3+ uses AudioSet labels in pre-training (its tokenizer teacher is fine-tuned on the same labelled set as the task), and the purely self-supervised iter2 gets 48.1 and 38.3, also above Audio-MAE Large [A1, A2, A3] (Sec. 4.2, p. 8, Sec. 4.3, Table 1, Table 2, pp. 9-10)
- C2: On ESC-50 BEATs reaches 98.1% with extra AS-2M supervised training (error 2.6% to 1.9%); without it iter3 gets 95.6% and iter3+ (whose tokenizer is trained with AudioSet labels, see C1) 96.2-97.1% (Table 2), and the text reports error 5.9% to 4.4% [A1, A2, A3] (Sec. 4.3, Table 1, Table 2, pp. 9-10)
- C3: The authors say BEATs_iter1, with a random-projection tokenizer alone, beats "previous works" on five of six tasks (AS-2M, ESC-50, KS1, KS2, ER) and take this as evidence that discrete label prediction beats reconstruction; against the self-supervised rows of Table 1 it wins on AS-2M (47.9 vs 47.4), KS1 (98.0 vs 96.9) and ER (65.9 vs 63.4), ties on KS2 (98.3, Audio-MAE) and loses on ESC-50 (94.0 vs Audio-MAE 94.1) and AS-20K (36.0 vs 37.6) [A1, A3] (Sec. 4.3, Table 1, pp. 9-10)
- C4: The self-distilled tokenizer (iter2) improves most on low-data tasks (AS-20K 36.0 to 38.3, ESC-50 94.0 to 95.1), while a third iteration with a self-supervised teacher adds little (iter3 AS-2M 48.0, AS-20K 38.3), i.e. it converges in about two iterations [A1, A3] (Sec. 4.3, Sec. 4.4, Table 2, p. 10)
- C5: A supervised-fine-tuned teacher helps more than an SSL teacher: tokenizer from AS-2M fine-tuned iter2 gives AS-2M 48.6, AS-20K 41.8 (using AS-2M labels in pre-training), ESC-50 97.1 without extra AS supervision; this uses downstream-style labels [A1, A3] (Sec. 4.4, Table 2, p. 10)
- C6: Ensembles reach 50.4 (5 models) and 50.6 (10 models) AS-2M mAP versus 49.6 for PaSST [A1, A2, A3] (Sec. 4.6, Table 3, pp. 11-12)
- C7: Fine-tuned on the speech tasks, the AudioSet-pretrained BEATs variants get KS1 97.7-98.1, KS2 98.1-98.3 and ER 64.5-66.1 (the iter3+ row also uses AS-2M labels in pre-training); Table 1 lists wav2vec 2.0 at KS1 96.2 and ER 63.4, but under the frozen-encoder SUPERB protocol, so the rows are not like for like and the paper makes no direct comparison [A1, A3] (Sec. 4.3, Table 1, p. 9)
- C8: In a t-SNE of ESC-50 clips under reverberation and noise (RIR, DNS), the spectrogram-reconstruction target scatters the same clip widely while the tokenizer targets cluster by class and are robust to the disturbance; qualitative, 10 clips [A1] (Sec. 4.5, Fig. 4, p. 11)

## Implementation notes

- Patch count and token rate: reader's arithmetic from the 10 ms shift and 16x16 patches of Sec. 4.2, p. 8; a 10 s clip is about 1000 frames, so about 62x8 = 496 patches of 160 ms x 16 mel bins if non-overlapping, i.e. one patch column per about 10 game frames at 60 Hz (source: assumption).
- Pre-training hardware: 16 GPUs, 400k steps; tokenizer training 8 GPUs, 400k steps (source: Table 4, p. 16).
- Fine-tuning hyperparameters per task: peak LR 3e-5 to 1e-4 (5e-4 is the pre-training and supervised-teacher tokenizer LR), layer-wise LR decay 0.2 to 1.0, SpecAugment 0.15-0.3, mixup 0.8 on the AudioSet and KS1 tasks (source: App. A.1, Table 4, p. 16).
- Code and checkpoints released at https://aka.ms/beats, not inspected (source: p. 1).

## Limitations / open questions

- The authors list scaling model and data, and audio-vision-language multimodality, as future work (Sec. 5, p. 12).
- iter3+ uses supervised labels in pre-training, so its numbers are not purely self-supervised (Sec. 4.2, p. 8).
- Open question: all evaluation is clip-level classification on 10 s (or 1-5 s) clips with a bidirectional encoder; nothing is shown for synchronised frame-by-frame use, game sound effects/chiptune (FM and PSG synthesis rather than recorded sound), or as an input to a policy.
- Open question: the targets are claimed robust to noise (C8); whether that carries to synthetic game audio, or helps a policy ignore music, is untested.
- Reader's note: inference latency, FLOPs and real-time factor are not reported anywhere in the paper.
- Reader's inference: frame-synchronous or streaming use would need a causal or windowed variant; the paper evaluates only whole clips.
- Reader's suggestion: BEATs is a candidate audio-side encoder for game-play models that take frames only (e.g. NitroGen); nothing in the paper tests this.

## Related

- related [[he2021masked]]: BEATs keeps MAE-style 75% masking and visible-only encoding but swaps the pixel/spectrogram reconstruction target for discrete labels.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
