---
id: chung2021w2vbert
title: "W2V-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training"
authors: [Yu-An Chung, Yu Zhang, Wei Han, Chung-Cheng Chiu, James Qin, Ruoming Pang, Yonghui Wu]
year: 2021
venue: IEEE ASRU 2021 (arXiv v2 read)
url: https://arxiv.org/abs/2108.06209
arxiv: 2108.06209
doi:
pdf_sha256: 36f0478d19c4b08124061c80bf8488b1a24fbd806f91da99f97f837a158ae439
tags: [speech, self-supervised, contrastive, masked-prediction, conformer, quantization, asr, librispeech]
status: read
read: 2026-11-26
---

# W2V-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training

## TL;DR

One model: conformers with a wav2vec 2.0 contrastive loss learn a codebook, and more conformers predict
its token ids at masked positions (BERT-style). Speech only (Libri-Light 60k): 0.6B/1B models get
test-other WER 2.9/2.8 with no LM or self-training, on test-other matching or beating HuBERT.

## Problem

MLM-style speech pre-training needs discrete targets. vq-wav2vec and DiscreteBERT freeze a separately trained
quantizer; HuBERT iterates k-means and re-training with hand-set cluster schedules. wav2vec 2.0 is end to end but
contrastive only (p. 1, Sec. 2, p. 2).

## Method

- Architecture (Sec. 3.1, Fig. 1, p. 2): a feature encoder (two 2D convolutions with stride (2, 2), 4x reduction
  in sequence length) on log-mel input; a contrastive module (linear projection plus N conformer blocks plus a
  quantizer); a masked-prediction module (M conformer blocks) with a softmax over token ids. Context vectors from
  the contrastive module feed the MLM module.
- Contrastive loss (Sec. 3.2, Eq. 1, pp. 2-3): wav2vec 2.0's task with codebook diversity loss (alpha 0.1);
  masked positions get random vectors instead of a learned mask vector; distractors come from other masked steps;
  the quantizer sees unmasked features and yields the token ids.
- MLM loss L_m: cross entropy of predicting those token ids at masked positions. Total L_p = beta L_c + gamma L_m
  with beta = gamma = 1, so both tasks train simultaneously (Eq. 2, p. 3).
- Sizes (Table 1, p. 3): XL 0.6B (12 contrastive + 12 MLM conformer blocks), XXL 1.0B (12 + 30); model dim 1024,
  8 heads, conv kernel 5, codebook 1024 x 1024.
- Input and masking (Sec. 4.1-4.2, p. 3): 80-d log-mel filterbank; start probability 0.065, span 10 steps (may overlap);
  Libri-Light unlab-60k pre-training; batch 2048, peak LR 2e-3, 25k warm-up.
- ASR fine-tuning (Sec. 3.3, Sec. 4.3, pp. 3-4): from the 400k-step pre-training checkpoints (Sec. 4.3, p. 3); sequence transducer with a two-layer LSTM decoder (640 hidden),
  projection block with Swish and batch norm; SpecAugment, Noisy Student self-training and LM fusion as in w2v-Conformer.

## Assumptions

- A1: Pre-training data is English read speech (Libri-Light 60k h) or Google Voice Search speech (34.3k h)
  (Sec. 4.1, p. 3, Sec. 5.3, p. 6).
- A2: Evaluation is ASR (WER) after fine-tuning a transducer on labeled speech with SpecAugment, and for the headline
  numbers self-training and an LM (Sec. 4.3, pp. 3-4, Table 2, p. 4).
- A3: Models are 0.6-1B parameters trained at large scale (batch 2048) (Sec. 4.2, Table 1, p. 3).
- A4: The contrastive module needs enough capacity and clean negatives; non-speech or short context makes the
  contrastive task easier to cheat (Sec. 5.2-5.3, pp. 5-6).

## Claims

- C1: Pre-training only, no self-training and no LM, w2v-BERT XL reaches test / test-other WER 1.5 / 2.9 on LibriSpeech (960 h labels), matching or beating other models that use an LM; XXL reaches 1.5 / 2.8 without LM [A1, A2, A3] (Sec. 5.1, Table 2, p. 4)
- C2: Against wav2vec 2.0 without LM, w2v-BERT XXL has 28%, 42%, 32% and 38% relative WER reduction on dev, dev-other, test, test-other, and 13% / 13% / 17% / 18% with LM; the authors note wav2vec 2.0's LM is much larger [A1, A2, A3] (Sec. 5.1, Table 2, p. 4)
- C3: With pre-training plus self-training and LM, XXL gets test-clean / dev-other / test-other 1.4 / 2.4 / 2.5, versus 1.4 / 2.6 / 2.7 for w2v-Conformer XXL, 1.4 / 2.6 / 2.6 for w2v-Conformer XXL+ and 1.5 / 2.7 / 3.1 for wav2vec 2.0 (no HuBERT row in this setting); in the pre-training-only setting with LM, XXL gets 1.5 / 2.7 test-clean / test-other vs HuBERT X-Large 1.8 / 2.9 and w2v-Conformer XXL 1.5 / 3.1, but on dev-other HuBERT X-Large is best at 2.5 (bolded) against 2.6 for XXL and 2.8 for XL. The abstract's "5% to 10% relative WER reduction" over published models such as conformer-based wav2vec 2.0 and HuBERT names no setting, and the table doesn't bear it out on every subset [A1, A2, A3] (p. 1, Sec. 5.1, Table 2, pp. 4-5)
- C4: Contrastive plus MLM beats contrastive alone: w2v-BERT XL (fewer parameters) beats w2v-Conformer XXL in the pre-training-only setting, especially on the other subsets [A1, A2, A3] (Sec. 5.1, Table 2, pp. 4-5)
- C5: Without the contrastive module the MLM module collapses the codebook: Fig. 2 (alpha = 0.5 only) shows masked prediction accuracy reaching 100%, loss near 0 and diversity loss near 1 early in training; the text says all variants tried (alpha 0.1, 0.3, 0.5, 0.7) were untrainable and collapsed, without showing the other curves; the authors conclude the contrastive loss is necessary for MLM to work in this framework [A1, A3] (Sec. 5.2, Fig. 2, p. 5)
- C6: With 24 total conformer layers on 100 h labels (no self-training or LM), contrastive depth n has a sweet spot: test-other 5.1 (C2), 5.1 (C4), 4.7 (C6), 4.6 (C8), 4.8 (C10), 4.6 (C12), 5.0 (C24); all beat wav2vec 2.0 (6.3), and XXL gets 4.3 [A1, A2, A3] (Sec. 5.2, Table 3, pp. 5-6)
- C7: On Google Voice Search (34.3k h pre-training, 1k h fine-tuning) Table 4's "Test (VS)" column gives w2v-BERT XL 6.2, w2v-Conformer-XL-tuned 8.9, w2v-Conformer-XL 10.8 and a 100M conformer without pre-training 10.7; the authors call the gain over the tuned contrastive baseline 30% relative, and say making contrastive learning work on this data needed less aggressive subsampling (stacking 3 frames as targets instead of a 4x convolutional stride) [A1, A2, A3, A4] (Sec. 5.3, Table 4, p. 6)
- C8: On noisy voice-search audio with more silence and 5 s average length, contrastive learning alone tends to cheat on easy negatives, and adding masked prediction mitigates this (authors' interpretation of the Table 4 gap) [A1, A4] (Sec. 5.3, p. 6)

## Implementation notes

- Frame rate: reader's arithmetic from the 4x sequence reduction of Sec. 3.1, p. 2; the paper doesn't state the log-mel hop, so with a conventional 10 ms hop the output is 40 ms per frame, about 2.4 game frames at 60 Hz (source: assumption).
- Table 4's "Test (VS)" metric is read as WER, like every other table; the paper doesn't name it (source: assumption).
- The 3-frame stacking described in Sec. 5.3 is read as the change that makes w2v-Conformer-XL "tuned"; the paper doesn't say so outright (source: assumption).
- XXL uses Adafactor (beta1 0.9, beta2 0.98); XL uses Adam with a transformer LR schedule; fine-tuning batch 256, peak LR 3e-4 (encoder) and 1e-3 (decoder) for XL (source: Sec. 4.2-4.3, pp. 3-4).
- Transcript tokenisation: 1024-token WordPiece from LibriSpeech transcripts (source: Sec. 4.1, p. 3).
- Hyperparameters not given here (self-training, LM fusion, SpecAugment) are those of w2v-Conformer ([21] in the paper, arXiv:2010.10504) (source: Sec. 4.3, pp. 3-4).

## Limitations / open questions

- Authors plan to search for the best training configuration (all hyperparameters were copied unchanged from w2v-Conformer) and to test on Libri-Light 10 min / 1 h / 10 h low-resource splits (Sec. 6, p. 6).
- The contrastive module's quantizer can collapse without the contrastive task; the paper notes codebook collapse as the risk of end-to-end discretisation (Sec. 2, p. 2).
- Reader's note: the paper mentions no code or model release, and reports no latency, streaming use or compute per second of audio.
- Open question: the codebook is learned on speech and sized (1024 entries x 1024) for phone-like units; there is no non-speech evaluation, so it is unknown whether it tokenises game audio meaningfully.
- Open question: the voice-search result shows contrastive learning degrading with silence and short context, a regime closer to game audio with long stretches of music and sparse effects (reader's inference).

## Related

- related [[hsu2021hubert]]: HuBERT is the closest MLM-with-discrete-targets method; w2v-BERT replaces its iterated k-means with a jointly trained quantizer and compares against HuBERT Large / X-Large with LM in Table 2 (see C3).
- related [[chen2023beats]]: BEATs's iterative tokenizer plus masked label prediction for general audio mirrors this idea for non-speech, using distillation rather than a contrastive codebook.
- related [[vaswani2017attention]]: multi-headed self-attention within the conformer blocks.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
