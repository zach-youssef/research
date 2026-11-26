---
id: hsu2021hubert
title: "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units"
authors: [Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed]
year: 2021
venue: IEEE/ACM TASLP 2021 (arXiv v1 read)
url: https://arxiv.org/abs/2106.07447
arxiv: 2106.07447
doi:
pdf_sha256: 3a919c3ede82b15c3c28f648ea8c8ef7417bab533b04f9b2b97a324c9be688d9
tags: [speech, self-supervised, masked-prediction, clustering, waveform, asr, librispeech, pseudo-labels]
status: read
read: 2026-11-26
---

# HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units

## TL;DR

A wav2vec 2.0-style CNN + Transformer on raw waveform predicts offline k-means ids (MFCC, then an
earlier model's layer) of masked 20 ms frames. Read English speech only; ASR WER is near or better
than wav2vec 2.0 from 10 min to 960 h labels, some cells 0.1-0.6 worse; 1B model 2.9 test-other.

## Problem

Speech SSL faces several sound units per utterance, no lexicon of units, and variable-length units with
no segmentation, which breaks instance classification and plain masked prediction (p. 1, Sec. I, p. 2).
wav2vec 2.0 needs a contrastive loss with negative sampling, a diversity loss and Gumbel-softmax
annealing (Sec. III, p. 4).

## Method

- Targets: frame-level hidden units from an offline clustering model h (k-means), z_t in [C] (Sec. II-A, p. 2).
  Iteration 1: k-means with 100 clusters on 39-d MFCC (13 + deltas). Later iterations: k-means with 500
  clusters on an intermediate Transformer layer of the previous model (layer 6 of iteration 1, layer 9 of
  iteration 2), fitted on 10% of the data (Sec. IV-B, p. 4).
- Masked prediction (Sec. II-B, Eq. 1-3, pp. 2-4): spans of l = 10 frames start at p = 8% of steps, as in
  SpanBERT and wav2vec 2.0; loss = alpha L_m + (1 - alpha) L_u; the paper's choice is alpha = 1 (masked only).
  Output logits are cosine similarity between a projected output and a codeword embedding, temperature 0.1.
- Architecture (Sec. II-E, Table I, p. 3): convolutional waveform encoder (seven 512-channel layers, strides
  5,2,2,2,2,2,2, total downsampling 320, so a 20 ms frame rate at 16 kHz), then a Transformer: BASE 12
  layers 768-d (95M), LARGE 24 layers 1024-d (317M), X-LARGE 48 layers 1280-d (964M).
- Cluster ensembles with several codebooks (one projection per clusterer) and product k-means are
  options (Sec. II-C, Eq. 2, p. 3).
- Pre-training: BASE two iterations on LibriSpeech 960 h, 32 GPUs, 250k then 400k steps; LARGE and X-LARGE one
  iteration on Libri-light 60k h with labels from layer 9 of the second BASE iteration (so in effect third
  iteration models), 128 / 256 GPUs, 400k steps (Sec. IV-C, p. 4).
- Fine-tuning for ASR with CTC, convolutional encoder frozen, 26 letters + space + apostrophe + blank
  (Sec. II-E, p. 3, Sec. IV-D, p. 5).

## Assumptions

- A1: Pre-training data is English read speech (LibriSpeech 960 h, Libri-light 60k h of audiobooks)
  (Sec. IV-A, p. 4).
- A2: Evaluation is ASR (WER) after supervised CTC fine-tuning; no frozen-feature, audio-event, music or
  non-speech evaluation (Sec. IV-D, Sec. V, pp. 5-7).
- A3: Cluster targets are consistent across utterances (fixed offline clustering per iteration) even when
  not accurate (p. 1, Sec. I, p. 2).
- A4: Input is a raw 16 kHz waveform processed by a convolutional encoder at a 20 ms frame rate (Sec. II-E, p. 3).

## Claims

- C1: The authors say HuBERT (BASE: two iterations from MFCC k-means; LARGE and X-LARGE: third-iteration labels) matches or beats wav2vec 2.0 across 10 min to 960 h fine-tuning, the only exceptions being 0.1 WER at 100 h (LARGE test-clean 2.1 vs 2.0, BASE test-other 8.1 vs 8.0). Tables II and III show more: BASE 10 min (4-gram) dev-clean 9.1 vs 8.9 and test-clean 9.7 vs 9.1, BASE 1 h clean sets 0.6 worse (5.6 vs 5.0, 6.1 vs 5.5) and dev-other 10.9 vs 10.8, BASE 10 h dev-clean 3.9 vs 3.8, and 960 h LARGE test-clean 1.9 vs 1.8; LARGE and X-LARGE are better or equal on every other-set cell [A1, A2, A3, A4] (Sec. V-A, Table II, Table III, pp. 5-7)
- C2: With 10 min of labels, HuBERT LARGE (LL-60k, Transformer LM) gets 4.7 / 7.6 WER on test-clean / test-other vs 4.8 / 8.2 for wav2vec 2.0 LARGE; X-LARGE gets 4.6 / 6.8 [A1, A2, A3, A4] (Sec. V-A, Table II, pp. 5-6)
- C3: With all 960 h labels, HuBERT LARGE gets 1.9 / 3.3 and X-LARGE 1.8 / 2.9 test-clean / test-other (Transformer LM), on par with the best pre-training results and below pre-training plus self-training (e.g. 1.5 / 3.1 for wav2vec 2.0 + self-training) [A1, A2, A3, A4] (Sec. V-A, Table III, pp. 6-7)
- C4: The 1B X-LARGE model gives up to 19% and 13% relative WER reduction over LARGE on dev-other and test-other (Libri-light 60k) [A1, A2, A3, A4] (p. 1, Sec. I, p. 2)
- C5: Predicting masked frames only (alpha = 1) is best when targets are poor (MFCC k-means C = 100: dev-other WER 17.86 vs 29.57 at alpha 0.5 and 96.37 at alpha 0 for BASE, 100k steps, 10 h labels); with better targets the gap closes, and for supervised-alignment (chenone) targets alpha = 0.5 is best (9.16) and alpha = 0 (9.79) also beats alpha = 1 (10.38) [A1, A2, A3] (Sec. V-D, Table V, p. 8)
- C6: Iterative refinement helps: clustering the iteration-1 layer-6 features gives PNMI 0.637 (C = 500) vs 0.276 for MFCC and dev-other WER 11.91 vs 18.40 (alpha = 1); iteration-2 layer 9 gives 10.75 [A1, A2, A3] (Sec. V-D, Table V, p. 8)
- C7: K-means unit discovery is stable: PNMI std at most about 0.012 across 10 trials, and more clustering data improves PNMI by at most 0.012; clustering HuBERT features gives PNMI 0.56-0.69 vs 0.25-0.29 for MFCC [A1, A3] (Sec. V-B, Table IV, p. 7)
- C8: Best layer for clustering is the middle (about layer 6-7) for iteration 1 and improves toward later layers for iteration 2 (Fig. 2); MFCC baselines are (cluster purity, phone purity, PNMI) = (0.099, 0.335, 0.255) for K = 100 [A1, A3] (Sec. V-C, Fig. 2, pp. 7-8)
- C9: Cluster ensembles improve over a single C = 100 k-means (dev-other WER 17.86): K-means {50,100} 17.81 (only 0.05 better), {50,100,500} 17.56, product k-means-{0,1,2}-100 16.73; single product codebooks are mixed (k-means-1 17.64 better, k-means-0 19.26 and k-means-2 18.46 worse) [A1, A2, A3] (Sec. V-E, Table V, Table VI, p. 8)
- C10: Mask start probability p = 8% is near-optimal, larger batch (more GPUs) helps a lot, and longer training helps (K-means C = 100: dev-other WER 17.86, 12.97, 12.32, 11.68 at 100k, 250k, 400k, 800k steps; C = 50: 18.68 to 11.82). Table VII's caption sets p = 6.5% for these runs, though its 100k values equal Table V's, which use the fixed ablation setup with p = 8% (Sec. IV-C) [A1, A2, A3] (Sec. V-F, Fig. 3, Table VII, pp. 8-9)
- C11: HuBERT beats DiscreteBERT (masked prediction of vq-wav2vec units with quantised inputs) by a large margin in every setup, e.g. 10 min: 9.7 / 15.3 (BASE) vs 16.3 / 25.2 test WER; the authors credit continuous waveform input and iterative refinement [A1, A2, A3, A4] (Sec. V-A, Table II, pp. 5-6)

## Implementation notes

- Frame rate 20 ms (CNN downsampling 320 at 16 kHz): 50 targets per second, about 1.2 per game frame at 60 Hz (source: Sec. II-E, p. 3).
- Batch per GPU: BASE up to 87.5 s of audio, LARGE 56.25 s, X-LARGE 22.5 s; Adam (0.9, 0.98), peak LR 5e-4 / 1.5e-3 / 3e-3, linear warm-up over 8% steps (source: Sec. IV-C, p. 4).
- k-means via scikit-learn MiniBatchKMeans, batch 10,000 frames, k-means++ with 20 starts (source: Sec. IV-B, p. 4).
- Fine-tuning hyperparameters swept per size/split on dev-other WER; decoding with wav2letter++ beam search with n-gram or Transformer LM (source: Sec. IV-D, p. 5).
- Code and models in fairseq at https://github.com/pytorch/fairseq/tree/master/examples/hubert, not inspected (source: p. 1).

## Limitations / open questions

- The authors plan a single-phase training procedure instead of iterations, and downstream tasks beyond ASR (generation, recognition) (Sec. VI, pp. 8-9).
- HuBERT lags pre-training plus self-training combinations on the 960 h setup (Sec. V-A, p. 6).
- Open question: all evidence is English read speech with WER; the 100-500 cluster targets are tied to phonetic structure (the paper measures purity against phones). How well the same recipe clusters non-speech audio is not shown here; BEATs applies the idea to general audio ([[chen2023beats]]).
- Reader's note: latency and streaming behaviour are not reported; the Transformer attends over the whole utterance (reader's reading of the BERT-style masked prediction, Sec. II-B, p. 2).
- Open question: a 20 ms frame rate and a convolutional waveform front-end give frame-aligned features, which fits a frame-synced consumer better than clip-level pooled encoders, but the encoder is bidirectional and was run on whole utterances.

## Related

- related [[chen2023beats]]: BEATs adapts HuBERT-style iterative masked prediction of discrete labels to general audio, with a learned tokenizer in place of k-means.
- related [[huang2022masked]]: contrasts reconstruction-based audio SSL with this discrete-target approach; Audio-MAE lists HuBERT in its speech comparison.
- related [[vaswani2017attention]]: the Transformer encoder is the BERT-style stack over CNN features.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
