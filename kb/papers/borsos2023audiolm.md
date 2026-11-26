---
id: borsos2023audiolm
title: "AudioLM: a Language Modeling Approach to Audio Generation"
authors: [Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, Neil Zeghidour]
year: 2023
venue: IEEE/ACM TASLP 2023 (arXiv v2 read; venue taken from other papers' citations, not printed in the PDF)
url: https://arxiv.org/abs/2209.03143
arxiv: 2209.03143
doi:
pdf_sha256: 16c8389ed725758ec76091137e98acdcab479ce84a40a11db5a82ccb65b054a2
tags: [audio, language-model, semantic-tokens, acoustic-tokens, speech, piano, generation, w2v-bert, soundstream]
status: read
read: 2026-11-26
---

# AudioLM: a Language Modeling Approach to Audio Generation

## TL;DR

Three decoder-only Transformers model semantic tokens (w2v-BERT, 25 Hz), then coarse and fine
acoustic tokens (SoundStream, 50 Hz x 12 codebooks). Trained on 60k h of speech (and 40k h of
piano), it continues 3 s prompts; raters told 7 s continuations from originals at chance (51.2%).

## Problem

Acoustic codec tokens give good reconstruction but poor long-term structure, while semantic tokens
from self-supervised models give structure but poor reconstruction (Sec. I, p. 1; Table I, p. 4). The
paper wants audio generation with both long-term consistency and high quality, without text or
labels (Abstract, p. 1).

## Method

- Tokenizer, language model, detokenizer; tokenizers frozen, token sequence 2-3 orders of magnitude
  shorter than the waveform (Sec. III-A, p. 3).
- Acoustic tokens: SoundStream at 16 kHz, 320x stride, 50 Hz; 12 RVQ layers with 1024 entries (6 kbps,
  600 tokens/s) (Sec. III-A, p. 3; Sec. IV-B, p. 6).
- Semantic tokens: normalised activations of the 7th layer of a 0.6B w2v-BERT XL (MLM module),
  k-means with K = 1024, 25 Hz (250 bps) (Sec. III-A, p. 3; Sec. IV-B, p. 6).
- Three stages with separate decoder-only Transformers (12 layers, 16 heads, dim 1024, 0.3B
  parameters each): semantic modeling p(z_t | z_<t); coarse acoustic modeling of the first Q' = 4
  quantizers conditioned on all semantic tokens (flattened row-major); fine acoustic modeling of
  the remaining 8 quantizers on 3 s chunks, conditioned on the coarse tokens only (Sec. III-C, pp. 4-5; Sec. IV-B, p. 6).
- Training: 16 TPUv4s, batch 256, 1M steps per stage; random crops of 30, 10 and 3 s; temperatures
  0.6, 0.8, 0.6; consecutive semantic repeats removed (Sec. IV-B, p. 6).
- Data: Libri-Light unlab-60k (60k h English speech) for every component; for piano an internal 40k h
  dataset, with a SoundStream of 3 layers and 2^14 codebooks and no fine stage (Sec. IV-A, pp. 5-6; Sec. IV-I, p. 9).

## Assumptions

- A1: Speech experiments use 60k h of English audiobook speech (Libri-Light) with 16 kHz audio; the piano experiment uses a separate internal 40k h dataset and separately trained components (Sec. IV-A, Sec. IV-I, pp. 5-9).
- A2: Prompts for continuation are 3 s from unseen speakers (LibriSpeech test-clean), and piano prompts are 4 s from the Maestro dataset (Sec. IV-B, p. 6, Sec. IV-I, p. 9).
- A3: In the speech experiments, the semantic tokenizer is a w2v-BERT trained on speech (Libri-Light) (Sec. III-A, p. 3, Sec. IV-A, p. 5).
- A4: The SoundStream codec is frozen and trained before the language models (Sec. III-A, p. 3).
- A5: The piano model retrains "all components of AudioLM" on 40k h of piano (the reader takes this to include w2v-BERT and its k-means), uses a SoundStream with 3 quantizers of 2^14 entries, and has two stages: semantic, then all 3 acoustic levels predicted in the second stage, with no fine stage (Sec. IV-I, p. 9).

## Claims

- C1: Acoustic tokens (SoundStream) reconstruct well (ViSQOL 3.3 at 2 kbps, 3.9 at 6 kbps) but have poor phonetic discriminability (ABX within/across 22.4/28.7 and 17.8/26.6), while w2v-BERT semantic tokens discriminate far better (6.7/7.6 at 250 bps, 5.6/6.2 at 6 kbps) but reconstruct poorly (ViSQOL 1.1-1.4) [A1, A3] (Sec. III-B, Table I, pp. 3-4).
- C2: A single-stage language model on acoustic tokens alone, prompted with 4 s, keeps speaker and recording conditions but produces inconsistent, babbling content [A1] (Sec. III-B, p. 4).
- C3: Token budget: 25 semantic tokens/s plus 50 x 4 = 200 coarse and 50 x 8 = 400 fine acoustic tokens/s per second of audio; the third stage has 2(Q - Q') tokens per semantic token [A1] (Fig. 2, p. 5, Sec. IV-B, p. 6).
- C4: The semantic tokens carry the linguistic content: ASR on acoustic generation from ground-truth semantic tokens gives WER 6.0 and CER 3.4 (original 2.5 / 0.8; SoundStream reconstruction 2.6 / 0.9; GSLM unit-to-speech 6.6 / 2.9) [A1, A3] (Sec. IV-C, Table II, p. 6).
- C5: Semantic tokens carry little speaker information: a speaker classifier gets 3.2% (chance 0.3%) on acoustic generation from fixed semantic tokens, vs 100% on SoundStream reconstruction; continuation with acoustic prompt tokens gets 92.6% [A1] (Sec. IV-D, Sec. IV-F, Table III, pp. 6-8).
- C6: On zero-resource linguistic probes AudioLM scores sWUGGY 71.5 (in-vocab 83.7) and sBLIMP 64.7 (with length-normalised log-likelihood; 67.5 without it, footnote 3), the best among systems without text supervision, ahead of GSLM (68.7 in-vocab, 57.1) and the best non-causal HuBERT-only/CPC-BERT models (sBLIMP about 59.9) [A1, A3] (Sec. IV-E, Table IV, p. 8).
- C7: In a 10 s original-vs-continuation task (100 LibriSpeech test-clean samples; half are SoundStream-compressed originals, half 3 s real prompts plus 7 s continuations; raters told that the first 3 s are real and to judge what follows), 10 raters labelled correctly 51.2% of 1000 ratings (binomial p = 0.23, not significantly different from chance) [A1, A2] (Sec. IV-G, p. 8).
- C8: A convolutional classifier detects AudioLM continuations against SoundStream-compressed originals with 98.6% accuracy [A1] (Sec. IV-H, pp. 8-9).
- C9: Piano continuation, with the two-stage piano model (semantic tokens, then 3 acoustic levels): raters preferred AudioLM over an acoustic-tokens-only model in 83.3% of 15 pairs of 20 s continuations (10 raters) [A2, A5] (Sec. IV-I, p. 9).
- C10: The authors argue the hierarchy rests on semantic tokens being nearly conditionally independent of past acoustic tokens given past semantic tokens, and that staging shortens each stage's sequence compared with interleaving the token types; the paper gives no direct measurement of the independence [A1, A3] (Sec. III-C, p. 4).

## Implementation notes

- Token counts per second of 16 kHz audio: 25 semantic; 200 coarse; 400 fine (and 25 + 200 = 225 tokens/s of context for the second stage) (derived from the stated rates) (source: assumption, reader's arithmetic from the rates in Sec. IV-B, p. 6).
- Consecutive repeated semantic tokens are removed in stages 1-2, so 25 semantic tokens/s is an upper bound on what the language models see (source: Sec. IV-B, p. 6).
- For a 60 fps game: 25 semantic tokens/s is 0.42 per video frame, coarse acoustic 3.3, fine acoustic 6.7; a frame-synced agent wanting only the semantic stream would add under 1 token per frame, but in the speech model those are speech-trained w2v-BERT tokens, and the piano model's are retrained on piano (A5) (reader's arithmetic) (source: assumption).
- Semantic layer choice: 7th layer of w2v-BERT XL, K = 1024; chosen by ABX/sWUGGY/sBLIMP across layers 6-9 and K = 256-2048 plus a small listening test (source: Sec. IV-B, Fig. 3, p. 6).
- Embeddings are normalised to zero mean and unit variance per dimension before k-means (source: Sec. III-A, p. 3).
- Speaker classifier: six conv blocks on 64-bin log-mel (25 ms window, 10 ms hop), 1 s crops (source: Sec. IV-D, p. 7).
- Code and weights: the paper links only audio samples (source: footnote 1, p. 2); no release is stated.

## Limitations / open questions

- Speech and piano only; the authors name multilingual speech, polyphonic music and audio events as future extensions (Sec. V, p. 9).
- Misuse risks (spoofing, impersonation) are acknowledged and a detection classifier is provided (Sec. VI, p. 9).
- Error sources in acoustic generation include proper nouns and misplaced end-of-sentence tokens (Sec. IV-C, p. 7).
- The 98.6% detector score uses SoundStream-compressed originals as negatives to avoid codec cues (Sec. IV-H, p. 8).
- The paper doesn't discuss the causality of the w2v-BERT features or any streaming latency (reader's observation).
- Open question: what a semantic tokenizer trained on game audio would give; the paper's only non-speech case is the piano model retrained on 40k h of piano (A5, C9), evaluated by a 15-pair preference test.
- Open question: whether a 3-stage generator is needed for policy inputs; only the tokenizers matter for a perception-only use.

## Related

- uses [[zeghidour2022soundstream]]: acoustic tokens come from SoundStream (12 layers, 50 Hz, 16 kHz).
- related [[defossez2023encodec]]: open-weights RVQ codec of the same family as the SoundStream used here.
- related [[kumar2023rvqgan]]: DAC's paper lists AudioLM as a downstream codec user.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
