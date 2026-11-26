---
id: defossez2024moshi
title: "Moshi: a speech-text foundation model for real-time dialogue"
authors: [Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, Neil Zeghidour]
year: 2024
venue: arXiv preprint
url: https://arxiv.org/abs/2410.00037
arxiv: 2410.00037
doi:
pdf_sha256: ea702589e103f8d02b4fbec733c556b08d136c2ccaa013eb48a91a7366c7e390
tags: [audio, neural-codec, streaming, speech, full-duplex, tokenizer, semantic-distillation, mimi, rq-transformer, open-weights, low-token-rate]
status: read
read: 2026-11-26
---

# Moshi: a speech-text foundation model for real-time dialogue

## TL;DR

Moshi: a 7B text LLM (Helium) plus a Depth Transformer predicting, per 80 ms step, text, semantic
and acoustic tokens of Mimi, a causal 24 kHz codec at 12.5 Hz x 8 codebooks (1.1 kbps), first one
WavLM-distilled. Latency 160 ms theoretical (abstract: "200ms in practice"); open weights.

## Problem

Spoken dialogue pipelines (VAD, ASR, text LLM, TTS) add seconds of latency, lose non-text audio
information and assume speaker turns (Sec. 1, p. 2). Audio language models also need several
tokens per timestep, and semantic tokens from self-supervised models are not causal, so they
can't be used for streaming (Sec. 3.3, p. 9). The paper wants one real-time, full-duplex speech-to-speech model.

## Method

Mimi codec (covered in depth; Sec. 3.3):
- SEANet-style causal conv encoder/decoder, strides (4, 5, 6, 8) plus a final stride-2 conv: 24 kHz
  audio becomes 12.5 latent frames/s of dimension 512; frame size and overall stride are 80 ms;
  all convolutions causal, so a first 80 ms input frame yields the first latent step (Sec. 3.3.1, pp. 10-11).
- Transformer bottleneck: 8-layer, 8-head causal Transformers (dim 512, RoPE, a 250-frame context,
  LayerScale) before and after quantization (Sec. 3.3.1, pp. 10-11). The paper gives the context as "20 seconds" on p. 10
  but as "10s (250 frames before the last downsampling layer)" on p. 11; 250 frames at 25 Hz is 10 s.
- Quantization: Q = 8 RVQ levels of 2048 entries (11 bits) at 12.5 Hz = 1.1 kbps; latent projected to
  256 d before RVQ; quantizer dropout, and quantization only applied 50% of the time per sequence
  (the decoder then gets unquantized embeddings) (Sec. 3.3.1, p. 11).
- Adversarial-only training: reconstruction losses are dropped, keeping feature-matching and discriminator
  losses of a multi-scale STFT discriminator (Sec. 3.3.1, p. 11).
- Semantic distillation (Sec. 3.3.2, p. 12): cosine distance between the output of the first quantizer (linear
  projection to 1024 d) and WavLM embeddings (16 kHz input, 50 Hz) average-pooled (stride 4, kernel 8, non-causal,
  used only in training) to 12.5 Hz. Split RVQ: the distilled level is a plain VQ and 7-level RVQ runs in
  parallel; both outputs are summed for reconstruction (Fig. 2, p. 10).
- Training: AdamW with weight decay only on Transformers, lr 8e-4, batch 128 of 12 s windows, 4M
  steps (Sec. 3.3.1, p. 11); Mimi's training data is not separately stated.

Moshi (briefly):
- Helium: 7B text LLM (32 layers, dim 4096, context 4,096, 32k SentencePiece vocabulary), pretrained on 2.1T
  tokens of filtered public English text (Sec. 3.2, pp. 7-8, Sec. 4.1, p. 17; 2.1T on p. 3 and Sec. 5.1, p. 22).
- RQ-Transformer: a Temporal Transformer (Helium weights) takes one step per 80 ms, and a 6-layer Depth
  Transformer (dim 1024) predicts the K sub-streams in order, with per-codebook weights (Sec. 3.4.1, pp. 12-14, Table 1, p. 8).
- Acoustic delay tau: acoustic tokens are delayed 1-2 steps relative to semantic ones (Eq. 4, p. 14).
- Multi-stream: user and Moshi each give 1 semantic + 7 acoustic tokens, plus a time-aligned text stream
  (Inner Monologue), K = 2Q + 1 = 17 streams (Eq. 6, p. 17, Fig. 4).
- Training: pretraining on 7M h of audio (transcribed with Whisper), diarization-based post-training,
  Fisher (2000 h) finetuning, instruction tuning on 20k h of synthetic dialogue (Sec. 4.2-4.4, pp. 18-21).

## Assumptions

- A1: Mimi and Moshi are trained and evaluated on speech (7M h mostly English speech); no non-speech or game audio is evaluated (Sec. 4.2, p. 18, Sec. 5.2, p. 23).
- A2: The semantic distillation target (WavLM) is a speech model, so the first Mimi codebook encodes speech phonetic content (Sec. 3.3.2, p. 12).
- A3: Latency figures are theoretical frame-size arithmetic (80 ms frame plus acoustic delay); "200ms in practice" is stated in the abstract only, with no measurement or setup described (p. 1, Sec. 5.3, p. 27).
- A4: Audio quality is judged by MUSHRA with 20 listeners (30 samples each), because VisQOL and MOSNet disagree with listeners when the training objective changes (Sec. 5.2, p. 23, p. 25).

## Claims

- C1: Mimi is causal at both ends: with 24 kHz input and 80 ms stride it outputs 12.5 latent steps/s of dimension 512, and the first 80 ms frame yields the first decodable latent [A1] (Sec. 3.3.1, pp. 10-11).
- C2: Mimi uses 8 codebooks of 2048 entries at 12.5 Hz, i.e. 100 audio tokens per second per stream and 1.1 kbps [A1] (Sec. 3.3.1, p. 11, Sec. 3.4.1, p. 13).
- C3: In Mimi, adding WavLM distillation to a single RVQ cuts ABX from 23.3% to 6.5% but lowers MUSHRA from 65.9 to 57.8; the split quantizer gives ABX 8.1% and MUSHRA 64.0, a better trade-off [A1, A2, A4] (Sec. 5.2, Table 3, pp. 23-24).
- C4: In Table 3, the Transformer in the encoder helps distillation (ABX 10.8% without it vs 8.1% with it, split quantizer), the decoder Transformer raises MUSHRA (48.4 without vs 64.0 with), and applying quantization only 50% of the time raises VisQOL (2.45 vs 2.82) but not MUSHRA (68.3 without vs 64.0 with) [A1, A4] (Sec. 5.2, Table 3, pp. 23-24).
- C5: Training Mimi with adversarial losses only gives MUSHRA 81.0 vs 58.8 with the EnCodec-style mix, although VisQOL falls to 1.84; this Mimi beats RVQGAN truncated to 2 levels (31.3, 1.5 kbps, 75 Hz), SemantiCodec (64.8, 1.3 kbps, 50 Hz) and SpeechTokenizer at 1.5 kbps (45.1) and 4 kbps (74.3); ground truth is 90.6 [A1, A4] (Sec. 5.2, Table 4, p. 24).
- C6: In that comparison Mimi is the only causal codec, and its 12.5 Hz frame rate is 4x lower than SemantiCodec's 50 Hz; a downsampled 16 kHz Mimi scores 77.7 [A1, A4] (Sec. 5.2, Table 4, p. 24).
- C7: Mimi's semantic-token ABX is 8.1-8.7% (SpeechTokenizer 3.3%, SemantiCodec 42.2%); the paper notes that VisQOL and MOSNet correlate poorly with MUSHRA when the loss mix changes [A1, A2, A4] (Sec. 5.2, Table 4, pp. 24-25).
- C8: Moshi's theoretical latency is 160 ms (one 80 ms frame plus a one-step acoustic delay), against a 230 ms average in natural conversation; the abstract adds "200ms in practice" without describing a measurement [A3] (p. 1, Sec. 1, p. 2, Sec. 5.3, p. 27).
- C9: With delay [0, 1, ..., 7] (640 ms) the RQ-Transformer helps perplexity only slightly (40.3 vs 42.2), but with delay [0, 2, 2, ..., 2] (240 ms) it cuts perplexity from 135.4 to 36.8 [A1] (Sec. 5.3, Table 5, pp. 25-26).
- C10: Delay [0, 0, ..., 0] (80 ms) gives transcript NLL 4.36 vs 4.12 and 4.09 for delays 1 and 2; semantic-token loss weight 100, depthwise parametrization and Inner Monologue then bring NLL to 3.75, 3.65 and 2.77, and transcript length from 486-602 characters (first five rows) to 1,920 [A1] (Sec. 5.3, Table 6, pp. 25-26).
- C11: Single-stream, from-scratch Moshi (audio only) reaches sWUGGY 74.8, sBLIMP 59.9, sTopic-StoryCloze 80.9, above GSLM and AudioLM except sBLIMP (AudioLM 64.7); after multi-stream instruction tuning sWUGGY falls to 63.0 [A1] (Sec. 5.4, Table 7, pp. 27-28).
- C12: Spoken QA (0-shot): Moshi gets 26.6 (Web Questions), 62.3 (LLaMA-Q) and 22.8 (audio Trivia QA); without Inner Monologue 9.2, 21.0, 7.3; Helium on text gets 32.3, 75.0, 56.4 [A1] (Sec. 5.5, Table 8, pp. 29-30).
- C13: In generated Fisher-like dialogues (1000 samples) Moshi's conditional perplexity is 41.9 at temperature 0.8, 56.7 at 0.9 and 79.3 at 1.0 (its ground truth 59.6), and overlap is 1.2 s at 0.8 and 4.1 s at 1.0 (ground truth 3.3 s); the best non-cascaded baseline (Nguyen et al., 50 samples, temperature 1.0) has 195.9 [A1] (Sec. 5.6, Table 9, pp. 30-31).
- C14: With a 2 s delay between text and audio streams the same model runs as streaming TTS (WER 4.7% LibriSpeech test-clean, VALL-E 5.9%) or ASR (5.7%, streaming FastConformer 3.6%) [A1] (Sec. 5.7, pp. 31-32).
- C15: Weights quantized to 4 bits (block 32) keep Helium within 2 MMLU points (52.97 vs 54.3) but Moshi loses 5-10 points; output audio quality by MOSNet is largely unchanged at 4 bits, and 2-bit weights give "significantly degraded audio quality"; the artifact rate depends on block size: W2A8 has 83.1% gibberish audio and 5.9% artifact-free segments at block 256, but 12.7% gibberish (40.9% noisy) and 45.4% artifact-free at block 32 (unquantized: 95.8% artifact-free) [A1] (Sec. 5.8, Table 10, Table 11, Fig. 5, Table 12, pp. 33-35).
- C16: Safety: Moshi's ALERT score is 83.05, mid-table among compared text models; in 100 h of generated dialogue the speaker model matches Moshi's own voice over the user's 98.4-99.3% of segments; the signal-based AudioSeal watermark is still detected after pink noise (0.71 at 10 s) but not after Mimi re-encoding (0.08 for marked audio vs 0.9999 before re-encoding), and the paper reports its token-level watermark as a negative result [A1] (Sec. 6, Table 14, p. 38, Table 15, p. 39, pp. 40-41).
- C17: In the Appendix A ablation, Mimi variants without WavLM distillation have ABX 23.3-31.4% whatever the quantization rate and Transformers, and every variant with distillation has 6.5-10.8%; the full model (split quantizer) gets ABX 8.1%, VisQOL 2.72, MOSNet 2.89 [A1, A2] (App. A, Table 17, p. 54).
- C18: Helium, trained on 2.1T tokens, is on par with or better than 7B models trained on under 2.5T tokens on most text benchmarks (MMLU 54.3 vs Llama 2 45.3, OLMo 52.0), and competitive with Mistral and Gemma on some (ARC, Open-Book QA, Natural Questions) [] (Sec. 5.1, Table 2, p. 22).
- C19: The authors state that Moshi is "the first real-time full-duplex spoken large language model" [A3] (p. 1).

## Implementation notes

- Tokens per second for a frame-synced agent: Mimi gives 12.5 steps/s with 8 codebooks, 100 tokens/s; the semantic codebook alone is 12.5 tokens/s. At 60 game frames/s that is 1.67 tokens per game frame for all 8 codebooks, 0.21 for the first, or one Mimi step per 4.8 game frames; the rate doesn't divide 60 (reader's arithmetic) (source: assumption, reader's arithmetic from Sec. 3.3.1, p. 11).
- Latency to count: Mimi alone adds one 80 ms frame (about 5 game frames at 60 fps) before the first latent is available (source: assumption, reader's arithmetic from Sec. 3.3.1, p. 11).
- Mimi training details beyond Sec. 3.3: loss weights and discriminator parameters are in the Audiocraft repository config (source: footnote 7, p. 11, facebookresearch/audiocraft compression/default.yaml).
- Weights and code: "We release both Mimi and Moshi" and github.com/kyutai-labs/moshi (source: p. 1, Sec. 7, pp. 41-42); licence not stated in the paper.
- Mimi's training data: taken to be the Sec. 4.2 speech audio, since Sec. 3.3 and Appendix A don't list it (source: assumption).
- Table 1: Temporal Transformer dim 4096, 32 layers; Depth Transformer dim 1024, 6 layers, 16 heads; audio cardinality 2048; acoustic delay 2 (pre-training) and 1 (fine-tuning) (source: Table 1, p. 8).

## Limitations / open questions

- Moshi's spoken QA is below its text-only Helium; the authors link this to training on audio and fine-tuning on oral-style conversations (Sec. 5.5, p. 30).
- Objective audio metrics correlate poorly with human ratings when losses change (Sec. 5.2, p. 25).
- Token-level watermarking failed because tokens are not stable under re-encoding (Sec. 6.4, pp. 40-41).
- Parts I did not read in full: Appendix B (audio matching and deduplication, pp. 54-56, but Appendix A's Table 17 on p. 54 was read), Appendix C (zero-shot ASR/TTS delay procedure, pp. 56-58, partly), Appendix D entropy-spectrum figures (pp. 59-62), Appendix E-F prompt and example tables (pp. 63-67), the safety tables beyond Table 12 and Table 14 cited in C15 and C16 (pp. 35-38) and the references (pp. 42-53); they don't bear on the codec.
- Open question: Mimi's first codebook is distilled from WavLM (speech), so its content for game sound effects and chip music is unknown; the acoustic codebooks may still carry it.
- Open question: the 1.1 kbps Mimi budget was tuned for speech with adversarial-only training; whether it reconstructs short synthesised events (square-wave beeps, noise bursts) at 80 ms resolution is not tested.

## Related

- extends [[zeghidour2022soundstream]]: Mimi is a SoundStream/EnCodec-type RVQ codec made causal at 12.5 Hz with Transformers in the bottleneck.
- extends [[defossez2023encodec]]: Mimi starts from EnCodec's setup and changes the losses and quantization.
- extends [[borsos2023audiolm]]: keeps the semantic-then-acoustic hierarchy, generated jointly with an RQ-Transformer and acoustic delay instead of three stages.
- related [[kumar2023rvqgan]]: RVQGAN (DAC) is a baseline truncated to two RVQ levels; Mimi also adopts its non-always-on quantization idea.
- related [[ji2025wavtokenizer]]: both lower the token rate, WavTokenizer to one codebook at 40-75 Hz, Mimi to 12.5 Hz with eight.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
