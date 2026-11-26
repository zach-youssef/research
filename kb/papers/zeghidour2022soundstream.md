---
id: zeghidour2022soundstream
title: "SoundStream: An End-to-End Neural Audio Codec"
authors: [Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, Marco Tagliasacchi]
year: 2022
venue: IEEE/ACM TASLP 2022 journal (vol. 30; arXiv v1 read, venue not printed in the PDF)
url: https://arxiv.org/abs/2107.03312
arxiv: 2107.03312
doi:
pdf_sha256: 060ce231a5762e1685bdbbf4f868f7db0ff9fa18618417183c59723b508d2db8
tags: [audio, neural-codec, rvq, streaming, speech, music, tokenizer, gan]
status: read
read: 2026-11-26
---

# SoundStream: An End-to-End Neural Audio Codec

## TL;DR

A fully causal convolutional encoder/decoder with a residual vector quantizer (RVQ), trained
end to end with adversarial and reconstruction losses on speech and music at 24 kHz. One model
spans 3-18 kbps via quantizer dropout; at 75 frames/s (13 ms) it runs in real time on a phone CPU.

## Problem

Waveform codecs (Opus, EVS) are general but produce artifacts at low bitrate; parametric codecs
assume speech (Sec. I, p. 1). Earlier neural codecs were speech-only, low bitrate, or needed one
model per bitrate (Sec. II, p. 3). SoundStream wants one learned codec for speech, music and general
audio at 3-18 kbps, usable for streaming (Abstract, p. 1).

## Method

- Encoder: SEANet-style causal 1D convolutions (no skip connections), ELU, no normalisation;
  B_enc = 4 blocks with strides (2, 4, 5, 8) give one D-dim embedding per M = 320 samples, so
  75 embeddings/s at 24 kHz. The decoder mirrors it with transposed convolutions. All
  convolutions are causal, also in training (Sec. III-A, Sec. III-B, Fig. 3, pp. 3-4).
- RVQ: N_q cascaded vector quantizers each with N codewords (default N = 1024, 10 bits). The
  bits per frame are N_q log2 N; at 6 kbps and 75 frames/s that is 80 bits, so N_q = 8
  (Sec. III-C, Alg. 1, p. 4). Codebooks use EMA updates, k-means init from the first batch,
  and replacement of unused codewords (pp. 4-5).
- Quantizer dropout: during training n_q is drawn uniformly from 1..N_q and only the first n_q
  quantizers are used, so one model serves every bitrate; at inference n_q selects the bitrate
  (Sec. III-C, p. 5).
- Discriminators: a multi-resolution waveform discriminator (3 scales) and an STFT discriminator
  (W = 1024, H = 256); hinge adversarial loss, a feature-matching loss from discriminator
  layers and a multi-scale mel reconstruction loss, weights 1, 100, 1 (Sec. III-D, Sec. III-E, Eq. 1-6,
  pp. 5-6).
- Optional FiLM conditioning (denoise on/off) for joint compression and denoising (Sec. III-F, p. 6).
- Data: 24 kHz clean speech (LibriTTS), noisy speech (LibriTTS mixed with Freesound noise), music
  (MagnaTagATune); eval on 200 clips of 2-4 s from four sets (clean, noisy, music, real-world
  reverberant speech). Metrics: MUSHRA-style crowd tests, and ViSQOL for ablations (Sec. IV, pp. 6-7).

## Assumptions

- A1: The training and test audio are speech and music (clean, noisy, reverberant speech; MagnaTagATune music) at 24 kHz; no game or sound-effect audio is evaluated (Sec. IV-A, p. 6).
- A2: Quality is measured by listening tests and ViSQOL (perceptual similarity to a reference), not by how useful the tokens are to a downstream model (Sec. IV-B, p. 7).
- A3: Bitrate is constant: every frame gets the same number of bits (Sec. V-B, p. 7).
- A4: Real-time figures are for a single CPU thread of a Pixel 4 phone (Sec. V-D, p. 9).

## Claims

- C1: At 3 kbps SoundStream beats Opus at 6 kbps and EVS at 5.9 kbps in the MUSHRA-style test; EVS needs at least 9.6 kbps and Opus at least 12 kbps to match it (3.2-4x more bits). It also beats Lyra at 3 kbps [A1, A2] (Sec. V-A, Fig. 5a, pp. 7-8).
- C2: At medium bitrates (6 kbps) EVS and Opus need 2.2-2.6x more bits to match SoundStream, and at high bitrates (12 kbps) 1.3-1.6x more [A1, A2] (Sec. V-A, Fig. 5, pp. 7-8).
- C3: At 3 kbps quality is consistent between clean and noisy speech, and music is coded better than Opus at 12 kbps and EVS at 5.9 kbps; music is the harder content type in ViSQOL [A1, A2] (Sec. V-A, Sec. V-B, Fig. 6, Fig. 7b, pp. 7-9).
- C4: ViSQOL stays above 3.7 at 3 kbps and rises to about 4.3 at 18 kbps for the whole test set; entropy coding the codes could save an estimated 7-20% [A1, A3] (Sec. V-B, Fig. 7a, p. 7, p. 9).
- C5: A bitrate-scalable model (quantizer dropout, trained at 18 kbps) is only slightly worse than a bitrate-specific one at 3 kbps and matches it at 6 and 12 kbps; a model trained at 18 kbps without dropout degrades increasingly at low rates [A1] (Sec. V-C, Fig. 7c, pp. 8-9).
- C6: Replacing the learned encoder with a fixed mel-filterbank drops ViSQOL at 6 kbps from 3.96 to 3.33, worse than the learned encoder at 3 kbps (3.76) [A1] (Sec. V-D, p. 8).
- C7: Default model (C_enc = C_dec = 32) runs encode at RTF 2.4x and decode 2.3x on one Pixel 4 CPU thread; C = 16 raises RTF above 7x with ViSQOL 3.98 vs 4.01 [A4] (Sec. V-D, Table I, p. 9).
- C8: Quantizer depth/codebook size trade-off at 6 kbps: N_q = 8 with N = 1024 gives ViSQOL 4.01, N_q = 16 with N = 32 gives 3.98, N_q = 80 with N = 2 gives 3.92 [A1] (Table II, p. 9).
- C9: Architectural latency equals the product of the strides M; strides (1,4,5,8), (2,4,5,8), (4,4,5,8) give 7.5, 13 and 26 ms and, with the per-frame bit budget kept at 6 kbps by using 4, 8 and 16 quantizers of fixed size, equal ViSQOL (4.01); the real-time factor improves as latency grows (encode RTF 1.6x, 2.4x, 4.1x) [A1, A4] (Sec. V-D, Table III, pp. 9-10).
- C10: One model can denoise and compress with no extra latency; denoising before quantization (encoder side) is cheaper in bits than at the decoder, and the joint model is close to cascading SEANet with SoundStream at half the compute [A1] (Sec. V-E, Sec. V-F, Fig. 8, Table IV, p. 10).

## Implementation notes

- Tokens per second: 75 frames/s times n_q codes; n_q = 8 at 6 kbps is 600 tokens/s, n_q = 4 at 3 kbps is 300 (derived from Sec. III-C and Table II; n_q for 3 kbps and the maximum of 24 at 18 kbps follow from 10 bits per code, not stated) (source: assumption).
- For a game agent at 60 game frames/s, 75 audio frames/s is 1.25 audio frames per video frame, so 8 codebooks add about 10 tokens per game frame; an integer alignment would need a different stride (reader's arithmetic) (source: assumption).
- Channel counts C, D and the embedding size D are not given in the text; Fig. 3 shows the layer layout with C symbolic and 32 as the default (source: Table I, p. 9).

## Limitations / open questions

- Evaluated only on speech and music; "general audio" is claimed in the abstract but the test sets are speech, noisy speech, music (Sec. IV-A, p. 6).
- Latency numbers are the architectural latency; the paper gives no end-to-end device latency (Table III, p. 9).
- The paper gives no optimiser, learning rate, batch size or training length, and doesn't say whether code or weights are released (reader's observation).
- Open question: whether tokens trained on speech/music reconstruct synthesised chip sound effects (square/noise channels) well; not tested.
- Open question: how useful the RVQ codes are as policy inputs; the paper only measures reconstruction.

## Related

- related [[he2021masked]]: both learn tokens or representations from raw modality data; MAE is for images.
- related [[srinivas2020curl]]: contrastive learning of pixel-observation features for RL; SoundStream is a reconstruction-based audio codec instead.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
