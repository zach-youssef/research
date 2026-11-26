---
id: huang2022masked
title: Masked Autoencoders that Listen
authors: [Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze, Christoph Feichtenhofer]
year: 2022
venue: NeurIPS 2022 (arXiv v3 read)
url: https://arxiv.org/abs/2207.06405
arxiv: 2207.06405
doi:
pdf_sha256: 5e55c8aff094d7baeeb6bccdda92d7e22480efab2cc6adcf3882e4d93c106b6e
tags: [audio, self-supervised, masked-autoencoder, spectrogram, audioset, general-audio, vit, reconstruction]
status: read
read: 2026-11-26
---

# Masked Autoencoders that Listen

## TL;DR

Audio-MAE applies MAE to 10 s Mel spectrograms: 16x16 patches, 80% random masking, visible-only ViT-B
encoder, a decoder with local (shifted-window) attention that reconstructs the spectrogram with MSE.
AudioSet-only pre-training gives 47.3 mAP on AS-2M and 94.1% on ESC-50 (clip-level classification).

## Problem

Transformer audio classifiers (AST, MBT, PaSST) mostly start from ImageNet-supervised weights, which
transfers a label bias and a questionable image-to-spectrogram prior; earlier audio SSL
(SS-AST) encodes the full view and needs extra speech data (Sec. 1, pp. 1-2; Sec. 2, p. 3).

## Method

- Input (Sec. 4.2, p. 5): mono 16 kHz audio, 128 Kaldi-compatible Mel bands, 25 ms window, 10 ms shift;
  a 10 s clip is a 1x1024x128 spectrogram. Non-overlapping 16x16 patches (conv kernel = stride) give
  64x8 = 512 patches, each fixed sinusoidal positional embedding (Sec. 3, p. 3; Fig. 6 caption, p. 9).
  A patch spans 16 frames, i.e. 160 ms (about 10 game frames at 60 Hz; reader's arithmetic).
- Masking (Sec. 3, Fig. 2, pp. 3-4): pre-training uses 80% unstructured random masking; the encoder sees
  only the 20% visible patches. Fine-tuning drops the decoder and masks again, best with structured
  time+frequency masking at 0.3 each (Sec. 4.2, p. 5; Fig. 4, p. 6).
- Encoder: 12-layer ViT-B (also ViT-S, ViT-L). Decoder: 16 Transformer layers, width 512, mask tokens
  inserted and order restored, shifted-window local attention (or hybrid local+global), linear head
  (Sec. 3, Fig. 3, p. 4; Sec. 4.2, p. 5; Table 1c-e, p. 7).
- Loss: MSE on masked patches (per-patch normalised in the main setup) (Sec. 3 "Objective", p. 4).
- Pre-training: AS-2M, 32 epochs, batch 512, LR 2e-4, 64 V100s, about 36 h; random start, cyclic crop
  of 10 s, +-6 dB gain jitter, no strong augmentation (Sec. 4.3, p. 5). Fine-tuning AS-2M 100 epochs
  with weighted sampling of 200K per epoch (about 12 h on 64 GPUs); AS-20K on 4 GPUs (pp. 5-6).
- Per-dataset input lengths: AudioSet 10 s, ESC 5 s, Speech Commands 1 s (128x128), VoxCeleb 10 s (App. B, p. 17).

## Assumptions

- A1: Pre-training data is AudioSet (about 2M 10 s YouTube clips, 527 event classes; speech, music, events),
  16 kHz mono; no game audio (Sec. 4.1, Sec. 4.2, p. 5).
- A2: Evaluation is clip-level classification after supervised fine-tuning of the full encoder, with the
  whole clip visible to a bidirectional encoder (Sec. 3, Sec. 4.1, pp. 4-5).
- A3: The speech numbers other than SUPERB-marked ones are end-to-end fine-tuned, not frozen-feature
  results (Sec. 4.5, Table 2, pp. 8-9).

## Claims

- C1: Audio-MAE (ViT-B, AudioSet only) reaches AS-2M 47.3 mAP and AS-20K 37.1 (text and Table 1; Table 2 prints 37.0 ± 0.11) (local decoder; global 46.8 / 36.6), best among in-domain self-supervised models and above ImageNet-supervised AST (45.9 / 34.7) [A1, A2] (Sec. 4.5, Table 2, p. 8)
- C2: ESC-50 accuracy is 94.1% (global decoder 93.6%), and 97.4% with an extra round of AudioSet supervised pre-training, above HTS-AT 97.0 [A1, A2] (Table 2, p. 8, App. C.1, Table 4, p. 18)
- C3: An 80% random pre-training mask ratio is best; random masking beats structured time/frequency masking for pre-training, while structured time+frequency masking at a lower ratio (0.3) is best for fine-tuning [A1, A2] (Sec. 4.4, Fig. 4, p. 6)
- C4: A decoder with shifted local-window attention beats global attention (AS-2M 47.3 vs 46.8; ESC-50 94.1 vs 93.6); an 8-layer decoder is best for global attention and 16 layers for local [A1, A2] (Sec. 4.4, Table 1c, Table 1d, p. 7)
- C5: Overlapping patches (16, stride 10) do not help (47.3 both) but cost 130.5 vs 48.6 FLOPs units (as tabulated); non-overlapping 16x16 is the default [A1, A2] (Sec. 4.4, Table 1a, pp. 6-7)
- C6: Encoder size: ViT-S 22M gets AS-20K 32.1 / AS-2M 45.0; ViT-B 86M 37.1 / 47.3; ViT-L 304M 37.6 / 47.4 [A1, A2] (Table 1b, p. 7)
- C7: Pre-training scales with data (AS-2M mAP 39.6 at 1%, 42.6 at 10%, 46.4 at 50%, 47.3 at 100%) and saturates after about 24 epochs [A1, A2] (Table 1f, Table 1g, p. 7)
- C8: ImageNet initialisation does not help and slightly hurts AudioSet SSL pre-training (AS-20K 37.1 from scratch vs 36.2-36.9 with ImageNet init; ImageNet alone 32.1-32.5); audio-only pre-training is best [A1, A2] (Sec. 4.4, Table 1h, pp. 7-8)
- C9: Adding an InfoNCE contrastive loss to reconstruction does not help (AS-2M 47.3 reconstruction alone, 46.8 combined, 46.6 contrastive alone) [A1, A2] (App. C.3, Table 5, pp. 19-20)
- C10: On speech tasks the AudioSet-only model gets SPC-2 98.3, SPC-1 96.9-97.6 and VoxCeleb speaker-ID 94.8 (end-to-end fine-tuned), above SS-AST and MAE-AST; wav2vec 2.0 / HuBERT numbers shown are SUPERB linear evaluation and not strictly comparable [A1, A3] (Sec. 4.5, Table 2, pp. 8-9)
- C11: Qualitatively a visualisation model (ViT-L encoder, 70% masking, MSE on non-normalised spectrograms, unlike the main setup) reconstructs masked spectrograms of speech, music and events, with unstructured masking easier than structured, and a 25% time-masked packet-loss demo is shown; no metric reported [A1] (Sec. 4.6, Fig. 6, p. 9, App. C.2, Fig. 8, pp. 18-19)

## Implementation notes

- Spectrogram normalisation uses per-dataset mean/std (AS: -4.268 / 4.569), estimated on the training split (source: App. B, Table 3, pp. 17-18).
- Pre-training: AdamW, betas 0.9/0.95, weight decay 1e-4, half-cycle cosine LR, 3 warm-up epochs, no SpecAug, no mixup, no drop path; fine-tuning adds SpecAug, mixup 0.5 (AudioSet, SPC) and drop path 0.1, per dataset (source: Table 3, p. 17).
- Effective LR = base LR x batch size / 256 (source: App. B, p. 17).
- Compute per sample: 512 patches, of which about 102 are visible at 80% masking; App. C.3 calls 102 the number of masked patches, which is arithmetically the visible count (about 410 are masked), a slip in the paper (source: App. C.3, p. 19).
- Code and checkpoints at https://github.com/facebookresearch/AudioMAE, not inspected (source: p. 1).

## Limitations / open questions

- Authors: AudioSet is two orders of magnitude smaller than text corpora; 10 s clips may not teach long-range temporal dependencies; may be sub-optimal on rare or unseen audio events; Transformers on long audio are computationally heavy (App. D, p. 20).
- Future work named: joint audio-visual MAE (Sec. 5, p. 10; see CAV-MAE, gong2023contrastive).
- Reader's note: inference latency and real-time factor are not reported.
- Reader's inference: streaming or frame-synced use would need a different window or masking scheme; the paper tests only whole clips.
- Open question: the encoder is bidirectional over fixed 10 s windows (1 s for keyword spotting); how it behaves on 1-2 s windows or a sliding window at frame rate is not tested.
- Open question: reconstruction targets spend capacity on low-level spectrogram detail, which for game audio (synthetic, loud music) may mean the representation tracks the music; unmeasured.

## Related

- uses [[he2021masked]]: direct extension of image MAE (visible-only encoder, high mask ratio, mask tokens in the decoder) to spectrograms.
- related [[chen2023beats]]: BEATs replaces this paper's reconstruction target with discrete tokenizer labels and reports higher AS-2M and AS-20K mAP.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
