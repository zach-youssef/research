---
id: gong2023contrastive
title: Contrastive Audio-Visual Masked Autoencoder
authors: [Yuan Gong, Andrew Rouditchenko, Alexander H. Liu, David Harwath, Leonid Karlinsky, Hilde Kuehne, James Glass]
year: 2023
venue: ICLR 2023 (arXiv v4 read)
url: https://arxiv.org/abs/2210.07839
arxiv: 2210.07839
doi:
pdf_sha256: 0d6d2986718beac040b4cd0605356ac1deccc07dd500eb416730bd01f8ed0547
tags: [audio-visual, self-supervised, masked-autoencoder, contrastive, multimodal, audioset, vggsound, retrieval]
status: read
read: 2026-11-26
---

# Contrastive Audio-Visual Masked Autoencoder

## TL;DR

CAV-MAE trains one audio-visual Transformer with masked reconstruction plus an audio-visual
contrastive loss (75% masking per modality, one RGB frame, 10 s spectrogram). Self-supervised, 4 GPUs:
VGGSound 65.5% and AudioSet-20K 40.5-42.0 mAP audio-visual; clip-level classification and retrieval.

## Problem

Contrastive audio-visual learning gives coordinated embeddings for retrieval but can discard modality-unique
information; masked autoencoding keeps information but has no explicit audio-visual correspondence
objective. They had never been combined for audio-visual learning (Sec. 1, pp. 1-2).

## Method

- Tokens (Sec. 2.1.1, p. 2): 10 s AudioSet/VGGSound clips. Audio: 128-d log-Mel fbank (25 ms window, 10 ms
  shift) = 1024x128, 512 patches of 16x16. Video: 10 RGB frames per clip (1 FPS) are sampled; training
  picks one at random, inference averages predictions over frames; 224x224 giving 196 patches.
- AV-MAE ("vanilla", Sec. 2.2, pp. 3-4): both token sets concatenated, 75% masked, one joint encoder and
  decoder reconstruct both (MSE on normalised masked patches).
- CAV-MAE (Sec. 2.3, Eq. 2-9, Fig. 1, pp. 4-5): modality-specific 11-layer encoders on the 25% visible tokens
  of each modality, then one shared joint-encoder layer run three times (audio only, video only, both)
  with separate LayerNorms. The single-modality streams are mean-pooled for an InfoNCE-style contrastive
  loss (tau 0.05); the joint stream feeds an 8-layer joint decoder (width 512) for reconstruction.
  Loss = L_r + lambda_c L_c, lambda_c = 0.01 (Sec. 2.3.1, p. 5).
- Sizes: Sec. 2.3.1 gives encoder about 164M and decoder about 27M parameters (p. 5), while App. B gives
  about 190M for the full model and about 160M for the audio-visual encoder (p. 14); the paper doesn't
  reconcile them. The single-modal branch used for single-modal fine-tuning is about 85M (p. 14).
- Initialisation: self-supervised ImageNet MAE weights (layers 1-11 copied to audio and visual encoders,
  layer 12 to the joint encoder) (Sec. 2.3.1, p. 6; App. E, p. 17).
- Pre-training: AudioSet-2M without labels or class balancing; 12 epochs, batch 48 on 4 Titan X for the
  main models; CAV-MAE-scale+ 25 epochs, batch 108 (contrastive batch 27) on 4 A5000; about a week (Sec. 3, p. 6; App. B, p. 14).
- Fine-tuning for event classification with audio (A), video (V) or both (AV) (Sec. 4, p. 6).

## Assumptions

- A1: Data is 10 s YouTube clips (AudioSet, VGGSound) of natural audio-visual events; the audio is
  loosely related to the frames (Sec. 2.1.1, p. 2, App. A, pp. 13-14).
- A2: Video is reduced to single 224x224 frames sampled at 1 FPS without temporal modelling across frames
  (Sec. 2.1.1, p. 2).
- A3: Downstream use is clip-level classification (fine-tuned) or clip-level retrieval, not
  frame-synchronous prediction (Sec. 4-5, pp. 6-9).
- A4: Pre-trained from self-supervised ImageNet MAE weights, which matter on small data (Sec. 2.3.1, p. 6, App. E, p. 17).
- A5: Contrastive batches are small (effective contrastive batch 27 for the best model), where audio-visual matching accuracy is about 72% at 75% masking (Sec. 2.3, p. 5).

## Claims

- C1: Contrastive and masked-reconstruction objectives are complementary: with the same architecture, CAV-MAE beats CAV (contrastive only) and AV-MAE (reconstruction only) in A-V fine-tuning, e.g. AudioSet-20K 40.5 vs 38.5 and 37.4, VGGSound 65.4 vs 64.1 and 64.1 [A1, A2, A3, A4] (Sec. 4, Table 1, pp. 6-7)
- C2: CAV-MAE-scale+ (larger batch, 25 epochs) reaches AudioSet-20K 37.7 (A) / 19.8 (V) / 42.0 (A-V), AudioSet-2M 46.6 / 26.2 / 51.2, VGGSound 59.5 / 47.0 / 65.5, which the authors call a new audio-only SOTA on AS-20K and VGGSound "without supervised pretraining and industry-level computational resources" [A1, A2, A3, A4] (Sec. 4, Table 1, p. 7)
- C3: Multi-modal pre-training improves single-modality fine-tuning: audio-only AS-20K 34.2 (Audio-MAE trained here) rises to 37.7 and visual 15.7 to 19.8; the authors hypothesise the modalities act as soft labels for each other [A1, A2, A3, A4] (Sec. 4, Table 1, p. 7)
- C4: Modality-specific encoders help: AV-MAE beats vanilla AV-MAE (single joint encoder) in A-V fine-tuning on all three sets (AS-20K 37.4 vs 36.5, AS-2M 49.6 vs 48.3, VGGSound 64.1 vs 63.4). The text says vanilla AV-MAE does not outperform the ensemble of single-modal MAEs, but Table 1 shows that only on AS-20K (36.5 vs 36.7 ens); vanilla wins on AS-2M (48.3 vs 46.9) and VGGSound (63.4 vs 63.1) [A1, A2, A3] (Sec. 4, Table 1, pp. 7-8)
- C5: Retrieval needs the contrastive loss: MAE-only models are near chance (visual-to-audio R@1 0.1 on AudioSet subset) while CAV-MAE-scale+ reaches 18.8 R@1, 50.1 R@10 (AudioSet) and 14.8 / 44.0 on VGGSound (zero-shot transfer); against CAV-only at the same lambda_c the result is mixed: at 0.1 CAV-MAE is better on most metrics but not AudioSet R@1 (16.1 vs 17.4) or VGGSound R@10 (45.9 vs 46.2), at 0.01 it is worse on AudioSet (R@1 12.3 vs 14.6) and better only on VGGSound R@1 (12.5 vs 10.9); the authors say the reconstruction loss "does not hurt, and in many cases, improves" retrieval [A1, A3, A5] (Sec. 5, Table 3, pp. 8-9)
- C6: Lambda_c matters: 0.1 / 0.01 / 0.001 give AS-20K 39.3 / 40.5 / 38.6; more epochs (1 to 12: 37.3 to 40.8) and larger batch (48 to 108: 40.5 to 40.8) help [A1, A2, A3, A4] (Table 2a, Table 2b, Table 2c, p. 8)
- C7: Fine-tuning on one modality is better than A-V fine-tuning with missing-modality inference for video (19.8 vs 14.4) and slightly for audio (37.7 vs 36.7); averaging per-frame predictions helps (video 19.8 vs middle-frame 17.4, A-V 42.0 vs 40.9); more frames help with saturation (Fig. 4) [A1, A2, A3] (Table 2g, Table 2h, p. 8, App. G, Fig. 4, p. 20)
- C8: ImageNet (self-supervised MAE) initialisation helps but is not essential: with AudioSet SSL pre-training, 40.6 AS-20K with vs 37.3 without; AudioSet-2M 50.5 vs 49.5; with neither, AS-20K fine-tune 8.0 (with no AudioSet pre-training, supervised ImageNet init is better for fine-tuning, 29.3 vs 25.6, but worse for linear probing, 7.2 vs 10.3) [A1, A3, A4] (App. E, Table 10, Table 11, pp. 17-18)
- C9: Masking ratio and strategy barely affect classification (AS-20K 40.3-41.3 for ratios 0.50-0.85; time-frequency 0.5 gives 41.2) but strongly affect retrieval (VGGSound A to V R@1 14.8 at 0.50 vs 5.6 at 0.85) [A1, A3] (App. F, Table 12-15, pp. 18-20)
- C10: Shuffling audio-visual pairs in pre-training leaves AV-MAE almost unchanged (it mostly uses single-modality information) but hurts CAV, and CAV-MAE only gains over AV-MAE with true pairs [A1, A3] (App. J, pp. 22-23)
- C11: CAV-MAE is not a strong visual sound-source localiser by cosine similarity of pooled audio and patch features (qualitative; "in some cases" the heatmaps point at the background); authors attribute it to masked contrastive learning [A1, A3] (App. I, Fig. 6, p. 22)

## Implementation notes

- Per-clip token budget: 512 audio + 196 visual tokens (one frame); with 75% masking the encoders see about 128 + 49 tokens (source: Sec. 2.1.1, p. 2, Sec. 2.3, p. 4).
- Time resolution per audio token: reader's arithmetic from the 10 ms shift and 16x16 patches of Sec. 2.1.1, p. 2; one patch column spans 16 frames = 160 ms, about 10 game frames at 60 Hz (source: assumption).
- Hyperparameters: Adam, wd 5e-7, betas (0.95, 0.999), LR 1e-4 (scale+) / 5e-5, input norm mean -5.081 std 4.485; fine-tuning per-dataset LRs, mixup, balanced sampling, weight averaging (source: App. B, Table 4, p. 14).
- Code, checkpoints and sample id lists at https://github.com/yuangongnd/cav-mae, not inspected (source: p. 1, App. A, p. 14).

## Limitations / open questions

- The authors note their visual branch is weaker than SOTA models that use supervised ImageNet weights and many frames, and that comparisons with those are not exactly fair (Sec. 4, p. 7; App. E, p. 18).
- Frame aggregation ignores inter-frame correlation, chosen for cost (Sec. 2.1.1, p. 2).
- Ethics: possible misuse for surveillance (Ethics statement, p. 10).
- Reader's note: latency, FLOPs and real-time factor are not reported.
- Reader's note: Appendices C (Kinetics-Sounds), D (audio-to-visual and MSR-VTT retrieval) and H (reconstruction figures) were read at table level only; their numbers are not in the claims.
- Open question: audio-visual correspondence is learned from web video where sound sources are visible (VGGSound filtered for it). In shmups the sound (music, explosions) is only loosely tied to a given frame; whether the contrastive term would find any signal is untested, and the 1 FPS visual stream is far below a game's frame rate.
- Open question: only clip-level (10 s) pooled outputs are evaluated; per-token outputs could be used as a frame-aligned feature, but nothing in the paper tests that.

## Related

- related [[huang2022masked]]: concurrent audio MAE; CAV-MAE compares against its reported numbers as a separate row, while its own Audio-MAE baseline is the authors' model (vanilla AV-MAE architecture, audio only).
- uses [[he2021masked]]: initialises encoders from image MAE weights and uses its masked reconstruction.
- uses [[gong2021ast]]: AST-style spectrogram tokenisation.
- related [[chen2023beats]]: BEATs reports CAV-MAE (AS-2M 44.9) in its comparison table.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
