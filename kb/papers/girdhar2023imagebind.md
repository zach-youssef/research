---
id: girdhar2023imagebind
title: "ImageBind: One Embedding Space To Bind Them All"
authors: [Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, Ishan Misra]
year: 2023
venue: CVPR 2023 (arXiv v2 read)
url: https://arxiv.org/abs/2305.05665
arxiv: 2305.05665
doi:
pdf_sha256: 0dcf74b737b46d82c304fb88fd16dfb3b1911975ac9439a23b9b0e82049e0ffb
tags: [multimodal, embedding, contrastive, clip, audio, zero-shot, audioset, depth, imu]
status: read
read: 2026-11-26
---

# ImageBind: One Embedding Space To Bind Them All

## TL;DR

Aligns six modalities (image/video, text, audio, depth, thermal, IMU) in the frozen OpenCLIP space by
training each new encoder only against images (InfoNCE), e.g. AudioSet video-audio pairs. Audio gets
emergent zero-shot ESC-50 66.9% and AudioSet 17.6 mAP via text prompts, from 2 s audio clips.

## Problem

Joint embeddings need all modalities paired together; real data pairs images with one modality at a
time. ImageBind asks whether image-paired data alone yields a shared space with emergent
cross-modal alignment (Sec. 1, pp. 1-2; Sec. 3.2, p. 3).

## Method

- Per-pair InfoNCE with an image/video embedding and a modality embedding, symmetric, in-batch negatives
  (Eq. 1, Sec. 3.1-3.2, p. 3). Image and text encoders are initialised from OpenCLIP ViT-H (630M) and a
  302M text encoder and kept frozen; the other encoders are trained (Sec. 4, p. 4).
- Audio (Sec. 3.3, p. 3; App. B.1, p. 13): 2 s of 16 kHz audio becomes a 128-bin log-Mel spectrogram (25 ms
  Hamming, 10 ms hop, 128x100t), encoded by a ViT-B with patch size 16, stride 10 (AST-style).
  At inference several 2 s clips are sampled uniformly to cover a longer input (ceil(5/2) = 3 for ESC);
  how their embeddings are combined is not stated.
- Video is treated as 2-frame clips sampled from 2 s with an inflated ViT patch projection (Sec. 3.3, p. 3).
  Depth as disparity, thermal as 1-channel images, IMU as 6x2000 (5 s) with a 1D conv and Transformer.
- Training data: AudioSet 2M unbalanced videos used without labels for (video, audio) matching;
  SUN RGB-D, LLVIP and Ego4D for the other pairs; web image-text pairs only through the frozen CLIP
  (Sec. 4, p. 4; App. A, p. 12). Pairs are trained one at a time with batch 2048 for (video, audio);
  Table 9 gives 64 epochs for the main audio model (p. 14), while Sec. 5 says the Sec. 4 models were
  trained for 32 epochs (pp. 6-7); the paper doesn't reconcile them.
- "Emergent zero-shot": classify audio with CLIP text prompt templates, with no (audio, text) pairs seen (Sec. 4, p. 4).

## Assumptions

- A1: The image/text encoders are the large frozen OpenCLIP ViT-H trained on billions of web pairs; the
  quality of audio alignment depends on this (Sec. 4, p. 4, Sec. 5.1, p. 7).
- A2: Audio training pairs are natural YouTube videos where the soundtrack relates to the content;
  audio classes are those of AudioSet/ESC (Sec. 4, p. 4, App. A, p. 12).
- A3: Audio is processed in 2 s clips; evaluation on longer clips uses several uniformly sampled 2 s
  clips (App. B.1, p. 13).

## Claims

- C1: With only (image, text) and (video, audio) supervision, ImageBind gives emergent zero-shot audio classification of ESC-50 66.9% top-1, AudioSet audio-only 17.6 mAP and VGGSound 27.8% (random 2.75 / 0.62 / 0.32); the supervised SOTA is 97.0 / 49.6 / 52.5 [A1, A2, A3] (Sec. 4.1, Table 2, p. 5)
- C2: Text-to-audio retrieval emergent zero-shot: Clotho R@1 6.0 / R@10 28.4, AudioCaps R@1 9.3 / R@10 42.3, twice AVFIC's Clotho R@1 (3.0) though no audio-text pairs are used (R@10 28.4 vs 17.5; AudioCaps 9.3 vs 8.7 R@1); ESC-50 66.9 is comparable to AudioCLIP's 68.6, which used AudioSet class names as supervision [A1, A2, A3] (Sec. 4.2, Table 3, pp. 5-6)
- C3: Few-shot linear probing on ESC-50 (fold-1 top-1) beats a self-supervised AudioMAE ViT-B by about 40% top-1 at <= 4 shots, and beats a supervised AudioMAE up to 4 shots; zero-shot exceeds supervised <= 2-shot (figure read from Fig. 3) [A1, A2, A3] (Sec. 4.3, Fig. 3, pp. 5-6)
- C4: Audio and image embeddings compose: adding them (L2-normalised, weights 0.5 each) retrieves images with both concepts; audio embeddings replace CLIP text embeddings in Detic (audio-prompted detector) and DALLE-2, with no retraining; qualitative [A1, A2] (Sec. 4.4, Fig. 1, Fig. 4, Fig. 5, p. 1, p. 6, App. B.4, p. 13)
- C5: Audio-based retrieval on MSR-VTT 1K-A text-to-video reaches R@1 6.8 / R@5 18.5 / R@10 27.2 from audio only, and 36.8 / 61.8 / 70.0 combining audio and video (0.05 / 0.95 embedding mix) [A1, A2, A3] (Sec. 4.2, Table 4, p. 5, App. C.1, p. 14)
- C6: Emergent performance rises with the image encoder: ViT-B to ViT-H gives about +4% on audio and +7% on depth zero-shot; a larger audio encoder helps (ESC fold-1 ViT-S 52.8, ViT-B 56.7 with a ViT-B image encoder; 54.8 / 60.3 with ViT-H) [A1, A2, A3] (Sec. 5.1, Fig. 6, Table 6, pp. 7-8)
- C7: Design choices for audio (ablations: ViT-B, 16 epochs, ESC fold-1): the fixed temperature 0.05 beats a learnable one (56.7 vs 54.8), but the other fixed values tried are worse than learnable (52.4 at 0.07, 45.4 at 0.2, 24.3 at 1.0), although the caption says fixed beats learnable for all modalities; linear head beats MLP (56.7 vs 51.0); longer training helps (16/32/64 epochs: 56.7/61.3/62.9); temporally aligned video-audio samples beat unaligned (56.7 vs 55.7); strong image augmentation cuts ESC to 22.6; batch 2K is best for audio (56.7 vs 39.4 at 512) [A1, A2, A3] (Sec. 5.2, Table 5, Table 7, pp. 7-8)
- C8: Emergent zero-shot quality is not tied to ImageNet accuracy of the frozen image encoder: self-supervised DINO gives ESC 44.7 and VGGSound 17.2 vs supervised DeiT 25.0 and 9.6 [A1] (Sec. 5.2, Table 8, p. 8)

## Implementation notes

- Audio token count per 2 s clip, reader's arithmetic from Sec. 3.3, p. 3 (the paper gives no count): with 200 frames, patch 16 and stride 10 (assuming stride 10 applies to both axes) a 128x200 spectrogram gives about 12 x 19 = 228 tokens, one column per 100 ms (source: assumption).
- Audio encoder ViT-B sizes: embedding 768, 12 heads, 12 layers; learning rate 1.6e-3, weight decay 0.2, batch 2048, temperature 0.05, 64 epochs, frequency masking 12 (source: Table 9, p. 14).
- Hardware: 32GB V100 or 40GB A100 (source: App. C.1, p. 14).
- Frozen OpenCLIP ViT-H and text encoder (source: Sec. 4, p. 4).
- Code and models at the project page https://facebookresearch.github.io/ImageBind, not inspected (source: p. 1).
- Embedding arithmetic uses the temperature-scaled features, L2-normalised and summed with weight 0.5 (source: App. B.4, p. 13).

## Limitations / open questions

- The embeddings are not trained for a downstream task and lag specialist models; adaptation for structured tasks such as detection is future work; the model is a research prototype not for real-world applications (Sec. 6, p. 8; App. F, p. 15).
- The model is a general-purpose embedding without task-specific tuning (Sec. 6, p. 8).
- Reader's note: inference latency and throughput are not reported.
- Concepts are limited to those of the datasets used (e.g. outdoor thermal, indoor depth) (App. F, p. 15).
- Absolute emergent audio numbers are far below supervised SOTA (ESC 66.9 vs 97.0) (Table 2, p. 5).
- Open question: all audio content is natural recordings; the text prompts name semantic classes (dog barking). Game sound effects and chiptune are outside the evaluated domain, and a 2 s clip embedding is clip-level, so use as a frame-synced policy input would be via a sliding window (reader's inference).
- Open question: the bind-to-image idea could in principle align game audio to game frames cheaply, but whether the frozen CLIP image side carries anything useful for pixel-art shmup frames is untested here.

## Related

- uses [[gong2021ast]]: audio tokenisation follows AST (patch 16, stride 10, 128 mel bins).
- related [[huang2022masked]]: Audio-MAE is the few-shot baseline for ImageBind's audio encoder.
- related [[gong2023contrastive]]: CAV-MAE also learns coordinated audio-visual representations from video, with a reconstruction loss and from-scratch pre-training instead of a frozen CLIP.
- related [[srinivas2020curl]]: contrastive objective for pixels in RL, a different use of InfoNCE.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
