---
id: he2021masked
title: Masked Autoencoders Are Scalable Vision Learners
authors: [Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick]
year: 2021
venue: arXiv preprint
url: https://arxiv.org/abs/2111.06377
arxiv: 2111.06377
doi:
pdf_sha256: 1b490443925c72a2b7c770f90dd797e248729ae34a57e1abfe9ed36751c4cc5b
tags: [self-supervised, representation-learning, vision-transformer, masked-image-modeling, pretraining, transfer-learning]
status: read
read: 2026-11-14
---

# Masked Autoencoders Are Scalable Vision Learners

<!-- Read from arXiv v3 (2021-12-19), 14 PDF pages; p. 9 (after broader impacts) and p. 10 are references; pp. 13-14 are uncurated sample figures. -->

## TL;DR

MAE masks 75% of random ViT patches, encodes only the visible 25% and reconstructs pixels with a
small decoder; this cuts pretraining time 3× or more, and an ImageNet-1K-only ViT-H reaches 87.8%
fine-tuned; transfer beats supervised pretraining on COCO and ADE20K and prior bests on iNat/Places.

## Problem

Self-supervised masked prediction scaled NLP models (BERT, GPT), but masked autoencoding in
vision lagged (Sec. 1, p. 1). The authors attribute the gap to (i) convolutional architectures,
which made mask tokens and positional embeddings awkward, now removed by ViT; (ii) images'
spatial redundancy, so a missing patch can be filled from neighbours without high-level
understanding; (iii) a pixel-reconstructing decoder outputs lower-level content than a
word-predicting one, so its design matters more (pp. 1-2). Large ViTs also overfit ImageNet-1K
when trained supervised from scratch (Sec. 4, p. 4; App. A.2, p. 11).

## Method

- **Masking** (Sec. 3, p. 3): split the image into non-overlapping ViT patches and keep a random
  subset sampled uniformly without replacement; the rest (75% by default) are removed. A high
  ratio removes redundancy; uniform sampling avoids a center bias.
- **Encoder** (Sec. 3, p. 3): a standard ViT (linear patch embedding plus sine-cosine positional
  embeddings) applied only to the visible patches, with no mask tokens, so it processes ~25% of
  the tokens in pretraining. At deployment it sees all patches of the uncorrupted image (Fig. 1, p. 1).
- **Decoder** (Sec. 3, pp. 3-4): Transformer blocks over the encoded visible tokens plus a shared
  learned mask token for each missing patch, all with positional embeddings; a linear projection
  outputs each patch's pixels. Used only in pretraining, so it can be small; the default (8 blocks,
  512-d) has 9% of ViT-L's FLOPs per token (p. 5).
- **Target and loss** (Sec. 3, p. 4): MSE on pixels, computed only on masked patches; a variant
  normalises each target patch by its own mean and std, which improves representations.
- **Implementation** (Sec. 3, p. 4): shuffle the patch tokens, keep the first part, encode, append
  mask tokens, unshuffle; no sparse operations are needed.
- **Evaluation** (Sec. 4, p. 4): ImageNet-1K pretraining, then end-to-end fine-tuning, linear
  probing, or partial fine-tuning of the last blocks (Sec. 4.3, p. 7); transfer to COCO (Mask
  R-CNN with a ViT-FPN), ADE20K (UperNet), iNaturalist and Places (Sec. 5, p. 8).

## Assumptions

- A1: Inputs are natural RGB images divided into a regular grid of non-overlapping patches, ViT-style (Sec. 3, p. 3)
- A2: Images are highly spatially redundant and have no word-like semantic decomposition, so only a very high masking ratio makes reconstruction a task that needs holistic understanding; information-dense signals such as language work with far lower ratios (Sec. 1, pp. 1-2, Sec. 4.1, p. 4, Sec. 6, p. 8)
- A3: The encoder is a Transformer over a set of patch tokens with positional embeddings, so masked patches can simply be dropped; convolutional networks do not take mask tokens or positional embeddings as easily (Sec. 1, p. 1, Sec. 3, pp. 3-4)
- A4: Masked patches are sampled uniformly at random without replacement, with new masks every iteration (Sec. 3, p. 3, Sec. 4.1, p. 6)
- A5: Asymmetric design: the encoder sees only the visible patches, mask tokens enter in a separate lightweight decoder, and the decoder is discarded after pretraining so the encoder sees full uncorrupted images downstream (Fig. 1, p. 1, Sec. 3, pp. 3-4)
- A6: The target is the masked patches' pixels with an MSE loss on masked patches only: unnormalised in the ablations, per-patch normalised in the Table 3 models used for the comparisons and transfer (Sec. 3, p. 4, Table 1, p. 5, Table 3, p. 7)
- A7: Pretraining uses only the ImageNet-1K training images, without labels or extra data (Sec. 4, p. 4, Table 3, p. 7)
- A8: Pretraining augmentation is minimal: random resized cropping with horizontal flips, no color jitter (Sec. 4.1, p. 6, Table 8, p. 11)
- A9: Pretraining runs for hundreds of epochs: 800 in the ablations and 1600 for the Table 3 models (Table 1, p. 5, Sec. 4.1, p. 6, Table 3, p. 7)
- A10: Backbones are vanilla ViT-B/16, ViT-L/16 and ViT-H/14, from 86M to 632M parameters (Table 3, p. 7, Table 12, p. 12)
- A11: Ablations change one factor at a time from the Table 1 defaults on ViT-L/16 (Table 1, p. 5)
- A12: Representations are evaluated after supervised training on labelled downstream data (end-to-end fine-tuning with a strongly regularised recipe, partial fine-tuning, or linear probing on normalised features), not zero-shot (Sec. 4, p. 4, Sec. 4.3, p. 7, App. A.1, p. 11)
- A13: The downstream tasks are COCO detection and segmentation, ADE20K segmentation, and iNaturalist and Places classification (Sec. 5, p. 8)
- A14: The supervised baselines are the authors' own from-scratch ViT recipe; supervised ViT-L/H on ImageNet-1K is unstable without it and overfits without EMA (Sec. 4, p. 4, App. A.2, pp. 11-12)
- A15: In the COCO and ADE20K comparisons every entry, competitors included, gets its own hyperparameter search in the authors' implementation (App. A.3, App. A.4, p. 12)
- A16: Dense tasks need the single-scale ViT adapted to a multi-scale FPN by up- and down-sampling, and ADE20K adds relative position bias only at transfer (App. A.3, App. A.4, p. 12)
- A17: Timings are measured on 128 TPU-v3 cores with the authors' TensorFlow implementation (Table 2, p. 5, Sec. 4.2, p. 7)
- A18: ImageNet accuracy is single-crop top-1 on the validation set, at 224 px unless noted (Sec. 4, p. 4)
- A19: The robustness results run the ImageNet-fine-tuned models on the variant validation sets without further training (App. C, p. 12)
- A20: Some comparison numbers come from other papers (previous bests, MoCo v3 timing and linear probing, JFT-300M and contrastive augmentation results), others from the authors' reproductions (BEiT, supervised) (Sec. 4.1, p. 6, Sec. 4.2, p. 7, Table 12, Table 13, p. 12)
- A21: Inputs are single static images with no temporal context, actions or rewards (reader's inference, Sec. 3, p. 3)
- A22: These downstream datasets are natural photographs close to ImageNet's domain (reader's inference, Sec. 5, p. 8)

## Claims

- C1: An MAE-pretrained vanilla ViT-H fine-tuned on ImageNet-1K reaches 86.9% top-1 at 224 px and 87.8% at 448 px, using only IN1K data; the previous best IN1K-only result was 87.1% at 512 px [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A18, A20, A21] (Table 3, Sec. 4.2, p. 7)
- C2: A 75% masking ratio works well for both protocols; linear-probing accuracy rises with the ratio up to that sweet spot (54.6% at 10% vs. 73.5% at 75%), while fine-tuning is less sensitive, with 40-80% all working well and every ratio beating ViT-L from scratch (82.5%) [A1, A2, A3, A4, A5, A6, A7, A8, A9, A11, A12, A14, A18, A21] (Fig. 5, Sec. 4.1, pp. 4-5)
- C3: Leaving mask tokens out of the encoder raises accuracy (fine-tuning 84.9 vs. 84.2, linear 73.5 vs. 59.6) and cuts pretraining FLOPs 3.3×, giving a 2.8× wall-clock speedup for ViT-L and 3.5-4.1× with a 1-block decoder, ViT-H, or both [A1, A2, A3, A4, A6, A7, A8, A9, A10, A11, A12, A17, A18, A21] (Table 1c, Table 2, p. 5)
- C4: Decoder depth matters for linear probing (65.5% with 1 block vs. 73.5% with 8, up to 8 points) but little for fine-tuning, where a single-block decoder gets 84.8% [A1, A2, A3, A4, A5, A6, A7, A8, A9, A11, A12, A18, A21] (Table 1a, Sec. 4.1, p. 5)
- C5: A decoder narrower than the encoder suffices: 512-d works well for both protocols, and narrower decoders still fine-tune well [A1, A2, A3, A4, A5, A6, A7, A8, A9, A11, A12, A18, A21] (Table 1b, p. 5)
- C6: Per-patch normalised pixels are the best target tested (85.4 ft, 73.9 lin); PCA coefficients are worse, and dVAE tokens as in BEiT match normalised pixels in transfer (statistically insignificant differences) while needing an extra pretraining stage [A1, A2, A3, A4, A5, A7, A8, A9, A10, A11, A12, A13, A15, A16, A18, A21, A22] (Table 1d, p. 5, Sec. 4.1, p. 6, Table 7, p. 8)
- C7: MAE works with cropping-only augmentation and decently with none (84.0 ft, 65.7 lin); adding color jitter hurts, unlike contrastive methods that depend heavily on augmentation [A1, A2, A3, A4, A5, A6, A7, A9, A11, A12, A18, A20, A21] (Table 1e, p. 5, Sec. 4.1, p. 6)
- C8: Uniform random masking beats block-wise (75%: 82.8 ft, 63.9 lin) and grid-wise (84.0 ft, 66.0 lin) masking [A1, A2, A3, A5, A6, A7, A8, A9, A11, A12, A18, A21] (Table 1f, p. 5, Sec. 4.1, p. 6)
- C9: Accuracy keeps improving with longer pretraining, from 100 to 1600 epochs (ViT-L fine-tuning 82.3 to 85.1, linear 57.3 to 75.1), with no saturation of linear probing seen at 1600 epochs [A1, A2, A3, A4, A5, A6, A7, A8, A11, A12, A18, A21] (Fig. 7, Sec. 4.1, p. 6)
- C10: On 128 TPU-v3 cores, 1600-epoch MAE ViT-L pretraining takes 31 hours vs. 36 hours for 300 epochs of MoCo v3; 800-epoch ViT-H takes 34.5 hours with an 8-block decoder and 29.3 with 1 block [A3, A4, A5, A7, A9, A10, A17, A20] (Sec. 4.2, p. 7, Table 2, p. 5)
- C11: MAE's gain over supervised pretraining grows with model size, following the trend of JFT-300M supervised pretraining, whereas supervised IN1K accuracy saturates [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A14, A18, A20, A21] (Fig. 8, Sec. 4.2, p. 7)
- C12: Linear probing and fine-tuning are largely uncorrelated: fine-tuning one ViT-L block lifts accuracy from 73.5% to 81.0% (79.1% with only its MLP), and 4-6 blocks come close to full fine-tuning; MoCo v3 has higher linear accuracy but all of its partial fine-tuning results are worse than MAE's (2.6 points at 4 blocks) [A1, A2, A3, A4, A5, A6, A7, A8, A9, A11, A12, A18, A20, A21] (Fig. 9, Sec. 4.3, p. 7)
- C13: In linear probing MAE ViT-L reaches 75.8% (ViT-B 68.0%, ViT-H 76.6%), well above earlier masked-modeling methods but behind MoCo v3's 77.6% for ViT-L [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A18, A20, A21] (Table 12, App. B, p. 12)
- C14: On COCO with Mask R-CNN, MAE beats supervised IN1K pretraining: box AP 50.3 vs. 47.9 (ViT-B) and 53.3 vs. 49.3 (ViT-L), with mask AP following the same trend [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A13, A14, A15, A16, A21, A22] (Table 4, Sec. 5, p. 8)
- C15: On ADE20K with UperNet, MAE reaches 48.1 (ViT-B) and 53.6 (ViT-L) mIoU vs. 47.4 and 49.9 for supervised pretraining, and beats token-based BEiT [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A13, A14, A15, A16, A20, A21, A22] (Table 5, Sec. 5, p. 8)
- C16: Fine-tuned on iNaturalist 2017/2018/2019 and Places205/365, MAE improves considerably with model size and its ViT-H (448) beats the previous best results, on Places those pretrained on 1-3.5B images [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A13, A20, A21, A22] (Table 6, Sec. 5, p. 8)
- C17: Without specialised fine-tuning, MAE models are far more robust on ImageNet variants than supervised ones, e.g. ViT-H on IN-Adversarial 68.2% vs. 33.1% [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A14, A19, A21] (Table 13, App. C, p. 12)
- C18: MAE is about 3.5× faster per epoch than BEiT and more accurate [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A17, A18, A20, A21] (Sec. 4.2, p. 7)
- C19: Without specialised fine-tuning, MAE ViT-H reaches 64.4% on IN-Rendition (66.5% at 448 px) and 49.6% on IN-Sketch (50.9%), vs. 50.3% and 38.0% for the authors' supervised ViT-H and previous bests of 48.7% and 36.0% [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A14, A19, A20, A21] (Table 13, App. C, p. 12)

## Implementation notes

- Pretraining: AdamW (β 0.9, 0.95), base lr 1.5e-4 with lr = base × batch / 256, batch 4096, weight decay 0.05, cosine schedule, 40 warmup epochs, RandomResizedCrop only; no color jitter, drop path or gradient clipping; xavier_uniform init (source: Table 8, App. A.1, p. 11).
- Encoder and decoder both use sine-cosine positional embeddings, no relative position or layer scale; a linear projection maps encoder width to decoder width; an auxiliary dummy token serves as class token, and average pooling works similarly (source: App. A.1, p. 11).
- Loss on all pixels instead of masked patches costs ~0.5% accuracy (source: footnote 1, p. 4).
- Patch sizes are 16 for ViT-B/L and 14 for ViT-H; evaluation uses a 224×224 crop (source: Table 3 caption, p. 7, Sec. 4, p. 4).
- Fine-tuning: AdamW, lr 1e-3, layer-wise lr decay 0.75, batch 1024, 100 (B) or 50 (L/H) epochs, RandAug, mixup, cutmix, label smoothing, drop path 0.1/0.2 (source: Table 9, p. 11).
- Linear probing needs a different recipe: no regularisation, LARS, lr 0.1, batch 16384, 90 epochs, and a BatchNorm without affine parameters before the classifier (source: Table 10, App. A.1, p. 11).
- MAE's 3×+ saving is a pretraining saving: at inference the encoder processes every patch (source: Fig. 1 caption, p. 1); the paper gives no per-frame inference cost (source: Sec. 4-5, pp. 4-8, which report only pretraining time).
- Parameter counts: ViT-B 86M, ViT-L 304M, ViT-H 632M (source: Table 12, p. 12).

## Limitations / open questions

- Pretraining and evaluation are on ImageNet-1K (curated, object-centric photos) and photo
  transfer sets; no video or game frames are studied (Sec. 4-5, pp. 4-8). The nearest test is
  zero-shot robustness on IN-Rendition and IN-Sketch (C19).
- MAE representations are less linearly separable than contrastive ones; they shine when some
  non-linear layers are tuned (Fig. 9, p. 7; App. B, p. 12).
- Masking random patches does not remove semantic segments, and pixels are not semantic
  entities; the authors argue semantics emerge anyway but call it a hypothesis (Sec. 6, p. 8).
- The model reflects training-data biases and may generate inexistent content (p. 9).
- Open question: a frozen encoder with a linear head is MAE's weakest protocol; should a policy
  fine-tune the last few blocks (C12) under RL's noisy gradients, or keep MAE as an auxiliary loss?
- Open question: does 75% masking transfer to game frames, which have large flat regions, HUDs and
  text, and so different redundancy from photos? Does it extend to frame stacks (masking in time)?
- Open question: SAM uses an MAE ViT-H as its starting point, but credits most of its ability to
  supervised mask training; how much of SAM's segment quality would survive on the MAE features alone?

## Related

- related [[serrano2024crossdomain]]: cross-domain RL transfer; MAE is a candidate source of a shared visual encoder across games.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
