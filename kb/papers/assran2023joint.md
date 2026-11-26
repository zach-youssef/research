---
id: assran2023joint
title: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
authors: [Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas]
year: 2023
venue: CVPR 2023 (arXiv v3 read)
url: https://arxiv.org/abs/2301.08243
arxiv: 2301.08243
doi:
pdf_sha256: eddbdc093eb4d48662bcf4fbd1c6735ccd606205cc9070977ce317d68aff3941
tags: [self-supervised, representation-learning, jepa, joint-embedding, masked-image-modeling, vision-transformer, latent-prediction]
status: read
read: 2026-11-26
---

# Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture

## TL;DR

I-JEPA predicts the embeddings (from an EMA target encoder) of large masked image blocks from one
context block: no pixel decoder, no view augmentations, images only (no actions, no simulator).
A ViT-H/14 trains on ImageNet in under 72 h on 16 A100s and beats MAE on linear probing.

## Problem

Invariance-based self-supervision (DINO, iBOT, SimCLR) relies on hand-crafted augmentations whose
biases may hurt some tasks or other modalities; generative masked methods (MAE, BEiT) need few
priors but learn lower-semantic-level features that lag in linear probing and low-label transfer
(Sec. 1, pp. 1-2). I-JEPA wants semantic features without hand-made views and without pixel-level
reconstruction.

## Method

- Setting: Joint-Embedding Predictive Architecture, like a generative masked model but with the
  loss in embedding space; collapse is avoided by an asymmetric design (EMA target encoder), not
  by negatives or regularisers (Sec. 2, Fig. 2c, pp. 2-3).
- Three ViTs: context encoder f_theta, target encoder f_theta-bar (EMA of the context encoder),
  and a narrow ViT predictor g_phi (Sec. 3, Fig. 3, p. 3).
- Targets: the whole image goes through the target encoder; M=4 blocks (random scale 0.15-0.2,
  aspect ratio 0.75-1.5) are cut from the encoder's output, not from the input (Sec. 3, pp. 3-4).
- Context: one block (scale 0.85-1.0, unit aspect ratio) with any overlap with the targets
  removed; only these patches go through the context encoder (Sec. 3, Fig. 4, pp. 3-4).
- Predictor: takes the context output plus a learnable mask token with positional embedding for
  each target patch, and is run once per target block (Sec. 3, p. 4).
- Loss: mean squared L2 distance between predicted and target patch representations, averaged
  over blocks (Sec. 3, p. 4). Target encoder weights follow an EMA of the context encoder.
- Evaluation uses the average-pooled target encoder output; there is no [cls] token (App. A.1, p. 12).

## Assumptions

- A1: Pretraining data is a large natural-image set (ImageNet-1K, or ImageNet-22K for the scaling rows) (Sec. 5, Table 5, p. 5, p. 7).
- A2: Representations are judged by frozen-encoder linear probing or by low-label fine-tuning, on image classification and Clevr counting/depth (Sec. 5-6, pp. 5-6).
- A3: The architecture is a ViT with patch tokens, so blocks of patches can be masked and predicted (Sec. 3, p. 3).
- A4: Inputs are single static images with no actions, rewards or temporal structure (reader's inference, Sec. 3, p. 3).
- A5: The reported compute is on A100 GPUs, with GPU hours as the unit (Fig. 1, Fig. 5, p. 1, p. 7).

## Claims

- C1: Linear probing on ImageNet-1K: I-JEPA ViT-H/14 reaches 79.3% top-1 in 300 epochs, above MAE ViT-H/14 (77.2%, 1600 epochs) and CAE ViT-L/16 (78.1%); ViT-H/16 at 448 px reaches 81.1%, vs iBOT ViT-L/16 81.0% which uses view augmentations [A1, A2, A3] (Table 1, p. 5).
- C2: With 1% of ImageNet labels, I-JEPA ViT-H/14 gets 73.3% and ViT-H/16 at 448 px gets 77.3%, vs MAE ViT-H/14 71.5% (the table's best, MSN ViT-B/4 with augmentations, is 75.7%; I-JEPA at 448 px surpasses it) [A1, A2, A3] (Table 2, p. 5).
- C3: Linear-probe transfer with ViT-H/14: CIFAR100 87.5, Places205 58.4, iNat18 47.6, versus MAE 77.3/55.0/32.9 and data2vec 81.6/54.6/28.1; I-JEPA beats DINO on CIFAR100 and Places205 but trails iBOT (88.3/60.4/57.3) [A1, A2, A3] (Table 3, p. 6).
- C4: On low-level tasks with a linear probe, I-JEPA ViT-H/14 scores Clevr/Count 86.7 and Clevr/Dist 72.4, above DINO (86.6/53.4) and iBOT (85.7/62.8) and tied with MAE on Dist (72.4); MAE is better on Count (90.5) [A1, A2, A3] (Table 4, p. 6).
- C5: Pretraining a ViT-H/14 takes under 1200 GPU hours on ImageNet, over 2.5x faster than a ViT-S/16 with iBOT and over 10x more efficient than a ViT-H/14 with MAE; the target computation adds about 7% per iteration but I-JEPA converges in roughly 5x fewer iterations [A1, A3, A5] (Sec. 1, Sec. 7, Fig. 5, p. 2, pp. 6-7).
- C6: Larger pretraining data and models help semantic transfer: ViT-H/14 on IN22K improves over IN1K on CIFAR100 (89.5 vs 87.5), iNat18 (50.5 vs 47.6), Clevr/Count (88.6 vs 86.7), but Places205 drops (57.8 vs 58.4), and the IN22K run trains for the equivalent of 900 IN1K epochs against 300, so data size is confounded with compute; ViT-G/16 further helps Place205 and iNat18 but not Clevr (low-level) tasks, which the authors attribute to its larger patches [A1, A2, A3] (Table 5, p. 7).
- C7: Predicting in pixel space instead of target-encoder output drops 1%-ImageNet top-1 for ViT-L/16 from 66.9 to 40.7 (500 vs 800 epochs) [A1, A2, A3] (Table 7, p. 8).
- C8: The multi-block masking is essential: it gets 54.2% top-1 (ViT-B/16, 300 epochs, 1% labels) versus 15.5% for rasterized, 20.2% for block and 17.6% for random masking [A1, A2, A3] (Table 6, p. 8).
- C9: Target block scale and count matter: scale (0.15, 0.2) gives 54.2 against 19.2 for (0.075, 0.2); 4 targets give 54.2 against 9.0 for 1; a context of scale (0.85, 1.0) gives 54.2 against 31.2 for (0.4, 1.0) [A1, A2, A3] (Table 8, Table 9, Table 10, p. 14).
- C10: Taking target blocks from the target encoder's output rather than masking its input improves 1% ImageNet top-1 for ViT-H/16 from 56.1 to 67.3 [A1, A2, A3] (Table 11, App. C, p. 15).
- C11: A narrower predictor (384 vs 1024 channels) is better (70.7 vs 68.4, ImageNet 1% fine-tuning, ViT-L/16, 600 epochs), and a 12-layer predictor beats 6 layers (66.9 vs 64.0) [A1, A2, A3] (Table 12, Table 14, p. 15).
- C12: Full-ImageNet fine-tuning: ViT-H/16 at 448 px reaches 87.1% in 300 epochs, versus MAE 87.8% in 1600 epochs [A1, A3] (Table 15, App. D, p. 16).
- C13: Decoding the predictor's output with a separately trained RCDM diffusion decoder yields samples that agree on object parts and pose and vary in low-level detail and background, which the authors read as the predictor capturing positional uncertainty [A1, A3] (Sec. 8, Fig. 6, pp. 7-8).

## Implementation notes

- Predictor depth 6 for ViT-B/16, 12 for ViT-L/16 and ViT-H, 16 for ViT-G/16; width 384, heads equal to the encoder's (source: App. A.1, p. 12).
- AdamW, batch 2048, LR warmup 1e-4 to 1e-3 over 15 epochs then cosine to 1e-6, weight decay 0.04 to 0.4, EMA momentum 0.996 rising to 1.0 (source: App. A.1, p. 12).
- Context and target mask sizes are made equal across a co-located GPU's batch for batching; masks are made in the data-loader collate function (source: App. A.1, p. 12).
- Linear probe: LARS, batch 16384, 50 epochs, sweep of three LRs and two weight decays (source: App. A.2, p. 12).
- Epoch counts for headline models: ViT-H/14 300 epochs; ViT-L/16 600; image resolution 224 unless stated (source: Table 1, Sec. 5, p. 5).

## Limitations / open questions

- Does not match view-invariance methods on all semantic tasks: iBOT is better on iNat18 and Places205, and MAE is better on Clevr/Count (Table 3, Table 4, p. 6).
- Larger patches (ViT-G/16) hurt low-level tasks (Table 5, p. 7).
- Weight decay trades fine-tuning for linear-probe quality (Table 13, p. 15).
- Open question: all evidence is on natural-image classification; nothing here tests small synthetic frames such as game screens, or whether tiny moving sprites survive the 0.15-0.2 block scale.
- Open question: the target is the EMA encoder's output and the paper gives no collapse analysis; stability for non-ImageNet data is untested.
- Open question: the representation is static; no temporal or action-conditioned prediction (the video extension, Assran et al. 2025, assran2025vjepa, is a sibling note not on main yet).

## Related

- related [[he2021masked]]: the pixel-reconstruction baseline whose masked-ViT setup I-JEPA keeps but replaces the loss target with embeddings.
- related [[srinivas2020curl]]: a contrastive, augmentation-based RL encoder objective; I-JEPA is the augmentation-free, non-contrastive alternative.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
