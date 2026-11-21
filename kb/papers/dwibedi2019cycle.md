---
id: dwibedi2019cycle
title: Temporal Cycle-Consistency Learning
authors: [Debidatta Dwibedi, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, Andrew Zisserman]
year: 2019
venue: CVPR 2019 conference paper
url: https://arxiv.org/abs/1904.07846
arxiv: 1904.07846
doi:
pdf_sha256: af6a8b394acde303650fded630450ee2dfe57929ced68ea1abc16a86d55d70c0
tags: [self-supervised, representation-learning, video, temporal-alignment, cycle-consistency, few-shot]
status: read
read: 2026-11-21
---

# Temporal Cycle-Consistency Learning

<!-- Read from arXiv v1 (2019-04-16), 14 PDF pages, all read: main paper pp. 1-8, references pp. 9-10 (skimmed), appendix pp. 11-14. -->

## TL;DR

TCC learns label-free per-frame video embeddings by making frames of two videos of one action cycle
back to themselves via soft nearest neighbours; trained from scratch it beats SaL and TCN on phase
classification, progression and alignment (Kendall's tau), and 1 labelled video matches ~50 supervised.

## Problem

Fine-grained temporal understanding of videos (which phase of an action a frame is in, how far
it has progressed) normally needs per-frame labels, which are costly and ill-defined (Sec. 1,
p. 2). Learning a shared embedding from aligned frame pairs (CCA, ranking losses, TCN's
multi-view setup) needs synchronised recordings, which most real videos lack (Sec. 1, p. 2;
Sec. 2, p. 2). Earlier cycle-consistency work either learned correspondences on fixed features
or used a non-differentiable cycle measure (Sec. 2, p. 2). TCC wants a self-supervised objective
that learns the representation itself from unaligned videos of the same process.

## Method

- An encoder φ maps each frame s_i of video S (length N) to u_i, and each frame t_j of video T
  (length M) to v_j (Sec. 3, p. 3). A frame u_i is cycle-consistent if its nearest neighbour v_j
  in V has u_i as its own nearest neighbour in U (Sec. 3.1, Fig. 2, p. 3). The goal is to
  maximise the number of cycle-consistent frames, which needs a differentiable relaxation.
- Soft nearest neighbour: ṽ = Σ_j α_j v_j with α = softmax over j of -||u_i - v_j||² (Eq. 1,
  Sec. 3.2, p. 3).
- Cycle-back classification: logits -||ṽ - u_k||² over the N frames of U, cross-entropy with the
  one-hot label i (Eq. 2, Sec. 3.2, pp. 3-4). It ignores how far in time the cycle lands.
- Cycle-back regression (the final TCC loss): β = softmax over k of -||ṽ - u_k||² (Eq. 3);
  with μ = Σ β_k k and σ² = Σ β_k (k - μ)², minimise |i - μ|²/σ² + λ log σ (Eq. 4,
  Sec. 3.3, p. 4). log σ is used instead of σ for numerical stability (p. 4). The MSE variant
  drops the variance and minimises |i - μ|² only (Sec. 5.2, p. 6).
- Training: losses over pairs of training videos and randomly selected frames within each
  (Sec. 3.4, p. 4). The paper aligns pairs "within an action category" (Sec. 1, p. 2).
- Encoder (Table 1, p. 4; Table 8, p. 14): a per-frame 2D base network (ImageNet ResNet-50 up to
  conv4c, 14×14×1024, or a 7-layer VGG-M-like net trained from scratch, 14×14×512) on 224×224
  frames; the frame and k context frames are stacked in time, then two 3×3×3×512 3D
  convolutions, global 3D max-pooling, two 512 FC layers and a linear 128-d embedding.
- Data (Sec. 4, pp. 4-6; Table 2, p. 6): Pouring (70 train / 14 val videos) and 13 Penn Action
  classes (40-134 train videos each); two classes (strumming guitar, jumping rope) were dropped
  because key events are ambiguous. The authors label key events and the phases between them
  for evaluation only (Fig. 4, p. 5).
- Evaluation on frozen features (Sec. 4.1, p. 5): per-frame phase classification by SVM; phase
  progression (normalised time to each key event) by linear regression, scored by R²; and
  Kendall's tau of nearest-neighbour alignment over all validation video pairs (no labels).
- Baselines (Sec. 5.1, p. 6): Shuffle and Learn (SaL, triplet order classification) and TCN
  (n-pairs loss, positives within a time window), plus TCC+SaL and TCC+TCN with loss weights
  searched over 0.25, 0.5, 0.75; all share the encoder.
- Applications (Sec. 6, p. 8; App. A-B, p. 11): annotation and sound transfer by nearest-frame
  lookup, fine-grained retrieval, anomaly detection by distance from typical trajectories, and
  synchronous playback using dynamic time warping on the embeddings.

## Claims

- C1: On Pouring, cycle-back regression beats both the MSE variant and cycle-back classification on all three metrics: phase classification 91.82% vs 86.16% and 88.06%, phase progression 0.8030 vs 0.6532 and 0.6636, Kendall's tau 0.8516 vs 0.6093 and 0.6707 (Table 3, Sec. 5.2, p. 6).
- C2: Training a VGG-M-like encoder from scratch, TCC beats SaL and TCN on phase classification at 10%, 50% and 100% of labels: Penn Action 74.68/76.39/77.30% vs at most 69.65/71.41/72.53%, Pouring 86.82/89.43/90.21% vs at most 76.03/83.27/84.57% (Table 4, Sec. 5.3, pp. 6-7).
- C3: From scratch, TCC beats supervised learning at 10% and 50% labels on both datasets and at 100% on Pouring (90.21 vs 88.41%), but not on Penn Action at 100% (77.30 vs 79.98%) (Table 4, p. 7).
- C4: Fine-tuning ImageNet ResNet-50, TCC is competitive on Penn Action (81.26/83.35/84.45% vs TCN 81.99/83.67/84.04%) and best on Pouring (89.23/91.43/91.82%); TCC+TCN is best on Penn Action (84.27/84.79/85.22%) (Table 5, Sec. 5.3, p. 7).
- C5: With fine-tuning, supervised learning on all labels still beats every self-supervised method on Penn Action (86.05%) and is within 0.3 points of TCC on Pouring (91.55 vs 91.82%); ImageNet or random features reach only 42.7-52.9% (Table 5, p. 7).
- C6: In few-shot phase classification on Golf Swing and Tennis Serve, TCC and TCC+TCN with one labelled video match supervised learning with about 50 densely labelled videos (Fig. 5, Sec. 5.3, p. 7, Sec. 7, p. 8).
- C7: From scratch, TCC has the best phase progression and Kendall's tau on both datasets: Penn Action 0.5383 and 0.6024, Pouring 0.7750 and 0.7504, against supervised learning from scratch at 0.5332/0.4997 and 0.5529/0.5282 (Table 6, Sec. 5.4, pp. 7-8).
- C8: With fine-tuning, TCC+TCN has the best Kendall's tau (Penn Action 0.7672, Pouring 0.8779) and Pouring progression (0.8307), TCC+SaL the best Penn Action progression (0.6839); all TCC variants beat supervised fine-tuning (0.6267/0.5582 Penn Action, 0.6986/0.6195 Pouring) (Table 6, Sec. 5.4, pp. 7-8).
- C9: Nearest neighbours in the TCC embedding retrieve frames of the same fine-grained state across videos, e.g. glass half full, or the pitcher's leg up before versus after the throw (qualitative) (Fig. 6, Sec. 6, p. 8, Fig. 9, App. D, p. 11, p. 13).
- C10: Distance from a video's frames to their nearest frames in typical videos flags anomalous activity: in one bench-press example it spikes around frame 20 (qualitative, single example) (Fig. 7, Sec. 6, p. 8).
- C11: During training on two Baseball Pitch videos, the frame-similarity matrix aligns along the diagonal more orderly ("more effective" per the caption) under cycle-back regression than under cycle-back classification (Fig. 10, App. F, p. 11, p. 14).
- C12: k = 2 frames per embedding (the frame plus one context frame) gives good performance on both datasets, with the two frames about 0.75 s apart on Penn Action (15 frames at 20 fps) and 0.3 s on Pouring (9 frames at 30 fps) (App. H, p. 12).

## Implementation notes

- Batch size 4, 20 frames, Adam, learning rate 1e-4, weight decay 1e-5, λ = 0.001, TCN positive window 5, 20 fps (Penn Action) or 30 fps (Pouring), SaL FC sizes 128 and 64, SaL fraction shuffled 0.75 (source: Table 7, App. G, p. 11).
- Batch size 4 is read as 4 videos per step with all pairs among them, and 20 frames as frames sampled per video (source: assumption; Table 7 gives only the values).
- Eqs. 1 and 3 use plain squared Euclidean distances in the softmax, with no temperature or embedding normalisation (source: Eqs. 1, 3, pp. 3-4); whether to add one is an open choice (source: assumption).
- Augmentation: random horizontal flip of a whole video, brightness offset in [-32, 32] on raw pixels, contrast factor in [0.5, 1.5], the same for all methods (source: App. E, p. 11).
- Context frames: k = 2 with a 15-frame (Penn Action) or 9-frame (Pouring) gap (source: App. H, p. 12).
- Fine-tuning: which ResNet-50 layers are trained, and for how many steps, isn't stated (source: Sec. 5.3, p. 7, silent); training runs "until convergence" (source: Sec. 3.4, p. 4).
- Whether one encoder is trained per Penn Action class or one for all classes isn't stated (source: Sec. 1, p. 2 says pairs are within a category; assumption: one per class).
- SVM kernel and regularisation for phase classification aren't given (source: Sec. 4.1, p. 5, silent).
- Synchronous playback picks the matching frame with dynamic time warping over the embeddings (source: App. A, p. 11).
- TensorFlow (source: Sec. 3.4, p. 4).

## Limitations / open questions

- Kendall's tau assumes no repeated frames; slow or periodic actions break it. The authors say
  this is not a problem for their datasets (Sec. 4.1, p. 5).
- Training data is small (tens to ~130 videos per class), which is why the from-scratch encoder
  is a small VGG-M (Sec. 3.4, p. 4; Sec. 5.3, p. 6); full supervision still wins on Penn Action
  with all labels (C3, C5).
- The method assumes every pair of videos shows the same process in the same order; actions
  without unambiguous key events were excluded from evaluation (Sec. 4, p. 5).
- Open question (reader): game trajectories loop, branch and revisit states (respawns, menus,
  periodic motion), which breaks the one-to-one, monotone correspondence TCC rewards; does the
  loss degrade gracefully, or does it need a DTW-style or segment-level variant?
- Open question (reader): TCC pairs videos of one task; across different games there is no
  shared process to align, so a cross-game embedding would need some other notion of
  correspondence (e.g. shared progress or goal events).
- Open question (reader): the embedding is per frame with k = 2 context frames; how would it
  compare with an N-frame transformer encoder as the state input to a large policy, and does
  the 128-d embedding keep the information a policy (not just a phase classifier) needs?

## Related

- related [[serrano2024crossdomain]]: the survey cites XIRL ([53]), which reuses TCC across embodiments as a reward.
- related [[he2021masked]]: another self-supervised visual pretraining objective, per image rather than across aligned videos.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
