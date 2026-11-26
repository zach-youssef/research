---
id: locatello2020slot
title: Object-Centric Learning with Slot Attention
authors: [Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, Thomas Kipf]
year: 2020
venue: NeurIPS 2020
url: https://arxiv.org/abs/2006.15055
arxiv: 2006.15055
doi:
pdf_sha256: ff310f9ee610ce0bad20bfc7517cb730b19e11b8f45195cd548b510616164b1f
tags: [object-centric, slot-attention, unsupervised-learning, set-prediction, segmentation, representation-learning]
status: read
read: 2026-11-25
---

# Object-Centric Learning with Slot Attention

Version read: arXiv v2 (14 Oct 2020), 27 pages including the supplementary material.

## TL;DR

Slot Attention maps CNN features to K exchangeable slots by T=3 rounds of attention normalised
over the slots (slots compete) plus a GRU. Trained by reconstruction alone it segments CLEVR6 /
Multi-dSprites / Tetrominoes (ARI 98.8 / 91.3 / 99.5); AP 94.3 on CLEVR10 set prediction.

## Problem

Learned representations of scenes are usually distributed rather than compositional, object-centric
ones usually need supervision or heavy iterative inference (IODINE, MONet), which is memory- and
time-hungry (Sec. 1, p. 1; Sec. 3, p. 4). The paper wants a simple, differentiable interface
between perceptual features and a set of exchangeable variables ("slots") that can bind to any
object (Sec. 1, p. 1).

## Method

- Module (Alg. 1, Eq. 1-2, p. 3): inputs (N x D_inputs, after LayerNorm) are projected by k and v;
  slots (K x D_slots) are projected by q. attn = softmax over the slot axis of k(inputs)q(slots)^T / sqrt(D)
  (Eq. 1), so each input vector distributes weight 1 across slots. Updates are the weighted mean
  of v(inputs) with weights attn/(sum over inputs) (Eq. 2). Each slot is updated by a shared GRU,
  then an optional residual MLP, for T iterations (T=3 in the experiments). Cost O(T D N K).
- Initial slots are sampled independently from one Gaussian with learned shared mean and
  variance, so K can change at test time (Sec. 2.1, pp. 2-3).
- Proposition 1 (p. 3, proof App. D, pp. 22-23): output is permutation invariant to input order and
  equivariant to slot order.
- Object discovery (Sec. 2.2, p. 4): CNN encoder plus position embedding, Slot Attention, then a
  per-slot spatial broadcast decoder producing RGB and an alpha mask; masks are softmaxed across
  slots and the reconstructions mixed; the loss is MSE reconstruction.
- Set prediction (Sec. 2.3, p. 4): a shared MLP classifier per slot, Hungarian matching of
  slots to targets.
- Position information is added to CNN features as a 4-channel soft-position grid projected
  linearly (App. E.2, pp. 23-24).

## Assumptions

- A1: Scenes are synthetic renders with plain or simple backgrounds and few (at most about 3-10) objects (CLEVR6/10, Multi-dSprites, Tetrominoes) (Sec. 4.1, p. 6; Sec. 5, p. 9)
- A2: No object-segmentation supervision is used: the signal is image reconstruction (discovery) or object property labels (set prediction); the downstream task decides what slots specialise on (Sec. 4, p. 6; App. A, p. 15)
- A3: Single static images: no temporal data is used (Sec. 3, p. 5)
- A4: The slot count K is set to the maximum object count, plus one for the background in object discovery (K = 7 for CLEVR6, 6 for Multi-dSprites, 4 for Tetrominoes; K = 10 for CLEVR10 set prediction) (Sec. 4.1, p. 6; Sec. 4.2, p. 8)
- A5: Position information is added to the CNN features by an absolute position embedding (App. E.2, pp. 23-24; App. A, p. 15)
- A6: Object discovery evaluation excludes the background label from ARI (Table 1, p. 6)

## Claims

- C1: Unsupervised object discovery ARI (%, mean over 5 seeds): Slot Attention 98.8 +- 0.3 (CLEVR6), 91.3 +- 0.3 (Multi-dSprites), 99.5 +- 0.2 (Tetrominoes, with one outlier seed excluded); IODINE 98.8 / 76.7 / 99.2; MONet 96.2 / 90.4 / not reported; a Slot MLP baseline is far lower (60.4 / 60.3 / 25.1) [A1, A2, A3, A6] (Table 1, p. 6)
- C2: Slot Attention is cheaper than IODINE: batch size 64 fits on one V100 with 16GB versus 4 for IODINE, and CLEVR6 training takes about 24 h on 8 V100s versus about 7 days for IODINE [A1, A2] (Sec. 4.1, p. 7)
- C3: Slot Attention can fail in a suboptimal solution on Tetrominoes (segmenting into stripes) in rare cases (1 of 5 seeds), identifiable by high training reconstruction error [A1, A2] (Sec. 4.1, p. 7)
- C4: More attention iterations at test time than in training (T=3) improve segmentation, more so on the larger-object-count CLEVR10 test scenes (K raised from 7 to 11) [A1, A2, A3, A4] (Fig. 2, p. 6, Sec. 4.1, p. 7)
- C5: The model usually spreads the uniform background evenly across slots instead of capturing it in one slot, and unused slots stay empty when there are more slots than objects; one of five seeds with K=11 put the background in its own slot [A1, A2, A4] (Sec. 4.1, p. 8, App. C.1, p. 19)
- C6: Attention already specialises to individual objects at the second iteration, while the first iteration's attention still merges parts of several objects into one slot [A1, A2] (Sec. 4.1, p. 8, Fig. 3, p. 7)
- C7: Segmentation does not rely only on colour: binarised Multi-dSprites ARI 69.4 +- 0.9 (K=4) versus IODINE 64.8 +- 17.2 and R-NEM 68.5 +- 1.7; greyscale CLEVR6 ARI 98.5 +- 0.3 [A1, A2, A6] (Sec. 4.1, p. 8, Fig. 4, p. 7)
- C8: On CLEVR10 set prediction (property + position) Slot Attention gets AP_inf 94.3, AP at distance 1 86.7, at 0.5 56.0, at 0.25 10.8, at 0.125 0.9, above DSPN (T=30) at AP_inf (85.2) and within confidence intervals at AP 1 and AP 0.5 (DSPN 81.1 +- 5.2 vs 86.7 +- 1.4; 47.4 +- 17.6 vs 56.0 +- 3.6; the text says "matches or outperforms", p. 9), with lower variance; DSPN uses a ResNet-34, the Slot Attention encoder a 4-layer CNN [A1, A2, A3, A4, A5] (Table 3, p. 21, Fig. 5, p. 8, Sec. 4.2, p. 9)
- C9: Without segmentation supervision the set-prediction attention maps segment the CLEVR10 objects (ARI 78.0 +- 2.9 on masks downscaled to 32x32) [A1, A2, A4] (Sec. 4.2, p. 9)
- C10: AP degrades gracefully when a CLEVR6-trained model (K=6) is tested on 7-10 objects with K raised to the object count, and error rises with the object count; more iterations (3 to 5) help most on crowded scenes, and very crowded scenes (10 objects) still give attribute/position errors and weaker segmentation [A1, A2, A4] (Fig. 5, p. 8, Fig. 21, Fig. 22, p. 21, App. C.2, p. 20)
- C11: Ablations (CLEVR6 ARI unless noted): softmax over the input axis instead of the slot axis drops ARI to about 60; no position embedding lowers ARI (about 91) and position AP; replacing the GRU by direct replacement lowers ARI (about 85); soft k-means (no GRU, no projections, identity maps) gets 75.5 +- 3.8 versus 98.8 +- 0.3; no warmup lowers ARI (about 92); sqrt(D) temperature helps (read off Fig. 8, 10, 11, 15, 18, not tabulated; values approximate) [A1, A2] (App. B, pp. 15-16, Fig. 8, Fig. 10, Fig. 11, Fig. 15, Fig. 18, pp. 17-18)
- C12: The weighted mean (Eq. 2) is more stable than a weighted sum; weighted sum plus LayerNorm gives a similar effect [A1, A2] (App. B, p. 15, Fig. 7, p. 17)
- C13: Per-slot initial parameters (instead of shared) slightly help supervised prediction, slightly hurt unsupervised discovery, and make adding slots at test time impossible without retraining; training with more slots than needed (K=11) is slightly harmful for discovery, so a rough upper bound on the object count is a reasonable choice for K [A1, A2, A4] (App. B, pp. 15-16, Fig. 9, p. 17, Fig. 19, p. 19)

## Implementation notes

- D = D_slots = 64, GRU hidden 64, residual MLP with one hidden layer of 128 and ReLU, T=3, attn epsilon 1e-8 (App. E, p. 23, Table 11a, p. 26)
- CNN encoder: four 5x5 conv layers (64 channels for CLEVR, 32 for Tetrominoes/Multi-dSprites), strides 1 (stride 2 on two layers for CLEVR10 property prediction), position embedding, flatten, LayerNorm, two per-location MLP layers (Tables 4-5, p. 23)
- Decoder: spatial broadcast (8x8 grid for CLEVR, full resolution for the others), five/three 5x5 deconvs, 4 output channels (Tables 6-7, p. 24)
- Adam lr 4e-4, beta (0.9, 0.999), exponential decay rate 0.5 over 100K steps (object discovery) / 50K (property prediction), warmup 10K / 1K iterations, 500K / 150K steps, batch 64 / 512 (Table 11, p. 26)
- Datasets: CLEVR (with masks) cropped to [29,221] x [64,256] then resized to 128x128, first 70K for training; Tetrominoes and Multi-dSprites first 60K, 320 test examples for object discovery (Sec. 4, p. 6, App. E.8, p. 26)
- Set-prediction head: one hidden layer of 64 with ReLU, sigmoid output, Huber loss (App. E.4, p. 25)
- Authors' code is at the google-research repo they cite (footnote 1, p. 2); not consulted by this reader.

## Limitations / open questions

- The background gets no special treatment; all slots share one format (App. A, p. 15)
- The absolute positional encoding is not translation-equivariant (App. A, p. 15)
- Slot Attention does not know about objects per se: it clusters whatever the task requires (objects, colours, regions) (App. A, p. 15)
- Slots interact only through the softmax competition, with no explicit slot-to-slot communication (App. A, p. 15)
- The authors expect struggles with complex backgrounds and textures, as in IODINE (Sec. 4.1, p. 8)
- Open question (reader's lens, a game-agnostic object space for a shmup policy): the slots here are permutation-equivariant and unsupervised, so no slot has a stable identity or class across images, let alone across games; a policy would need the order-agnostic read-out (set transformer or per-slot MLP) and some way to align slot semantics across games.
- Open question: every dataset has at most about 10 large, distinct, 3D-rendered objects. Bullet hell has dozens of small near-identical sprites; the paper shows error rising with the object count and merged objects on crowded scenes (C10) but nothing with tiny items, so whether K bullets can each get a slot is untested.
- Open question: a shared art style across games would be encoded by the CNN features, not removed by slots; whether reconstruction of slots from one game transfers to another game's sprites is untested here.

## Related

- related [[lin2020space]]: SPACE, another unsupervised object-decomposition model (per-cell foreground objects plus background segments)
- related [[delfosse2023continuity]]: adds motion and object-continuity supervision to SPACE or Slot Attention to make them work on Atari
- related [[delfosse2024ocatari]]: provides RAM-derived object lists for Atari, usable to check what slots bind to
- related [[yoon2023investigation]]: compares object-centric representations (including slots) for RL
- related [[ravi2024sam]]: promptable video segmentation that gives class-agnostic masks with supervision, an alternative route to a universal object space
- related [[zambaldi2018relational]]: a relational policy over entities extracted by attention, a consumer of slot-like sets

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
