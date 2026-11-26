---
id: wu2018compressed
title: Compressed Video Action Recognition
authors: [Chao-Yuan Wu, Manzil Zaheer, Hexiang Hu, R. Manmatha, Alexander J. Smola, Philipp Krähenbühl]
year: 2018
venue: CVPR 2018
url: https://arxiv.org/abs/1712.00636
arxiv: 1712.00636
doi:
pdf_sha256: 01bb6ecc71def7f5de1f42982710698d803d98b7ffe3cb19d896e98fa184702d
tags: [video, action-recognition, compressed-domain, motion-vectors, video-codec, temporal-redundancy, representation-learning, efficiency]
status: read
read: 2026-11-25
---

# Compressed Video Action Recognition

<!-- Read from arXiv v2 (2018-03-29), 14 PDF pages: pp. 1-8 main text, pp. 9-10 references, p. 11 Apps. A-C, pp. 12-14 App. D confusion matrices. -->

## TL;DR

CoViAR trains CNNs directly on MPEG-4 streams: a ResNet-152 on I-frames, ResNet-18s on P-frames'
motion vectors and residuals, each traced back to the I-frame, scores summed. With ImageNet pretraining
only it reaches 90.4% UCF-101 and 59.1% HMDB-51 at 4.2 GFLOPs (4.6× fewer than Res3D) without flow.

## Problem

Video has low information density (1 h of 720p is ~222 GB raw, ~1 GB compressed) and RGB-only
CNNs struggle to learn temporal structure, so the best models add precomputed optical flow, which is
slow (Sec. 1, p. 1). Earlier compressed-domain work used codec signals only as hand-crafted features,
or still decoded the full RGB stream (MV-CNN distils from a flow network) (Sec. 2.2, p. 3).

## Method

- **Codec background** (Sec. 2.2, Eq. 1, pp. 2-3): an I-frame is a normal image; a P-frame stores
  block motion vectors T(t) and a residual Δ(t), with I(t)_i = I(t-1)_{i - T(t)_i} + Δ(t)_i. B-frames
  are left out: the paper uses only I- and P-frames (backward references).
- **Back-tracing** (Sec. 3, Eq. 2, Fig. 3, p. 4): every P-frame's motion vectors are traced back to
  the reference I-frame and residuals accumulated along the way, giving accumulated motion D(t) and
  residual R(t) with I(t)_i = I(0)_{i - D(t)_i} + R(t)_i. Each P-frame then depends only on the
  I-frame, so P-frames are processed in parallel; it is computed in linear time while decoding.
  The accumulated signals are smoother and carry longer-range motion (Fig. 2, p. 3).
- **Network** (Fig. 5, p. 4): three CNNs, φ_RGB on I(0), φ_motion on D(t), φ_residual on R(t);
  predictions are fused by summing class scores (late fusion). Most information is in I-frames, so
  they get ResNet-152 and the P-frame streams get ResNet-18 (both pre-activation) (Sec. 3, p. 4).
  Temporal Segments (TSN, k = 3 segments, features averaged) give video-level predictions (p. 4).
- **Training** (Sec. 4, p. 5): ImageNet (ILSVRC-2012) pretraining, Adam, batch 40, MPEG-4 videos with
  ~11 P-frames per I-frame. Test: 25 uniformly sampled frames, flips plus 5 crops, scores averaged.
- **Tokens per frame and redundancy (reader's framing for the KB lens):** CoViAR has no tokens; per
  sampled frame it produces one CNN feature/score from a heavy keyframe network or two light delta
  networks. Temporal redundancy is removed by the codec itself (only changes are stored), and the
  model exploits it by spending most compute on the rare I-frames.

## Assumptions

- A1: Videos are re-encoded as MPEG-4 with only I- and P-frames (no B-frames), about 11 P-frames per I-frame on average (Sec. 2.2, p. 3, Sec. 4, p. 5).
- A2: Backbones are pretrained on ImageNet ILSVRC-2012 only; no Kinetics pretraining (Sec. 4, p. 5, Sec. 4.3, p. 6).
- A3: The task is action recognition: one label per trimmed clip under 10 s (UCF-101, HMDB-51) or per-frame labels on ~30 s untrimmed videos (Charades) (Sec. 4, p. 5).
- A4: Test predictions average 25 uniformly sampled frames, each with flips and 5 crops (Sec. 4, p. 5).
- A5: Training uses Temporal Segments with k = 3 segments (Sec. 3, p. 4).
- A6: The videos are natural camera footage, where codec motion vectors roughly resemble optical flow (reader's inference, Sec. 3, Fig. 2, p. 3).

## Claims

- C1: Fusing I-frame, motion-vector and residual streams (I+M+R) averages 90.4% on UCF-101 and 59.1% on HMDB-51 over 3 splits, vs 87.7% and 53.3% for the I-frame RGB stream alone (+2.7 and +5.8 points) [A1, A2, A3, A4, A5, A6] (Table 1, Sec. 4.1, p. 5).
- C2: Residuals alone reach 80.9% (UCF-101) and 44.1% (HMDB-51) and motion vectors alone 65.0% and 38.8%, yet adding either to the I-frame stream raises accuracy (I+M 89.9% and 58.9%) [A1, A2, A3, A4, A5, A6] (Table 1, Sec. 4.1, p. 5).
- C3: Back-traced (accumulated) signals beat the original ones on UCF-101 split 1: motion vectors 58.3% to 63.9%, residuals 79.0% to 79.9%, full model 90.4% to 90.8% [A1, A2, A3, A4, A5] (Table 2, p. 5).
- C4: CoViAR costs 4.2 GFLOPs averaged over frames vs 11.3 for ResNet-152 and 19.3 for Res3D (2.7× and 4.6× fewer), while scoring 90.4%/59.1% vs Res3D's 85.8%/54.9% on UCF-101/HMDB-51 [A1, A2, A3] (Table 3, Sec. 4.2, p. 6).
- C5: Preprocessing (accumulated motion and residuals on CPU) takes 2.87 ms per frame single-threaded and 0.46 ms multi-threaded, vs 75.0 ms for TV-L1 optical flow on a P100; CNN time is 1.3 ms sequential and 0.3 ms concurrent vs 7.5 and 4.0 ms for a ResNet-152 two-stream network; up to 100× faster overall, about 1,300 frames per second [A1] (Table 4, Sec. 4.2, p. 6).
- C6: Adding a TSN temporal stream on optical flow (BN-Inception, 7 segments) by late fusion reaches 94.9% on UCF-101 and 70.2% on HMDB-51, above every listed flow-based method [A1, A2, A3, A4] (Table 5, Table 6, p. 6, p. 7).
- C7: Among methods without optical flow, CoViAR's 90.4%/59.1% is above all listed ones (e.g. P3D ResNet 88.6% UCF-101, Res3D 54.9% HMDB-51) under the same ImageNet pretraining [A1, A2, A3, A4] (Table 6, Sec. 4.3, p. 7).
- C8: On Charades, CoViAR reaches 21.9 mAP and 29.4 wAP without flow and 24.1 mAP and 32.3 wAP with flow, above the listed methods, using only action labels [A1, A2, A3, A4] (Table 7, Sec. 4.3, p. 7).
- C9: A Conv-LSTM over the original (non-accumulated) P-frame chain scores 89.1% on UCF-101 split 1, below CoViAR's 90.8% but above RGB-only 88.4%; adding a skip connection drops it to 87.8% [A1, A2, A3] (App. A, Table 8, p. 11).
- C10: Late fusion (summed scores) beats feature-level fusion on UCF-101 split 1: 90.8% vs max 87.9%, mean 88.1%, multiplicative 87.8% and concatenation 89.7% [A1, A2, A3] (App. B, Table 9, p. 11).
- C11: Without temporal segments, CoViAR still scores 88.9% on UCF-101 split 1 (I 84.7%, M 63.4%, R 76.6%) [A1, A2, A3, A4] (App. C, Table 10, p. 11).
- C12: In t-SNE, two "Jumping Jack" videos are separated in RGB space but overlap in motion-vector and residual space, and motion-vector trajectories form circles where RGB trajectories retrace one path (qualitative) [A1, A6] (Fig. 7, p. 8, Sec. 4.1, p. 5).

## Implementation notes

- Frames resized to 340 × 256 (UCF-101, HMDB-51) or 256 × 256 (Charades), random 224 × 224 crops and colour jitter (source: Sec. 4, p. 5).
- Learning rate 0.001 (UCF-101, HMDB-51) or 0.03 (Charades), divided by 10 on plateau; pretrained layers use a 100× smaller rate; hyper-parameters chosen on other splits (source: Sec. 4, p. 5).
- Optical-flow baselines use TV-L1 flow (source: Sec. 4, p. 5).
- Codec, encoder settings (GOP length, macroblock size, motion-search range) and how motion vectors are normalised before the CNN are not stated; a 16 × 16 macroblock grid is the MPEG-4 default (source: assumption).
- How fused score weights are set (equal or tuned) is not stated; "simple summing of scores" suggests equal (source: assumption, Sec. 3, p. 4).
- Back-tracing must follow each block's reference pointer chain to the I-frame; with no B-frames that chain is strictly backward (source: Sec. 3, Eq. 2, p. 4).
- For a game-frame policy (the KB's lens): game video can be encoded by a hardware H.264/HEVC encoder at capture time, which yields motion vectors and residuals with no extra model; an action chunk of 18 frames would span about 1.5 of the paper's 12-frame GOPs, so the GOP could be aligned to the chunk (one I-frame per chunk) (source: assumption).

## Limitations / open questions

- Only I- and P-frames are modelled; arbitrary decoding orders with B-frames are out of scope (Sec. 2.2, p. 3).
- Motion vectors alone lack spatial detail and score poorly (65.0% UCF-101) (Table 1, Sec. 4.1, p. 5).
- The Conv-LSTM over the P-frame chain lost to the decoupled model; the authors attribute it to the long dependency chain (Sec. 3, p. 3, App. A, p. 11).
- Pretraining on Kinetics would score higher; the comparison is restricted to ImageNet pretraining (Sec. 4.3, p. 6).
- Open question: codec motion vectors optimise bitrate, not true motion. In shmups, bullets are often smaller than a 16 × 16 macroblock and the background scrolls; do motion vectors capture bullets, or only the scroll, leaving bullets to the residual?
- Open question: CoViAR's memory is one GOP. Nothing here carries strategy (weapon level, economy, rank) beyond ~12 frames; the keyframe-plus-delta idea compresses the frame stream feeding a policy, but long-term state still needs a separate memory (see ryoo2023turing).
- Open question: accumulated residuals grow with distance from the I-frame (Fig. 2, p. 3); for a long chunk, does a delta encoder degrade towards the chunk's end?
- Open question: game RAM can be used offline (never as policy input) to check what the motion stream encodes, e.g. by regressing RAM object positions and velocities from motion-vector features.

## Related

- related [[delfosse2023continuity]]: also uses motion (optical flow) as a cheap signal, for object discovery in Atari frames.
- related [[magne2026nitrogen]]: a single-frame game policy that found more past frames gave no benefit; CoViAR's I/P split is one cheap way to add motion context.
- related [[hafner2025scalable]]: a learned causal video tokenizer for game frames, in contrast to CoViAR's codec-given deltas.
- related [[ryoo2023turing]]: a bounded token memory over time, complementary to CoViAR's per-GOP compression of the input.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
