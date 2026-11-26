---
id: kipf2022conditional
title: Conditional Object-Centric Learning from Video
authors: [Thomas Kipf, Gamaleldin F. Elsayed, Aravindh Mahendran, Austin Stone, Sara Sabour, Georg Heigold, Rico Jonschkowski, Alexey Dosovitskiy, Klaus Greff]
year: 2022
venue: ICLR 2022
url: https://arxiv.org/abs/2111.12594
arxiv: 2111.12594
doi:
pdf_sha256: ac7cae9602bf0e3b27996bdb7ed82562853f6bdb13c390b70c7ec7f8739ed689
tags: [object-centric, slot-attention, video, tracking, optical-flow, weak-supervision, segmentation]
status: read
read: 2026-11-25
---

# Conditional Object-Centric Learning from Video

Version read: arXiv v2 (15 Mar 2022, "Published as a conference paper at ICLR 2022"), 32 pages
including the appendix. Its name in the paper is SAVi (Slot Attention for Video).

## TL;DR

SAVi runs Slot Attention per video frame plus a slot-wise transformer predictor, so slots track
objects; slots can start from first-frame hints (box or point) and train on optical flow alone.
On realistic MOVi++ this gives 78.3% FG-ARI with centre-of-mass hints.

## Problem

Unsupervised object-centric models (Slot Attention, IODINE, SCALOR, SIMONe) work on toy data and
fail on complex textures; the notion of an object is also task-dependent, so one wants a way to
steer the granularity during training and to query the model at inference (Sec. 1, pp. 1-2).

## Method

- Architecture (Fig. 1, p. 2; Sec. 2, pp. 2-3): per frame, a small CNN encoder with position
  encoding gives N feature vectors; K slots S_t are refined by the corrector (Slot Attention,
  softmax over slots, GRU update; Eq. 1, p. 3; one iteration, or two on CATER, App. A.6, p. 28);
  the predictor, a transformer encoder over slots (self-attention + MLP, Eq. 2, p. 3), maps the
  corrected slots to the initial slots of frame t+1. Both are permutation equivariant.
- Decoder: a slot-wise spatial broadcast decoder gives per-slot RGB-coded optical flow (or RGB)
  plus an alpha mask, softmaxed across slots and used to mix per-slot outputs (Eq. 3, p. 3). The loss
  is squared error to the optical flow (or frame), summed over time and pixels (Eq. 4, p. 4).
- Initialiser (Sec. 2, p. 3; App. A.6, p. 29): conditional slots are encoded from first-frame
  hints (bounding box or centre of mass through an MLP; segmentation mask through a CNN); slots
  without a hint get a fixed value, so all unconditioned slots share one initialisation.
  Unconditional variants sample from a Gaussian or use learned initial slots.
- Training uses 6-frame subsequences with the hint on the first frame (Sec. 4, p. 5). Flow
  comes from the simulator/renderer, or from an unsupervised estimator (SMURF) in an ablation (p. 8).
- Evaluation: FG-ARI (clusters per object across the whole video, foreground only) and mIoU
  (strict slot-to-object alignment, no matching, missing-object frames count 1.0 or 0.0), skipping
  frame 1 (Sec. 4, p. 5; App. A.8, p. 32).

## Assumptions

- A1: Videos are synthetic, rendered scenes of rigid objects with at most 10 objects (CATER, MOVi, MOVi++); MOVi++ uses real HDR backgrounds and scanned objects (Sec. 4.2, p. 6; Sec. 4.5, p. 9)
- A2: In the conditional setting a first-frame hint (box, centre of mass or mask) for each object to track is available at training and test time (Sec. 4.3, p. 6)
- A3: Training uses a motion signal: ground-truth or estimated optical flow of moving objects (or RGB reconstruction on the simplest data) (Sec. 4.3, p. 8; Sec. 4.5, p. 9)
- A4: Objects in the MOVi datasets all move, and the main experiments use a static camera; the slot count K=11 exceeds the object count (at most 10) (Sec. 4.5, p. 9; App. A.6, p. 29)
- A5: Low resolution inputs (64x64 for MOVi/CATER, 128x128 for MOVi++) and 6-frame training clips (Sec. 4.2, p. 6; Sec. 4, p. 5)
- A6: Metrics skip the first frame and use foreground objects only (FG-ARI) (Sec. 4, p. 5; App. A.8, p. 32)

## Claims

- C1: Unsupervised, with RGB reconstruction on CATER, SAVi (uncond.) gets FG-ARI 92.8 +- 0.8 versus SIMONe 91.8, S-IODINE 66.8, MONet 41.2 and per-frame Slot Attention 7.3 (the last two are per-frame models without temporal consistency); with segmentation-mask conditioning 97.9 [A1, A3, A5, A6] (Table 1a, p. 6, Sec. 4.1, p. 5)
- C2: On MOVi++ unconditional models trained on RGB degenerate: SCALOR reaches 22.7 +- 0.9 FG-ARI and SIMONe 32.7 (reimplementation, about 33 in the text), and SAVi's slots represent fixed image regions (stripes or patches) instead of objects; unconditional SAVi gets 47.6 +- 0.2 with flow (MOVi: SCALOR 81.2, SIMONe 74.8, SAVi 78.2) [A1, A3, A5, A6] (Sec. 4.2, p. 6, Table A.2, p. 26)
- C3: With flow targets and first-frame hints on MOVi++, FG-ARI / mIoU are 70.4 / 43.0 (segmentation hint), 77.4 / 45.9 (bounding box), 78.3 / 43.5 (centre of mass); on MOVi 93.7 / 72.0, 93.7 / 71.2, 93.8 / 72.1; a segmentation-propagation baseline with the same CNN and flow training gets 37.8 / 33.3 (MOVi++) and 61.5 / 49.8 (MOVi) [A1, A2, A3, A5, A6] (Table 1b, p. 6)
- C4: Against VOS baselines with stronger ResNet backbones at higher resolution on MOVi++ mIoU, SAVi is slightly worse at default size (43.0 vs T-VOS 46.4 and CRW 50.9) and a SAVi (ResNet) with bounding boxes reaches 82.8 FG-ARI / 50.7 mIoU, closing the gap to CRW; on MOVi SAVi's 72.0 mIoU beats T-VOS 50.4 and CRW 42.4 [A1, A2, A3, A5, A6] (Table 1b, p. 6, Sec. 4.3, p. 7)
- C5: A very weak hint suffices: centre-of-mass points work about as well as boxes, and Gaussian noise added to the hints (training and test) is tolerated up to a scale of about 20% of the mean object size before mIoU decays faster than FG-ARI; removing conditioning drops performance markedly [A1, A2, A3, A5, A6] (Sec. 4.3, p. 7, Fig. 3, p. 6)
- C6: Optical flow is needed on MOVi++: clustering flow (k-means with centre-of-mass init) gets only 32.6 FG-ARI / 9.3 mIoU, RGB-reconstruction training without flow fails to learn meaningful segmentation there, while estimated flow (SMURF) gives nearly the same benefit as ground-truth flow and flow is unnecessary on the simple MOVi [A1, A3, A5, A6] (Sec. 4.3, p. 8, Table 1b, p. 6, Fig. 3, p. 6)
- C7: Trained on 6-frame clips, SAVi's FG-ARI stays about stable (even increases for unconditional models) over 24-frame videos, while the segmentation-propagation baseline's performance decays over time [A1, A2, A3, A5, A6] (Sec. 4.4, p. 8, Fig. 6a, p. 8)
- C8: Trained on MOVi++ with boxes, SAVi shows no significant drop on unseen objects or backgrounds (FG-ARI 82.0 default, 82.0 unseen objects, 82.2 unseen backgrounds, 80.4 both; mIoU 54.3 / 54.4 / 54.0 / 52.6) and gets 83.7 FG-ARI on MOVi at test time [A1, A2, A3, A5, A6] (Sec. 4.4, p. 8, Table A.1, p. 26)
- C9: Hint granularity steers the decomposition: one box around two fists gives one slot tracking the composite, two boxes give two slots tracking the parts, though the dataset annotates only whole objects (anecdotal; also a laptop screen vs whole laptop) [A1, A2, A3] (Sec. 4.4, pp. 8-9, Fig. 7, p. 9, Fig. A.2, p. 18)
- C10: Architecture ablations on MOVi++ with boxes: no predictor FG-ARI 83.0 but mIoU 44.7 (vs default 82.0 / 54.3), MLP predictor 82.8 / 49.5, softmax over inputs instead of slots in the corrector 78.3 / 11.4, so slot-to-slot interaction and slot-axis competition help hint-object correspondence [A1, A2, A3, A5] (Table A.3, p. 27)
- C11: Using the boxes as a matching-based training target with unconditioned slots (Hungarian matching) gives clearly worse FG-ARI (83.0 learned init, 82.3 Gaussian init) than conditioning (92.9) on MOVi [A1, A2, A3, A5] (Table A.4, p. 27)
- C12: Tracking a subset of objects works with partial conditioning (remaining slots randomly initialised), but the model's ability to bind hints to the right objects decays as more objects must be explained by unconditioned slots; training on up to 6 conditioned objects is a reasonable trade-off [A1, A2, A3] (App. A.4, p. 25, Fig. A.10, p. 25)
- C13: Slots retain some semantic information beyond position and motion: an MLP probe (gradients stopped) predicting colour, shape and material from slots gets 54.6 +- 1.5% accuracy on MOVi with flow training, and 85.7 +- 0.7 with RGB reconstruction training [A1, A2, A3] (App. A.4, pp. 27-28)
- C14: Failure and caveats: with heavy occlusion by a similar-looking object SAVi can lose track and swap identities; on a moving-camera MOVi++ variant FG-ARI is about 65.5 +- 0.5 because flow also encodes background motion; SAVi applied to repeated first frames (no motion) reaches 88.7 +- 0.2 FG-ARI on MOVi++ [A1, A2, A3, A4] (Fig. 5b, p. 7, App. A.3, p. 18, App. A.4, p. 25)
- C15: SCALOR and the segmentation-propagation baseline lose or merge small objects: SCALOR merges small objects that come close and picks up background texture on MOVi++, and segmentation propagation tends to lose small objects or merge them into larger ones [A1, A3] (App. A.3, p. 24, Fig. A.7, Fig. A.8, Fig. A.9, pp. 23-24)

## Implementation notes

- Training: 100k steps (200k for unsupervised video decomposition), batch 64, Adam lr 2e-4, gradient clipping 0.05, 2.5k-step linear warmup then cosine decay, 11 slots; SAVi (ResNet) on MOVi++ trained 1M steps (Sec. 4, p. 5; App. A.6, p. 29)
- 12 h (64x64) or 30 h (128x128) on 8 V100s with bounding box conditioning; trained in JAX/Flax on TPU v3 (Sec. 4, p. 5; App. A.6, p. 30)
- Encoder: four 5x5 conv layers (32 channels, 64 and stride 2 for MOVi++), position embedding, LayerNorm, two 1x1 convs (Table A.5, p. 28); corrector projection size D=128, one iteration (two on CATER), optional MLP of 256 hidden units only with multiple iterations (App. A.6, p. 28)
- Predictor: 4-head self-attention, size 128, MLP 256 hidden, ReLU; identity predictor on CATER (App. A.6, pp. 28-29)
- Decoder: 8x8 spatial broadcast, four ConvTranspose 5x5 layers, outputs mask logits and flow/RGB via 1x1 convs (Table A.6, p. 29)
- Initialiser: MLP with 256 hidden units to size 128 for boxes ([ymin, xmin, ymax, xmax]) and centre points ([y, x]), -1 for unconditioned slots; mask conditioning uses a CNN (Table A.7, p. 30; App. A.6, p. 29)
- Datasets: Kubric-generated MOVi and MOVi++ (9000 train, 1000 validation videos of 24 frames at 12 fps; 64x64 and 128x128), MOVi++ has 380 HDR backgrounds and 1028 scanned objects; CATER with masks from the SIMONe authors (Sec. 4.2, p. 6; App. A.5, p. 28)
- Authors' code: project page https://slot-attention-video.github.io/ (p. 1); not consulted by this reader, and the exact SAVi code release is an assumption of the reader.

## Limitations / open questions

- Training assumes optical flow (not available for real video); estimated flow from SMURF is a partial remedy (Sec. 4.5, p. 9)
- Only rigid objects with simple physics and, in MOVi, only moving objects; flow gives indirect foreground-background supervision, and static objects are a problem when training only on flow (Sec. 4.5, p. 9)
- Real-world video is far more complex; a Sketchy robotic grasping qualitative result is shown only unsupervised with RGB reconstruction (Sec. 4.5, p. 9; App. A.2, pp. 16-17)
- The corrector's attention masks have artifacts for large objects (limited encoder receptive field) (App. A.3, p. 18)
- Open question (reader's lens, a universal object space for shmup frames): SAVi's slot-object binding is set by the first-frame hint (box or point), so a game-agnostic pre-encoder needs a source of hints per game (a detector, SAM-style prompts [[ravi2024sam]], or RAM-derived boxes offline as in [[delfosse2024ocatari]], never read by the agent at run time). The paper measures hints at about 10 objects, not at dozens of bullets.
- Open question: unhinted slots are randomly initialised in the C12 subset experiments (App. A.4, p. 25), so they are unordered and C12 shows binding decays as their number grows; a bullet stream is the extreme case. Is a per-class slot set (player, enemies, bullets as one aggregate) workable?
- Open question: flow supervision needs motion; scrolling shmup backgrounds make flow encode the camera (cf. moving-camera result C14), and the paper does not test scrolling backgrounds.
- Open question: whether slots from one game's art style align with another's is untested; the paper's generalisation (C8) is to new objects and backgrounds within one renderer, not to a new visual style.

## Related

- extends [[locatello2020slot]]: adds recurrent slot initialisation from the previous frame via a transformer predictor, optical-flow targets and conditional first-frame slots
- related [[delfosse2023continuity]]: also adds motion and continuity cues to object discovery on Atari frames
- related [[delfosse2024ocatari]]: RAM-derived object boxes, a possible offline source of conditioning hints and of validation
- related [[lin2020space]]: SPACE, an unsupervised object decomposition model for single frames
- related [[ravi2024sam]]: SAM 2, supervised promptable video segmentation with a memory bank, achieving similar prompt-then-track behaviour with strong supervision
- related [[yoon2023investigation]]: object-centric slot representations used for RL

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
