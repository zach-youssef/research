---
id: ravi2024sam
title: "SAM 2: Segment Anything in Images and Videos"
authors: [Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, Christoph Feichtenhofer]
year: 2024
venue: arXiv preprint
url: https://arxiv.org/abs/2408.00714
arxiv: 2408.00714
doi:
pdf_sha256: 664f5c4254db441569ecfe36a2a2fa91a1ecec5f4d31b3bcc5d451193efc2c0d
tags: [segmentation, video, foundation-model, promptable, memory, tracking, dataset, zero-shot, state-representation]
status: read
read: 2026-11-20
---

# SAM 2: Segment Anything in Images and Videos

## TL;DR

Extends SAM to video: a streaming Hiera encoder plus a memory bank of past-frame features and
object pointers lets a prompted object be segmented and tracked (a "masklet") across frames.
Trained with a model-in-the-loop data engine (SA-V, 35.5M masks); beats SAM+tracker baselines.

## Problem

SAM segments images, but video adds motion, deformation, occlusion, lighting changes and lower
image quality, and needs efficient processing of many frames (Sec. 1, p. 1). The paper defines
Promptable Visual Segmentation (PVS): clicks, boxes or masks on any frame define a target whose
spatio-temporal mask (masklet) the model must return over the whole video, refinable by more
prompts (Sec. 3, p. 3; App. B, p. 15). Prior interactive pipelines (SAM + a mask tracker) can't
correct a tracker's mistakes except by re-annotating with SAM and restarting (Sec. 2, p. 3).
Like SAM, valid targets are objects or parts with clear boundaries (App. B, p. 15).

## Method

- **Image as one-frame video.** With an empty memory the model behaves like SAM on an image
  (Sec. 1, p. 1).
- **Image encoder.** MAE-pretrained Hiera, hierarchical, run once per frame in streaming fashion,
  producing unconditioned frame tokens (Sec. 4, p. 4). An FPN fuses stride-16/32 features
  (stages 3-4) into a 256×64×64 embedding; stride-4/8 features skip the memory attention and go
  to the decoder's upsampling (App. D.1, p. 16; Fig. 8, p. 17). No relative positional encoding;
  windowed absolute embeddings (App. D.1, p. 16). Sizes T, S, B+ (default), L (App. D.1, p. 16).
- **Memory attention.** L=4 transformer blocks; each does self-attention over the current frame,
  cross-attention to memories and object pointers, then an MLP; vanilla attention so efficient
  kernels apply (Sec. 4, p. 5; App. D.1, p. 16). 2d-RoPE plus sinusoidal absolute embeddings;
  object pointers are excluded from RoPE (App. D.1, p. 16).
- **Prompt encoder and mask decoder.** As in SAM (two-way transformer); several masks for
  ambiguous prompts, and in video the highest-predicted-IoU mask is propagated if prompts don't
  resolve it (Sec. 4, p. 5; App. D.1, p. 17). A new occlusion token and MLP score whether the
  object is visible in the frame (Sec. 4, p. 5; App. D.1, p. 16; Fig. 8, p. 17).
- **Memory encoder.** The predicted mask is conv-downsampled, summed with the frame's
  unconditioned embedding and fused by light conv layers; no separate encoder (Sec. 4, p. 5;
  App. D.1, p. 17). Memory features are projected to 64 dims (App. D.1, p. 17).
- **Memory bank.** FIFO of up to N recent frames and FIFO of up to M prompted frames, both as
  spatial maps, plus object pointers (the decoder's mask output token per frame, split into 4
  tokens of 64 dims). Temporal position embeddings only on the N recent frames. A learned
  occlusion embedding is added to memories of frames predicted occluded (Sec. 4, p. 5;
  App. D.1, pp. 16-17). Default N=6 past frames (App. A.2.1, p. 14).
- **Multiple objects** are processed independently: encoder features are shared, but memory bank
  and decoder run once per object (App. D.1, p. 17).
- **Training.** Pre-train on SA-1B images, then train jointly on images and video, alternating
  full batches from one source (App. D.2, pp. 17-18). Simulated interaction on 8-frame sequences,
  up to 2 prompted frames; first prompt is the GT mask (p 0.5), a positive click (0.25) or a box
  (0.25), then corrective clicks (Sec. 4, p. 6; App. D.2.2, p. 18). Loss: focal + dice for masks,
  MAE for IoU, cross-entropy for occlusion, weighted 20:1:1:1; only the lowest-loss mask of a
  multi-mask output is supervised (App. D.2.2, p. 18).
- **Data engine.** Phase 1: SAM per frame at 6 FPS; Phase 2: SAM plus a mask-only SAM 2 for
  propagation; Phase 3: full SAM 2 in the loop (Sec. 5.1, p. 6). Later, auto masklets from grid
  prompts on the first frame, verified by annotators (Sec. 5.2, p. 7; App. E.1, p. 20). SA-V: 50.9K
  videos (avg. 13.8 s, 240p to 4K, 47 countries), 190.9K manual + 451.7K auto masklets
  (Sec. 5.2, p. 7; App. E.1, p. 20). Masklets have no class labels (App. H.2, p. 33).

## Assumptions

- A1: Video is consumed as a stream of frames, one at a time; an image is treated as a one-frame video with an empty memory (Sec. 1, p. 1, Sec. 4, p. 4).
- A2: The main model resizes frames to a 1024×1024 square (Sec. 6, p. 9, App. D.2.1, p. 17).
- A3: The image encoder is an MAE-pretrained hierarchical Hiera whose stride-16/32 features are fused by an FPN into a 256×64×64 embedding, with stride-4/8 features skipping memory attention into the decoder (Sec. 4, p. 4, App. D.1, p. 16).
- A4: The encoder is Hiera-B+ unless another size is stated (Sec. 6, p. 9, App. A.2.1, p. 14).
- A5: A target is defined by clicks, boxes or masks on frames, only prompted objects are segmented and tracked, and valid targets are objects or parts with a clearly defined visual boundary (Sec. 3, pp. 3-4, App. B, p. 15).
- A6: Multiple objects are processed independently: encoder features are shared, but memory bank and decoder run per object, with no inter-object communication (App. C, p. 16, App. D.1, p. 17).
- A7: Each frame's features are conditioned by memory attention on a memory bank of spatial memory features of recent frames (with temporal position embeddings) and of up to M prompted frames (Sec. 4, p. 5, App. D.1, p. 17).
- A8: The memory keeps N = 6 recent frames by default (App. A.2.1, p. 14).
- A9: Memory attention also cross-attends to object pointers, the decoder's mask output tokens from other frames (Sec. 4, p. 5, App. D.1, pp. 16-17).
- A10: Training simulates interaction: up to 2 prompted frames per sequence, the first prompt a ground-truth mask (p 0.5), a positive click (0.25) or a box (0.25), then corrective clicks from the ground truth, with at most 3 masklets per sequence (Sec. 4, p. 6, App. D.2.2, p. 18).
- A11: Training sequences are 8 frames long, plus a 16-frame fine-tune on the most-edited masklets, shorter than the videos seen at inference (App. D.2.2, p. 18).
- A12: The main model is pre-trained on SA-1B, then trained on SA-V manual masklets, the licensed Internal videos, a 10% subset of SA-1B and DAVIS, MOSE and YouTubeVOS (the released model omits the open VOS sets) (Sec. 6, p. 9, App. D.2.2, p. 18).
- A13: SA-V videos are "in-the-wild" everyday footage captured by crowdworkers in 47 countries (54% indoor), 240p to 4K, 4 s to 2.3 min (Sec. 5.2, p. 7, App. E.1, p. 20).
- A14: Masklets are class-agnostic, covering whole objects and parts with no semantic constraints and no class labels (Sec. 5, p. 6, App. H.2, p. 33).
- A15: The data engine relies on human annotators with the model in the loop and separate verifiers who send back incorrect masklets and reject ones on objects that are not well defined; auto masklets come from grid prompts on the first frame and enter SA-V only when verified; SA-V val/test and Internal-test were annotated with per-frame SAM (Phase 1) (Sec. 5.1, pp. 6-7, Sec. 5.2, p. 9, App. E.1, p. 20).
- A16: Clicks are simulated from ground truth: the first at the mask center (the pixel farthest from the boundary), later ones at the center of the error region (App. F.1.2-F.1.3, p. 23).
- A17: Interactive evaluation uses 3 clicks per prompted frame for up to 8 frames, choosing the worst-IoU frame each pass (offline) or pausing at frames with IoU < 0.75 (online), on 9 densely annotated datasets, each object independently (Sec. 6.1, p. 9, App. F.1.2, pp. 22-23).
- A18: Semi-supervised VOS gives prompts (clicks, box or ground-truth mask) on the first frame only (Sec. 6.2, p. 10, Sec. 7, p. 10).
- A19: The 17 zero-shot video datasets span surgery, robot-camera, long, open-vocabulary, deformation, parts, synthetic driving (Virtual KITTI 2, rendered by a game engine), panoptic, wildfire, egocentric, driving, wildlife and microscopy video, preprocessed so every object has a mask on its video's first frame (App. F.1.1, p. 22, App. F.1.3, p. 24, Table 16, p. 30).
- A20: The image suite is SAM's 23 datasets plus frames sampled from 14 video datasets, scored as single images; OVIS is not strictly zero-shot since its videos are in MOSE (Sec. 6.3, p. 10, App. F.3.1, p. 27, p. 28).
- A21: The video baselines are SAM chained with a mask tracker (XMem++ or Cutie), SAM turning clicks and boxes into masks; with mask prompts the trackers run without SAM (Sec. 6.1, pp. 9-10, App. F.1.3, p. 24).
- A22: Speed is measured on one A100 with bf16 autocast and a torch.compile'd image encoder, at batch size 1 for video and 10 for images (Sec. 7, p. 10, App. D.3, p. 19).
- A23: Data ablations pre-train on SA-1B and fix the iteration count and batch size, changing only the training data (App. A.1, pp. 12-13).
- A24: Ablations use 512² inputs unless resolution is the varied factor (App. A, p. 12, App. A.2, p. 13).
- A25: Model ablations train on SA-V manual plus a 10% subset of SA-1B (App. A, p. 12).
- A26: Ablations are scored by 3-click first-frame J&F on SA-V val, Internal-test, MOSE dev (200 MOSE training videos held out) and the 9 zero-shot datasets, and by 1-click mIoU on SA-23 (App. A, p. 12).
- A27: Fairness is measured on 20-second third-person Ego-Exo4D clips with the subjects' self-reported gender and age, annotated like SA-V val/test (App. E.1.1, p. 20).
- A28: Training runs on 256 A100 GPUs, and the image encoder is frozen during the 16-frame fine-tune to fit the sequence into the 80 GB A100 memory (App. D.2.2, p. 18, Table 18, p. 32).
- A29: The phase comparison is a controlled experiment: three annotator subsets annotate the same 452 masklets in 169 videos, with Phase 1 masks as the quality reference (Sec. 5.1, p. 7, App. E.2.2, p. 22).
- A30: Neither training nor evaluation data contain non-photorealistic game imagery (sprites, pixel art, HUD overlays); the only game-engine data is photorealistic Virtual KITTI 2 (reader's inference, Sec. 5.2, p. 7, App. F.1.1, p. 22).

## Claims

- C1: SA-V has 35.5M masks on 50.9K videos, 53x more masks than any existing video segmentation dataset, and 642.6K masklets with auto ones included [A13, A14, A15] (Sec. 1, p. 2, Table 3, p. 9).
- C2: With SAM 2 in the loop (Phase 3) annotation takes 4.5 s per frame vs. 37.8 s with per-frame SAM (Phase 1), 8.4x faster, with 89.1% of masks agreeing with Phase 1 at IoU above 0.75 [A14, A15, A29] (Table 1, p. 7, Sec. 5.1, p. 6).
- C3: In interactive offline and online PVS over 9 densely annotated zero-shot datasets, SAM 2 beats SAM+XMem++ and SAM+Cutie on every dataset, reaching better accuracy with over 3x fewer interactions [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A16, A17, A19, A21, A30] (Fig. 5, p. 9, Sec. 6.1, p. 10, App. F.1.2, p. 23).
- C4: Averaged over 8 interacted frames with 3 clicks each, SAM 2 scores 80.3 J&F offline and 79.8 online vs. 74.7 and 74.0 for SAM+Cutie [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A16, A17, A19, A21, A30] (Fig. 12, p. 24, Fig. 13, p. 25).
- C5: In semi-supervised VOS over 17 zero-shot video datasets, SAM 2 averages 64.7, 75.3, 77.6, 74.4 and 79.3 J&F for 1, 3 and 5 clicks, a box and a GT mask on the first frame, vs. 56.7, 70.1, 72.2, 69.4 and 74.1 for SAM+Cutie [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A16, A18, A19, A21, A30] (Table 4, p. 10).
- C6: On the image SA task, SAM 2 trained on SA-1B alone gets 58.9 1-click mIoU on SAM's 23 datasets vs. SAM's 58.1, at 130.1 vs. 21.7 FPS (6x faster); trained on the image+video mix it reaches 61.9, and 69.6 vs. SAM's 59.1 on 14 new video-frame datasets [A1, A2, A3, A4, A5, A14, A16, A20, A22, A30] (Table 5, p. 11, Sec. 6.3, p. 10).
- C7: On semi-supervised VOS benchmarks SAM 2 (Hiera-B+/Hiera-L) scores 76.6/77.9 on MOSE val, 90.2/90.7 on DAVIS17 val and 76.8/77.9 on SA-V val, vs. 71.7, 88.1 and 61.3 for Cutie-base+ [A1, A2, A3, A5, A6, A7, A8, A9, A10, A11, A12, A18, A30] (Table 6, p. 11).
- C8: SAM 2 runs at 43.8 FPS (Hiera-B+) and 30.2 FPS (Hiera-L) on one A100 at batch size 1, which the authors call real-time [A1, A2, A3, A6, A7, A8, A9, A22] (Sec. 7, p. 10).
- C9: Training only on existing VOS datasets does well on in-domain MOSE dev but poorly elsewhere (48.1 J&F on SA-V val, 59.7 on 9 zero-shot); adding data-engine data gives +12.1% on 9 zero-shot, and mixing all data is best [A1, A3, A4, A5, A7, A8, A9, A10, A11, A23, A24, A26, A30] (Table 7, p. 12, App. A.1, pp. 12-13).
- C10: Video segmentation accuracy follows a consistent power law in the number of SA-V training masklets on SA-V val, 9 zero-shot datasets and MOSE dev [A1, A3, A4, A5, A7, A8, A9, A10, A11, A13, A14, A23, A24, A26, A30] (Fig. 6, p. 13, App. A.1, p. 13).
- C11: Training on the 50k most-edited SA-V masklets beats 50k random ones (66.2 vs. 63.7 J&F on SA-V val) but is worse than all 190k (69.9) [A1, A3, A4, A5, A7, A8, A9, A10, A11, A13, A15, A23, A24, A26, A30] (Table 8, p. 13).
- C12: Training on 8-frame instead of 4-frame sequences raises SA-V val J&F from 60.0 to 68.3 at equal speed; resolution 1024 vs. 512 improves MOSE dev (77.0 vs. 73.0) and SA-23 (61.5 vs. 59.7) at 0.22x speed [A1, A3, A4, A5, A7, A8, A9, A10, A25, A26, A30] (Table 9, p. 14, App. A.2.1, p. 13).
- C13: More memories (N) generally help with some variance; 64-dim memory channels cost little accuracy vs. 256 and make memory storage 4x smaller [A1, A3, A4, A5, A7, A9, A10, A11, A24, A25, A26, A30] (Table 9, p. 14, App. A.2.1, p. 14).
- C14: Scaling the image encoder generally improves image and video metrics, and scaling memory attention generally improves video metrics only; there are exceptions in the table (Hiera-L lowers SA-V val to 66.3 from B+'s 68.3) [A1, A3, A5, A7, A8, A9, A10, A11, A24, A25, A26, A30] (Table 9, p. 14, App. A.2.1, p. 14).
- C15: Cross-attending to object pointers leaves the 9 zero-shot average unchanged (70.7 vs. 70.9) but raises SA-V val from 64.5 to 68.3 and LVOSv2 val from 67.0 to 71.6; a GRU memory gives no improvement except slightly on LVOSv2 [A1, A3, A4, A5, A7, A8, A10, A11, A24, A25, A26, A30] (Table 11, p. 15, App. A.2.3, p. 15).
- C16: Removing all relative positional biases from the image encoder causes no regression on SA-23 and minimal regression on video while enabling FlashAttention-2, and 2d-RoPE in memory attention is beneficial [A1, A3, A4, A5, A7, A8, A9, A10, A11, A24, A25, A26, A30] (Table 10, p. 14, App. A.2.2, p. 14).
- C17: SAM 2 (Hiera-B+) surpasses SAM on 29 of 37 image-task datasets in 1-click mIoU, despite a smaller encoder [A1, A2, A3, A4, A5, A12, A14, A16, A20, A30] (App. F.3.2, p. 28, Fig. 15, p. 29).
- C18: On Virtual KITTI 2 (synthetic driving rendered by a game engine) SAM 2 averages 74.1 J&F in interactive offline evaluation vs. 70.3 for SAM+Cutie, and gains 4.6 1-click mIoU over SAM on its frames [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A16, A17, A19, A20, A21, A30] (Table 16, p. 30, Fig. 12, p. 24, Fig. 15, p. 29).
- C19: A variant trained only on SA-1B, SA-V and Internal data (no open VOS sets) scores higher on SA-V test (79.5 vs. 78.4 for Hiera-L) but lower on MOSE val (74.6 vs. 77.9) [A1, A2, A3, A5, A6, A7, A8, A9, A10, A11, A18, A30] (Table 17, p. 31).
- C20: For people in Ego-Exo4D, 3-click and GT-mask accuracy shows minimal gender or age gap; the 1-click gap (81.9 male vs. 75.1 female J&F) shrinks to 94.3 vs. 92.7 on clips where the person is segmented, attributed to part-vs-person prompt ambiguity [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A16, A18, A27, A30] (Table 13, p. 20, App. E.1.1, p. 20).
- C21: The released SAM 2 was trained on 256 A100 GPUs for 108 hours, about 12,165 kWh and 3.89 t CO2e [A28] (Table 18, p. 32).

## Implementation notes

- Results are for "SAM 2.1", an improved version of the initial release; code at https://github.com/facebookresearch/sam2, Apache 2.0; SA-V under CC BY 4.0 (source: p. 2 footnote 1, Table 18, p. 32, App. H.2, p. 33).
- Pre-training: SA-1B, ~90k steps, 1024² input, bf16, AdamW (0.9, 0.999), lr 4e-4 reciprocal-sqrt schedule, 1k warmup and 5k cooldown, weight decay 0.1, batch 256, drop path 0.1-0.3 and layer decay 0.8-0.925 by size, up to 64 masks per image, 7 correction clicks, masks covering >90% of the image filtered (source: Table 12, p. 19, App. D.2.1, p. 17).
- Full training: ~150k steps, lr 6e-5 for the image encoder and 3e-4 otherwise, cosine, 7.5k warmup, batch 256, max 3 masklets per 8-frame video sequence, 50% time reversal, 10% mosaic 2x2 tiling of the same video; data mix ~15.2% SA-1B, ~70% SA-V, ~14.8% Internal, only manual SA-V masklets (source: Table 12, p. 19, App. D.2.2, p. 18).
- 16-frame fine-tune: 50k iterations at half lr on the 50% most-edited masklets, image encoder frozen to fit 80 GB A100 memory (source: App. D.2.2, p. 18).
- Global attention blocks per encoder size: 5-7-9 (T), 7-10-13 (S), 12-16-20 (B+), 23-33-43 (L) (source: Table 12, p. 19).
- The default M (prompted-frame memories) isn't stated in the paper; check the sam2 config files (source: assumption, Sec. 4, p. 5).
- Auto masklets: a 32×32 point grid on the first frame, plus a 16×16 grid on each of 4 zoomed crops (2×2 overlapped window) and a 4×4 grid on each of 16 zoomed crops (4×4 overlapped window); components and holes under 200 px removed or filled (source: App. E.1, p. 20).
- Ablations use 512² input, SA-V manual plus 10% of SA-1B, 200k iterations, batch 128 (source: App. A, p. 12, App. A.1, p. 12).
- Speed: one A100, PyTorch 2.3.1, CUDA 12.1, bf16 autocast, torch.compile on the image encoder; image FPS at batch 10, video at batch 1 (source: App. D.3, p. 19).
- Inconsistency: the text says the data mix lifts SA-23 to 61.4% (Sec. 6.3, p. 10, App. F.3.2, p. 27), while Tables 5 and 15 give 61.9; the note uses the table (source: Table 5, p. 11).
- Inconsistency: App. F.3.2 says SAM 2 beats SAM on 29 datasets by up to 53.9 mIoU, but Fig. 15 shows 31 positive deltas with a maximum of 41.1; C17 follows the text (source: Fig. 15, p. 29, App. F.3.2, p. 28).
- Memory footprint per object of a game-frame state is not reported; the reader estimates 6 recent + prompted spatial maps of 64×64×64 plus pointers (source: assumption from App. D.1, pp. 16-17).

## Limitations / open questions

- Authors' limitations: may fail across shot changes; can lose or confuse objects in crowded
  scenes, after long occlusions or in long videos; struggles with thin or fast-moving details and
  similar-looking nearby objects; no inter-object communication; the data engine still needs
  human annotators (App. C, p. 16).
- SAM 2 tends to segment parts on a first click where benchmarks want whole objects (App. F.2,
  p. 27; App. E.1.1, p. 20).
- Not scaled beyond Hiera-L (App. F.3.2, p. 28).
- Lens answer (to the question on [[kirillov2023segment]]): SAM 2 does fix SAM's missing
  identity across frames, since each prompted object keeps one masklet with an occlusion score
  and a pointer vector. But it tracks only objects it was prompted with, one at a time; there is
  no automatic all-object segmentation of every frame (auto masklets only prompt a grid on the
  first frame of the video) (App. D.1, p. 17; App. E.1, p. 20). As a policy state it would be a
  base (per-object pointer tokens plus mask features), not a ready encoding; whether that beats
  the raw Hiera embedding is untested.
- Open question (lens): temporal context is 6 recent memories plus prompted frames, trained on
  8-16 frame clips; new objects entering mid-game would need re-prompting (reader inference
  from Sec. 4, p. 5, App. D.2.2, p. 18).
- Open question (lens): cost scales with object count (memory attention and decoder per object);
  for many sprites the 43.8 FPS single-object figure is an upper bound (reader inference from
  App. D.1, p. 17, Sec. 7, p. 10).
- Open question (lens): the only game-engine data tested is photorealistic Virtual KITTI 2;
  sprites, pixel art, HUD and text remain untested, as for SAM.
- Open question (lens): training is fully supervised by masks; nothing here learns
  reward-relevant or action-relevant features, so an RL objective or a self-supervised one
  (CURL-style) would still be needed on top.

## Related

- extends [[kirillov2023segment]]: same task family and prompt encoder/decoder, generalised to video with memory; SA-1B is reused for pre-training.
- uses [[he2021masked]]: the Hiera image encoder is MAE-pretrained (Sec. 4, p. 4).
- uses [[vaswani2017attention]]: memory attention and the decoder are standard transformer blocks.
- related [[srinivas2020curl]]: a self-supervised RL encoder, the alternative to SAM 2's supervised object-level state.
- related [[serrano2024crossdomain]]: per-object masklets are one candidate shared state space across games.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
