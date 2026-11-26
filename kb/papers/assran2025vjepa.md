---
id: assran2025vjepa
title: "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"
authors: [Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas]
year: 2025
venue: arXiv preprint
url: https://arxiv.org/abs/2506.09985
arxiv: 2506.09985
doi:
pdf_sha256: 9cfcfde5fb0d9730637da5b9e7317825c3f3d09e91f3553e22eeba42c74d2226
tags: [world-model, jepa, latent-prediction, video, self-supervised, planning, robotics, action-conditioned, mpc]
status: read
read: 2026-11-26
---

# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

## TL;DR

Stage 1 pretrains a ViT (up to 1B) with masked latent prediction on 1M+ hours of action-free video;
stage 2 freezes it and trains a 300M action-conditioned predictor (V-JEPA 2-AC) on 62 h of Droid
robot video. Plans by CEM on latent L1 to image goals, zero-shot on Franka arms; 16 s per action.

## Problem

World models trained from interaction data alone are limited by how little real interaction data
exists; video-generation world models spend capacity on pixel detail and are costly to plan with,
and prior ones mostly show visual quality, not control (Sec. 1, pp. 1-2). The paper asks whether
action-free internet video plus a little interaction data gives a latent world model that can
understand, predict and plan.

## Method

- Stage 1, V-JEPA 2 (Sec. 2, pp. 4-8): mask-denoising in representation space. Encoder E_theta and predictor P_phi
  minimise the L1 distance between the predicted and the stop-gradient EMA-encoder embeddings of
  the masked tubelets (Eq. 1, p. 4); the loss touches only masked patches. Video is cut into 2x16x16
  tubelets, with 3D-RoPE and multi-block masking (Sec. 2.1, pp. 4-5).
- Four scaling ingredients: data 2M to 22M videos (VideoMix22M: SSv2, Kinetics, Howto100M,
  curated YT-Temporal-1B, ImageNet; Table 1, p. 6), encoder 300M to 1B, 90K to 252K iterations with a
  warmup-constant-cooldown schedule, and a cooldown at higher resolution and 64 frames (Sec. 2.1-2.4, pp. 5-8).
- Stage 2, V-JEPA 2-AC (Sec. 3, pp. 8-10): the encoder is frozen; a block-causal transformer
  predictor (24 layers, 16 heads, width 1024, about 300M parameters) takes per-frame feature maps,
  the end-effector state and the action (a 7-D change in end-effector pose) interleaved in time and
  predicts the next frame's feature map (Fig. 2, Fig. 6, p. 4, p. 10). Loss = teacher-forcing L1
  (Eq. 2, T=15) + a 2-step rollout L1 (Eq. 3, Eq. 4, p. 9).
- Planning (Sec. 3.2, pp. 10-11): at each step, minimise Eq. 5, the L1 distance between the
  imagined latent T steps ahead and the goal image's latent, over action sequences with the cross-entropy
  method (CEM); execute the first action and replan (receding horizon, Fig. 7, p. 11).
- Probes and language: attentive probes on the frozen encoder for classification and EK100
  anticipation, and an LLaVA-style MLLM with the encoder for video QA (Sec. 5-7, pp. 15-21).

## Assumptions

- A1: Stage-1 training data is internet video and images at the 1M-hour scale, with no actions or labels (Sec. 2.3, Table 1, p. 6).
- A2: Stage-2 (AC) data is real teleoperated tabletop Franka video from Droid: 4 fps, 256x256, clips of at least 4 s, with end-effector state and 7-D action deltas (Sec. 3.1, p. 9).
- A3: Tasks are specified by goal images (up to two sub-goal images for pick-and-place), not by reward or language (Sec. 4.2, Sec. 4.3, pp. 13-15).
- A4: A single fixed exocentric monocular camera is used, positioned by hand-tuning (Sec. 4.3, p. 14).
- A5: Planning is greedy, with a short horizon (1 in Table 3; an optimised sequence is executed one action at a time) and continuous actions bounded to an L1 ball of radius 0.075 (Sec. 4.1, Table 3, p. 12, p. 15).
- A6: The pretrained encoder stays frozen during stage 2 and during planning (Sec. 3, p. 8).
- A7: Compute for pretraining is large (ViT-g, up to 252K iterations, global batch 3072; a full-resolution 64-frame ViT-g run would be roughly 60 GPU-years) (Table 9, p. 33, Sec. 2.4, p. 7).
- A8: Evaluation is on real robots, with success rates from 10 trials per cell (Table 2, p. 14).

## Claims

- C1: Scaling lifts average accuracy over six probe tasks from 84.2 (ViT-L baseline) by +1.0 with data (to 22M videos), +1.5 with model size (1B), +0.8 with 252K iterations, and +0.7 with 64-frame, higher resolution, reaching 88.2 [A1, A7] (Fig. 3, Sec. 2.2, p. 5).
- C2: Progressive-resolution training (16 frames at 256 px, then a 12K-iteration cooldown at 64 frames and 384 px) is up to 8.4x cheaper than training the 64x384x384 ViT-g from scratch [A7] (Fig. 5, Sec. 2.4, pp. 7-8).
- C3: Under the shared attentive-probe protocol, V-JEPA 2 ViT-g at 256 px reaches 75.3 top-1 on SSv2 and 87.5 average over six tasks, above DINOv2 (81.1 average), PEcoreG (82.3), SigLIP2 (81.1), V-JEPA ViT-H (85.2) and InternVideo2s2-1B (87.0); ViT-g384, which differs in protocol (384 px, 512 px for ImageNet, and 64 SSv2 frames), gets 77.3 and 88.2. It is only competitive on appearance tasks (ImageNet 84.6 at 256 px vs PEcoreG 87.6* and SigLIP2 88.0) and InternVideo2s2 is better on K400 (89.4) and COIN (93.8); the encoders are trained on different data, so the paper calls this a system-level comparison [A1] (Table 4, p. 16).
- C4: Anticipating EK100 actions 1 s ahead, a probe on frozen V-JEPA 2 ViT-g at 384 px gets 39.7 mean-class recall@5 for actions (vs PlausiVL 8B 27.6, +12.1, 44% relative), and recall scales with encoder size from 32.7 (ViT-L) [A1] (Table 5, Sec. 6, p. 17).
- C5: With the same Qwen2-7B LLM and a frozen encoder, V-JEPA 2 ViT-g512 averages 52.3 over seven VidQA benchmarks against 49.1 (PE G) and 48.1 (SigLIP2), though SigLIP2 and PE are better on PerceptionTest (72.4, 72.3 vs 72.0); an encoder trained without language supervision is therefore competitive [A1] (Table 6, p. 20).
- C6: With 88.5M alignment samples and a Llama 3.1 8B backbone, ViT-g384 reaches 84.0 PerceptionTest, 76.9 TempCompass and 36.7 TemporalBench, which the paper calls state of the art at the 8B scale; V-JEPA 2's PerceptionTest figure is on the test set after supervised fine-tuning while the other models are zero-shot, and PerceptionLM 8B wins TVBench and MVBench (63.5, 77.1), with InternVL-2.5 also ahead on TVBench (61.6 vs 60.6) [A1] (Table 8, Sec. 7.4, p. 21).
- C7: V-JEPA 2-AC is trained on under 62 h of Droid video with no reward or task labels and no teleoperation action labels (actions are computed as end-effector state deltas from the recorded 7-D state), including failed attempts (the paper's footnote counts 23k trajectories of successes and failures), with a frozen encoder, and needs no data from the deployment robots or labs [A2, A6] (Sec. 1, Sec. 3.1, p. 3, p. 9, p. 11).
- C8: Zero-shot on Franka arms in two labs, V-JEPA 2-AC with image goals reaches 100% on reach and 65% to 80% (averaged over labs) on cup grasp, reach with cup and cup pick-and-place and 25% to 75% on the box variants, while Octo (behavior cloning on Droid) gets 15% on cup grasp, 15% on cup pick-and-place and 0% on box grasp; 10 trials per cell [A2, A3, A4, A8] (Table 2, p. 14).
- C9: Against the Cosmos action-conditioned video-generation world model in one lab, V-JEPA 2-AC needs 16 s of planning per action with 800 samples, 10 iterations and horizon 1 on one RTX 4090, while Cosmos with 80 samples needs 4 min; success: reach 100 vs 80, cup grasp 60 vs 0, cup pick-and-place 80 vs 0, box 20 vs 20 and 50 vs 0 [A2, A3, A4, A5] (Table 3, p. 15).
- C10: In single-goal reaching, the end-effector gets within 4 cm of its goal with monotonically falling error, and the one-step energy landscape over (dx, dy) is smooth and locally convex with its minimum at about (dx, dy) = (0, -0.05) against a ground-truth action of (0, -0.1), i.e. near but at half the true displacement [A2, A3, A4] (Fig. 8, Fig. 9, Sec. 4.2, pp. 12-13).
- C11: The AC model is sensitive to camera position: the inferred coordinate axis is rotated relative to the true one by a systematic amount, from a few degrees to over 30 degrees across camera positions of about 35 to 85 degrees; mean action error is about 1.6 cm against a roughly 5 cm true delta [A4] (Sec. 4.3, App. B.4, Fig. 16, p. 14, pp. 40-41).
- C12: Long-horizon planning is not shown: autoregressive latent predictions lose accuracy with rollout length and the action search space grows exponentially with horizon; the planned tasks reach about 16 s into the future and pick-and-place needs sub-goal images [A2, A3, A5] (Sec. 4.3, Sec. 9, pp. 14-15, p. 23).
- C13: A separately trained feed-forward frame decoder (not part of training) applied to AC rollouts shows the arm and cup moving and the background unchanged, with accumulating error (cup ends slightly low), and an open gripper leaves the cup in place [A2, A6] (App. B.3, Fig. 15, pp. 37-39).
- C14: Curating YT1B by cluster retrieval gains +1.4 average over uncurated for ViT-L (abbreviated schedule), but with ViT-L mixed data the curated mix is worse on some tasks, and at ViT-g VM22M beats the uncurated mix on all four tasks [A1] (Fig. 4, Table 13, Fig. 12, pp. 6-7, pp. 35-36).

## Implementation notes

- Pretraining: batch 3072, LR 1e-4 warmup (12K steps) to 5.25e-4 then constant, cooldown to 1e-6 over 12K steps at 64 frames and crops 256/384/512, EMA 0.99925, weight decay 0.04, spatial mask scale 0.15-0.7 (source: Table 9, p. 33).
- Encoders: ViT-L/H/g with widths 1024/1280/1408 and depths 24/32/40; the predictor is a 22M ViT-s, width 384, depth 12 (source: Table 12, p. 35).
- AC training: AdamW, LR 7.5e-5 warmed to 4.25e-4 over 4500 iterations, 85500 constant, 4500 decay to 0; batch 256 of 4 s clips; feature maps 16x16x1408; left exocentric camera only (training on both cameras hurt) (source: App. B.1, p. 37, Sec. 3.1, p. 9).
- Planning: CEM with 800 samples, 10 refinements from the top-10 elites, horizon 1; pick-and-place switches goals after 4 and 14 steps (source: App. B.2, p. 37, Sec. 4.2, p. 13).
- Robot low-level control: the paper says only that the same inference code and "similar low-level controllers" based on operational-space control were used, with blocking control for V-JEPA 2-AC and Cosmos (source: Sec. 4.1, p. 12).

## Limitations / open questions

- Authors: sensitive to camera pose, short horizons only, image goals only, scaled to only 1B parameters (Sec. 4.3, Sec. 9, pp. 14-15, p. 23).
- EK100 anticipation is kitchen-only with a closed vocabulary and a 1 s anticipation time (Sec. 6, p. 18).
- Open question: for a shmup policy, there is no real-robot analog of a goal image; is a latent L1 distance to a goal frame a usable objective when the task is score and survival?
- Open question: the planner needs the world model and many samples per action (16 s each on a 4090); at emulator speeds (tens of thousands of fps) this is far too slow, so the model would more likely supply an encoder or a value shaping signal than a planner.
- Open question: Table 2 is out of 10 trials per cell, with no confidence intervals.
- Open question: pretraining on natural video is not game-screen data; whether the latent features keep small sprites is untested here (compare [[assran2023joint]]'s block-scale ablations, Table 8).

## Related

- extends [[assran2023joint]]: the same latent-prediction principle on video, with an L1 loss, 3D-RoPE, and an action-conditioned stage 2 on top.
- related [[he2021masked]]: masked pixel reconstruction, the contrast the JEPA approach is built against.
- related [[hafner2024mastering]]: DreamerV3 learns its latent dynamics jointly with a decoder and a policy from interaction; V-JEPA 2-AC learns dynamics on a frozen encoder and plans instead.
- related [[jiang2026olaf]]: uses a frozen V-JEPA 2 as the teacher whose feature changes supervise latent actions (per that note).
- related [[bruce2024genie]]: also learns from internet video, but generates pixels and infers latent actions; V-JEPA 2-AC uses real actions and predicts in embedding space.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
