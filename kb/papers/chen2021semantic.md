---
id: chen2021semantic
title: Semantic Audio-Visual Navigation
authors: [Changan Chen, Ziad Al-Halah, Kristen Grauman]
year: 2021
venue: CVPR 2021 (arXiv v2 read)
url: https://arxiv.org/abs/2012.11583
arxiv: 2012.11583
doi:
pdf_sha256: 642e6b4bc63ccb10573576eb591e22618eceb80b5015c98b3f70b867285127a8
tags: [audio-visual, navigation, semantic-audio, transformer, scene-memory, goal-descriptor, intermittent-sound, embodied-ai]
status: read
read: 2026-11-26
---

# Semantic Audio-Visual Navigation

## TL;DR

SAvi: task and model for finding an object that makes a short, non-periodic sound (21 classes,
Matterport3D). A CNN predicts a goal descriptor (location, category) from audio; a transformer over a
150-step memory attends conditioned on it. Beats SoundSpaces baselines (24.0 vs <=23.7 SPL, heard).

## Problem

Prior audio-visual navigation assumes a constantly sounding, periodic target with no visual form and
no semantic link to the scene; audio is only a beacon (Sec. 1, p. 1). Real events are short
(door slam, dishwasher chime), so the agent must infer what object sounded and where, and keep
going after silence (Sec. 1-3, pp. 1-3).

## Method

- Data: SoundSpaces Matterport3D scenes with sounds attached to object instances of 21 ObjectGoal
  categories (chair, sink, toilet, ...), from freesound.org clips (mean 81 s), split evenly into
  train/val/test clips; object-emitted and object-related sounds. Acoustic event duration per
  episode ~ Gaussian (mean 15 s, std 9 s), clipped to 5-500 s, clip replayed if shorter
  (Sec. 3, p. 3).
- Task: no goal vector or label; sensors are RGB, depth, binaural audio and the agent's pose;
  success needs Stop within 1 m of the object's boundary viewpoints for the specific instance
  (Sec. 3, p. 3).
- Observation encoder: CNNs encode image and spectrogram; each step's embedding
  [e^I, e^B, pose, a_{t-1}] goes into a memory M of the last s_M steps (Sec. 4.1, p. 4).
- Goal descriptor network f_D(B_t): CNN on the current spectrogram predicts location L_t (relative
  to the agent) and object category C_t; aggregated with the previous descriptor
  (1-lambda)D_hat + lambda f_p(D_{t-1}, dp), lambda=0.5; when sound is silent the descriptor is only
  carried forward by the pose change (Sec. 4.2, pp. 4-5).
- Policy: transformer encoder (one layer) over M gives M_e; a decoder (one layer, 8 heads) queries
  M_e with D_t to give state s_t; linear actor-critic (Sec. 4.3, p. 5).
- Training: category classifier pre-trained off-policy on 3.5M spectrogram-category pairs then frozen;
  location predictor trained on-policy; DD-PPO in two stages (s_M=1 to train the encoders, then freeze
  them and train the rest with s_M=150); reward +10 success, +1 geodesic progress, -0.01 per step
  (Sec. 4.4, p. 5).
- Inputs: 1 s of audio per step as two 65x26 spectrograms; RGB/depth center cropped to 64x64;
  simplified ResNet-18 CNNs; hidden size 256; Adam 2.5e-4 (policy) and 1e-3 (descriptor);
  300M training steps for all methods (Sec. 4.4, p. 5).

## Assumptions

- A1: Matterport3D scans with audio only at 1 m grid nodes and discrete actions; no actuation noise (Sec. 3, p. 3, Sec. 5, p. 6).
- A2: The goal is one specific instance of one of 21 ObjectGoal categories, and stopping near another instance of the class fails (Sec. 3, p. 3).
- A3: The sound may stop at any time (duration about N(15 s, 9 s) clipped to 5-500 s, clip replayed if shorter); episodes need geodesic distance over 4 m and Euclidean/geodesic ratio over 1.1 (Sec. 3, p. 3, Sec. 4.4, p. 5).
- A4: The agent's pose sensor is used as given, assumed noise-free (reader's inference, Sec. 3, p. 3).
- A5: Training uses ground-truth location and category labels from the simulator for the goal descriptor (Sec. 4.4, p. 5).
- A6: Sounds are everyday object sounds (clips from freesound.org searched by the 21 category names), not speech, music or game sounds (Sec. 3, p. 3).
- A7: The sounding object is static (reader's inference, Sec. 3, p. 3).

## Claims

- C1: SAvi has the best success and SPL on both splits. Heard: success 33.9, SPL 24.0 (Gan 29.3/23.7, Chen 21.6/15.1, AV-WaN 20.9/16.8, SMT+Audio 22.0/16.8); unheard: success 24.8, SPL 17.2 (best baseline 18.0/13.4 for Chen). Random 1.4/3.5 [A1, A2, A3, A4, A5, A6, A7] (Table 1, p. 6).
- C2: SPL gains stated are 0.3, 8.9, 7.2, 7.2 points on heard sounds and 4.9, 3.8, 4.0, 5.3 on unheard over Gan, Chen, AV-WaN and SMT+Audio [A1, A2, A3, A4, A5, A6, A7] (Sec. 5, p. 6).
- C3: Success-when-silent (SWS): SAvi 21.5 heard / 14.7 unheard; baselines heard Gan 14.4, Chen 10.7, AV-WaN 8.3, SMT+Audio 8.7; unheard Gan 8.0, Chen 6.9, AV-WaN 6.9, SMT+Audio 8.5 [A1, A2, A3, A4, A5, A6, A7] (Table 1, p. 6).
- C4: In the cumulative success vs silence ratio plot, baselines' success barely rises past ratio 1 while SAvi continues to increase past ratios 1 and 2, i.e. it copes with long silence [A1, A2, A3, A4, A5, A6, A7] (Fig. 4, p. 8).
- C5: With an unheard distractor sound (a periodic SoundSpaces clip at a random location, and the target's one-hot class given to all models), SAvi gets success 11.8, SPL 7.4 vs the best baseline (SMT+Audio) 4.2 / 2.9; all methods drop strongly from 24.8 [A1, A2, A3, A4, A5, A6, A7] (Table 2, p. 7).
- C6: Goal descriptor ablation (unheard): category-only 20.5/13.5 (success/SPL), location-only 23.9/16.2, no aggregation 21.9/14.3, full 24.8/17.2; location matters more than category [A1, A2, A3, A4, A5, A6, A7] (Table 3, p. 8).
- C7: The descriptor predictions are weak: mean location error 8.1 m and category accuracy 64.5% with aggregation vs 8.4 m and 53.6% without [A1, A2, A3, A5, A6, A7] (Sec. 5, p. 8).
- C8: In successful episodes, 56% stop at the agent's own estimate of the goal location; the other 44% stop elsewhere at the correct object, which the authors read as use of visual context and category. Replacing C_t with a random label at the start makes success and SPL drop by up to 8 points (the paper: "drop up to 8%") [A1, A2, A3, A5, A6, A7] (Sec. 5, p. 8).
- C9: Transformer vs RNN+MLP policy (unheard): SAvi with transformer 24.8 success / 17.2 SPL, with RNN+MLP 21.4/15.4; Chen et al. (RNN) 18.0/13.4, SMT+Audio 16.7/11.9; so the descriptor gives gains in both, the transformer more [A1, A2, A3, A4, A5, A6, A7] (Table 4, p. 12, Sec. 7.3, p. 11).
- C10: Given the true goal category and location, SPL rises to 65% vs 24% with predicted descriptors (the paper names no split; 24.0 is the heard SPL in Table 1), given as an upper bound for the descriptor network, i.e. the task then reduces to PointGoal [A1, A2, A3, A4, A5, A6, A7] (Sec. 7.7, p. 12).
- C11: Off-policy training of the location predictor, as in Gan et al., lowers success by 4.8 and SPL by 4.7 points (unheard) vs on-policy [A1, A2, A3, A5, A6, A7] (Sec. 7.6, p. 12).
- C12: The ObjectGoal RL baseline, given the true category label and perfect stopping but no audio, reaches only success 1.5, SPL 0.8: the category alone does not give the instance [A1, A2, A3, A4, A6, A7] (Table 1, p. 6, Sec. 7.2, p. 11).
- C13: The authors do not observe the large margins of AV-WaN over Chen et al. that AV-WaN reported on the standard AudioGoal task (here heard SPL 16.8 vs 15.1, unheard 13.2 vs 13.4), attributed as speculation to temporal gaps in its waypoint-level memory [A1, A2, A3, A6, A7] (Table 1, p. 6, Sec. 7.2, p. 11).

## Implementation notes

- Spectrogram settings follow [[chen2020soundspaces]] (1 s clips, 65x26); the STFT parameters are not repeated here (source: Sec. 4.4, p. 5; [[chen2020soundspaces]]).
- Memory size 150 equals the policy update interval; the transformer is one encoder layer plus one decoder layer (source: Sec. 4.4, p. 5).
- Episodes 0.5M/500/1000 for train/val/test across the 85 scenes (source: Sec. 4.4, p. 5).
- Gan et al. baseline is adapted with the same lambda update; Gan et al.'s goal predictor gets the same amount of training data as this paper's category predictor (source: Sec. 7.2, p. 11).
- Distractor sounds: 102 periodic SoundSpaces clips split 73/11/18 (source: Sec. 5, p. 7).
- Silent-period handling: the descriptor is kept by the pose transform alone, and the audio input is then zero (source: Sec. 4.2, pp. 4-5).

## Limitations / open questions

- Authors: failures when the sound stops too early (little evidence) and premature Stop near the target (Sec. 5, p. 7); distractors are not explicitly separated, left for future work (Sec. 5, p. 7).
- Real-world transfer left to future work (Sec. 6, p. 8).
- Open question: all rows are in one simulator with simulated sound, one object per episode; the sounds are everyday object sounds, and nothing is shown on music, speech or game sound effects (reader's inference).
- Open question: the descriptor head needs simulator ground truth for location and category; a game has no such labels (reader's inference).
- Open question: the memory is 150 steps of pooled per-step features; the paper doesn't compare memory lengths.

## Related

- extends [[chen2020soundspaces]]: same simulator and policy family, adds intermittent semantic sounds and a transformer memory.
- related [[chen2021waypoints]]: AV-WaN is a baseline in Table 1 that this paper outperforms.
- uses [[vaswani2017attention]]: transformer encoder/decoder over the memory.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
