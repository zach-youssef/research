---
id: chen2020soundspaces
title: "SoundSpaces: Audio-Visual Navigation in 3D Environments"
authors: [Changan Chen, Unnat Jain, Carl Schissler, Sebastia Vicenc Amengual Gari, Ziad Al-Halah, Vamsi Krishna Ithapu, Philip Robinson, Kristen Grauman]
year: 2020
venue: ECCV 2020 (arXiv v3 read)
url: https://arxiv.org/abs/1912.11474
arxiv: 1912.11474
doi:
pdf_sha256: d180d06c21f58139620581c8a6ba2d15f9af2a786822175f78a882f3087d97bc
tags: [audio-visual, navigation, embodied-ai, reinforcement-learning, ppo, spectrogram, binaural, simulation, sound-generalization]
status: read
read: 2026-11-26
---

# SoundSpaces: Audio-Visual Navigation in 3D Environments

## TL;DR

Adds precomputed binaural room-acoustics renderings to Habitat (Replica, Matterport3D) and trains
a PPO policy (separate CNNs on spectrogram and depth/RGB, concatenated into a GRU) to reach a
sounding target. Audio plus vision beats vision-only point-goal navigation and tolerates GPS noise.

## Problem

Embodied navigation agents were "deaf": only vision (and a goal-displacement vector) was used,
although sound reveals unseen targets and room geometry (Sec. 1, pp. 1-2). No photorealistic
simulator rendered acoustics for agent training (Sec. 2, pp. 4-5).

## Method

- Simulator: RIRs between all pairs of points on a grid (0.5 m Replica, 1 m Matterport, at 1.5 m
  height), by geometric bidirectional path tracing with per-surface acoustic materials; arbitrary
  mono waveforms are convolved with the RIR on the fly and rendered as binaural audio
  (Sec. 3, pp. 5-6; App. 8.1, pp. 21-23). The agent moves on the grid graph, so audio exists only
  at grid nodes (Sec. 3, p. 6).
- Tasks: PointGoal (vector to goal), AudioGoal (only sound from the goal, no GPS) and
  AudioPointGoal (both). Actions MoveForward/TurnLeft/TurnRight/Stop, 500-step horizon, no actuation
  noise, success needs Stop at the goal (Sec. 4, pp. 6-7).
- Audio input: STFT of 1 s of audio per step per ear (window 512, hop 160), magnitude, downsample
  by 4, log; stacked to 65x65x2 (Replica, 44.1 kHz) or 65x26x2 (Matterport, 16 kHz). Vision
  128x128 RGB and/or depth (Sec. 5, p. 8).
- Policy: audio CNN and vision CNN (conv 8x8, 4x4, 3x3, linear; separate weights) give 512-d
  vectors, concatenated with the 2-d displacement (1026-d), single-layer bidirectional GRU (hidden 512), linear
  actor and critic; PPO with reward +10 for Stop at goal, -0.01 per step, +/- geodesic-distance
  shaping (Fig. 3, Sec. 5, pp. 8-9). Memory is only the GRU state (reader's inference).
- Training: Adam 2.5e-4, discount 0.99, 30M steps on Replica (105 GPU h), 60M on Matterport
  (210 GPU h) (Sec. 5, p. 9).
- Sounds: 102 copyright-free clips (phones, music, fans, speech, door, ...), split 73/11/18;
  default source is a telephone ring (Sec. 6, p. 10; App. 8.5, p. 26).

## Assumptions

- A1: Audio comes from one static point source at the goal, rendered from precomputed RIRs at grid nodes only, with no noise unless added (Sec. 3, Sec. 4, p. 6, p. 7).
- A2: Navigation is in unmapped but indoor, static, scanned environments with discrete actions and no actuation noise (Sec. 4, p. 7).
- A3: The agent has two microphones at human height and gets binaural spectrograms (Sec. 4, p. 7).
- A4: Sources are 102 copyright-free natural sound clips from freesound.org; the default source is a telephone ring (Sec. 6, p. 10).
- A5: Test scenes are never seen in training; in Table 2 the test sound is the same heard telephone clip (Sec. 6, p. 10).

## Claims

- C1: Adding audio helps: AudioPointGoal beats PointGoal in SPL for every visual sensor. Replica blind 0.480 to 0.681, RGB 0.521 to 0.632, depth 0.601 to 0.709; Matterport blind 0.426 to 0.473, RGB 0.466 to 0.521, depth 0.541 to 0.581. Non-learning baselines are at most 0.197 [A1, A2, A3, A4, A5] (Table 2, p. 12).
- C2: AudioGoal (sound only, no GPS) matches or beats PointGoal with perfect GPS on Replica (depth SPL 0.756 vs 0.601) and about equals it on Matterport (0.552 vs 0.541), and is immune to GPS noise, while PointGoal drops from about 0.6 towards about 0.2 as Gaussian noise rises to sigma 1.5 m [A1, A2, A3, A4, A5] (Fig. 5a, Table 3, pp. 12-14).
- C3: Under 40 dB SNR microphone noise, AudioGoal SPL falls only from 0.756 to 0.753 (Replica) and 0.552 to 0.550 (Matterport) [A1, A2, A3, A4, A5] (Sec. 6, p. 13).
- C4: The learned audio features encode distance and angle to the goal (t-SNE coloured by distance and angle) [A1, A2, A3, A4, A5] (Fig. 5b, p. 12).
- C5: Spectrograms matter more than loudness: replacing the audio CNN with left/right RMS intensity drops AudioGoal-depth SPL from 0.756 to 0.291 (Replica) and 0.552 to 0.014 (Matterport); blind/RGB likewise (Replica 0.276/0.177 vs 0.673/0.626; MP3D 0.173/0.003 vs 0.438/0.479) [A1, A2, A3, A4, A5] (Sec. 6, p. 13, Table 4, p. 25).
- C6: In the two AudioGoal episodes shown, the modality-importance bars (replacing one input with its training mean) change over the episode: audio drives the early turns and the stop in one episode, vision is used near obstacles. The paper's general statement is only that the agent draws dynamically on either or both modalities [A1, A2, A3, A4, A5] (Fig. 6, p. 13).
- C7: Generalisation to unheard sounds is weaker for AudioGoal than for AudioPointGoal. Replica depth AG: same sound 0.756, varied heard 0.645, varied unheard 0.454; APG 0.709, 0.724, 0.707. Matterport depth AG 0.552, 0.448, 0.338; APG 0.581, 0.570, 0.538 (PG 0.541). Blind Replica AG unheard 0.277 vs APG 0.649. Averaged over 7 test runs, std at most 0.01 [A1, A2, A3, A4, A5] (Table 3, p. 14).
- C8: With 102 natural sounds split 73/11/18 into train/val/test sounds, APG almost always beats PointGoal even on unheard test sounds; the authors expect larger sound repositories to reduce AG's decline (not tested) [A1, A2, A3, A4, A5] (Table 3, Sec. 6, p. 14).
- C9: Depth is the stronger visual input; in AudioPointGoal the RGB-depth gap is smaller than in PointGoal, which the authors read as audio-visual learning recovering geometry from RGB (Replica 0.632 vs 0.709, PG 0.521 vs 0.601) [A1, A2, A3, A4, A5] (Table 2, p. 12).

## Implementation notes

- Audio at each step is the first 1000 ms of the source waveform as rendered at the agent's pose (Sec. 5, p. 8); how audio continues across steps for long sources is not stated (source: assumption).
- A step is one 0.5 m / 1 m move; the audio rate per step is one 1-s spectrogram, not a streaming rate (source: Sec. 5, p. 8).
- Importance bars (Fig. 6) are computed by replacing a modality with its training-sample average (source: Sec. 6, p. 13).
- The supplement says 78 sounds for training in some places while the split is 73/11/18; inconsistent in the paper (source: App. 8.5, p. 26).
- Code and data released with the project page; no commit stated in the paper (source: assumption).

## Limitations / open questions

- No sim-to-real transfer, single static source, no moving sources or multi-agent; named as future work (Sec. 7, p. 14).
- AG's accuracy declines on unheard sounds (Table 3, p. 14).
- Open question: the realism of the geometric-acoustics rendering is untested against real audio, and the sources are natural sounds (phones, music, fans, speech), not game sound effects (reader's inference).
- Open question: all sounds here are spatial, localisable sources; a game's mixed, non-spatial music and sound effects give no direction or distance cue, so the benefit seen here may not carry over.
- Open question: evidence is for 1-s spectrograms processed by a small CNN and GRU; no transformer, no latency measure.

## Related

- uses [[schulman2017proximal]]: PPO is the training algorithm.
- related [[he2021masked]]: MAE-style audio models would be an alternative spectrogram encoder; this paper trains the CNN end to end from reward.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
