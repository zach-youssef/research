---
id: chen2022soundspaces
title: "SoundSpaces 2.0: A Simulation Platform for Visual-Acoustic Learning"
authors: [Changan Chen, Carl Schissler, Sanchit Garg, Philip Kobernik, Alexander Clegg, Paul Calamia, Dhruv Batra, Philip W Robinson, Kristen Grauman]
year: 2022
venue: NeurIPS 2022 Datasets and Benchmarks (arXiv v2 read)
url: https://arxiv.org/abs/2206.08312
arxiv: 2206.08312
doi:
pdf_sha256: 3709b85870f8d6491c192b033b258872c9fd7b59b21960d64b004edef101cf14
tags: [audio-visual, simulation, room-acoustics, ray-tracing, navigation, sim2real, speech-recognition, benchmark, dataset]
status: read
read: 2026-11-26
---

# SoundSpaces 2.0: A Simulation Platform for Visual-Acoustic Learning

## TL;DR

On-the-fly geometric acoustic rendering for any 3D mesh, with continuous source/listener positions,
configurable materials and microphones. Validated on one real apartment; continuous AV navigation SPL
49.3 and far-field ASR WER 29.1 to 12.48 after fine-tuning on its IRs.

## Problem

SoundSpaces 1.0 used precomputed RIRs on a grid (0.5 m in Replica, 1 m in Matterport3D, see
[[chen2020soundspaces]]) for a fixed set of 100 environments: agents could only hop between grid
points, nothing generalised to new meshes, and the data was terabytes and not configurable (Sec. 1,
p. 2). It also implicitly treated the source as silent between steps, so sound was not continuous
(Sec. 3.2, p. 5).

## Method

- Engine (RLR-Audio-Propagation) integrated with Habitat-Sim: given mesh, source and receiver
  positions, a bidirectional path tracer computes an energy-time histogram in M log-spaced frequency
  bands with spherical-harmonic directivity, converted to ambisonic or binaural pressure IRs, then
  convolved with the source waveform (Sec. 3.1, p. 4; App. 8.2, p. 15).
- Changes from 1.0: edge diffraction (smooth occlusion), and a sqrt(4*pi) bias fix in indirect
  sound that corrects the direct-to-reverberant ratio (Sec. 3.1, p. 4).
- Spatial continuity: arbitrary source/receiver/heading, R(s, r, theta) (Sec. 3.2, p. 5).
- Acoustic continuity: per step, N = F*dt audio samples are taken from the source segment aligned by
  propagation delay and convolved with the IR at the new position, with linear crossfade between
  consecutive steps over an overlap window T (Sec. 3.2, p. 5).
- Configurability: sampling rate, frequency bands, rays, which phenomena are on; mono, stereo,
  binaural, quad, surround and ambisonic microphones, custom HRTFs; 29 acoustic materials with
  per-frequency absorption/scattering/transmission, air absorption, user mappings from object
  category to material (Sec. 3.3, pp. 5-6; App. 8.3-8.4, p. 16).
- Generalisation to any mesh (Gibson, HM3D, Matterport3D, Replica) and to shoebox rooms
  (Sec. 3.4, p. 6).
- Modes: high-quality (all rays, no temporal coherence) and high-speed (fewer rays, reuses earlier IRs
  assuming continuous motion) (Sec. 3.5, p. 6).
- SoundSpaces-PanoIR: 10M panorama (RGB-D) plus IR pairs from 750 environments (Sec. 4, p. 7;
  App. 8.5, p. 16).

## Assumptions

- A1: Scene meshes are high quality (no large holes); otherwise rays leak and RT60s come out too low, as in many Matterport3D meshes (Sec. 6, p. 10).
- A2: Acoustics are geometric (rays, no wave effects beyond a diffraction approximation); acoustic materials are assigned from object category, not measured (Sec. 6, p. 10).
- A3: Real-world validation is one apartment (Replica FRL), 7 source/receiver positions, bands 250 Hz to 4 kHz (Sec. 5.2, p. 7, Fig. 3b, Fig. 3c, p. 8).
- A4: The navigation benchmark uses the AV-Nav agent trained with DD-PPO on the existing AudioGoal dataset, in high-speed rendering mode (Sec. 5.3, p. 8, App. 8.7, p. 17).
- A5: The ASR benchmark uses speech (LibriSpeech) convolved with IRs, a transformer ASR model (SpeechBrain), and BUT ReverbDB real IRs for testing (Sec. 5.4, p. 9).
- A6: The navigation goal is a single static sound source (reader's inference, Sec. 5.3, p. 8).
- A7: For the real-IR comparison the measurements were scaled to match the simulated direct-sound level and the mesh's acoustic material properties were optimised to match the measurements (p. 7).

## Claims

- C1: High-speed mode runs 7.7 FPS (1 thread) and 33.5 FPS (5 threads) vs 0.9 and 4.0 FPS for high-quality mode, at 9.5% relative RT60 error, on a Xeon Gold 6230 CPU along 15 m paths in 20 Matterport3D scenes. The text states "8x" single-thread and "33x" with 5 threads; the table's ratios are about 8.6x and 8.4x, so the 33x appears to be only the 5-thread FPS figure [A1, A2] (Table 2, p. 7, Sec. 5.1, p. 7).
- C2: A navigation model trained in high-speed mode loses under 1% when tested on high-quality mode [A1, A2, A4, A6] (Sec. 5.1, p. 7).
- C3: Compared with measured IRs in the Replica FRL apartment, SoundSpaces 2.0 reduces the DRR error from 11.0 dB to 0.98 dB on average, with the same relative RT60 error of 12.4% for both versions; this is after the scaling and material optimisation to that one apartment's measurements (footnote 7) [A2, A3, A7] (Fig. 3b, Sec. 5.2, pp. 7-8, Fig. 5, p. 17, App. 8.6, p. 16).
- C4: Continuous AV navigation (agent moves 0.15 m per step or turns 10 degrees; success within 1 m): the old-SoundSpaces baseline (trained on the discrete setup with data rendered from the enhanced simulation, 90-degree turns and 1 m steps) gets 64.2% success / 27.5% SPL when only space is continuous, but 0.9% / 0.3% (DTG 12.9 m) when sound is continuous too; trained on SoundSpaces 2.0 it gets 64.7% success / 49.3% SPL (DTG 5.9 m). One training seed per row, evaluated on 3 seeds; high-speed rendering mode [A1, A4, A6] (Table 3, p. 8, Sec. 5.3, p. 9).
- C5: The authors attribute the collapse in C4 to the old agent relying on a direct-sound cue that is always present in 1.0, while in continuous rendering the direct sound is mixed with reverberation (an explanation, not tested separately) [A4, A6] (Sec. 5.3, p. 9).
- C6: Far-field ASR (SpeechBrain transformer, LibriSpeech, 60 epochs): pretrained WER 29.10%; fine-tuned on real IRs 13.32%, Pyroomacoustics 16.24%, SoundSpaces 1.0 18.48%, SoundSpaces 2.0 12.48% (the clean-test WER of the pretrained model is 2.4%) [A5] (Table 4, p. 9, Sec. 5.4, p. 9).
- C7: Acoustic randomization (random material per category plus N(0, 0.1) noise on coefficients) lowers WER from 12.48% to 12.04%, while uniform random coefficients in [0,1] give 12.58% [A5] (Sec. 5.4, pp. 9-10).
- C8: Table 1 compares platforms: SoundSpaces 2.0 is the only listed one that is audio-visual, geometric, configurable and renders arbitrary new meshes; the speed comparison cites TDW at 60 FPS vs SoundSpaces (1.0, precomputed) at 500+ FPS [] (Table 1, p. 3, Sec. 5.1, p. 7).

## Implementation notes

- Navigation benchmark: AV-Nav agent with DD-PPO, 80M steps on 32 GPUs for 46 h (source: App. 8.7, p. 17).
- ASR fine-tuning: 28,539 IRs per setting, batch 24 on 8 GPUs, 60 epochs, warmup 1000 steps (source: App. 8.8, p. 17).
- The hop between two steps uses sampling rate F and time dt, with crossfade window T; the values of dt and T used in the navigation benchmark are not stated (source: assumption, Sec. 3.2, p. 5).
- Code: github.com/facebookresearch/sound-spaces and rlr-audio-propagation; commit not stated in the paper (source: p. 1, p. 4).
- Audio is rendered per agent step; the per-step latency is only given as FPS (Table 2), not as milliseconds per step (source: Table 2, p. 7).

## Limitations / open questions

- Needs high-quality meshes; geometric acoustics lacks room modes; materials cannot be inferred from the mesh (Sec. 6, p. 10).
- Validation limited to one real apartment (Sec. 6, p. 10).
- Open question: all demonstrations are speech or spatial sounds; nothing about non-spatial mixed audio like game soundtracks.
- Open question: relevance to the game-audio goal is mostly the lesson in C4-C5: an audio policy trained against a rendering shortcut (always-present direct sound) failed when the rendering became more realistic (reader's inference).

## Related

- extends [[chen2020soundspaces]]: continuous, on-the-fly successor of the precomputed SoundSpaces.
- related [[chen2021waypoints]]: its benchmark agent is the AV-Nav of 1.0, not the waypoint model.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
