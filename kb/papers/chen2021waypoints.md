---
id: chen2021waypoints
title: Learning to Set Waypoints for Audio-Visual Navigation
authors: [Changan Chen, Sagnik Majumder, Ziad Al-Halah, Ruohan Gao, Santhosh K. Ramakrishnan, Kristen Grauman]
year: 2021
venue: ICLR 2021 (arXiv v3 read)
url: https://arxiv.org/abs/2008.09622
arxiv: 2008.09622
doi:
pdf_sha256: 44be3759ce450aeee9d09316aa7b703842e9bbb074002c4e5ade847bb5091b27
tags: [audio-visual, navigation, hierarchical-rl, acoustic-memory, waypoints, ppo, spectrogram, distractor-sounds, embodied-ai]
status: read
read: 2026-11-26
---

# Learning to Set Waypoints for Audio-Visual Navigation

## TL;DR

AV-WaN: a PPO policy that, instead of primitive actions, picks waypoints in a local grid from a
GRU over encoded depth-map, binaural spectrogram and an "acoustic memory" (a map of past sound
intensity); a Dijkstra planner walks to each. Beats SoundSpaces baselines on Replica/Matterport3D.

## Problem

AudioGoal agents either act step by step with a GRU over audio (Chen et al. 2020) or predict the
final goal location from audio and plan to it (Gan et al. 2020); both fix the granularity of
decisions, and a GRU gives an unstructured audio memory (Sec. 1, pp. 1-2).

## Method

- Observation: depth plus binaural spectrograms (same STFT as [[chen2020soundspaces]]: 1 s clip,
  65x65x2 / 65x26x2) (Sec. 3.1-3.2, pp. 4-5; App. 6.4, p. 13).
- Geometric map G_t: depth back-projected to a 3x3 m egocentric occupancy+explored map, fused into an
  allocentric map (Sec. 3.2, pp. 4-5).
- Acoustic memory A_t: ground-plane map storing a moving average of direct-sound intensity (RMS of the
  first 3 ms of non-zero audio, mean of both ears) at positions the agent has visited (Sec. 3.2, p. 5;
  App. 6.4, p. 13).
- Encoders f_g, f_b, f_a (small CNNs, 512-d each) for G_t, spectrogram B_t and A_t; concatenated into
  a one-layer bidirectional GRU (512 hidden); actor outputs a distribution over an s_w x s_w grid of
  waypoints around the agent (masked to free space), critic outputs the value (Fig. 2, Sec. 3.3, pp. 5-6;
  App. 6.2, 6.5, pp. 13-14).
- Planner: Dijkstra on G_t (unexplored = free), low-level action by analytic rule; the loop breaks on
  reaching the waypoint, no path (then a random action), or a step limit; waypoint (0,0) means Stop
  (Sec. 3.4, p. 6).
- Reward +10 for Stop at goal, +0.25 for geodesic progress, -0.01 per executed action; the waypoint's
  reward is the sum collected while the planner executes it; PPO updated every 150 waypoint steps
  (Sec. 3.5, p. 6).

## Assumptions

- A1: Environments are the SoundSpaces Replica/Matterport3D scans with audio only at grid nodes and discrete actions, so no noisy actuation can be tested (Sec. 3.3, p. 5).
- A2: Depth is available and used to build the occupancy map; audio is binaural spectrograms of a single sound source at the goal (Sec. 3, pp. 4-5).
- A3: The agent has no GPS or goal vector (AudioGoal) (Sec. 3, p. 4).
- A4: Heard-sound experiments use one telephone sound; unheard use 102 clips split 73/11/18 with test scenes unseen in training (Sec. 4, p. 6, p. 16).
- A5: Low-level motion comes from a hand-written Dijkstra shortest-path planner on the agent's own occupancy map (Sec. 3.4, p. 6).
- A6: Map building uses the agent's pose, which is assumed available (reader's inference, Sec. 3.2, pp. 4-5).

## Claims

- C1: On heard sounds AV-WaN has the best SPL/SR/SNA: Replica 86.6/98.7/70.7 vs Chen et al. 78.2/94.5/52.7 and Gan et al. 57.6/83.1/47.9; Matterport3D 72.3/93.6/54.8 vs Chen 55.1/71.3/32.6 and Gan 22.8/37.9/17.1. The text gives SPL gains of 8.4% and 29% (Replica) and 17.2% and 49.5% (Matterport) over Chen and Gan [A1, A2, A3, A4, A5, A6] (Table 1, p. 7, Sec. 4, p. 8).
- C2: On unheard sounds AV-WaN is best on most metrics: Replica SPL 34.7 (table value equal to Chen's 34.7; the text says 2.8% lower), SR 52.8 vs 50.9, SNA 27.1 vs 16.7; Matterport 40.9/56.7/30.6 vs Chen 25.9/40.1/12.8. Absolute performance falls for all methods vs heard sounds [A1, A2, A3, A4, A5, A6] (Table 1, p. 7, Sec. 4, p. 8).
- C3: AV-WaN beats the three hierarchical baselines (Direction Follower, Frontier Waypoints, Supervised Waypoints) in every column of Table 1, e.g. heard Replica SPL 86.6 vs 54.7, 44.0 and 59.1, and unheard Replica SPL 34.7 vs 11.1, 6.5 and 14.1; the authors expect (hedged) that the gap to Supervised Waypoints is due to its narrow supervised definition of the optimal waypoint [A1, A2, A3, A4, A5, A6] (Table 1, p. 7, Sec. 4, p. 8).
- C4: Ablation SPL, heard Replica / heard Matterport: full 86.6/72.3; no A_t 85.7/70.2; no G_t 85.1/70.2; neither 84.3/68.8; no waypoints (primitive-action actor) 79.8/44.3. Unheard Replica: full 34.7, no A_t 34.5, no G_t 27.0, neither 34.0, no waypoints 25.5. Unheard Matterport: full 40.9, no A_t 36.7, no G_t 25.4, neither 20.5, no waypoints 25.5. Even the neither-map variant beats Chen et al. in the heard setting; the authors call the acoustic memory critical for unheard sounds, but on Replica unheard the no-A_t SPL is 34.5 vs 34.7, so the gap is clear only on Matterport; results average 5 runs with std at most 0.5 [A1, A2, A3, A4, A5, A6] (Table 2, p. 8, Sec. 4, p. 8).
- C5: With a distractor sound at a random location (heard setting), AV-WaN gets SPL 83.1 (Replica) and 70.9 (Matterport) vs 71.7 and 53.3 for the best baseline Chen et al. [A1, A2, A3, A4, A5, A6] (Sec. 4, p. 9).
- C6: Under added Gaussian microphone noise, AV-WaN degrades far less than Chen and Gan, and removing A_t makes it degrade faster (at the highest noise levels it falls below Chen et al.), so the acoustic memory helps in noise; exact values are read only from the plot [A1, A2, A3, A4, A5, A6] (Fig. 4b, p. 9).
- C7: Waypoints are chosen further away when the goal is far and closer when near; they cluster at obstacles and doors, typically 2-3 m apart [A1, A2, A3, A4, A5, A6] (Fig. 4a, Fig. 4c, Sec. 4, p. 9, App. 6.8, p. 16).
- C8: Depth noise (Redwood model, test time only) changes SPL by under 1% and SR by under 0.5% for all methods [A1, A2, A5, A6] (App. 6.3, p. 13).
- C9: Failure cases: goal cornered next to obstacles (oscillation from audio reflections or mapping errors) and premature Stop near the goal [A1, A2, A5, A6] (Sec. 4, pp. 8-9).

## Implementation notes

- Adam, lr 2.5e-4, entropy coefficient 0.02, 7.5M policy steps, planning step limit 10 (source: App. 6.2, p. 13).
- Geometric map 200x200 cells at 0.1 m; acoustic map s_a = 20; action map s_w = 9 (source: App. 6.2, p. 13).
- CNN encoders: three conv layers (f_g and f_b, the geometric-map and spectrogram encoders: kernels 8,4,3, strides 4,2,1; f_a, the acoustic-memory encoder: kernels 5,3,3, strides 2,1,1), channels from 32 doubling, final FC to 512 (source: App. 6.5, p. 14).
- Planner and map projections use ground-truth pose; the paper does not discuss pose noise (source: assumption).
- Baseline Gan et al. is the authors' reimplementation (no public code), with prediction every N steps (N=20 Replica; 50/60 Matterport) (source: App. 6.7.1, p. 15).
- Direction Follower: K=2 (Replica), K=4 (Matterport) in Sec. 4, while App. 6.7.2 says a 4-step (2 m) intermediate goal; the paper does not reconcile these (source: Sec. 4, p. 7, App. 6.7.2, p. 15).
- App. 6.9 says 78 sounds for training right after the 73/11/18 split, the same slip as in [[chen2020soundspaces]] (source: App. 6.9, p. 16).
- Train/val/test scenes: 9/4/5 Replica, 73/11/18 Matterport (source: Sec. 4, p. 6).

## Limitations / open questions

- Future work named: semantic sounds, moving sources, real-world transfer; sim-to-real untested (Sec. 5, p. 9).
- Unheard-sound performance is much lower for all methods (Table 1, p. 7).
- Open question: the acoustic memory is a spatial map and needs agent localization; a game sidescroller or shmup has no ground-plane geometry for such a map. A temporal buffer (the GRU alone) is the analogue.
- Open question: only a single point sound plus synthetic distractors; no game audio, music or speech content.
- Open question: the text (2.8% lower unheard Replica SPL) and Table 1 (equal 34.7) disagree.

## Related

- extends [[chen2020soundspaces]]: same AudioGoal task and audio rendering; adds waypoints and an acoustic map.
- uses [[schulman2017proximal]]: PPO.
- related [[bacon2017option]]: cited as inspiration for hierarchical policies; this paper's waypoints are not options with learned termination.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
