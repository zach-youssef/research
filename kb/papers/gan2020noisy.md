---
id: gan2020noisy
title: "Noisy Agents: Self-supervised Exploration by Predicting Auditory Events"
authors: [Chuang Gan, Xiaoyu Chen, Phillip Isola, Antonio Torralba, Joshua B. Tenenbaum]
year: 2020
venue: arXiv preprint
url: https://arxiv.org/abs/2007.13729
arxiv: 2007.13729
doi:
pdf_sha256: 23c1276878250a3d7eb91b5dd19752f7e4d81634c57b14d7048faf3796a52cca
tags: [audio, multimodal, curiosity, exploration, intrinsic-reward, atari, sound-events, clustering, game-sound]
status: read
read: 2026-11-26
---

# Noisy Agents: Self-supervised Exploration by Predicting Auditory Events

## TL;DR

Clusters short game-audio clips (sound textures) with K-means into "auditory events", then rewards an agent for
events a frame-and-action forward model mispredicts. With no extrinsic reward it beats ICM, RND, RFN and
disagreement in 15 of 20 Atari games (Gym Retro). The paper describes audio only in the reward and predictor.

## Problem

Visual future prediction is hard and stochastic, and feature-space prediction error of sounds is a poor signal and diminishes (Sec. 1, p. 1-2, Sec. 3.2, p. 4). The paper wants an intrinsic reward from sound as a causal effect of actions.

## Method

- Two stages (Fig. 1, p. 3, Sec. 3.3, p. 4-5).
  1. Sound clustering: a first policy is rewarded for sound novelty, r_t = min_i ||Phi(s_{s,t+1}) - c_i||_2 over the current cluster centres (Eq. 2); clusters grow online. When the number of clusters saturates, K-means on the collected data (about 10K interactions) defines the auditory event classes.
  2. Auditory event prediction: a forward network f(Psi(image_t), a_t) predicts the class of the next sound (cross-entropy, Eq. 3); the prediction error is the intrinsic reward for the exploration policy.
- Audio representation: "sound textures" Phi(s) of each clip (Sec. 3.2, p. 4); regressing the textures directly was less effective (Sec. 3.2, p. 4, Sec. 4.5, p. 8).
- Atari via Gym Retro with its audio API: 84x84 gray frames, 60 ms audio clips, frame skip 4, PPO, 4-layer CNN policy, 3-layer CNN plus 2-layer MLP predictor (Sec. 4.1, p. 5-6).
- 3D physics: a rolling robot on a billiard table (Unity, TDW), 84x84 image, 50 ms audio, 9 actions, 200K steps (Sec. 4.1, p. 5-6, Sec. 4.4, p. 7-8).
- Hard-exploration runs combine intrinsic and extrinsic rewards RND-style with two value heads and normalised intrinsic reward (Sec. 4.3, p. 7).

## Assumptions

- A1: Audio is the game's own sound effects from the emulator, time-aligned with frames (Gym Retro audio API) (Sec. 4.1, p. 5).
- A2: The predictor sees frames and actions and audio defines event classes and rewards; whether the policy also receives audio is not stated (reader's inference, Sec. 3.1, p. 4, Sec. 3.3, pp. 4-5, Sec. 4.1, p. 6).
- A3: Sound effects correlate with game events or achievements, which the authors say is a bias of Atari design (p. 9).
- A4: Atari games are drawn from 20 games with sound effects and 5 hard-exploration games; training lengths are 10M steps (no extrinsic reward), 4M (hard exploration) and 2M (ablations) (Sec. 4.2, p. 6, Sec. 4.3, p. 7, Sec. 4.5, p. 8).
- A5: The 3D physics scene uses TDW-synthesised collision sounds from object physical properties (Sec. 4.1, p. 6).

## Claims

- C1: Using intrinsic reward only (extrinsic reward used for evaluation), the method is significantly better than ICM, RFN, RND and DIS in 15 of 20 Atari games and comparable in the other 5, 10M steps, 8 environments, 3 seeds [A1, A2, A3, A4] (Sec. 4.2, pp. 6-7, Fig. 3, p. 6).
- C2: It falls short in games dominated by background music or noise that reflects no auditory event (examples named in the text: Freeway, Time Pilot) [A1, A3] (Sec. 4.2, p. 7).
- C3: App. A sorts the 20 games into event-driven, action-driven and background sounds and says the method helps on both event- and action-driven games, more on event-driven ones, and falls short of baselines on six games it calls background-noise or music games (CrazyClimber, MsPacman, Gopher, Tutankham, WizardOfWor, BattleZone). Two inconsistencies in the paper: its own Table 1 lists only CrazyClimber and BattleZone as background-dominated (MsPacman is event-driven; Gopher, Tutankham and WizardOfWor action-driven), and "falls short on six" conflicts with C1's "comparable in the other 5" [A1, A3] (App. A, p. 12, Table 1, Sec. 4.2, p. 7).
- C4: Combined with extrinsic reward on 5 hard-exploration games (Venture, Solaris, PrivateEye, Pitfall!, Gravitar; 4M steps, 32 environments), the agent earns about 4x the maximum episodic return on Venture and almost 30x on Private Eye against plain PPO [A1, A2, A3] (Sec. 4.3, p. 7, Fig. 4, p. 7).
- C5: In the 3D physics scene, auditory-event reward leads to more object interactions (collision events) than ICM and RND, whose intrinsic reward diminishes quickly [A2, A5] (Sec. 4.4, pp. 7-8, Fig. 5, p. 7).
- C6: Predicting event classes earns more extrinsic reward than predicting sound-texture features, and sound clustering alone is similar to feature prediction and worse than event prediction (Amidar, BattleZone, Carnival; 2M steps) [A1, A2, A3] (Sec. 4.5, p. 8, Fig. 6, p. 8).
- C7: Active, novelty-driven collection of sounds for clustering beats random exploration with the same 10K samples and K-means, and yields more diverse clusters [A1, A2] (Sec. 4.5, p. 8, Fig. 7, p. 8).
- C8: Giving baselines the sound by concatenating audio and visual features to predict the next image embedding does not match the method on 5 Atari games, so the gain is not just from multimodal input [A1, A2] (App. C, p. 13, Fig. 8, p. 13).

## Implementation notes

- PPO: rollout length 128, 4 minibatches, learning rate 2.5e-4, clip 0.1, entropy coefficient 0.01, lambda 0.95, gamma 0.99 (source: Table 2, App. B, p. 12).
- Number of clusters K, the sound-texture statistics, the audio sampling rate and the number of exploration samples before clustering (only "e.g. 10K") are not given (source: assumption; the project page noisy-agent.csail.mit.edu is the likely place, not checked).
- Sound textures follow McDermott and Simoncelli (the paper's reference [29], not in the KB) (source: Sec. 3.2, p. 4).

## Limitations / open questions

- Only synthetic environments; Atari sound correlates with achievements more than natural sound does (p. 9).
- Background music and noise break the method; "advanced audio processing" is left as future work (Sec. 4.2, p. 7).
- A two-stage pipeline: event classes are fixed after the first 10K interactions, so later new sounds are not new classes (reader's inference from Sec. 3.3, p. 4-5).
- The paper contradicts itself about which games the method fails on (C3), so its evidence on which kinds of game sound help is weak.
- Per-game numbers are only in curves; no tables, and results come from 3 seeds.
- Open question (reader's): shmups mix constant music with frequent action-driven effects; the paper's own split suggests music-heavy games are its failure case.

## Related

- related [[dean2020hear]]: concurrent audio exploration using audio-visual alignment, cited by this paper as concurrent work (Sec. 2, p. 3).
- related [[hegde2021listen]]: puts audio in the policy's input, which this paper does not.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
