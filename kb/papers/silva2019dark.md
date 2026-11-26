---
id: silva2019dark
title: "Playing Games in the Dark: An approach for cross-modality transfer in reinforcement learning"
authors: [Rui Silva, Miguel Vasco, Francisco S. Melo, Ana Paiva, Manuela Veloso]
year: 2019
venue: arXiv preprint
url: https://arxiv.org/abs/1911.12851
arxiv: 1911.12851
doi:
pdf_sha256: 068ddfe5e52f3c4ecee73cd23946db458cda10661490e007d6a0bbe985a24cfc
tags: [audio, multimodal, cross-modality-transfer, zero-shot, vae, latent-representation, synthetic-sound, reinforcement-learning]
status: read
read: 2026-11-26
---

# Playing Games in the Dark: An approach for cross-modality transfer in reinforcement learning

## TL;DR

Trains an associative multimodal VAE on (image, sound) pairs, learns a policy on the image latent only, then swaps
in the sound encoder at test time for zero-shot "playing in the dark". On a Doppler-effect pendulum and a synthetic
Space-Invaders-like game it beats an untrained agent; the sound is simulated, not recorded game audio.

## Problem

Generative-model approaches such as DARLA assume source and target domains have similar inputs (both camera images); the paper wants a policy that is learned on one input modality and reused when only a different modality (sound) is available (Sec. 1, p. 1-2, Sec. 4.3, p. 7).

## Method

- Three stages (Sec. 3, p. 3-4, Fig. 5, p. 4): (1) learn a multimodal latent space Z with latent maps F (encoders) and inverses on a dataset of simultaneous observations collected with a random controller; (2) learn a policy pi: Z -> A with any continuous-state RL algorithm using only the training modalities' encoder, VAE weights frozen; (3) at test time encode whatever modality is available into Z and reuse pi with no further training (zero-shot).
- Model: Associative VAE (AVAE), a VAE per modality plus a symmetric-KL term (weight alpha) pulling the two latents together (Sec. 2.1.2, p. 3). JMVAE and M2VAE are mentioned as options, not used (Sec. 4.3, p. 8).
- Sound is synthetic (Sec. 4.1-4.2, p. 5-7). Pendulum: the tip emits a constant 440 Hz tone heard by 3 microphones; each hears the Doppler-shifted frequency and an inverse-square amplitude, and the sound input is just these (frequency, amplitude) numbers, not a waveform. Hyperhot: a pygame shooter with 4 emitter types (left enemies, right enemies, enemy bullets, own bullets) at 261, 329, 392, 466 Hz, 4 microphones, summed sinusoids of 1047 samples per frame at 31,400 Hz and 30 fps, 16-bit, amplitude decaying with distance.
- Reward in hyperhot: +10 for killing all enemies, -1 for death or timeout, else 0 (Sec. 4.2, p. 6).
- Pendulum: AVAE with 60x60 gray images (2 conv, 2 FC) and a 2-layer FC(50) sound encoder; DDPG. Hyperhot: 80x80 gray images (3 conv, 2 FC), FC(512) sound encoder; DQN (Sec. 4.1.1-4.2.3, p. 5-7). Latent size 10 (pendulum) and 40 (hyperhot) (Tables 3-4, p. 8).

## Assumptions

- A1: The sound is simulated from game state (physics formulae or summed sinusoids), clean, not recorded game audio (Sec. 4.1, p. 5, Sec. 4.2, pp. 6-7).
- A2: A paired image-sound dataset (20,000 samples pendulum, 32,000 hyperhot) is collectable with a random controller that covers the state space (Sec. 4.1.1, p. 5, App. A.1, p. 8).
- A3: Sound is informative enough about the state to act on: each emitter type has its own fixed frequency and the microphones are placed to localise sources (Sec. 4.2, p. 6).
- A4: Source and target tasks are the same MDP; only the observation modality changes at test time (Sec. 3, p. 3).
- A5: Both tasks are tiny (60x60 or 80x80 grayscale, pendulum and a custom shooter with few entities) (Sec. 4.1.1, p. 5, Sec. 4.2.1, p. 7).

## Claims

- C1: On the pendulum with sound only at test, AVAE-image-trained DDPG gets -2.00 +- 0.97 average reward over a 300-frame episode (10 seeds, 75 episodes), against -6.30 +- 0.29 for an untrained agent and -1.41 +- 0.91 for DDPG trained directly on sound [A1, A2, A3, A4, A5] (Sec. 4.1.3, p. 6, Table 1).
- C2: The paper states "an out-of-box performance improvement of over 300%" over the untrained agent in pendulum [A1, A2, A3, A4, A5] (Sec. 4.1.3, p. 6).
- C3: On hyperhot with sound only at test, the transferred DQN gets discounted reward 0.15 +- 0.16 and 36.10 +- 10.38 percent wins (10 seeds, 100 episodes), against -0.33 +- 0.16 and 8.30 +- 5.75 for random, 0.10 +- 0.22 and 27.30 +- 21.44 for DQN trained on sound, 0.21 +- 0.11 and 43.20 +- 7.03 for the same pipeline run on images, and 1.54 +- 0.20 and 75.00 +- 5.33 for DQN trained directly on images [A1, A2, A3, A4, A5] (Sec. 4.2.3, p. 7, Table 2).
- C4: In hyperhot the transferred policy's win rate is 36.1 vs 8.3 percent, about 4.3 times random's (the paper says 5-fold), and comparable to (the paper says slightly above in reward) a sound-trained DQN of the same architecture and steps; the authors note the sound-DQN might improve with tuning [A1, A2, A3, A4, A5] (Sec. 4.2.3, p. 7).
- C5: The best score is by the agent trained on the most informative modality directly (image DQN); cross-modal transfer costs performance, and the paper argues the trade-off may be worthwhile [A1, A5] (Sec. 4.2.3, p. 7, Table 2).
- C6: The results are from two custom environments; the paper does not test real game audio or Atari audio, despite the opening motivation with video games [A1] (Sec. 4, pp. 5-7, Sec. 5, p. 8).

## Implementation notes

- Pendulum: lambda_image = lambda_sound = beta = alpha = 1.0, batch 128, 500 epochs, AVAE lr 1e-3; DDPG batch 128, lr actor 1e-4, critic 1e-3, gamma 0.99, replay buffer 25,000, 150,000 frames, tau 1e-3, frame stack 2 (source: Table 3, p. 8).
- Hyperhot: sound amplitudes a0 = 1.0 (a_M 1.0 as printed), delta 0.025, latent 40, lambda_image 0.02, lambda_sound 0.015, beta 1e-5, alpha 0.05, batch 128, 250 epochs, AVAE lr 1e-3, 32,000 samples; DQN lr 1e-5, gamma 0.99, replay 350,000, 1,750,000 frames, 450-frame episodes, frame stack 2 (source: Table 4, p. 8).
- Images are grayscale and resized, sounds normalised to [0, 1] by sample min and max (source: Sec. 4.1.1, p. 5, Sec. 4.2.1, p. 7).
- Swish activations (pendulum), ReLU (hyperhot) (source: Sec. 4.1.1, p. 5, Sec. 4.2.1, p. 7).
- No code link is given in the paper (source: reader's reading of the whole PDF).

## Limitations / open questions

- Sound in both tasks is a clean synthetic signal; whether the approach works for recorded game audio (sound effects plus music, mixed) is untested.
- Only a single-game, same-MDP modality swap; no transfer across levels or games.
- High variance in the hyperhot sound-trained baseline (+-21.44 win percent) makes comparisons noisy (Table 2, p. 7).
- Only AVAE was run, though the abstract claims results hold with different multimodal generative models (abstract, p. 1, Sec. 4.3, p. 8); two RL algorithms were used, DDPG and DQN.
- Open question (reader's): the latent z must be informative from sound alone; in a shmup with continuous music and overlapping effects, sound likely does not determine player and bullet positions.

## Related

- related [[gan2020noisy]]: also uses game sound, but for exploration rewards on Atari.
- related [[hegde2021listen]]: audio as direct policy input in a real game (Doom) rather than via a shared latent.
- related [[kirk2023survey]]: zero-shot generalisation survey; this paper's cross-modality transfer is a case of changed observations.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
