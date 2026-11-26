---
id: omidshafiei2018crossmodal
title: "Crossmodal Attentive Skill Learner"
authors: [Shayegan Omidshafiei, Dong-Ki Kim, Jason Pazis, Jonathan P. How]
year: 2018
venue: AAMAS 2018 conference
url: https://arxiv.org/abs/1711.10314
arxiv: 1711.10314
doi:
pdf_sha256: 1849458707d05d4f2d0721ede7e4afc6398680e0fe6efa93e613b7d97e0b615f
tags: [audio, multimodal, attention, options, hierarchical-rl, spectrogram, atari, transfer]
status: read
read: 2026-11-26
---

# Crossmodal Attentive Skill Learner

## TL;DR

CASL adds a learned per-step attention over video and audio-spectrogram features to the A2OC option-critic agent.
It speeds up learning in a door puzzle (and transfer to a randomised variant) and is needed in a Minecraft-like
mining task. A preliminary Atari result (audio+video, no options) scores 900 in Amidar and 32985 in H.E.R.O.

## Problem

Agents with many sensors should attend to useful modalities and time-scales instead of processing everything; audio and video had not been used for crossmodal skill learning in RL (Sec. 1, p. 1-2). Note: the brief called this "Kim et al. 2019"; the paper is Omidshafiei, Kim, Pazis and How, AAMAS 2018 (arXiv v3, May 2018).

## Method

- Per modality m, a feature extractor gives x_m (3 conv layers of 32 3x3 filters, stride 2, ReLU) (Sec. 4.1, p. 3-4). The paper shows the audio input as a spectrogram image (Fig. 5a, p. 6).
- Attention layer: z_t = tanh(sum_m W_m^T x_m + b_m + W_h^T h_{t-1} + b_h) mixes exogenous (feature-driven) and endogenous (LSTM-state-driven) terms; alpha_t = softmax(W_z^T z_t + b_z) lies on the simplex (Eq. 4-5, Sec. 4.1, p. 3-4).
- Attended features are summed or concatenated (Eq. 6), fed to an LSTM; heads give option values Q_Omega, intra-option policies and terminations (Eq. 7-9). Entropy regularisation of alpha encourages exploring attention behaviours (Sec. 4.1, p. 4).
- Built on A2OC (asynchronous advantage option-critic with deliberation cost) (Sec. 3.2, p. 3).
- Analysis tool: forget/input-gate residuals and a "pseudo correlation" of each input's contribution (Eq. 10-15, Sec. 5.2, p. 5-6).
- Domains (Sec. 5, p. 4-7): a door puzzle (fixed sound near the key, noise otherwise; audio not necessary), a 2D mining domain (ore type visible only through audio), and ALE modified to return audio (Amidar, H.E.R.O.).

## Assumptions

- A1: The task is Atari or custom toy 2D domains with a recurrent (LSTM) policy; the toy domains' audio is designed and synthetic (Sec. 5.1-5.2, pp. 4-5).
- A2: In the toy domains the audio carries a fixed, clean cue (a tone near the key or ore) that is easy to separate from noise (Sec. 5.1, p. 4, Sec. 5.2, p. 5).
- A3: In Atari, audio cues exist that mark game events (previously uncrossed vertices in Amidar), while background audio plays throughout (Sec. 5.3, p. 7).
- A4: The Atari numbers are preliminary, from a single option, 30 test runs in Amidar, and are not direct comparisons with video-only methods (Sec. 5.3, p. 7).

## Claims

- C1: In the door puzzle, all three LSTM-based option learners converge to the optimal value, the video-only one converges faster than the audio-video one (audio overwhelms it), and CASL converges fastest with roughly half the training data; the feed-forward nets do not reach the optimum [A1, A2] (Sec. 5.1, p. 4, Fig. 2a, p. 4).
- C2: In a transfer variant (randomised key position, options pretrained on the fixed position), all networks gain a jumpstart of nearly 0.2 return and CASL converges fastest; asymptotic return is only slightly above V-A-O-LSTM [A1, A2] (Sec. 5.1, pp. 4-5, Fig. 2b, p. 4).
- C3: In the mining domain, where images of the two ore types are identical and audio must be stored over time, the network without attention fails to learn while CASL succeeds; the agent raises audio attention as it approaches the ore and drops it to 0 after hearing the cue [A1, A2] (Sec. 5.1, p. 5, Fig. 4, p. 5, Fig. 5a, p. 6).
- C4: Gate analysis shows audio's contribution to the LSTM forget and input gates is about zero before the cue (despite positive attention), then forget activation drops and input rises at t = 6 as audio overwrites memory, and audio's contribution returns to zero afterwards [A1, A2] (Sec. 5.2, pp. 5-6, Fig. 5).
- C5: In Atari with audio (no options), CASL scores 900 mean in Amidar (30 test runs) and 32985 in H.E.R.O., against DQN 740 / 19950, A3C 284 / 28766, GA3C 218 / -, A2OC 880 / 20100, FeUdal over 2500 / about 36000; the authors stress the comparisons are not direct because CASL has extra inputs [A3, A4] (Sec. 5.3, p. 7, Table 1, p. 7).
- C6: In Amidar the agent increases audio attention near path vertices that make a sound when first crossed, which are hard to see when the agent sprite overlaps them [A3] (Sec. 5.3, p. 7, Fig. 6, p. 7).

## Implementation notes

- Code: github.com/shayegano/CASL, a hybrid CPU-GPU implementation building on GA3C (source: footnote 1, p. 1).
- ALE was modified to support audio queries (source: Sec. 5.3, p. 6-7).
- How the audio is converted to a spectrogram (window, hop, sample rate, bins) is not stated, nor are the Atari frame size, option count or training budgets (source: assumption; check the authors' code).
- The audio feature extractor is assumed to match the 3-conv visual one (source: assumption, from Sec. 4.1, p. 4).

## Limitations / open questions

- The Atari results are called preliminary; only 2 games, and no ablation of audio against video-only in Atari with the same architecture (Sec. 5.3, p. 7).
- In the door puzzle, adding audio without attention slowed learning (Fig. 2a).
- FeUdal beats CASL in both games; combining the approaches is future work (Sec. 5.3, p. 7).
- Audio is fed as a spectrogram image to a CNN with a recurrent state; per-step latency or token rates are not reported.
- Open question (reader's): the attention result is on cleanly designed toy audio; whether it filters game music and overlapping effects is untested.

## Related

- uses [[bacon2017option]]: builds on the option-critic framework through A2OC (Sec. 3.2, p. 3).
- related [[hegde2021listen]]: audio as policy input in Doom with FFT features.
- related [[dean2020hear]]: Atari audio via Gym Retro for curiosity.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
