---
id: nadeem2021musical
title: "Learning from Musical Feedback with Sonic the Hedgehog"
authors: [Faraaz Nadeem]
year: 2021
venue: ISMIR 2021 conference
url: https://archives.ismir.net/ismir2021/paper/000059.pdf
arxiv:
doi:
pdf_sha256: 1d0083bea068052b6568b1cedc914256fe8786c0af462c65b1fb6929466465d2
tags: [audio, multimodal, mel-spectrogram, sonic, genesis, gym-retro, zero-shot, generalization, ppo, game-sound, music]
status: read
read: 2026-11-26
---

# Learning from Musical Feedback with Sonic the Hedgehog

## TL;DR

Exposes Genesis audio in Gym Retro and feeds the last 1 s of audio as a mel-spectrogram image to a CNN beside
one video frame, PPO on 47 Sonic levels. Audio+video beat a 2-frame video agent by 6.6% on training levels
and, by Table 2, 38% on 11 unseen levels (the text says 20.4%). Small scale: 30M steps, 3 seeds, large stds.

## Problem

Video-game RL almost always ignores audio, though humans play with it (Sec. 1, p. 1). The paper builds tooling to get emulator audio and tests whether audio helps joint training and zero-shot transfer to unseen levels in Sonic The Hedgehog (Genesis).

## Method

- Tasks (Sec. 4.2, p. 3): joint PPO training on 47 of 58 levels for 30M steps (evaluated every 5M); zero-shot evaluation of that agent on the 11 test levels, with no fine-tuning; each test level shares a Zone with at least one training level.
- Environment: Gym Retro 0.8.0 patched to expose emulator audio (code released), with the Gotta Learn Fast setup (Nichol et al.): save states, stochastic frame skips, reduced action space, reward for progress rightwards. Compute-cut: horizon 512 (not 8192), 3 workers per level (not 4), 30M steps (not 400M) (Sec. 4.3-4.4, p. 3).
- Agent variants (Sec. 4.5, p. 3, Table 1, p. 4): A1 = benchmark-like RGB agent, 14M parameters; A2 = grayscale plus a random start of 16-45 random actions, 14M; A3 = A2 plus the previous frame, hidden size 512, 28M; A4 = one gray frame plus audio, 31M; A5 = like A4 but a coarser spectrogram, 18M.
- Audio features: the past 16 frames of audio without frame skip, at 44,100 Hz is 735 samples per frame, 47,040 per observation (about 1 s). Mel spectrogram, 256 mels: A4 window 256, hop 128 gives 367x256 (text elsewhere says 368x256); A5 window 1024, hop 512 gives 92x256. Power to dB with an 80 dB range, divided by 80 into [0, 1]. A separate CNN (same architecture as the video one) encodes audio, hidden size 256 per stream, concatenated to 512 (Sec. 4.5, p. 3, Sec. 5.3.1, p. 5).
- Baselines: "hold right" policy; the human and Nichol et al. results from the benchmark paper (Sec. 4.6, p. 3-4).
- Analysis: Score-CAM heat maps on audio and video (Sec. 5.1, p. 4).

## Assumptions

- A1: The game is one 2D platformer series (Sonic 1-3 and Knuckles on Genesis) with a progress-based reward; audio is the emulator's native 44.1 kHz mixed output including music and effects (Sec. 4.3-4.5, p. 3); that it is stereo is not stated.
- A2: Each unseen level shares a Zone with at least one training level (Sec. 4.2, p. 3); sharing of art and sounds is the reader's inference.
- A3: Scores are means over three seeds, with the standard deviation taken across per-level averages, which is large (Sec. 5.1, p. 4).
- A4: Training is 30M steps with a reduced horizon and worker count, so absolute scores are far below the benchmark's 400M-step agent (Sec. 4.4, p. 3, Table 2, p. 4).
- A5: Reward and action space are the benchmark's modified versions, not the game's score (Sec. 4.4, p. 3).

## Claims

- C1: On the joint task A5 (one gray frame plus a 92x256 mel spectrogram of the last second of audio, 18M parameters) reaches 2031.1 +- 501.9 final average, against 1905.0 +- 286.9 for the 2-frame video agent A3 (28M parameters), +6.6% [A1, A3, A4, A5] (Sec. 5.1, p. 4, Table 2).
- C2: On the 11 unseen levels A5 reaches 936.6 +- 220.8 against 678.5 +- 238.3 for A3; this is +38%, whereas the text (p. 4) and the abstract (p. 1) state +20.4%, which matches A4 (817.3) against A3 instead [A1, A2, A3, A4, A5] (Sec. 5.1, p. 4, Table 2, p. 1).
- C3: A4 (larger spectrogram, 31M parameters) is on par with A3 on the joint task (1893.7 vs 1905.0) and higher on zero-shot (817.3 vs 678.5); the finer spectrogram was more prone to noise and overfitting than A5's coarser one [A1, A3, A4, A5] (Table 2, p. 4, Sec. 5.3.1, p. 5).
- C4: Using best checkpoints per level, the audio agents' advantage largely disappears: joint 3041.4 (A4) and 2901.5 (A5) against 3013.0 (A3); zero-shot 1779.2 (A4) and 1756.9 (A5) against 1686.9 (A3), with standard deviations above 1100 [A1, A2, A3, A4, A5] (Table 2, p. 4).
- C5: The authors conclude that game audio informs decision making and that audio features transfer to unseen levels more easily than video features [A1, A2, A3, A4] (Sec. 5.1, p. 4, Sec. 6, p. 6).
- C6: Ablations among video-only variants: grayscale plus random starts raises A1 to A2 (1344.6 to 1479.5 joint, 435.6 to 578.5 zero-shot); adding the previous frame raises A2 to A3 by 29% and 17% [A1, A3, A4] (Sec. 5.2, p. 4, Table 2).
- C7: Score-CAM analysis finds the audio CNN picks up ring-collect and ring-loss sounds (hard to see visually) and raises the "move right" likelihood on them, and the drowning theme, after which the agent seeks an air bubble; the video-only agents do not act on drowning cues; the agent also leaves water after the 5-second ping in some levels [A1] (Sec. 5.3.2, p. 5, Fig. 5, Fig. 6, p. 6).
- C8: Many features Score-CAM highlights are from the background music, which the author hypothesises are spurious reward attribution; jump and enemy-defeat sounds, and consonance versus dissonance, were not learned (a companion character makes the same effects, and the sound set is small) [A1] (Sec. 5.3.2, p. 5, Sec. 5.3.3, p. 6).
- C9: Hold-right scores 1099.1 +- 1092.8 joint and 321.9 +- 277.5 zero-shot; the benchmark's 400M-step agent 5083.6 joint and about 1000 zero-shot; humans 7438.2 zero-shot [A1, A4, A5] (Table 2, p. 4).

## Implementation notes

- Hop and window: A4 256/128, A5 1024/512, 256 mels, 44.1 kHz, dB range 80 then divide by 80 (source: Sec. 4.5, p. 3).
- Audio window is the past 16 frames with no frame skipping on the audio path (source: Sec. 4.5, p. 3).
- PPO from Kostrikov's pytorch-a2c-ppo-acktr (source: Sec. 4.1, p. 3, reference 25, p. 7); learning rate and other hyperparameters are not given (source: assumption).
- Code and gameplay videos at github.com/faraazn/meng (source: footnote, p. 1).
- Spectrogram size is given as 367x256 in Sec. 4.5 and 368x256 in Sec. 5.3.1; the discrepancy is the paper's (source: p. 3, p. 5).
- Rate: 47,040 samples = 16 x 4 x 735 implies frame skip 4 (about 67 ms per step); each step recomputes a spectrogram over about 1.07 s of audio (source: reader's derivation from Sec. 4.5, p. 3; the paper doesn't state latency).

## Limitations / open questions

- Single series of games, 3 seeds, 30M steps; the standard deviation is across levels, so no significance test; the text's 20.4% does not match Table 2 for A5 (C2).
- A3 has a bigger model than A5 but A3 differs from A5 in the second input; there is no audio-with-previous-frame variant, so audio is not compared against a matched 1-frame video agent except A2 (A2 1 frame, 14M: joint 1479.5, zero-shot 578.5).
- The author notes that the audio CNN may attribute reward to the wrong cues and that music cues could be reinforced coincidentally (Sec. 5.3.2, p. 5, Sec. 5.3.4, p. 6).
- Transfer across games (Sonic to Mario) is future work (Sec. 6, p. 6).
- Open question (reader's): zero-shot here is unseen levels of the same game and Zone; nothing is shown about other games.

## Related

- related [[hegde2021listen]]: spectrogram or FFT of short windows as policy input in Doom.
- related [[omidshafiei2018crossmodal]]: AAMAS 2018 conference version of the "Kim et al." JAAMAS 2020 work this paper cites as [4]; same idea of an audio spectrogram as a second CNN stream.
- related [[kirk2023survey]]: zero-shot generalisation survey; this is a level-generalisation setting.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
