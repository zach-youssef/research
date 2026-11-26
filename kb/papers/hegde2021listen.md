---
id: hegde2021listen
title: "Agents that Listen: High-Throughput Reinforcement Learning with Multiple Sensory Systems"
authors: [Shashank Hegde, Anssi Kanervisto, Aleksei Petrenko]
year: 2021
venue: IEEE CoG 2021 conference
url: https://arxiv.org/abs/2107.02195
arxiv: 2107.02195
doi:
pdf_sha256: 59d8add80886165f0d3d7a19c0e23e6aa651e895d63e14727352cb36705301c4
tags: [audio, multimodal, reinforcement-learning, vizdoom, spectrogram, fft, high-throughput, game-sound]
status: read
read: 2026-11-26
---

# Agents that Listen: High-Throughput Reinforcement Learning with Multiple Sensory Systems

## TL;DR

Adds software-rendered stereo audio to ViZDoom at about 0.8x the throughput of the silent version, and compares
three encoders on raw 114 ms audio windows. A log-FFT-magnitude MLP trains best; with it, agents solve
sound-finding and spoken-instruction tasks and beat a vision-only agent in Doom duels (53-31-16).

## Problem

RL research mostly uses vision only; environments with sound are either missing or too slow for large-scale training (Sec. I, p. 1). ViZDoom's own audio is tied to real-time playback (Sec. II, p. 2). The paper wants a fast audio-visual environment and a simple way to feed raw audio to a policy.

## Method

- Environment: ViZDoom audio through OpenAL Soft with the loopback extension, so sound is rendered on the CPU decoupled from audio hardware and runs faster than real time; the thread-local-context extension lets hundreds of game processes render sound at once (Sec. III, p. 2).
- Raw stereo samples in [-1, 1] are exposed with configurable audio frame-stacking and sampling rate; the paper used 22050 Hz. One sim step is 29 ms of game time (35 fps); with frameskip 4 the observation is 2520 samples = 114 ms (Sec. III-IV, p. 2).
- Encoders, applied to left and right channels separately and concatenated (Sec. IV, p. 2-3, Fig. 2):
  - 1D Conv: keep every 8th sample (drops content above about 3000 Hz), two 1D conv layers; the naive baseline.
  - Fourier transform: log of FFT magnitudes, 1D max-pool downsampling, two-layer MLP; discards temporal structure inside the 114 ms.
  - Mel-spectrogram: STFT with 25 ms window, 10 ms hop, 80 mel bins (parameters from a speaker-recognition paper), then two 2D conv layers.
- The audio vector is concatenated with the 3-layer conv image features, then a GRU and a linear head for actions and value; PPO with V-trace in the Sample Factory framework, with hyperparameters from the Sample Factory paper (Sec. V, p. 3).
- Scenarios, one map with six coloured pillars in four rooms (Sec. V-A, p. 3, Fig. 4): Music Recognition (each pillar loops a music track, touch the one with the target track, high attenuation so the agent must get close); Sound Instruction (a spoken English command "go to the green pillar" repeats through the episode); Sound Instruction Once (the command is said once at the start); Duel (1v1 self-play in full Doom).

## Assumptions

- A1: The setting is a 3D first-person game (ViZDoom) with a recurrent policy trained by on-policy PPO at hundreds of millions to billions of frames (Sec. V, pp. 3-4).
- A2: Audio is rendered by the game's own engine (OpenAL, with reverberation, Doppler and gaze-dependent dampening) from a limited bank of sounds (Sec. III, p. 2, Sec. VIII, p. 5).
- A3: Training and evaluation use the same sound bank and map layout, with randomised object placement and start (reader's inference, Sec. V-A, p. 3, Sec. VIII, p. 5).
- A4: Frames are 128x72 RGB with frameskip 4 (2 in Duel) (Sec. V-B, p. 4).

## Claims

- C1: Software-rendered OpenAL audio in ViZDoom gives training throughput of 1.2e5 frames/s per experiment with sound against 1.5e5 without, on a 36-core, 4-GPU machine with 576 parallel environments per experiment, 4 experiments at once [A1] (Sec. VII, p. 4).
- C2: In the multi-agent Duel setup, throughput is the same with and without sound (6.7e4 combined for a 4-policy population), since network communication between game instances is the bottleneck [A1] (Sec. VII, p. 4).
- C3: On Music Recognition, the Fourier-transform encoder learned fastest, followed by the mel-spectrogram encoder, then the 1D-conv encoder (high variance), while the no-sound agent stayed near zero success; all sound encoders reached about 1.0 within 5e8 frames, by reading Fig. 3a [A1, A2, A3] (Sec. VI, p. 4, Fig. 3a, p. 3).
- C4: With the FFT encoder, Sound Instruction reaches about 0.5 mean success and Sound Instruction Once about 0.8 after 1e9 frames (read off Fig. 3b-c), with most runs reaching optimal performance but high variance across 5 seeds; vision-only stays near zero [A1, A2, A3] (Sec. VI, p. 4, Fig. 3b, Fig. 3c, p. 3).
- C5: In 100 four-minute Duel matches, the sound agent beat a separately trained vision-only agent 53 wins, 31 losses, 16 draws; the same agent with audio replaced by silence at evaluation lost 74-17 with 9 draws against the normal version [A1, A2] (Table I, p. 4).
- C6: The Duel agent used less loud ammunition (hypothesised by the authors as concealing its position) and used stereo to locate the opponent's gunfire; the paper calls these observations and does not test them [A1] (Sec. VI, p. 4).
- C7: The authors conclude that log-FFT of a short window is sufficient for fast RL training with a recurrent policy; they call the work preliminary, with one RL algorithm and three encoders, untuned [A1, A2] (Sec. VIII, pp. 4-5).

## Implementation notes

- Sampling rate 22050 Hz, window per action 2520 samples (frameskip 4); stacking length is configurable and could be longer for feed-forward policies (source: Sec. III-IV, p. 2).
- Image 128x72, frameskip 4 (2 for Duel); training budgets 5e8 (Music Recognition), 1e9 (Instruction), 2e9 (Duel) frames (source: Sec. V-B, p. 4).
- Duel uses population-based training with 4 policies and self-play (source: Sec. VI, p. 4).
- Layer widths for each encoder are in Fig. 2 (e.g. 1D conv 32 filters, kernel 16, stride 8 then 16 filters; FC 256/512); full details deferred to the Sample Factory appendix (source: Fig. 2, p. 2; Sample Factory, Petrenko et al. 2020, not in the KB).
- Code and the augmented simulator are said to be in the process of being merged into ViZDoom; demos and experiment code at a project page (source: abstract, p. 1).
- The FFT magnitude scaling beyond "natural log" (no epsilon given) is unstated (source: assumption).

## Limitations / open questions

- Preliminary; one algorithm, three encoders, little tuning; a small sound bank, so overfitting to exact samples cannot be ruled out; the authors suggest noise augmentation (Sec. VIII, p. 5).
- It is unclear how exactly the agent uses sound, or how well it localises the opponent in 3D (Sec. VIII, p. 5).
- Unclear whether agents understand long audio sequences (Sec. VIII, p. 5).
- (Reader's) No transfer test to unseen sounds, maps or games; this is a single-game (Doom) result, with speech only as a spoken command and music only as loops.
- Open question (reader's): the duel comparison is one trained agent per condition, with no seeds or confidence intervals (Table I, p. 4).
- Open question: for shmup sound effects, only the FFT-versus-mel comparison on 114 ms windows is informative; the paper has no ablation of window length.

## Related

- related [[he2021masked]]: MAE-style pretraining is a possible way to learn audio features instead of a fixed FFT front end (reader's link, not in this paper).
- related [[lee2022multigame]]: a transformer multi-game policy from frames, which could take audio tokens alongside frames.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
