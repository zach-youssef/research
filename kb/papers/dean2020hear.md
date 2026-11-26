---
id: dean2020hear
title: "See, Hear, Explore: Curiosity via Audio-Visual Association"
authors: [Victoria Dean, Shubham Tulsiani, Abhinav Gupta]
year: 2020
venue: NeurIPS 2020 conference
url: https://arxiv.org/abs/2007.03669
arxiv: 2007.03669
doi:
pdf_sha256: 7733a003b99be239734bb965ac36a3a182595ff392cf179f699193b97f7bb649
tags: [audio, multimodal, curiosity, exploration, intrinsic-reward, atari, habitat, fft, game-sound]
status: read
read: 2026-11-26
---

# See, Hear, Explore: Curiosity via Audio-Visual Association

## TL;DR

Intrinsic reward from an audio-visual discriminator: it learns whether a frame feature and an FFT audio feature
are aligned, and the agent (which sees only frames) is rewarded where it is surprised. Without extrinsic
reward it beats future-prediction curiosity in 8 of 12 Atari games and covers a Habitat apartment about 3x faster.

## Problem

Prediction-based curiosity needs a forward model, which is hard to learn and gets stuck on stochastic inputs (the "couch potato" TV) (Sec. 1, p. 1, Sec. 3.1, p. 3). The paper replaces prediction with a simpler question: does this image co-occur with this sound?

## Method

- Per step the environment gives a visual feature v_t and a sound feature s_t; the policy sees only v_t, so comparisons with visual-only baselines are fair (Sec. 3, p. 3).
- Discriminator D(v_t, s_t) outputs the probability that the pair is aligned. Positives are real pairs; negatives (half of the samples) pair v_t with a sound feature drawn uniformly from the same 128-step trajectory. The negative-case cross-entropy is weighted by the L2 distance between true and false audio features, normalised by the batch mean, so similar-sounding fakes are not penalised (Sec. 3.3, p. 4-5, Fig. 2, p. 4).
- Intrinsic reward r_t = -log D(v_t, s_t); the agent maximises its discounted sum with PPO and never sees extrinsic reward, which is used only for evaluation (Sec. 3.3, p. 5).
- Audio features: waveform to FFT, max-pooled to 512 dimensions; visual features: a fixed random CNN, 512-d, as in large-scale-curiosity; the discriminator is 3 dense layers (Sec. 4.1, p. 5, App. B, p. 13).
- Atari audio comes from Gym Retro, a clip spanning 4 steps (1/15 s at 60 fps); 530 samples per step, 2120 with frameskip 4. Habitat uses the audio-visual navigation extension with a fixed clip from a fixed location, at most 1 s, zero-padded to 66150 samples (Sec. 4.1, p. 5, App. B, p. 13).
- Baselines: future prediction, disagreement, RND, all with the same policy hyperparameters (Sec. 4.2, p. 6).

## Assumptions

- A1: Audio is the game's own sound captured from the emulator (Gym Retro) or a simulator, aligned in time with frames (Sec. 4.1, p. 5).
- A2: Games with no audio, or with background music, are excluded (Amidar, Pong, RoadRunner, Super Mario Bros) (Sec. 4.1, p. 5).
- A3: The task is reward-free exploration, with score used only as an exploration proxy, trained 200M frames (Sec. 4.3, p. 6).
- A4: Atari uses the Gym Retro action space Discrete(6) with frameskip 4 and 84x84 grayscale stacked frames (Sec. 4.1, p. 5, App. B, p. 13).
- A5: The audio-visual pairing is informative: sounds follow visible events without long delay (reader's inference, Sec. 4.3, p. 6).

## Claims

- C1: On 12 Atari games (3 seeds, 200M frames, extrinsic reward used only for evaluation), SHE beats disagreement in 9 and future-prediction curiosity in 8 [A1, A2, A3, A4] (Sec. 4.3, p. 6, Fig. 4, p. 7).
- C2: SHE underperforms where association is trivial (Qbert, Atlantis: the same sound for every move, so reward falls) or where background sounds are periodic and hard to tie to frames (SpaceInvaders, BeamRider), or delayed (Alien) [A1, A2, A4, A5] (Sec. 4.3, p. 6).
- C3: On Gravitar, a hard-exploration game, SHE explores well while the baselines' extrinsic reward barely grows; other hard-exploration games were not solved by any method [A1, A2, A3, A4] (Sec. 4.3, p. 7, Fig. 5, p. 7).
- C4: In Habitat (Replica Apartment 0, 211 locations, 3 seeds), SHE has coverage similar to RND and reaches full coverage about 3x faster than future prediction, and visits rare states about 8x more than the next-best baseline [A1, A3] (Sec. 4.4, pp. 7-8, Fig. 6, p. 8).
- C5: Adding audio to the baselines does not help: audio-visual prediction performs like the visual one, and audio-visual RND on Gravitar likewise, so the gain comes from association and not from extra input [A1, A2, A3, A4] (Sec. 4.5, p. 8, App. D, p. 14, Fig. 9, Fig. 10).
- C6: With Gaussian noise added to audio and visual feature vectors, future-prediction curiosity degrades in Asterix and SpaceInvaders while SHE stays similar (3 environments only) [A1, A2, A3, A4] (Sec. 4.5, p. 8, Fig. 7, p. 9).
- C7: Summing SHE and future-prediction losses beats future prediction alone in 10 of 12 games, e.g. final reward 593 against 108 (SHE) and 359 (future prediction) in Alien [A1, A2, A3, A4] (Sec. 4.5, p. 8, App. C, pp. 13-14, Fig. 8).
- C8: With sticky actions (repeat probability 0.25) SHE shows trends like the deterministic runs [A1, A2, A3, A4] (App. E, p. 15, Fig. 11).

## Implementation notes

- Learning rate 1e-4, gamma 0.99, 3 optimisation epochs per rollout, 128 parallel environments (Atari) and 1 (Habitat); 84x84 grayscale, 4-frame stack (source: App. B, p. 13).
- Code at github.com/vdean/audio-curiosity, built on openai large-scale-curiosity, exploration-by-disagreement and random-network-distillation (source: App. A, p. 13).
- FFT window, sampling rate of Atari audio and the max-pool factor are not stated beyond "512-dimensional" (source: assumption; check the authors' code).
- Discriminator input is the concatenated 512-d visual and 512-d audio features, 2 hidden layers of 512 units (source: Fig. 2, p. 4, App. B, p. 13).

## Limitations / open questions

- Audio is used only to compute reward; the policy never hears it, so nothing here says how to feed audio into a policy.
- (Reader's) Games with background music or no sound were excluded; results are Atari (sound effects from Gym Retro) and Habitat only (Sec. 4.1, p. 5).
- Failure cases are games with trivial or delayed associations (Sec. 4.3, p. 6); the paper offers no remedy besides combining with prediction curiosity.
- Open question (reader's): for shmups, with constant background music and many simultaneous effects, the 2 failure modes (periodic background, trivial association) might dominate.
- No evidence on transfer across games or levels.

## Related

- related [[hegde2021listen]]: also uses FFT audio features, but as policy input in Doom.
- related [[kirk2023survey]]: surveys zero-shot generalisation, a different goal from exploration.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
