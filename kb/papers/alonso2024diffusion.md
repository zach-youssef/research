---
id: alonso2024diffusion
title: "Diffusion for World Modeling: Visual Details Matter in Atari"
authors: [Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, François Fleuret]
year: 2024
venue: NeurIPS 2024 (arXiv v2 read)
url: https://arxiv.org/abs/2405.12399
arxiv: 2405.12399
doi:
pdf_sha256: 842ff972404c74ca4bac9d2f37a6d309506c19eb3f2efa3e0daf75d6abc19707
tags: [world-model, diffusion, model-based-rl, atari, imagination, pixel-space, u-net, game-engine]
status: read
read: 2026-11-26
---

# Diffusion for World Modeling: Visual Details Matter in Atari

## TL;DR

DIAMOND trains an actor-critic entirely inside an EDM-diffusion world model (a U-Net conditioned on
the last 4 frames and actions, 3 denoising steps) that predicts the next frame in pixels. It reaches mean
human-normalised score 1.46 on Atari 100k, best among world-model agents; no transformer, no discrete latents.

## Problem

World models for RL mostly compress frames into discrete latents, which can drop visual details
that matter for the policy, and raising the token count costs compute (Sec. 1, pp. 1-2). The paper
asks whether a diffusion model, the dominant image generator, can serve as a stable and cheap
world model, and whether better visual detail improves agents (Sec. 1, p. 2).

## Method

- The world model is a conditional diffusion model of the next observation given past
  observations and actions (Eq. 5, Sec. 2.3, p. 4), with the denoiser trained by a plain L2
  loss (Eq. 4, p. 3) in the EDM parameterisation (Sec. 3.1, Eq. 6-7, p. 4; preconditioners Eq. 9-13, App. C, p. 16).
- Network: a 2D U-Net; the L=4 previous frames are stacked channel-wise with the noisy next frame;
  actions and noise level enter through adaptive group norm (Sec. 3.1, pp. 4-5; Table 2, p. 17).
- Sampling: Euler's method with n=3 denoising steps (3 network evaluations per frame) in all
  Atari experiments (Sec. 3.1, Sec. 5.2, p. 5, p. 7).
- Reward and termination come from a separate model (CNN plus LSTM) with cross-entropy losses on
  sign(reward) and end flags; the diffusion model doesn't predict them (Sec. 3.2, p. 5; Alg. 1, p. 20).
- Agent: shared CNN-LSTM actor-critic, REINFORCE with a value baseline and lambda-returns, trained
  only on imagined 15-step rollouts (Sec. 3.2, App. F, Eq. 14-16, p. 5, p. 19).
- Loop: each epoch collects 100 real steps with the current policy, then updates the diffusion
  model, the reward/termination model and the actor-critic from the replay dataset (Alg. 1, p. 20; Table 3, p. 18).
- CS:GO extension (Sec. 6, pp. 8-9): a model trained only on static human gameplay data, no RL.

## Assumptions

- A1: The environment emits image observations and has discrete actions (up to 18 in Atari); the paper's main evaluation is discrete control (Sec. 8, p. 10, Table 3, p. 18).
- A2: The agent can interact with the real environment for a small budget of 100k steps (about 2 hours of play) to collect the training data, with reward available (Sec. 4.1, p. 5).
- A3: Observations are 64x64x3 frames with frameskip 4, termination on life loss and rewards clipped to {-1, 0, 1} (Table 3, p. 18).
- A4: Conditioning on the last 4 frames is enough memory for the game; the authors call frame stacking a minimal memory mechanism (reader's inference, Sec. 8, p. 10).
- A5: A single consumer GPU (RTX 4090, about 12 GB of VRAM) is available per run, with about 2.9 days of training (Sec. 4.1, p. 5, Table 5, p. 22).
- A6: Atari games are close to deterministic from the frame stack and action, with the multimodality that remains handled by n=3 denoising (reader's inference, Sec. 5.2, p. 7).

## Claims

- C1: On Atari 100k (26 games, 5 seeds, 100k environment steps), DIAMOND reaches mean human-normalised score 1.459 and IQM 0.641, versus STORM 1.266/0.636, DreamerV3 1.097/0.497, IRIS 1.046/0.501, TWM 0.956/0.459 and SimPLe 0.332/0.130, and is superhuman on 11 games [A1, A2, A3, A4] (Table 1, Fig. 2, pp. 5-6).
- C2: It is not the best method on the benchmark: EfficientZero (lookahead search) has mean 1.943 and BBF (network resets) 2.247, both above DIAMOND's 1.459, though DIAMOND wins some individual games; the authors call these techniques orthogonal [A1, A2, A3, A4] (Table 6, App. J, p. 23, Sec. 4.1, p. 5).
- C3: The world model is smaller and faster to train than IRIS's: 13M parameters in total (the Atari diffusion model alone is 4M, Sec. 6, p. 9) against 30M (IRIS) and 18M (DreamerV3), with 2.9 days against 4.1 and under 1 day per run [A5] (Table 4, App. H, p. 21, Sec. 6, p. 9).
- C4: With the same network and a fixed 100k-frame expert dataset in Breakout, the DDPM-based model drifts into severe compounding error within 1000 steps at 10 or fewer denoising steps (the fewer the worse), while the EDM-based model stays stable even with one step [A1, A3] (Fig. 3, Sec. 5.1, pp. 6-7, Fig. 8, App. K, p. 24).
- C5: With one denoising step the model averages possible outcomes, so unpredictable motion (Boxing's black player) is blurred; three steps give crisp frames, and agents with n=1 score a mean HNS of 1.962 against 3.052 for n=3 on DIAMOND's 10 best games (the n=1 results are a single seed) [A1, A2, A3, A6] (Fig. 4, Sec. 5.2, p. 7, Table 7, App. L, p. 24).
- C6: Compared with IRIS (discrete tokens plus transformer) on a shared dataset of 100k frames, DIAMOND's imagined frames show fewer visual inconsistencies (enemies shown as rewards, bricks and score inconsistent) with 3 network evaluations per frame against 16 for IRIS; the authors attribute part of DIAMOND's score gain to this, noting the agent components are similar [A1, A3] (Fig. 5, Sec. 5.3, p. 8).
- C7: The diffusion model alone, trained on 87 hours of static CS:GO gameplay (5M frames; world model at 56x30 resolution plus a 51M-parameter upsampler, 381M parameters total, 12 days on an RTX 4090), serves as a playable game engine at about 10 Hz on an RTX 3090 (stochastic sampling and 10 steps for the upsampler, 3 for the dynamics) [A1] (Sec. 6, pp. 8-9).
- C8: The CS:GO model is stable for hundreds of steps but drifts out of distribution in rarely seen areas, forgets the state when walls fill the view (limited memory), and generalises a jump wrongly; no quantitative measure is given [A4] (Sec. 6, p. 9).
- C9: On held-out 64x64 CS:GO and motorway-driving data (150k frames, 120k updates), frame-stacking DIAMOND beats IRIS (K=16 and K=64) and DreamerV3 on FVD (CS:GO 34.8 vs 85.7 for IRIS K=64 vs 509.1) and also beats a cross-attention variant on every listed metric (CS:GO FVD 34.8 vs 81.4); sampling (20 denoising steps, measured on one RTX A6000) is 7.4 Hz against DreamerV3's 266.7 Hz and IRIS K=16's 4.2 Hz [A1, A4] (Table 8, App. M.3, App. M.4, pp. 25-26).
- C10: Given action sequences not in the data, the model follows common actions for a few frames (steering, speed up, slow down, fire, look) and then degrades or snaps to another map area; the authors call this causal confusion on offline data, not a flaw specific to DIAMOND [A1, A4] (Fig. 11, Fig. 12, App. M.4, pp. 27-28).
- C11: One training step costs 543 ms on an RTX 4090 (diffusion 88 ms, reward/termination 115 ms, actor-critic 340 ms including 15 imagination steps at 20.4 ms each, of which 3 denoising steps are 12.7 ms) [A5] (Table 5, App. I, p. 22).

## Implementation notes

- Diffusion model: 2D U-Net with 4 stages of 2 residual blocks and 64 channels, conditioning dimension 256; reward model has 32-channel blocks, LSTM 512; actor-critic blocks [1,1,1,1] with channels [32,32,64,64], LSTM 512 (source: Table 2, p. 17).
- Training: 1000 epochs of 400 updates, batch 32, 100 environment steps per epoch, epsilon 0.01, horizon H=15, gamma 0.985, lambda 0.95, entropy weight 0.001, AdamW LR 1e-4, weight decay 1e-2 (diffusion and reward) and 0 (actor-critic) (source: Table 3, p. 18).
- Noise: log(sigma) from a normal with mean -0.4 and std 1.2, sigma_data 0.5 (source: App. C, Eq. 13, p. 16).
- Burn-in: conditioning frames and actions initialise the LSTM states before each imagination (source: App. D, p. 17).
- Atari preprocessing: 64x64 RGB, frameskip 4, noop max 30, termination on life loss (source: Table 3, p. 18).
- Code, agents and playable models: diamond-wm.github.io (source: abstract, p. 1); the commit is not given (source: assumption: not checked).
- Which games' seeds produced Table 1's per-game numbers beyond "5 seeds" is not stated (source: assumption).

## Limitations / open questions

- Authors: evaluation is mostly discrete control; frame stacking is a minimal memory and a transformer over environment time could give longer memory (a cross-attention variant was no better early on); reward and termination are kept in a separate model (Sec. 8, p. 10).
- Authors: CS:GO limits are memory and rare-region drift; the authors expect scale to address these except the model's memory (Sec. 6, p. 9).
- Open question: the world model reproduces pixels, so it must be trained per game; nothing here shows transfer between games or an embedding useful for a cross-game policy (the diffusion model's representations are not extracted; Sec. 8 calls extracting them non-trivial, p. 10).
- Open question: at n=3 and 4 stacked frames the model costs 12.7 ms per imagined frame on an RTX 4090, orders of magnitude slower than an emulator, so it is a training-time simulator for sample-poor settings, not a speed-up for an exact fast emulator.
- Open question: shmups have many small fast sprites; Atari 64x64 results (Asterix, Breakout, Road Runner gains) hint that detail fidelity matters, but no bullet-heavy game is tested.

## Related

- related [[hafner2024mastering]]: the DreamerV3 baseline (RSSM with discrete latents); DIAMOND beats it on Atari 100k mean HNS, DreamerV3 trains faster (under 1 day).
- related [[bruce2024genie]]: concurrent playable generative world model (a discrete-token transformer rather than diffusion; the paper cites it as concurrent work, Sec. 7, p. 10).
- related [[samsami2024mastering]]: addresses the long-memory weakness DIAMOND names (frame stacking over 4 frames) by an SSM-based world model.
- related [[hafner2025scalable]]: Dreamer 4 trains agents in imagination inside a fast video world model, a successor setting to DIAMOND's diffusion world model.
- related [[gao2025adaworld]]: a diffusion video world model pretrained on action-free video with latent actions, where DIAMOND uses real actions and one model per game.
- related [[hansen2024tdmpc2]]: a decoder-free alternative; the opposite choice of not modelling pixels.
- related [[assran2025vjepa]]: predicts in embedding space instead of pixels; DIAMOND argues visual details matter.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
