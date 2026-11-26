---
id: wang2026audiovisual
title: "Audio-Visual World Models: Learning Physically Grounded Multisensory Dynamics"
authors: [Jiahua Wang, Leqi Zheng, Jialong Wu, Yaoxin Mao, Shijie Cheng]
year: 2026
venue: arXiv preprint (v4, 2026-08-04)
url: https://arxiv.org/abs/2512.00883
arxiv: 2512.00883
doi:
pdf_sha256: 40b11f43a3a76320fd2616e69680bfc9c8fbba910a1566073fc479a41bde3669
tags: [world-model, audio-visual, diffusion-transformer, binaural-audio, embodied-navigation, planning, simulation, action-conditioned]
status: read
read: 2026-11-26
---

# Audio-Visual World Models: Learning Physically Grounded Multisensory Dynamics

## TL;DR

Defines an audio-visual world model (POMDP) and AVW-4k (30 h, Matterport3D/SoundSpaces 2.0, one
ringtone). AV-CDiT (diffusion transformer, modality experts, 3-stage training) beats baselines and,
as a lookahead planner, lifts a pretrained AV-Nav agent's SPL from 45.08 to 47.38.

## Problem

World models are visual (DIAMOND, Genie, Cosmos, ...); audio-video generators are not conditioned on
low-level actions, and datasets lack synchronised binaural audio with action labels (pp. 1-2,
Table 1, p. 1, Table 5, p. 9). The paper wants action-conditioned prediction of both modalities that is
physically consistent (p. 2).

## Method

- Formulation: POMDP (S, O, A, p) with o_t = {o^v (H x W x 3), o^a (L x 2 binaural segment)}; action
  a_t = (translation, heading change); prediction
  o_hat_{t+dt} ~ p_theta(o_{t+dt} | o_{t-m+1:t}, a_{t->t+dt}, dt), with skip-step offset dt in [Tmin, Tmax]
  as in NWM (Eq. 1, p. 2).
- AVW-4k: Matterport3D + SoundSpaces 2.0; one source with a looping ringtone at a known location;
  frame = 128x128 RGB and 0.15 s of binaural audio at 16 kHz; actions forward 0.15 m, turn 10 degrees
  left or right, stop; at most 500 frames; ~30 h, 76 scenes, 4,500 trajectories, scene-disjoint
  6:1:2 splits; trajectories sampled from source-approaching, source-receding and source-independent
  patterns (p. 3, p. 8).
- AV-CDiT: frozen pretrained encoders (Stable Diffusion VAE for frames, a SoundStream tokenizer trained
  on AVW-4k for audio), separate adaptors into a shared latent space; visual and audio tokens of the
  target frame are concatenated (Eq. 2); context frames from both modalities (Eq. 3); conditioning
  (action, dt, diffusion step) via AdaLN; shared self- and cross-attention over target and context
  tokens, then a modality-expert feed-forward for each of audio and vision and separate noise heads
  (Fig. 3, pp. 3-4).
- Training: DDPM with independent per-modality noise, epsilon-prediction loss (Eq. 5) plus the learned-
  covariance variational term; three stages: vision only, audio only (shared attention frozen),
  joint end-to-end fine-tuning (Fig. 4, Eq. 4-5, pp. 4-5).
- Initialisation from a pretrained NWM visual model (Bar et al. 2025), context 4 frames, AdamW with
  learning rates 1.6e-4, 8e-4, 1.6e-4 and batch sizes 512, 768, 128 for the 3 stages; 8 A100 GPUs
  (p. 5, p. 10).
- Metrics: LPIPS, DreamSim, PSNR (vision); log-spectral distance and spectral SSIM (audio); ILD error
  (binaural level difference); AV-Lag and AV-Corr errors from cross-correlating a visual motion-energy
  sequence with an audio-energy sequence (p. 5, Eq. 6-17, pp. 8-9).
- Planning: a reward-annotated AV-CDiT variant (progress-reward token, r = d_{t-1} - d_t, Eq. 18-20)
  scores n sampled actions from a pretrained AV-Nav policy with a k-step rollout, combining predicted
  progress rewards with the policy's value; beam search keeps the top B; the first action is executed
  (p. 7, p. 12).

## Assumptions

- A1: The environment has one stationary sound source (a looping ringtone in the main benchmark) at a known location, rendered by SoundSpaces 2.0 (p. 2, p. 3).
- A2: Actions are discrete with equal duration; audio is a short (0.15 s) binaural segment per frame (p. 2, p. 3).
- A3: Synthetic indoor scenes only (76 Matterport3D scenes); test scenes are disjoint from training scenes (p. 3).
- A4: The visual backbone is a pretrained visual navigation world model (NWM) fine-tuned on AVW-4k (p. 5).
- A5: Audio fidelity and consistency are judged with proxy metrics (ILD, AV-Lag, AV-Corr) computed against ground truth, not human or task-level audio quality (p. 5).
- A6: The planner uses a pretrained AV-Nav policy, and the reward head is trained on ground-truth geodesic-distance progress (p. 7, p. 12).

## Claims

- C1: In the fixed-step setting AV-CDiT matches the visual baseline NWM on LPIPS (0.382) and is slightly better on DreamSim (0.255 vs 0.258), PSNR 16.504 vs 16.539 (NWM+AudioLDM slightly higher); audio SSIM 0.547 vs 0.485 (AudioLDM) and 0.328 (AVDiT+action), LSD 1.311 vs AudioLDM 2.593 and AVDiT+action 1.252 (AVDiT+action is slightly better on LSD) [A1, A2, A3, A4, A5] (Table 2, p. 6).
- C2: AV-CDiT has the lowest physical-consistency errors: ILD error 2.471 (vs 22.894 for DIAMOND+AudioLDM, 3.003 for AVDiT+action), AV-Lag 1.218 (vs 1.665-1.678), AV-Corr 0.194 (vs 0.279-0.294) [A1, A2, A3, A4, A5] (Table 2, p. 6).
- C3: Ablations (fixed-step): removing modality experts or stagewise training leaves vision unchanged (LPIPS 0.378-0.382) but worsens audio LSD (1.366, 1.371 vs 1.311), SSIM (0.485 vs 0.547) and consistency (AV-Lag 1.618/1.611 vs 1.218; AV-Corr 0.236/0.238 vs 0.194). Rollout shows the same trend [A1, A2, A3, A4, A5] (Table 3, p. 7, Table 8, p. 11).
- C4: Stage by stage (fixed-step), stage 1+2 gives LSD 1.297, SSIM 0.531, ILD 2.529, AV-Lag 1.249, AV-Corr 0.200; adding stage 3 gives LSD 1.311, SSIM 0.547, ILD 2.471, AV-Lag 1.218, AV-Corr 0.194, with visual LPIPS 0.382 retained (the pretrained NWM before fine-tuning: LPIPS 0.628). Unfreezing everything in stage 2 diverged despite tuning [A1, A2, A3, A4, A5] (Table 6, p. 10, p. 11).
- C5: Under autoregressive rollout the full model has LPIPS 0.407, DreamSim 0.272, PSNR 16.019, LSD 1.620, SSIM 0.577, ILD 2.654, AV-Lag 0.867, AV-Corr 0.322; the rollout-mode ablations lie close to it in vision and are worse in SSIM/ILD/lag for no-experts (0.515 SSIM) [A1, A2, A3, A4, A5] (Table 7, p. 10, Table 8, p. 11).
- C6: A model trained and tested separately on a mixed-source version of AVW-4k (door, fan, electric buzz, elevator, engine instead of the ringtone) gets broadly stable fixed-step metrics (the paper's wording): LSD 1.140 (vs 1.311 for the ringtone), ILD 1.971 (vs 2.471), AV-Corr 0.208 (vs 0.194), but SSIM 0.431 (vs 0.547) and AV-Lag 1.240 (vs 1.218). In rollout mode AV-Lag worsens from 0.867 to 1.557 and SSIM from 0.577 to 0.533. The authors read this as generalising across sound categories; it is not one model tested on unseen sounds [A1, A2, A3, A4, A5] (Table 9, Table 10, p. 12).
- C7: As a lookahead planner (n = 2 candidate actions, B beams, k rollout steps) AVWM raises a pretrained continuous AV-Nav agent's SPL from 45.08 to between 45.70 and 47.38 (best B=3, k=5 at 47.38; B=4: 46.46 at k=4 and 46.74 at k=5), reduces actions NA from 332.8 to 317.2 (B=3, k=5), and raises SNA from 6.44 to 7.09. An oracle world model gets SPL 50.5 to 52.0 [A1, A2, A3, A4, A6] (Table 4, p. 7).
- C8: Reward prediction for planning is checked on held-out trajectories as a sanity check (no baseline given): cumulative reward MSE 0.746 (fixed-step) and one-step reward MSE 0.052 (rollout); the two are not comparable in scale [A1, A2, A3, A6] (Table 11, p. 12).
- C9: Existing audio-visual or action-labelled datasets lack one of: synchronised audio, physical grounding of audio, low-level actions; AVW-4k is the only one in the paper's Table 5 with all three, but it is synthetic [A1, A3] (Table 5, p. 9).

## Implementation notes

- Per-frame audio: 0.15 s at 16 kHz, 2 channels, tokenised by SoundStream trained on AVW-4k with one extraction block removed; the latent token count and token rate per frame are not given in the main text (source: p. 10).
- Visual encoder is the Stable Diffusion VAE as in DiT/CDiT; image resolution 128x128 (source: p. 3, p. 10).
- Inference is diffusion denoising; the number of denoising steps and latency per predicted frame are not reported (source: p. 5; the paper has no latency table).
- Fixed-step mode predicts t+dt directly, dt swept over a 16-frame horizon; rollout feeds back generated frames (source: p. 5).
- Training sampling: each sample chooses 4 random goals within +/- 16 frames of the current timestep (source: p. 10).
- Planning reward weights (success bonus, step penalty) are not given numerically (source: assumption; see Eq. 18, p. 12).

## Limitations / open questions

- Synthetic simulated indoor scenes with stationary sound sources and a telephone ringtone as the main source; future work includes richer sound events, moving sources, broader acoustic variability and other generative backbones (p. 7).
- No human evaluation of audio; consistency metrics are proxies (p. 5).
- Open question: the planning gain is about 2.3 SPL points on one agent and the oracle gap is large; Table 4 gives no variance across seeds, although the paper says results were tested with t-tests at p < 0.05 (reader's observation from p. 5 and Table 4, p. 7).
- Open question: no wall-clock latency or token-rate figures; a diffusion model predicting future audio per step is likely too slow for a real-time shmup loop without evidence either way.
- Open question: how this transfers to game audio (sprites, music) is untested; the main supervision signal is sound-source geometry (distance, direction), which in 2D games has no counterpart.

## Related

- uses [[chen2022soundspaces]]: the simulator used to render binaural audio and to build AVW-4k.
- related [[chen2020soundspaces]]: the AudioGoal task and baseline policy lineage (AV-Nav).
- related [[bruce2024genie]]: action-conditioned visual world model; this paper's Table 1 lists it as vision-only.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
