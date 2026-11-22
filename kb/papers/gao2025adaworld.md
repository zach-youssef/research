---
id: gao2025adaworld
title: "AdaWorld: Learning Adaptable World Models with Latent Actions"
authors: [Shenyuan Gao, Siyuan Zhou, Yilun Du, Jun Zhang, Chuang Gan]
year: 2025
venue: ICML 2025 conference paper (read as arXiv v4)
url: https://arxiv.org/abs/2503.18938
arxiv: 2503.18938
doi:
pdf_sha256: ebaded596ac3941608f0faebfbb8e25e72baf2f8c758f7e08f464d31250ce26e
tags: [world-model, latent-actions, video-pretraining, diffusion, action-conditioning, adaptation, transfer-learning, cross-domain, vae, visual-planning, model-predictive-control, procgen, gym-retro]
status: read
read: 2026-11-21
---

# AdaWorld: Learning Adaptable World Models with Latent Actions

## TL;DR

A β-VAE extracts continuous latent actions from frame pairs of ~2000M unlabelled frames (1016 games,
robot, human video); an SVD world model pretrained on them adapts to unseen environments (100
samples per action, 800 steps) better than action-free, flow or Genie-style discrete pretraining.

## Problem

- World models need many action labels and costly training to become controllable in a new
  environment, and a unified action format across environments is hard to define; pretraining on
  action-agnostic video leaves action control to be learned at finetuning (Sec. 1, pp. 1–2,
  Sec. 2.1, p. 3).
- Goal: put action information into pretraining without labels, so a world model can transfer a
  demonstrated action to new contexts without training and adapt to a new action space with few
  interactions and finetuning steps (p. 2, Fig. 1, p. 1).

## Method

- **Latent action autoencoder** (Sec. 2.1, p. 3, Fig. 2): a transformer encoder takes two
  consecutive frames f_t, f_{t+1} as 16×16 patches plus two learnable tokens; a spatiotemporal
  transformer (spatial attention within a frame, temporal attention across the two frames at the
  same position, rotary embeddings in temporal attention) mixes them; only token a_{t+1} is
  projected to a Gaussian posterior (μ, σ). The sampled latent action ã is attached to f_t and a
  spatial-transformer decoder predicts f_{t+1} in pixels. Trained as a β-VAE (Eq. 2, p. 4); the
  compact latent is the information bottleneck that should keep only the dominant change, i.e. the
  action. β = 2×10⁻⁴, latent dimension 32, 500M parameters, 16 encoder and 16 decoder blocks
  (App. B.1–B.2, p. 16).
- **Action-aware pretraining** (Sec. 2.2, pp. 4–5, Fig. 3, p. 3): the frozen encoder labels every
  frame pair of the corpus with ã. The world model is initialised from Stable Video Diffusion (EDM),
  denoises one frame at a time, gets ã concatenated with SVD's timestep and CLIP image embeddings,
  uses the last memory frame as SVD's condition image and concatenates up to 6 encoded history
  frames (random length, memory-length condition) with the noisy latent; history frames get noise
  augmentation during training. Loss: x0-prediction MSE (Eq. 3, p. 5). 3D UNet, 1.5B trainable
  parameters, 256×256 (App. B.1, p. 16).
- **Adaptation** (Sec. 2.3, p. 5): for N discrete actions, encode a few labelled transitions per
  action, average their latent actions and use the N averages to initialise the action embeddings,
  then finetune the whole model briefly. For continuous actions, a light MLP maps raw actions to the
  latent-action interface, initialised by fitting it to action / latent-action pairs.
- **Transfer, composition, creation** (Sec. 2.3, pp. 5–6): reuse a demonstration's latent-action
  sequence from a new start frame; average two latents to compose actions; k-means over latent
  actions to create any number of discrete controls (App. C.3, p. 18).
- **Data** (Sec. 3, p. 6, App. A, p. 15, Table 7): SSv2, Ego4D, Open X-Embodiment, MiraData, plus
  random-action rollouts from 1000 Gym Retro ROMs (1M transitions each) and 16 Procgen games (10M
  each, 9000 training levels, 1000 held out); about 2000M frames, ~10 Hz (App. B.2, p. 17), mixed by video count
  (Retro 49%, OpenX 30%, MiraData 3D 14%).
- **Baselines** (Sec. 3, p. 6): same architecture with zero actions (action-agnostic); UniMatch flow
  downsampled to 16×16 as condition; VQ-VAE latent actions with 8 codes as in Genie. All compared
  world models trained 50K iterations.

## Assumptions

- A1: The game pretraining data are random-policy rollouts (biased random action sampling), not human or expert play (App. A.1, p. 15)
- A2: Pretraining uses about 2000M frames from 1000 Gym Retro and 16 Procgen games plus SSv2, Ego4D, Open X-Embodiment and MiraData video, at about 10 Hz (Sec. 3, p. 6, App. A, p. 15, App. B.2, p. 17)
- A3: The adaptation targets have small discrete action spaces (Procgen: 4 actions; the counts for Habitat, Minecraft and DMLab aren't given) or low-dimensional continuous actions (nuScenes displacements) (reader's inference, Sec. 3.2, pp. 7–8)
- A4: Adaptation gets about 100 labelled samples per discrete action, or 100 trajectories for continuous actions (Sec. 3.2.1, p. 7, Sec. 3.2.2, p. 8)
- A5: Adaptation is scored by video fidelity (PSNR, LPIPS) or MPC planning success, not by the return of a trained policy (Sec. 3.2, pp. 7–8)
- A6: The Procgen planning scenes are test scenes of games whose other levels are in the pretraining data (reader's inference, App. A.1, p. 15, App. E, p. 19)
- A7: The planning goal is an image of the final state and the reward is RGB cosine similarity to it (Sec. 3.2.2, p. 8, App. B.4, p. 17)
- A8: Action-transfer pairs share an action (same LIBERO task or same SSv2 label among the top 10) (Sec. 3.1, p. 6, App. D, p. 19)
- A9: A latent action is inferred from two consecutive frames (Sec. 2.1, p. 3)
- A10: The controlled agent's action drives the dominant change between consecutive frames (Sec. 2.1, p. 3)
- A11: The SSv2 transfer pairs come from a dataset whose videos are in the pretraining corpus; whether the test videos are held out isn't said (reader's inference, Sec. 3.1, p. 6, Table 7, p. 15)
- A12: The planning scenes are those whose goal is reachable within 20 steps by goal-image matching in the true environment (Sec. 3.2.2, p. 8, App. E, p. 19)
- A13: The VP2 models are 64×64 low-resolution variants of AdaWorld, adapted with 5K Robosuite and 35K scripted RoboDesk trajectories; their pretraining data aren't stated separately (Sec. 3.2.3, p. 8, App. B.5, p. 18)
- A14: The iVideoGPT comparison pretrains on Open X-Embodiment only and adapts to BAIR robot pushing (App. B.6, p. 18)

## Claims

- C1: The latent actions are context-invariant and transfer across contexts: AdaWorld transfers 20-step demonstrated action sequences to new scenes without training [A2, A8, A9, A10] (p. 2, Sec. 2.3, p. 5, Fig. 4, p. 4, Sec. 3.1, p. 6)
- C2: A standard VAE latent action autoencoder struggles to express diverse transitions; β trades expressiveness against disentangling action from context, and β = 2×10⁻⁴ was chosen for that balance [A2, A9, A10] (p. 3, Eq. 2, p. 4, App. B.2, p. 16)
- C3: In a UMAP of 1000 samples per action from Habitat, Minecraft and DMLab, the same actions cluster together across environments (noise comes from actions that can't execute in some states); with β = 2×10⁻⁶ latents are more separable but overlap less across environments [A2, A9, A10] (Sec. 3.3, p. 9, Fig. 7)
- C4: Using the latent action decoder as the world model gives coarse single-pass predictions that degrade markedly after several interactions, hence the separate diffusion world model [A2] (Sec. 2.2, p. 5)
- C5: Noise augmentation of history frames during training alleviates long-term drift even with no noise at inference [A2] (Sec. 2.2, p. 5)
- C6: On action transfer over 1300 pairs from LIBERO and SSv2 (the paper says unseen; SSv2 is in the pretraining corpus) (20 autoregressive frames), AdaWorld gets FVD 767.0 / 473.4, ECS 0.804 / 0.639 and human-judged success 70.5% / 61.5% (LIBERO / SSv2), against at best FVD 1409.5 / 702.8, ECS 0.724 / 0.611 and 3.5% / 21.5% for the baselines; human evaluation used 50 pairs per dataset and 4 volunteers [A2, A8, A9, A10, A11] (Sec. 3.1, p. 6, Table 1)
- C7: Adapting to four environments absent from training with 100 samples per discrete action (Habitat, Minecraft, DMLab) or 100 trajectories (nuScenes) and 800 finetuning steps, AdaWorld has the best PSNR and LPIPS in every case (23.58, 21.59, 22.92, 21.60 PSNR vs 20.34, 19.44, 20.96, 20.86 action-agnostic; discrete latents next at 23.31, 21.33, 22.36, 21.28); all action-aware variants significantly beat the action-agnostic baseline, though on nuScenes flow conditioning is within 0.08 PSNR of action-agnostic (20.94 vs 20.86; no error bars) [A2, A3, A4, A5, A10] (Sec. 3.2.1, p. 7, Table 2)
- C8: Across Minecraft with 50–200 samples per action and nuScenes with 100–300 samples, the paper says AdaWorld performs much better at the start and improves faster; read off the figure, on nuScenes at step 0 action-agnostic (about 21.45 PSNR) is at or above AdaWorld with MLP initialisation (about 21.2–21.35), and with 100 samples AdaWorld peaks at steps 200–400 (about 21.95) then falls to about 21.6 by step 800 [A2, A3, A5, A10] (Sec. 3.2.1, p. 7, Fig. 6)
- C9: On Minecraft, read off the figure, AdaWorld with averaged-latent initialisation starts near 19.8 PSNR against about 18.7 for action-agnostic and ends near 21.2–21.8 [A2, A3, A5, A10] (Fig. 6, p. 7)
- C10: With a randomly initialised control interface, AdaWorld starts slightly below the action-agnostic baseline but surpasses it after about 200 steps on Minecraft and nuScenes; read off the figure, the start gap is about 0.25 PSNR on Minecraft (about 18.5 vs 18.75) and about 0.6 on nuScenes (about 20.85 vs 21.45) [A2, A3, A5] (Sec. 3.3, p. 9, Fig. 6, p. 7)
- C11: With limited interactions (e.g. 50), the approach is significantly more efficient than pretraining on action-agnostic video [A2, A3] (p. 2)
- C12: Goal-image planning (CEM MPC) on 30 Procgen test scenes in each of Heist, Jumper, Maze and CaveFlyer (chosen by exhaustive search in the true environment, survival games excluded), after 500 finetuning steps on 100 samples per action: AdaWorld averages 56.67±2.16% success, action-agnostic 26.00±0.98% (about random, 26.17±2.55%), the ground-truth simulator 80.67±2.11%; 5 seeds, standard error [A1, A2, A3, A4, A5, A6, A7, A10, A12] (Sec. 3.2.2, p. 8, Table 3, App. E, p. 19)
- C13: Without any weight updates, using only averaged latent actions as action embeddings, AdaWorld averages 44.83±1.37%, above the finetuned action-agnostic model; it beats its finetuned self on Jumper (68.00 vs 58.67%) but equals random on Maze (41.33%) [A1, A2, A3, A4, A5, A6, A7, A10, A12] (p. 8, Table 3)
- C14: Tabular Q-learning on the same samples (quantised images, cosine reward) averages 27.17±1.27%; the paper says AdaWorld significantly dominates it, though on CaveFlyer Q-learning (34.00±6.17%) is slightly above finetuned AdaWorld (33.33±3.80%), with overlapping ranges [A1, A2, A3, A4, A5, A6, A7, A10, A12] (p. 8, Table 3)
- C15: On VP2 robot planning (64×64 variant, MPPI, 1K finetuning steps on 5K Robosuite and 35K scripted RoboDesk trajectories, 4 runs per task), AdaWorld's GT-simulator-normalised aggregate is 21.54 vs 5.03 for action-agnostic, e.g. Robosuite push 63.50±1.71% vs 17.50±0.50%; on Open slide the ranges overlap (5.83±2.85% vs 1.67±1.67%), and Flat block and Open drawer were omitted as giving no meaningful scores [A5, A13] (Sec. 3.2.3, p. 8, Table 4)
- C16: Latent action autoencoders trained 40K steps on OpenX, Retro or both reconstruct unseen Procgen at 25.51 / 26.43 / 26.62 PSNR (LPIPS 0.318 / 0.250 / 0.234): adding real-robot OpenX video helps generalisation to 2D games, which the authors read as more data diversity possibly helping [A1, A9, A10] (Sec. 3.3, p. 9, Table 5)
- C17: Latent-action pretraining also helps iVideoGPT: on BAIR after finetuning, 17.40 vs 16.59 PSNR and 0.204 vs 0.220 LPIPS [A5, A14] (Sec. 3.3, p. 9, Table 6, App. B.6, p. 18)
- C18: Averaging two latent actions (right, jump) gives an action combining both (jump right), which the authors read as a semantically continuous latent space (qualitative) [A2, A9, A10] (Fig. 5, p. 5)
- C19: k-means over the latent actions of the Procgen and Retro training set gives 4–12 control options whose ΔPSNR, measured on the latent action decoder's predictions, is 5.15–8.23, against 6.47 for the 8-code discrete model (whose count is fixed); with 8 clusters AdaWorld's 6.26 is slightly below it [A1, A9, A10] (App. C.3, p. 18, Table 9)
- C20: Biased random action sampling (raising one action's probability for a short period, then alternating) explores longer horizons and gives more diverse scenes than uniform sampling (qualitative) [A1] (App. A.1, p. 15, Fig. 8)
- C21: Limitations: not real-time; struggles to create new content once a rollout leaves the initial scene; falls short on extremely long rollouts; failures with real-world physics, dynamic agents, long rollouts and large view changes [A2] (Sec. 4, p. 9, App. C.1, p. 18, Fig. 21, p. 28)

## Implementation notes

- Training: latent action autoencoder 200K steps from scratch, batch 960, AdamW lr 2.5×10⁻⁵, weight decay 0.01; world model 80K steps, batch 64, lr 5×10⁻⁵, 16 A100s, cosine schedule with 10K warmup, EMA; noise augmentation level drawn from 0.0–0.7 in steps of 0.1; brightness jitter and centre crop for both models (source: App. B.2, pp. 16–17)
- Steps: the comparisons train all world models 50K iterations (p. 6, and Fig. 10, p. 20, says AdaWorld at 50K), while App. B.2 gives 80K; which checkpoint Tables 2–4 use isn't said beyond that (source: assumption, Sec. 3, p. 6 vs App. B.2, p. 16)
- Sampling: 5 steps, classifier-free guidance 1.05, history conditioned on augmentation level 0.1 without added noise, timestep shifting (source: App. B.3, p. 17)
- Procgen MPC: CEM with 2 iterations, 100 sequences of length 15, top 10, execute the first 5 actions, search limit 20 steps, 3 denoising steps, no guidance; reward is RGB cosine similarity to the final-state image; the 120 scenes were picked by exhaustive search in the true environment, excluding survival games (source: App. B.4, pp. 17–18, App. E, p. 19)
- VP2: 5K Robosuite trajectories and 35K scripted RoboDesk trajectories, 1K steps, batch 32, lr 5×10⁻⁴, success when cost < 0.05 (source: App. B.5, p. 18)
- Adaptation finetuning: batch 32, lr 5×10⁻⁵ with 0.1× on pretrained weights; action-agnostic gets random action embeddings; nuScenes MLP is two layers on displacements, fitted 3K steps in under 30 s on one GPU (source: Sec. 3.2.1, p. 7)
- Game data are random-policy rollouts, not human play (source: App. A.1, p. 15); the Procgen planning scenes are test scenes of games whose other levels are in the training set, so Table 3 is not transfer to an unseen game (source: assumption, App. A.1, p. 15, App. E, p. 19)
- Table 7's frame counts sum to about 1971M, consistent with "about 2000 million" (source: assumption, arithmetic on Table 7, p. 15)
- Not stated: the prior p(ã) (presumably standard normal, as in the cited VAE), how the averaged embeddings replace ã at the input, seeds or error bars for Tables 1, 2, 5, 6, compute for the autoencoder, and inference speed (source: assumption after a full read; code is at github.com/Little-Podi/AdaWorld per the project page, not read)
- For the survey's embedding, the reusable parts are the two-frame β-VAE latent action encoder as a label-free action interface across games, averaged latents as a few-shot mapping to a new game's action space, and biased random rollouts over many Retro ROMs as cheap multi-game data (source: assumption)

## Limitations / open questions

- Adaptation is measured as video fidelity (PSNR, LPIPS) and MPC planning with a goal image; no policy is trained on top, by RL or BC (Sec. 3.2, pp. 7–8)
- Authors' limitations: speed, content beyond the initial scene, very long rollouts (Sec. 4, p. 9)
- The latent action encodes one frame pair; how it separates camera motion from agent action, or other agents' motion from the player's, isn't analysed beyond Fig. 7 and the failure cases (p. 9, Fig. 21, p. 28)
- Open question: would a cross-game transformer policy predicting AdaWorld latent actions (then mapped by averaged latents) work as a label-free BC target, where the paper only uses latents to condition a world model?
- Open question: Fig. 6's nuScenes curves show no step-0 advantage for the MLP initialisation; does the benefit for continuous actions depend on the 3K-step MLP fit?

## Related

- related [[hafner2025scalable]]: also pretrains a world model mostly on unlabelled video, but feeds a learned null action for unlabelled frames and grounds real actions from a labelled subset instead of inferring latent actions
- related [[baker2022video]]: the pseudo-labelling alternative (an inverse dynamics model trained on labelled data) that AdaWorld cites as hard to unify across action formats (p. 1)
- related [[liu2025prototypes]]: also pretrains on random-action frames from several domains, for an RL encoder rather than a world model
- related [[magne2026nitrogen]]: recovers real gamepad labels from gameplay video for BC across games, where AdaWorld uses unsupervised latent actions
- uses [[vaswani2017attention]]: the transformer of the latent action autoencoder (Sec. 2.1, p. 3)

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
