---
id: lin2020space
title: "SPACE: Unsupervised Object-Oriented Scene Representation via Spatial Attention and Decomposition"
authors: [Zhixuan Lin, Yi-Fu Wu, Skand Vishwanath Peri, Weihao Sun, Gautam Singh, Fei Deng, Jindong Jiang, Sungjin Ahn]
year: 2020
venue: ICLR 2020 conference paper (read as arXiv v3)
url: https://arxiv.org/abs/2001.02407
arxiv: 2001.02407
doi:
pdf_sha256: d9076acc59e77647da8b996b483d745d9acc271a2b5f533c029b2dd798bcb3be
tags: [object-centric, unsupervised-learning, object-detection, scene-decomposition, vae, atari, representation-learning]
status: read
read: 2026-11-22
---

# SPACE: Unsupervised Object-Oriented Scene Representation via Spatial Attention and Decomposition

## TL;DR

A VAE that explains a frame as parallel per-cell foreground objects (presence, box, depth, appearance)
plus K GENESIS-style background segments. Matches SPAIR's box quality on 3D-Room (AP 0.90 at IoU 0.5)
at far lower step latency; on Atari it is shown only qualitatively (no ground-truth boxes there).

## Problem

Spatial-attention models (AIR, SPAIR) give explicit per-object position and scale but can't model
backgrounds of complex shape and process objects sequentially; scene-mixture models (MONet, IODINE,
GENESIS) segment flexible backgrounds but encode objects only implicitly as full-image components
and also infer sequentially, so neither scales to many objects (Sec. 1, pp. 1-2).

## Method

- **Generative model.** Image likelihood is a pixel-wise mixture: foreground with weight α = f(z_fg),
  K background components with weights (1 - α)π_k (Eq. 1-2, Sec. 2.1, p. 2). Foreground and
  background latents are independent (Sec. 2.1, p. 2).
- **Foreground.** An H×W grid; each cell has z_pres (Bernoulli), z_where (scale and shift relative to
  the cell), z_depth (resolves occlusion) and z_what (glimpse appearance and mask); glimpses are
  decoded and placed with a spatial transformer, then combined by a depth-weighted softmax into one
  foreground image and mask (Eq. 3, Sec. 2.1, p. 3, Alg. 1-2, pp. 16-17).
- **Background.** GENESIS-style: K components, each with a mask latent z_m (autoregressive LSTM prior)
  and a colour latent z_c; masks via stick-breaking (Eq. 4, p. 4, Alg. 3-4, p. 17).
- **Inference.** ELBO with reparameterisation (Eq. 6, p. 4, App. B, p. 15); cell latents are
  mean-field, so all cells are inferred in parallel, unlike SPAIR's autoregressive cells (Eq. 7, p. 4).
  z_pres uses a Gumbel-Softmax relaxation during training (App. B, p. 15).
- **Boundary loss.** Optional auxiliary loss penalising glimpse masks near the glimpse border, to stop
  a large object being split across boxes; turned off once the model converges (Sec. 2.2, p. 4, App. C, p. 15).
- **Evaluation.** Atari: 60,000 frames per game from a pretrained agent (50k/5k/5k split) for Space
  Invaders, Air Raid, River Raid and Montezuma's Revenge, plus a 10-game joint dataset of 8,000 train
  images per game; 3D-Room (MuJoCo): small (4-8 objects) and large (18-24 objects) sets with ground-truth
  boxes; baselines SPAIR (with an added background VAE and parallel rendering), SPAIR-P, IODINE,
  GENESIS, compared at equal "decomposition capacity" (Sec. 4, p. 5, Sec. 4.2, p. 8, App. E, p. 22).

## Assumptions

- A1: Foreground objects are small enough to fit a bounding-box glimpse attached to one grid cell, at most one object per cell (Sec. 2.1, p. 3).
- A2: Foreground and background are modelled independently, and what counts as foreground is decided by the model's capacity and priors, not by labels (Sec. 2.1, p. 2, Sec. 4.1, p. 8).
- A3: Cells don't need to communicate: nearby objects interact little because two objects can't occupy the same position, so a mean-field posterior suffices (Sec. 2.2, p. 4).
- A4: Hyperparameters (box-size prior, z_pres prior, fixed-α warmup, K) are tuned per dataset (App. D.2, p. 18).
- A5: Atari training images come from a pretrained agent's play, resized to 128×128 (App. E, p. 22).
- A6: Quantitative detection metrics are measured only on synthetic 3D-Room scenes, where ground-truth boxes come from a 2-D camera projection (Sec. 4.2, p. 9, App. E, p. 22).
- A7: Each model is shown at its best hyperparameter setting from a search (Sec. 4.1, p. 6).

## Claims

- C1: On 3D-Room Large with a 16×16 grid, SPACE reaches AP 0.8927 ± 0.0027 at IoU 0.5 vs SPAIR's 0.9072 ± 0.0003, AP 0.4445 vs 0.4364 averaged over IoU 0.5-0.95, and object-count error 0.0446 vs 0.0360 (5 best seeds) [A1, A3, A6] (Table 1, p. 8).
- C2: On 3D-Room Small with an 8×8 grid, SPACE reaches AP 0.9027 ± 0.0009 at IoU 0.5 vs SPAIR's 0.9081, AP 0.5069 vs 0.5068 over IoU 0.5-0.95, and count error 0.0397 vs 0.0209 [A1, A3, A6] (Table 1, p. 8).
- C3: SPACE's gradient-step latency grows much more slowly with the number of cells (4×4 to 24×24) than SPAIR's, which grows because of its sequential inference; IODINE is the slowest, and IODINE and GENESIS ran out of memory at 256 or more components on a 22 GB GPU [A3] (Fig. 5, p. 9, Sec. 4.2, p. 9).
- C4: SPACE converges (held-out pixel MSE over wall clock) much faster than IODINE and GENESIS in all 3D-Room settings, as fast as SPAIR at 4×4 and 8×8, and much faster than SPAIR at 16×16 [A3, A6] (Fig. 5, p. 9).
- C5: On Atari, IODINE and GENESIS fail to capture the foreground or put all objects in one component, which the authors attribute to Atari objects being small and irregular [A5, A7] (Sec. 4.1, p. 6, Fig. 3, p. 7).
- C6: SPAIR-P (trained on 32×32 patches) often splits one object over several boxes and detects background parts (the Space Invaders ground strip) as objects, because its receptive field is limited to a 32×32 glimpse [A5, A7] (Sec. 4.1, pp. 6-8, Fig. 3, p. 7).
- C7: On River Raid, whose background keeps changing, SPACE segments the river and detects the foreground objects, while SPAIR often fails to separate foreground from background (qualitative) [A1, A2, A5, A7] (Sec. 4.1, p. 8, Fig. 3, p. 7).
- C8: One SPACE model trained jointly on 10 Atari games still detects foreground objects and segments the background cleanly (qualitative) [A1, A2, A4, A5] (Sec. 4.1, p. 8, Fig. 4, p. 7, Fig. 7, p. 12).
- C9: SPACE tends to detect static but important objects (Space Invaders shields, Montezuma's Revenge key) as foreground where SPAIR treats them as background, and in Atlantis detects parts of the static base as foreground; the authors attribute this to the weak spatial-broadcast background decoder [A2, A5] (Sec. 4.1, p. 8, Fig. 4, p. 7, Fig. 6, p. 12).
- C10: A small box-size prior makes SPACE split large objects and a large one makes it miss small objects; the boundary loss mitigates splitting but loosens boxes and lowers AP, so it is turned off after convergence [A1, A4] (Sec. 2.2, p. 4, App. C, p. 15).

## Implementation notes

- Image 128×128, batch 12-16, foreground RMSprop lr 1e-5 (1e-4 for Fig. 5), background Adam lr 1e-3, gradient clipping at norm 1.0, up to 160,000 steps for quantitative results (source: App. D.2, p. 18).
- Joint 10-game Atari setting: z_pres prior 1e-10, z_scale prior mean -2.5 (std 0.1), σ_fg 0.20, σ_bg 0.10, K = 3, Gumbel temperature 2.5 → 1.0 over 10k steps, α fixed at 0.1 for the first 4,000 steps, no boundary loss (source: App. D.2, p. 18).
- Latent sizes: z_what, z_m, z_c 32-d, z_where 4-d, glimpse 32×32; the 16×16 foreground encoder is a strided conv stack with GroupNorm/CELU; the 8×8 version adds a stride-2 last layer (source: App. D.3, pp. 18-20).
- Boundary loss: glimpse 32×32, boundary gap b = 6, optimised with RMSprop alongside the ELBO (source: App. C, p. 15).
- Code: the project page is https://sites.google.com/view/space-project-page (source: p. 1); a public implementation is linked from there (source: assumption, not checked).
- Per-game Atari hyperparameters other than the joint-training table are "finetuned" but not listed (source: App. D.2, p. 18).

## Limitations / open questions

- No quantitative detection results on Atari: Atari frames have no ground-truth boxes in this paper (Sec. 4.2, p. 9).
- What is foreground is ambiguous; static objects may land on either side (Sec. 4.1, p. 8).
- Background inference is still sequential (LSTM over K); natural images are future work (Sec. 5, p. 9).
- Open question (reader): Atari F1 for SPACE was measured later, by [[delfosse2024ocatari]] and the SPOC paper [[delfosse2023continuity]], and is poor on Pong, Tennis and MsPacman; this note can't say whether that reflects SPACE or per-game tuning.
- Open question (reader): the per-cell z_what is a ready object embedding for a policy; this paper doesn't test it in RL.

## Related

- related [[delfosse2024ocatari]]: its Table 2 compares RAM extraction's F1 with SPACE's on Atari.
- related [[delfosse2023continuity]]: adds motion and object-continuity losses to SPACE and measures its per-game F1 on Atari.
- related [[yoon2023investigation]]: another route from pixels to object slots for RL, via slot-based encoders.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
