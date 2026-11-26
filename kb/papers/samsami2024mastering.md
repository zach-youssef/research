---
id: samsami2024mastering
title: Mastering Memory Tasks with World Models
authors: [Mohammad Reza Samsami, Artem Zholus, Janarthanan Rajendran, Sarath Chandar]
year: 2024
venue: ICLR 2024 conference paper
url: https://arxiv.org/abs/2403.04253
arxiv: 2403.04253
doi:
pdf_sha256: 940dc2d4f1d61f1246375df31b7a1ff50bf93be16c89dcf95c8648d0cef252cc
tags: [reinforcement-learning, model-based-rl, world-model, state-space-model, s4, memory, credit-assignment, partial-observability, memory-maze, popgym, dreamer]
status: read
read: 2026-11-25
---

# Mastering Memory Tasks with World Models

<!-- Read from arXiv v1 (2024-03-07), 37 PDF pages: main text pp. 1-9, references pp. 10-14, appendices A-T pp. 15-37 (all read). -->

## TL;DR

R2I swaps DreamerV3's GRU for a stack of diagonal MIMO SSMs computed by parallel scan, with a
non-recurrent encoder; it beats DreamerV3 on BSuite and POPGym memory tasks, exceeds human scores on
3 of 4 Memory Maze sizes, matches DreamerV3 on Atari 100K and DMC, and trains up to 9x faster.

## Problem

Model-based agents struggle with long-term dependencies, both recalling distant observations and
assigning credit across long gaps (p. 1). Their RNN backbones suffer vanishing gradients, and
transformer world models are quadratic in sequence length and unstable on long sequences (Sec. 1,
p. 1). SSMs such as S4 capture very long dependencies in supervised learning (up to 16K, p. 2), but
had not been used in a model-based RL agent (Sec. 1, p. 2).

## Method

- **S3M world model** (Sec. 3.1, p. 4, Fig. 1): DreamerV3's RSSM with the GRU replaced by f_θ, a
  stack of SSM layers (Eq. 2), each followed by GeLU, a GLU transform and LayerNorm. It maps
  (a_{t-1}, z_{t-1}) and hidden state x_{t-1} to the deterministic output h_t and new hidden x_t.
- **Non-recurrent representation** (Sec. 3.1, p. 4): the posterior is q(z_t | o_t), not
  q(z_t | o_t, h_t), so all z_{1:T} can be computed at once and h_{1:T}, x_{1:T} come from one
  parallel scan (Eq. 7). App. M (p. 27) tests the change on DreamerV3.
- **Loss** (Eq. 3-6, pp. 4-5): DreamerV3's prediction (observation, reward, continue), dynamics
  and representation losses with KL clipped below 1 (free bits), KL balancing and β weights.
- **Parallel scan, not convolution** (Sec. 3.1, p. 5, Table 1): the scan returns the hidden states
  x_t (needed by the policy), avoids burn-in for imagination, and allows resetting states at
  episode boundaries inside a sampled sequence via a modified associative operator after Lu et al.
  (App. C, pp. 17-18).
- **SSM design** (App. B.1, p. 17): diagonal state matrices (not DPLR) and MIMO SSMs (after S5),
  bilinear discretization (Table 3, p. 19).
- **Actor-critic** (Sec. 3.2, pp. 5-6, App. D, p. 18): DreamerV3's procedure, imagining H steps
  from every time step of the batch with the SSM in recurrent mode; REINFORCE for discrete actions,
  backprop through dynamics for continuous; λ-returns (Eq. 8). The policy takes (z_t, h_t)
  ("output state"), (z_t, x_t) ("hidden state") or (z_t, h_t, x_t) ("full state"), chosen per domain.
- **Pseudocode**: Alg. 1 (App. R, p. 35).

## Assumptions

- A1: World-model hyperparameters are mostly reused from DreamerV3 and fixed across domains, with three S3M sizes (Small Memory for BSuite and POPGym, Small for Atari and DMC, Medium Memory for Memory Maze) (Sec. 4, p. 6, Table 2, p. 19).
- A2: Training uses a 10M-step FIFO replay buffer, batch length L = 1024 and batch size 4, imagination horizon 15, γ 0.997 (Sec. 4, p. 6, Table 3, p. 19).
- A3: The policy input is chosen per domain: output state for BSuite, Atari and DMC-proprio; hidden state for POPGym and DMC-vision; full state for Memory Maze (Sec. 4.1, p. 7, Sec. 4.2, p. 8, Sec. 4.3, p. 9). The paper is inconsistent for BSuite: p. 7 says output state (z_t, h_t), but Fig. 17 (App. N, p. 28) shows that policy failing Discounting Chain while (z_t, x_t) learns it, matching the main-text R2I curve, so the BSuite policy input was probably (z_t, x_t) (reader's inference, Fig. 17, p. 28).
- A4: Training batches hold whole episodes where possible; App. O argues this is needed (App. O, p. 30).
- A5: BSuite results use 10 seeds (median, 25-75th percentiles); POPGym, Atari and DMC curves use 3 seeds (Fig. 3, p. 6, Fig. 9, p. 22, Fig. 10-12, pp. 23-25).
- A6: DreamerV3's POPGym baseline was tuned only over network sizes and replay size, picking 1024 RSSM units, GRU size 1024, 5 encoder/decoder layers and a 10M replay (App. L, p. 26); model-free POPGym baselines are taken from Morad et al. (Sec. 4.1, p. 7).
- A7: Memory Maze runs train for 400M environment steps or two weeks; the Dreamer baseline is DreamerV2/V3 with truncated BPTT at Pasukonis et al.'s size, and IMPALA is the model-free baseline (Sec. 4.2, p. 8).
- A8: The human baseline in Memory Maze comes from Pasukonis et al. (reader's inference, Fig. 5, p. 8, Table 6, p. 37).

## Claims

- C1: R2I learns BSuite Memory Length and Discounting Chain over a wider range of memory steps and reward delays than DreamerV3, which handles dependencies up to about 30 environment steps [A1, A2, A3, A5] (Fig. 3, p. 6, Sec. 4.1, p. 7)
- C2: In Memory Length training curves R2I's median reward (Fig. 7) stays at or above about 0.5 up to 91 memory steps (dipping at 41) and falls to about 0.3 at 101, while DreamerV3's drops to about 0.4 at 21-31 steps and near zero from 41; the authors summarize R2I as memorizing up to about 100 steps [A1, A2, A3, A5] (Fig. 3, p. 6, Sec. 4.1, p. 8, Fig. 7, p. 20)
- C3: In Discounting Chain R2I's success rate falls roughly linearly with reward delay to near zero at 100; at a 30-step delay it improves slowly over 2M steps, and neither method learns a 100-step delay [A1, A2, A3, A5] (Fig. 3, p. 6, Fig. 8, p. 20)
- C4: In POPGym R2I sets a new best in Autoencode Easy and Medium and RepeatPrevious Medium and Hard against DreamerV3 and all 13 model-free baselines, roughly matches DreamerV3 in Concentration Easy and is slightly better in Medium; no method solves Autoencode Hard [A1, A2, A3, A5, A6] (Fig. 4, p. 7, Sec. 4.1, p. 8, Fig. 9, p. 22)
- C5: The POPGym tasks R2I solves need up to 64 (RepeatPrevious) and 104 (Autoencode Medium) memorized events, consistent with BSuite's roughly 100-step memory [] (Sec. 4.1, p. 8, Table 4, p. 21)
- C6: In Memory Maze R2I outperforms Dreamer (TBTT) and IMPALA at every size, is similar to Dreamer in 9x9, and exceeds the human baseline in 9x9, 11x11 and 13x13 [A1, A2, A3, A7, A8] (Fig. 5, p. 8, Sec. 4.2, p. 8)
- C7: Final Memory Maze scores are 33.55, 51.96, 58.14 and 40.86 (9x9 to 15x15), i.e. 127%, 117%, 104.7% and 60% human-normalized and 96%, 89%, 78% and 46% oracle-normalized, averaging 102% and 77% [A7, A8] (Table 6, p. 37)
- C8: R2I matches DreamerV3 on most Atari 100K and DMC (proprio and vision) tasks, with a drop in the fraction of runs scoring 500-900 in both DMC-vision and DMC-proprio; the authors conclude it keeps generality [A1, A3, A5] (Fig. 6, p. 9, Sec. 4.3, p. 9, Fig. 25, App. Q, p. 34)
- C9: A training step of R2I is up to 9x faster than DreamerV3 on vector (POPGym) environments and up to 6x on image (Memory Maze) environments, growing with batch sequence length [A1] (Fig. 2, p. 6)
- C10: Passing (z_t, h_t) to the policy breaks policy learning in memory-intensive environments while (z_t, x_t) works; Sec. 3.2 says the full state (z_t, h_t, x_t) is no remedy, attributing this to non-stationary features as the world model trains, but App. N says memory-intensive environments generally prefer policies that include x_t, either (z_t, x_t) or (z_t, h_t, x_t), the full state works on POPGym (Fig. 18) and is used for Memory Maze [A3] (Sec. 3.2, p. 5, Sec. 4, p. 6, Fig. 17-18, App. N, p. 28)
- C11: In Memory Maze the output-state policy does worse than the other two; with a probe head that predicts object positions from h_t, all three policy inputs perform about the same [A3] (Fig. 19-20, App. N, pp. 28-29)
- C12: Making DreamerV3's representation model non-recurrent, q(z_t | o_t), keeps or improves Memory Maze returns (better in 11x11, 13x13, 15x15 without a probe) and leaves Atari 100K and DMC-proprio unchanged; the authors call it an inductive bias that favours memory [] (Fig. 14-16, App. M, p. 27)
- C13: Training sequence length is critical: with 64-step batch sequences R2I fails BSuite Memory Length once episodes exceed 64 steps while 256-step sequences keep working, and Memory Maze returns rise with sequence length from 64 to 1024; caveats: the 256-step model also fails at about 100 memory steps (Fig. 22), and in Memory Maze 15x15 the 64-step run ends above the 128-step one (Fig. 23) [A1, A2] (Fig. 22-23, App. O, p. 30)
- C14: A higher world-model likelihood does not imply higher return: the non-recurrent DreamerV3 variant has higher image loss but higher Memory Maze return [] (Fig. 24, App. P, p. 31)
- C15: Diagonal and DPLR state matrices perform comparably for the agent, but DPLR makes imagination about 2-3 times slower [] (App. P.1, p. 33)
- C16: A Memory Maze run uses two 40 GB A100s and 40 environment workers at about 350 FPS, with 51 replayed steps per sampled step; superhuman 9x9, 11x11 and 13x13 take 17M, 66M and 206M steps (0.6, 2.2 and 6.9 wall-clock days) [A7] (Table 6, App. T, p. 37)
- C17: The authors list as limitations that batch sequence length and imagination horizon are not very long, and suggest SSM-attention hybrids [] (Sec. 5, p. 9)

## Implementation notes

- Website recall2imagine.github.io; the implementation is in JAX, inspired by the Annotated S4 blog and the S5 codebase (source: p. 1, p. 10). The paper gives no code repository URL.
- S3M sizes: h_t 512/512/2048, x_t per layer 512/192/512, SSM layers 3/5/5, SSM units 1024/512/1024 for Small Memory/Small/Medium Memory (source: Table 2, p. 19).
- World model: 32 categorical latents of 32 classes, unimix 0.01, learning rate 1e-4, β_pred 1, dynamics 0.5, representation 0.1, gradient clip 1000, 8 HiPPO blocks, discretization range (1e-3, 1e-1); actor-critic: λ 0.95, entropy 3e-4, critic EMA 0.98, return normalization Per(R, 95) − Per(R, 5) with decay 0.99, gradient clip 100 (source: Table 3, p. 19). The dynamics weight is labelled β_pred in Table 3, likely a typo for β_dyn (source: assumption, from Eq. 5, p. 5).
- SSM matrices train at the same learning rate as the rest of the world model, which worked best (source: p. 5).
- Shmup lens: the memory length R2I actually exploits is bounded by the training sequence (1024 steps in Memory Maze, whole episodes elsewhere, App. O); at 60 Hz that is about 17 s of frames, or about 4.6 to 5.1 min if each step were one action chunk (16 frames in the NitroGen paper, 18 in the released checkpoint's config). The recurrent state runs over the full episode at inference, but App. O suggests dependencies longer than the training sequence aren't learned (source: assumption, from Table 3, p. 19, App. O, p. 30).
- Shmup lens: memory and credit assignment are separated by construction. Memory sits in the world model, trained by reconstruction and reward prediction over the whole sequence; credit assignment is the actor-critic's 15-step imagination plus λ-returns from the critic. BSuite Discounting Chain is the only pure credit-assignment probe, and it degrades with delay (C3) (source: assumption, from Sec. 3, pp. 3-6, App. D, p. 18).
- Shmup lens: the bounded-context requirement maps onto R2I's fixed-size recurrent state with O(1) inference per step (Table 1, p. 5), but a transformer policy reading frames would need that state as an input or as memory tokens (source: assumption).
- Shmup lens: Memory Maze's probe-head variant (predict object coordinates from h_t, App. M-N) is the analogue of decoding game RAM (weapon level, rank) from a shmup policy's memory offline; the probe variant of the benchmark trains on the probe, which the agent-never-reads-RAM rule would allow only offline for validation (source: assumption, from Fig. 15, Fig. 20, pp. 27-29).
- Shmup lens: of the benchmarks used, Memory Maze (pixels, episodes up to 4K steps, egocentric, repeatedly sent to one of several objects) is the closest test of a frames-in policy's long memory; POPGym and BSuite are tabular (source: assumption, from Sec. 4.2, p. 8).

## Limitations / open questions

- Batch sequence length and imagination horizon are short; SSM-attention hybrids are left to future work (Sec. 5, p. 9).
- Policy input has to be chosen per domain, and no single choice works everywhere (App. N, p. 28); the paper's BSuite policy input is inconsistent between p. 7 and Fig. 17 (A3).
- 15x15 Memory Maze stays below human (60% human-normalized, Table 6, p. 37).
- Open question: no ablation separates S3M's memory from the non-recurrent encoder on R2I itself; App. M tests the encoder change only on DreamerV3 (App. M, p. 27).
- Open question: credit assignment beyond about 30-100 steps (Discounting Chain, C3) is not solved here; shmup rank and economy effects may pay off over minutes, so the actor's 15-step horizon plus critic may be the bottleneck rather than memory, as ni2023decoupling argues for transformers.
- Open question: would a NitroGen-style BC policy benefit from an R2I-style SSM state learned by a world model on gameplay video, given that NitroGen found no gain from more past frames?
- Open question: does the 1024-step training limit (App. O) still bind if steps are action chunks rather than frames, and what is lost within a chunk?

## Related

- extends [[hafner2024mastering]]: R2I is DreamerV3 with the GRU replaced by an SSM stack and a non-recurrent posterior, keeping its losses and actor-critic.
- related [[ni2023decoupling]]: R2I picks its POPGym tasks after Ni et al. and tests memory (Memory Length) and credit assignment (Discounting Chain) separately.
- related [[parisotto2020stabilizing]]: cited as prior memory work in RL; GTrXL keeps attention memory where R2I keeps a recurrent SSM state.
- related [[schulman2017proximal]]: the POPGym model-free baselines are PPO with MLP, GRU, LSTM and S4D backbones.
- related [[cherepanov2026recurrent]]: RATE, an offline transformer with memory tokens, is also evaluated on Memory Maze and POPGym; another bounded-state route to long memory.
- related [[magne2026nitrogen]]: a short-context game policy; R2I shows a fixed-size SSM state can carry 100-1000-step memory at O(1) cost per step.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
