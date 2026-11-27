---
id: cherepanov2026recurrent
title: Recurrent Action Transformer with Memory
authors: [Egor Cherepanov, Aleksei Staroverov, Alexey K. Kovalev, Aleksandr I. Panov]
year: 2026
venue: ICLR 2026
url: https://arxiv.org/abs/2306.09459
arxiv: 2306.09459
doi:
pdf_sha256: b97c47e6688b460d833bcd472ed7cb606cb1fed5bbd88beefe595e514d32dbdd
tags: [memory-tokens, offline-rl, decision-transformer, pomdp, recurrence, long-horizon, memory-gate]
status: read
read: 2026-11-25
---

# Recurrent Action Transformer with Memory

## TL;DR

RATE adds RMT-style memory tokens, Transformer-XL cached states and a cross-attention Memory
Retention Valve to a Decision Transformer, trained offline on 90-150 step segments. Keeps sparse
cues over T-Maze corridors up to 9,600 steps and leads on memory tasks, competitive on Atari/MuJoCo.

## Problem

Decision Transformer has a fixed context window and fails in POMDPs when a cue (for example the
T-Maze's first observation) falls outside it; plain recurrent state (Transformer-XL caching, RMT)
can overwrite sparse long-range information as intermediate segments update the memory
(Sec. 1, p. 1, Sec. 3, pp. 3-4).

## Method

- The trajectory (R_t, o_t, a_t) is encoded per modality (conv for images, linear for vectors and
  returns, embedding or linear for actions) and split into N = T/K segments of K timesteps; the
  effective context is K_eff = N*K (Alg. 1, Sec. 3, pp. 3-4, Table 11, p. 26).
- Each segment is wrapped by memory embeddings M_n (m tokens of dimension d): S~_n =
  concat(M_n, S_n, M_n), Eq. 1. The prefix copy is read access (tokens attend backward to it),
  the suffix copy is write access (it attends forward to the segment) (Sec. 3, p. 4).
- The transformer outputs actions and M_{n+1} (Eq. 2); M_0 is initialised from N(0,1)
  (Alg. 1, p. 3). Hidden states of earlier segments are also cached as in Transformer-XL, with no
  backpropagation through the cached states (Sec. 3, p. 3, fn. 1).
- Memory Retention Valve (MRV): MRV(M_n, M_{n+1}) = FFN(MultiHead(Query=M_n, Key=M_{n+1},
  Value=M_{n+1})) (Eq. 3, Alg. 2, pp. 3-4). This is the variant MRV-CA-2; the others tried are
  CA-1 (roles swapped), a GTrXL-like gate (MRV-G), GRU and LSTM (Sec. 5, pp. 9-10, Fig. 15, p. 27).
- Theorem 1: if each memory row is L2-normalised and satisfies alpha-alignment (each row of M_n
  has some row of V W_M with inner product at least alpha), one MRV update keeps ||M_{n+1}-M_n||_F
  at most sqrt(2(1-alpha/m)) ||M_n||_F, and ||M_{n+1}||_F above a matching lower bound
  (Sec. 3.1, Eq. 4, pp. 4-5).
- Training is supervised behaviour cloning on offline trajectories with a return-to-go input,
  with cross-entropy (MSE for MuJoCo) on actions (Table 8, p. 23). How many segments gradients
  flow back through memory is not stated; only the cached states are stop-gradiented
  (Sec. 3, fn. 1, p. 3; reader's reading).

## Assumptions

- A1: Setting is offline RL: fixed datasets of successful or competent-policy trajectories, with returns-to-go as input and a supervised action loss (Sec. 2, p. 3, App. C, pp. 16-18, Table 8, p. 23).
- A2: In T-Maze (T_train 9-900), ViZDoom-Two-Colors and Minigrid-Memory, training sequences start at t = 0 so the cue is always in them, and are split into N = 3 segments in the main results (App. E.1-E.4, pp. 21-22).
- A3: The evaluation tasks are memory puzzles with a cue (T-Maze, ViZDoom-Two-Colors, Minigrid-Memory, Memory Maze, POPGym); Atari and MuJoCo are the non-memory tests, Atari restricted to 4 games (Sec. 4, pp. 5-6, App. C.2, p. 18).
- A4: Models are small (3-10 layers, embedding 32-128, about 1.6-6.8M parameters) and trained on one A100 80GB (Table 8, p. 23, Table 12, p. 29, App. I, p. 29).
- A5: The memory content is learned from the action loss only; no reconstruction or probing loss is used (reader's inference, Sec. 3, pp. 3-4).
- A6: Theorem 1 holds only under the alpha-alignment condition and L2-normalised memory rows, which the paper says holds empirically in trained models (Sec. 3.1, pp. 4-5).
- A7: Actions are discrete in all memory tasks (Table 7, p. 17).
- A8: RATE's D4RL MuJoCo scores use a non-standard protocol that resets the return-to-go to the maximum at each segment, while the baselines' numbers are taken from their papers under their own protocols (App. E.6, p. 22, Sec. 4, p. 8).
- A9: Memory Maze models train on 90-step subsequences sampled with stride 90 from 1,000-step trajectories, so a training sequence need not contain the episode start (reader's inference; App. E.3, p. 21).

## Claims

- C1: In T-Maze (clue at the first step, trained on corridors up to 900 steps), RATE keeps 100% success on in-distribution corridor lengths and the paper reports good success up to 9,600 inference steps (28,800 tokens); the figures show it declining at the longest lengths (about 0.7 in the last cells of Fig. 6, read off the figure), while DT and LSDT fall to about 50% beyond their context and BC-LSTM generalises when trained on sequences up to 150 but collapses to about 50% when trained on 600 or more [A1, A2, A3, A5, A7] (Fig. 5, Fig. 6, Sec. 4.1, pp. 6-7, App. A, p. 15).
- C2: In ViZDoom-Two-Colors (pillar vanishes at step 90, T_train 150, 2100-step inference) RATE has the highest average return and the lowest red-green imbalance among all baselines [A1, A2, A3, A4, A7] (Fig. 4, Sec. 4.1, p. 6).
- C3: Removing the pillar degrades all models, and DT's return in red-pillar episodes drops about 50% once the cue leaves the window, while RATE, RMT and TrXL stay stable; RATE gives the highest and most stable return when the pillar is present [A1, A2, A3, A4, A7] (Fig. 3, Sec. 4.1, pp. 5-6).
- C4: On Memory Maze (9x9, 1000-step episodes, trained on 90-step subsequences) the average return is 7.64 +- 0.41 for RATE against 7.27 (RMT), 7.12 (TrXL), 6.83 (DT) and 4.75 (BC-LSTM) [A1, A3, A4, A7, A9] (Table 1, p. 7).
- C5: On Minigrid-Memory (trained on 41x41 grids, evaluated 11x11 to 501x501) RATE keeps consistently high return across sizes, with TrXL also good on average but higher variance [A1, A2, A3, A4, A7] (Fig. 7, p. 7).
- C6: On the 48 POPGym tasks RATE's aggregated score is 9.5 overall and 0.5 on the 33 memory tasks (DT -3.5, BC-LSTM -0.2), and 9.1 on the 15 reactive tasks, about equal to DT (9.3) and BC-LSTM (9.1) [A1, A3, A4] (Table 2, p. 8).
- C7: On standard benchmarks RATE is competitive, not best: MuJoCo D4RL average (RATE with per-segment max-return reset, baselines as published) 78.5 vs 83.2 for the best DMamba variant and 74.7 for DT; Atari raw scores beat DT on 3 of 4 games (Breakout 111.0, Qbert 12486.9, Pong 18.8; SeaQuest 1037.9 vs DT 1129.3) [A1, A4, A8] (Table 3, p. 8, Table 4, p. 9).
- C8: Corrupting memory at inference hurts: in T-Maze, replacing memory embeddings with noise drops success to about 50% (the agent navigates but loses the cue); in ViZDoom, noising memory tokens or cached states each lowers return, with cached states more important for dense continuous feedback and memory tokens for sparse discrete cues [A1, A2, A3, A7] (Fig. 8, Sec. 5, pp. 8-9, Fig. 16, p. 27).
- C9: An OracleDT given the cue as an extra input bounds what RATE could reach: T-Maze T=480 and T=900 success 1.00 (OracleDT) vs 0.90 (RATE) vs 0.50 (DT); ViZDoom total reward 56.5 (Oracle) vs 41.5 (RATE) vs 24.8 (DT) [A1, A2, A3, A7] (Table 5, p. 9).
- C10: Trained on corridors of at most 150 with N = 5, K = 30 (not the main N = 3 setup), without MRV T-Maze success decays to about 0.61 at corridor 900 (from 1.00 at 150); MRV-CA-2 holds 0.90, while MRV-G (0.65), MRV-GRU (0.55), MRV-LSTM (0.47) and MRV-CA-1 (0.49) are worse at 900 [A1, A2, A3, A7] (Table 6, p. 10, Fig. 17, p. 28).
- C11: Theorem 1 gives a guaranteed fraction (1 - sqrt(2(1 - alpha/m))) of memory norm preserved per MRV update, and an upper bound on memory change [A6] (Sec. 3.1, Eq. 4, p. 5).
- C12: Splitting into segments with fixed K_eff = 90 matters: N=1, K=90 collapses to about 0.1 success on long corridors, while N=3, K=30 keeps about 0.9 on long corridors [A1, A2, A3, A7] (Fig. 18, p. 28, App. F.4, p. 26).
- C13: RATE uses 1.0-3% fewer parameters than DT, trains up to about 6x faster per epoch in T-Maze (16.2 s vs 95.8 s), and needs about 2.7x less GPU memory there (3,148 vs 8,608 MiB), but trains slower on Memory Maze (110 s vs 83 s) and has similar per-step latency (e.g. 10.35 vs 10.45 ms in ViZDoom) [A4] (Table 12-13, p. 29).
- C14: In an action-associative-retrieval diagnostic on 180-step validation (trained at 90 with K_eff = 90), DT's action probabilities correlate negatively (-0.97) while RATE's stay positive (0.80), taken as evidence of memory beyond the context [A1, A2, A3] (Fig. 12, App. D, pp. 20-21).

## Implementation notes

- Curriculum learning over segment counts gave RATE no significant gain on T-Maze (source: App. F.2, pp. 24-25).
- Defaults the authors recommend: 5 memory tokens, 1 MRV head, ReLU, N=3, K=T//3; cached-state length mem_len = (3K + 2*num_mem_tokens)*N for dense-reward settings and 0 for sparse ones such as T-Maze (source: App. H, pp. 27-28).
- Per-environment hyperparameters: memory tokens 5-30, cached tokens 0-360, 3-10 layers, embedding 32-128, AdamW, 10-500 epochs (source: Table 8, p. 23).
- Encoders: three Conv2D layers (32, 64, 64 channels; kernels 8, 4, 3) for image tasks, linear for returns and vectors (source: Table 11, p. 26).
- Datasets: ViZDoom 5,000 90-step A2C trajectories; T-Maze 6,000 oracle trajectories; Memory Maze 5,000 Dreamer trajectories (return about 4.7); Minigrid-Memory 10,000 PPO-TrXL trajectories; Atari and MuJoCo standard DQN replay and D4RL (source: App. C, pp. 16-18).
- MuJoCo evaluation restarts each segment from the max return (source: App. E.6, p. 22).
- Whether the memory tokens' gradients flow across all N segments in training, and how the 15-30 memory tokens are initialised at test time, are not stated (source: assumption, the authors' code, sites.google.com/view/rate-model, not inspected).
- A policy over game frames would use the image path of Table 11; no pretraining or frame-compression step is described (source: assumption, user's lens).

## Limitations / open questions

- Admitted: the architecture's complexity may be unnecessary in fully observable or short-horizon settings where simpler recurrent models suffice (Sec. 7, p. 10).
- Admitted: memory benefits concentrate on memory puzzle tasks; on reactive POPGym tasks gains are small (Sec. 5, p. 8).
- Reader's inference: Theorem 1's bound is vacuous for m >= 2 memory tokens: with alpha <= 1 the lower-bound fraction 1 - sqrt(2(1 - alpha/m)) is positive only for alpha/m > 1/2, i.e. m = 1, and the upper bound exceeds ||M_n||_F (about 1.26x at m = 5, alpha = 1); the proof also treats rows of M_{n+1} as unit norm (Eq. 6), which MRV doesn't enforce. The recommended m = 5 (App. H, p. 28) is thus outside the guarantee (Eq. 4, p. 5).
- The paper is inconsistent: POPGym is 48 tasks on pp. 6, 8, 10 but 46 on pp. 7, 18; the ViZDoom pillar vanishes at step 90 on p. 6 but at step 45 in App. C.1.1 / App. E.1 and Fig. 3.
- Open question (user's lens): evaluations use at most a handful of cues (a colour, a turn); nothing tests many simultaneous long-term facts (weapon tier, shop state, rank) or whether 5-30 memory tokens suffice for them.
- Open question: memory is written only through the action loss; for a RAM-blind frame policy one could probe memory tokens against RAM variables offline to see what was stored (the paper does no probing, only noise corruption).
- Open question: tested horizons are far beyond K_eff only for sparse single-cue tasks, and BPTT depth through memory is not stated; the cost to train on real video frames at 60 fps is not measured (the largest image tasks are ViZDoom and Memory Maze at 90-step training).
- Open question: Atari uses 4 games with DT's setup; frame-level performance on complex games with richer long-term state (shmups) is untested.

## Related

- extends [[bulatov2022recurrent]]: adds Transformer-XL cached states and the MRV gate to RMT-style read/write memory tokens, for offline RL.
- extends [[chen2021decision]]: a recurrent-memory variant of Decision Transformer that handles cues beyond the context.
- related [[chevalier2023adapting]]: another descendant of RMT, accumulating summary vectors for language models.
- related [[lee2022multigame]]: Atari Decision Transformer with a fixed context window, the setting RATE's memory would extend.
- related [[magne2026nitrogen]]: a frames-to-gamepad policy with a short context; the target use for memory tokens.

## Experiments

<!-- kb:begin experiments -->
- [2026-11-27-memory-token-toys](../../experiments/2026-11-27-memory-token-toys/plan.md) (approved): tests C1, C8, C12
<!-- kb:end experiments -->
