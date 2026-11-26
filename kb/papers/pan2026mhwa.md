---
id: pan2026mhwa
title: "MHWA: Multi-timescale Hierarchical World-Action Model"
authors: [Pengcheng Pan, Guoqing Ma, Yuhan Zhang, Yang Chen, Yichen Liu, Ziheng Li, Shan Yu]
year: 2026
venue: arXiv preprint (v1, 10 Oct 2026)
url: https://arxiv.org/abs/2610.14727
arxiv: 2610.14727
doi:
pdf_sha256: a5aa632c56c75bb7ca958a0d7cf8efc779bf4d9773e50651f840eeb89e1367a8
tags: [world-model, hierarchical, multi-timescale, mixture-of-experts, offline-rl, atari, transformer, slow-fast-context, vq-tokens]
status: read
read: 2026-11-25
---

# MHWA: Multi-timescale Hierarchical World-Action Model

## TL;DR

Adds to JOWA's tokenised world-action transformer a stride-4 manager (stop-gradient continuous context, skip-step
loss) that routes a 4-expert top-2 MoE worker. On 14 Atari games with 10% offline data, 77M parameters give IQM
HNS 0.884 (matched JOWA-70M 0.529, published JOWA-150M 0.833) at 70.88% fewer FLOPs than JOWA-150M.

## Problem

Autoregressive world models drift under open-loop rollout, and past hierarchical world models are complex and
often don't beat flat baselines, because of "conceptual" interference (a high-level target that moves as the low
level learns) and "structural" interference (one shared low-level network for heterogeneous dynamics) (Sec. 1, p. 2).
Scaling (JOWA-150M) is the usual remedy and is costly (Sec. 1, p. 2).

## Method

- **Base**: JOWA's tokenised world-action model (frames to K = 36 VQ-VAE tokens, interleaved with action tokens,
  heads for next observation token p_o, reward p_r, termination p_d, and a C51 Q-head h_Q trained with TD + CQL)
  (Sec. 3.1, p. 4; Sec. 3.5, Eq. 5-6, p. 6; App. B, p. 14).
- **High-Level Manager** (Sec. 3.3, Eq. 1, p. 5): a causal transformer reads the interleaved sequence downsampled by stride k = 4 (every k-th token,
  x_1, x_{1+k}, ...), giving h^high_m; a Context Projector on stopgrad(h^high_m) yields a continuous context vector z^ctx_m,
  "slow-varying context of the current dynamics regime rather than ... a discrete subgoal". It is broadcast and held constant for the
  next k atomic steps (refresh interval r = k by default).
- **Low-Level Worker** (Sec. 3.4, Eq. 2-4, p. 5-6): 4-layer causal transformer whose FFNs are MoE layers with N = 4 experts, Top-K = 2. The gate sees
  concat(x_t, z^ctx_m) so routing is mode-aware and locally reactive. Load-balancing KL to uniform (Eq. 8, p. 7).
- **Losses** (Sec. 3.5, Eq. 5-9, p. 6-7): L_base + L_critic + lambda_h L_high + lambda_a L_aux (lambda_h = 1, lambda_a = 0.1). L_high is a skip-step
  head predicting the observation token at the next stride position from the manager state alone (Eq. 7). Stop-gradient blocks low-level losses from
  the manager.
- **Theory sketch** (App. C, Eq. 10-27, p. 15-16): under Lipschitz dynamics, slow-context drift rho and manager context error beta_H, a strided
  model's one-step open-loop error bound is no larger than a flat model's when beta_H + (k-1) rho <= beta_F. Framed as intuition, not a guarantee.

## Assumptions

- A1: Observations are stacked frames tokenised by JOWA's pretrained VQ-VAE (K = 36 tokens per frame), and actions are discrete Atari actions (Sec. 3.1, p. 4, Sec. 5, p. 10).
- A2: Offline training on 10% of the RL Unplugged (rlu-atari) data of 14 games, 2 of 5 runs (Sec. 4.1, p. 7).
- A3: The stride k = 4 is fixed, not adaptive; adaptive temporal abstraction is future work (Sec. 3.3, p. 5, Sec. 5, p. 10).
- A4: The 77M model uses the same context window L = 8 frames as JOWA (Sec. 4.1, p. 7).
- A5: The theory assumes Lipschitz dynamics, slowly drifting context and bounded manager/local errors (App. C, Eq. 12-17, p. 15).
- A6: The ablations use a 44M variant on a 6-game subset, so ablation numbers don't transfer to the 77M model (Sec. 4.4, pp. 8-9, App. A, p. 14).
- A7: Results are scoped to multi-game offline Atari; the authors treat the hierarchy as an engineering hypothesis (Sec. 1, p. 2).

## Claims

- C1: MHWA-77M gets IQM HNS 0.884 (95% CI [0.736, 1.031]) over 3 seeds on 14 Atari games at 10% data, versus JOWA-70M 0.529 [0.459, 0.602] trained on the same data and batch setting, and the published JOWA-150M at 0.833 [A1, A2, A3, A4, A7] (Table 1, Table 2, p. 8, Sec. 4.2, pp. 7-8).
- C2: Forward+backward training cost is 48.199 GFLOPs per sequence versus 84.238 (JOWA-70M) and 165.524 (JOWA-150M), that is 42.78% and 70.88% fewer; MHWA has 77.27M parameters, 60.49M active per forward pass [A1, A3, A4, A7] (Table 2, p. 8, Sec. 4.1, p. 7, Table 6, p. 15).
- C3: Median HNS is 0.676 and mean superhuman games 5.7 of 14, against JOWA-150M's 0.582 median and 6; per game it wins some and loses others (e.g. Atlantis 1.69M vs JOWA-150M's 2.69M raw) [A1, A2, A3, A4, A7] (Table 1, p. 8).
- C4: In open-loop prediction from t = 5 over t = 5-8, JOWA-70M degrades fastest, while MHWA-77M is stable and matches or exceeds JOWA-150M on MSE and SSIM; a long-horizon check to t = 54 shows smoother degradation than JOWA-70M but no stability claim beyond that [A1, A2, A3, A4, A7] (Fig. 2, p. 9, Fig. 4, p. 21, Sec. 4.3, p. 8, App. G, p. 20).
- C5: Under JOWA's held-out protocol (5 games, 5k transitions), MHWA reaches IQM DQN-normalised 0.643 +/- 0.007 versus JOWA-150M 0.647 and JOWA-70M 0.603; weaker on Robotank, competitive on Pong and YarsRevenge [A1, A2, A3, A4, A7] (App. E, Table 9, p. 18, Sec. 4.2, p. 8).
- C6: Stride matters most in the 44M / 6-game ablation: k = 1 drops IQM from 0.564 to 0.214 (62.1% drop), k = 2 to 0.446, k = 8 to 0.517, and k = 4 is best [A1, A2, A3, A4, A6, A7] (Table 4, p. 10, Sec. 4.4, p. 9).
- C7: Gradient isolation helps: with L_high on, stop-gradient lifts IQM from 0.533 to 0.564; with L_high off, from 0.387 to 0.503; replacing the manager's hidden state with zeros or noise lowers it from 0.503 to 0.406 or 0.381 [A1, A2, A3, A4, A6, A7] (Table 3, p. 10, Sec. 4.4, p. 9, App. F.1, Table 11, p. 19).
- C8: Removing the state input of the router gives the largest routing drop (IQM 0.359, 36.3%); dense FFN instead of MoE 0.450 (20.2%); no L_aux 0.444 (21.3%); no L_high 0.503 (10.8%); MoE-only without manager 0.445 (21.1%) [A1, A2, A3, A4, A6, A7] (Table 4, p. 10, Sec. 4.4, p. 9).
- C9: Expert activations of the last MoE layer give similar PCA positions for Assault and Phoenix (both vertical shooters) in the full model, which is lost without L_high or with k = 1 [A1, A2, A3, A4, A6, A7] (Fig. 3, p. 10, Sec. 4.5, p. 10).
- C10: Retraining with context-refresh interval r = 2 (k = 4) gives IQM 0.558 and median 0.484, versus 0.564 and 0.463 for r = 4; evaluating the default checkpoint at r = 2 gives 0.568, at r = 1 0.533 [A1, A2, A3, A4, A6, A7] (Table 14, p. 20, App. F.3, pp. 19-20).
- C11: Top-2 of 4 experts is best (IQM 0.564) among tested expert configurations; 4 experts Top-1 gets 0.542 and 8 experts Top-2 0.511 [A1, A2, A3, A4, A6, A7] (Table 12, p. 19, App. F.2, p. 19).
- C12: Under Lipschitz dynamics with slow context drift rho, the strided model's one-step error bound is at most the flat model's when beta_H + (k-1) rho <= beta_F [A5] (App. C, Eq. 27, p. 16).

## Implementation notes

- Manager: 4-layer, 8-head, 512-dim transformer with dense MLP (expansion 4), output detached then a trainable 512-512 Context Projector (source: App. B, p. 14). Worker: 4 layers, 8 heads, 512-dim, MoE with N = 4, Top-2, expert 512-2048-512 GELU, bias-free gate, no capacity factor or token dropping (source: App. B, p. 14).
- Sequence: 37 tokens per block (36 visual plus 1 action), 8 blocks, 296 tokens per sequence; critic head with MLP dim 768, dropout 0.01 (source: App. B, p. 14).
- Training: 2 NVIDIA A40, AdamW, learning rate 1e-4, batch 1024 for 0.875M steps (JOWA used 512 and 1.75M), 21.6 GPU-days; 10% of the data is drawn at random per game (about 10M transitions) (source: Sec. 4.1, p. 7).
- Games: Assault, Atlantis, BeamRider, Carnival, Centipede, ChopperCommand, DemonAttack, NameThisGame, Phoenix, Seaquest, SpaceInvaders, StarGunner, TimePilot, Zaxxon; Berzerk excluded because rlu-atari omits it; ablation subset: Assault, Carnival, Centipede, NameThisGame, Phoenix, SpaceInvaders (source: Sec. 4.1, p. 7; App. A, Table 5, p. 14).
- Evaluation: 16 episodes per game; IQM and Median HNS by RLiable (source: Sec. 4.1, p. 7).
- Code and tokenizer weights: the paper promises a supplementary code release (source: App. H, p. 21); no URL in the PDF. The tokenizer is JOWA's (source: Sec. 3.1, p. 4; [[cheng2025jointly]]).

## Limitations / open questions

- Authors: evidence is scoped to discrete-action offline Atari with a pretrained tokenizer and a fixed stride; adaptive abstraction, continuous control and other domains are future work (Sec. 5, p. 10).
- Authors: the open-loop diagnostic "does not eliminate eventual manifold departure" (Fig. 4, p. 21).
- Authors: held-out fine-tuning is one seed-set; a batch-32 run reached 0.668 on one seed and is not used (App. E, p. 18).
- Reader, skill lens: the "high-level" signal is a continuous context refreshed on a fixed 4-token stride; there is no discrete skill, no learned termination, and the manager does not choose actions (its loss L_high predicts observation tokens, and it only conditions the worker's routing). The agent does act through the hierarchy: the worker output feeds the critic head h_Q, which is the action-part trained by TD/CQL (Sec. 3.1, p. 4; Sec. 3.2, p. 5), and the HNS in Table 1 are average returns over 16 episodes per game (Sec. 4.1, p. 7). So the 0.884 versus 0.529 result is a score for a policy whose fast path is conditioned on the slow context, though the ablation that removes the manager but keeps the MoE (0.445, C8) is on the 44M / 6-game setting only. How actions are picked from the model (argmax or planning) is deferred to JOWA (Sec. 4.1, p. 7).
- Reader: 14 games from one benchmark with each of Table 1's cells averaged over 3 seeds; the ablations use 44M, 6 games and each variant's IQM comes from six numbers, so differences of a few hundredths are not distinguishable (Table 4, p. 10).
- Reader: whether the manager's context carries anything like an in-game economy or rank is untested; Fig. 3's PCA gives only a loose genre signal across 6 games.

## Related

- extends [[cheng2025jointly]]: same tokenizer, data protocol and critic as JOWA, adding a manager and MoE worker; compared with JOWA-70M/150M throughout.
- related [[lee2022multigame]]: the MGDT baseline (200M) in Table 1, p. 8; MTBC (120M), EDT and Scaled-QL are other baselines there.
- related [[kujanpaa2023hierarchical]]: contrasts discrete subgoal codes with a continuous slow context.
- related [[bruce2024genie]]: another latent-action world model (optional comparison; Genie's tokens stand for actions, MHWA's context is a slow regime summary).
- related [[cherepanov2026recurrent]]: another way to carry long-range information (memory tokens) at fixed context.
- related [[hafner2022director]]: a manager-worker split MHWA cites as conceptual interference.
- related [[vezhnevets2017feudal]]: a manager-worker split MHWA cites as conceptual interference.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
