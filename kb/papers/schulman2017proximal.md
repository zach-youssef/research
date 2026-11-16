---
id: schulman2017proximal
title: Proximal Policy Optimization Algorithms
authors: [John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov]
year: 2017
venue: arXiv preprint
url: https://arxiv.org/abs/1707.06347
arxiv: 1707.06347
doi:
pdf_sha256: e78feadadbdbb0b601b3c2bcc81404722cd431a489b307545f9b7bea1e8c4f5b
tags: [reinforcement-learning, policy-gradient, actor-critic, on-policy, trust-region, atari, continuous-control]
status: read
read: 2026-11-16
---

# Proximal Policy Optimization Algorithms

<!-- Read from arXiv v2 (2017-08-28), 12 PDF pages; p. 9 is references, pp. 10-12 are App. A-B (hyperparameter tables, 49-game Atari curves and scores). -->

## TL;DR

PPO runs several epochs of minibatch SGD/Adam per batch of on-policy data on a clipped
probability-ratio surrogate (ε = 0.2 best, 0.82 normalized score on 7 MuJoCo tasks); it beats
TRPO, A2C and CEM on almost all of them, and on 49 Atari games wins 30 by learning speed (ACER 18).

## Problem

Deep Q-learning fails on many simple (continuous) problems and is poorly understood; vanilla
policy gradient is data-inefficient and not robust; TRPO is relatively complicated and not
compatible with architectures that include noise (dropout) or parameter sharing between policy
and value function or with auxiliary tasks (Sec. 1, p. 1). Running several optimization steps on
the plain policy-gradient loss L^PG over the same trajectory is not well-justified and empirically
gives destructively large updates (Sec. 2.1, p. 2). The goal is TRPO's data efficiency and
reliability with first-order optimization only (Sec. 1, p. 1).

## Method

- **Policy gradient** (Sec. 2.1, Eq. 1-2, p. 2): ĝ = Ê_t[∇ log π_θ(a_t|s_t) Â_t], with Ê_t an
  empirical average over a finite batch and Â_t an advantage estimate.
- **TRPO background** (Sec. 2.2, Eq. 3-5, p. 2): maximize Ê_t[r_t Â_t] subject to mean KL ≤ δ;
  the theory suggests a penalty β·KL instead (Eq. 5), but one fixed β does not work across
  problems or over the course of learning.
- **Clipped surrogate** (Sec. 3, Eq. 6-7, p. 3): with r_t(θ) = π_θ(a_t|s_t)/π_θold(a_t|s_t),
  L^CLIP = Ê_t[min(r_t Â_t, clip(r_t, 1−ε, 1+ε) Â_t)]. Clipping removes the incentive to move r_t
  outside [1−ε, 1+ε]; the min makes it a pessimistic lower bound on the unclipped objective L^CPI:
  a ratio change is ignored when it would improve the objective and counted when it makes it
  worse. For Â > 0 the term flattens above 1+ε, for Â < 0 below 1−ε (Fig. 1, p. 3).
- **Adaptive KL penalty** (Sec. 4, Eq. 8, p. 4): optimize L^KLPEN = Ê_t[r_t Â_t − β KL[π_old, π_θ]]
  for several epochs; then compute d = Ê_t[KL]; halve β if d < d_targ/1.5, double it if
  d > 1.5·d_targ; the new β is used for the next update. Offered as a baseline or an addition to
  clipping.
- **Combined actor-critic loss** (Sec. 5, Eq. 9, p. 5): with a network sharing parameters between
  policy and value function, maximize Ê_t[L^CLIP − c1 L^VF + c2 S[π_θ](s_t)], where L^VF =
  (V_θ(s_t) − V_t^targ)² and S is an entropy bonus for exploration.
- **Advantages** (Sec. 5, Eq. 10-12, p. 5): fixed-length segments of T steps (T much less than the
  episode length), with truncated GAE Â_t = δ_t + (γλ)δ_{t+1} + … up to δ_{T−1},
  δ_t = r_t + γV(s_{t+1}) − V(s_t); λ = 1 gives the A3C-style T-step estimator (Eq. 10).
- **Algorithm 1** (p. 5): each iteration, N parallel actors run π_old for T steps and compute
  Â_1…Â_T; optimize L on the NT samples for K epochs with minibatch size M ≤ NT, using SGD or
  "usually for better performance" Adam; then θ_old ← θ.
- **Experiments**: (1) objective comparison on 7 MuJoCo Gym v1 tasks, 1M steps, 3 seeds each,
  MLP 64-64 tanh policy outputting a Gaussian mean with variable std, no policy/value parameter
  sharing and no entropy bonus; scores normalized so random = 0 and best = 1, averaged over 21 runs
  (Sec. 6.1, p. 6); (2) PPO (ε = 0.2) vs tuned TRPO, CEM, vanilla PG with adaptive stepsize, A2C
  and A2C + trust region (Sec. 6.2, pp. 6-7); (3) Roboschool humanoid run/steer/get-up tasks
  (Sec. 6.3, p. 7); (4) 49 Atari games vs tuned A2C and ACER with the same policy network as
  [Mni+16], 3 trials, 40M frames (Sec. 6.4, p. 8; App. B, pp. 10-12).

## Claims

- C1: On the 7-task MuJoCo benchmark (1M steps, 3 seeds per task, 21 runs averaged), clipping with ε = 0.2 gives the best average normalized score, 0.82, of all settings tried (Table 1, Sec. 6.1, p. 6)
- C2: The other settings score: clipping ε = 0.1 → 0.76, ε = 0.3 → 0.70; adaptive KL d_targ = 0.003/0.01/0.03 → 0.68/0.74/0.71; fixed KL β = 0.3/1/3/10 → 0.62/0.71/0.72/0.69, with β initialized at 1 (Table 1, p. 6)
- C3: With no clipping or penalty the average normalized score is −0.39, because on HalfCheetah it ends far worse than the random initial policy (Sec. 6.1, Table 1, p. 6)
- C4: The KL penalty (fixed or adaptive) performed worse than the clipped surrogate in the authors' experiments; it is kept as an important baseline (Sec. 4, p. 4, Table 1, p. 6)
- C5: With the adaptive KL rule, updates whose KL differs significantly from d_targ occur but are rare and β adjusts quickly; the constants 1.5 and 2 are heuristic and the algorithm is not very sensitive to them, and the initial β is not important in practice (Sec. 4, p. 4)
- C6: Clipping in log space performed no better than the clipped surrogate (Sec. 6.1, p. 6)
- C7: Multiple optimization steps on the unclipped L^PG over the same trajectory gave results similar to or worse than the "no clipping or penalty" setting (results not shown) (Sec. 2.1, p. 2)
- C8: A fixed penalty coefficient β optimized with SGD is not sufficient to emulate TRPO's monotonic improvement; additional modifications are required (Sec. 2.2, p. 2)
- C9: L^CLIP equals L^CPI to first order around θ_old (r = 1) and is a lower bound on it, penalizing too large a policy update (Sec. 3, p. 3, Fig. 2, p. 4)
- C10: On the first Hopper-v1 update with the Sec. 6.1 hyperparameters, the policy after one PPO iteration has KL about 0.02 from the initial one, and L^CLIP is maximal at that point along the interpolation (Fig. 2, p. 4)
- C11: At 1M timesteps PPO (clip, ε = 0.2) outperforms tuned TRPO, CEM, vanilla PG with adaptive stepsize, A2C and A2C + trust region on almost all of the 7 MuJoCo environments (Sec. 6.2, Fig. 3, p. 7)
- C12: PPO learns 3D humanoid running (RoboschoolHumanoid, 50M steps), running toward a target that moves every 200 steps or on reaching it (Flagrun, 100M steps) and the same while pelted by cubes and getting up (FlagrunHarder, 100M steps), shown by learning curves without baselines (Sec. 6.3, Fig. 4, Fig. 5, pp. 7-8)
- C13: On 49 Atari games (3 trials), PPO wins 30 games on average reward over all of training vs 18 for ACER and 1 for A2C; on average reward over the last 100 episodes ACER wins 28, PPO 19, A2C 1, with 1 tie (Table 2, Sec. 6.4, p. 8)
- C14: PPO's final Atari scores (last 100 episodes, 40M frames = 10M timesteps) are the best of the three in 19 games and A2C's in 1 (Krull); in Venture all three score 0 (Table 6, App. B, p. 12)
- C15: PPO beats A2C significantly in sample complexity on Atari and performs similarly to ACER while being much simpler (Sec. 1, p. 1)
- C16: Synchronous A2C performed the same as or better than asynchronous A3C in the authors' tests (Sec. 6.2, p. 7)
- C17: PPO keeps the stability and reliability of trust-region methods while needing only a few lines of change to a vanilla policy-gradient implementation, applies to more general settings such as a joint policy-value architecture, and has better overall performance (Sec. 7, p. 8, p. 1)

## Implementation notes

- MuJoCo (Table 3, p. 10): horizon T = 2048, Adam 3e-4, 10 epochs, minibatch 64, γ = 0.99, λ = 0.95; ε = 0.2 (Sec. 6.2, p. 7); the number of actors is not given (source: App. A, p. 10, Sec. 6.2, p. 7).
- Roboschool (Table 4, p. 10): T = 512, 15 epochs, minibatch 4096, γ = 0.99, λ = 0.95, 32 actors (locomotion) or 128 (flagrun), log std of the action distribution linearly annealed from −0.7 to −1.6; Adam stepsize adjusted from a KL target whose value is not given (source: App. A, p. 10).
- Atari (Table 5, p. 10): T = 128, 8 actors, 3 epochs, minibatch "32 × 8", γ = 0.99, λ = 0.95, Adam 2.5e-4·α, ε = 0.1·α, c1 = 1, c2 = 0.01, with α annealed linearly from 1 to 0 over training; both stepsize and clip range are annealed (source: App. A, p. 10).
- Reading of "32 × 8": 256 samples per minibatch, i.e. 4 minibatches from the 8 × 128 = 1024 samples per iteration (source: assumption from Table 5, p. 10).
- 40M frames = 10M timesteps implies a frame skip of 4; other Atari preprocessing (frame stacking, resize, reward clipping) and the network are only given as "same policy network as [Mni+16]" (source: Table 6 caption, p. 12, Sec. 6.4, p. 8; the frame-skip reading is an assumption).
- Atari's c1 = 1 indicates a shared policy-value network with the combined Eq. 9 loss; the MuJoCo runs use separate networks and no entropy bonus (source: Table 5, p. 10, Sec. 6.1, p. 6; the shared-network reading is an assumption since Sec. 6.4 doesn't say it).
- Left unstated by the paper: V_t^targ (e.g. GAE return Â_t + V(s_t) vs n-step return), advantage normalization, value-loss clipping, observation and reward normalization, gradient-norm clipping, weight initialization, minibatch shuffling, how GAE handles episode ends inside a segment, and the KL estimator for Sec. 4 (source: assumption, a reading of Sec. 5, p. 5 and App. A, p. 10 which are silent; the later implementation-details literature covers them).
- Eq. 10-11 print exponents T−t+1 on the last terms; the reader takes T−t−1 as intended, matching the series' pattern (source: assumption from Eq. 10-11, p. 5).
- Score metric: average total reward of the last 100 episodes, normalized per environment (random 0, best 1) for MuJoCo; Atari reports per-game raw score and win counts averaged over three trials (source: Sec. 6.1, p. 6, Sec. 6.4, p. 8).
- For the user's design (reader's inference): a large transformer policy with a shared trunk and policy and value heads fits Eq. 9 directly, and the intro's point that first-order PPO, unlike TRPO, tolerates dropout and parameter sharing is what makes it a candidate (source: assumption from Sec. 1, p. 1, Sec. 5, p. 5).
- For the user's design (reader's inference): the Atari recipe (T = 128, 8 actors, 3 epochs, ε annealed from 0.1) is the paper's only pixel-game setting; none of its numbers was tuned for transformers or for multi-game training, so they are a starting point only (source: assumption from Table 5, p. 10).

## Limitations / open questions

- The hyperparameter search over ε, β and d_targ was done only on the cheap MuJoCo benchmark (1M steps, MLPs without sharing or entropy), so it does not test the shared-network, entropy-bonus or Atari settings (Sec. 6.1, p. 6).
- On Atari final performance ACER wins more games than PPO (28 vs 19), and results are 3 trials per game with no error bars in Table 6 (Table 2, p. 8, Table 6, p. 12).
- No theoretical guarantee is given for L^CLIP; the motivation is the lower-bound argument and Fig. 1-2 (Sec. 3, pp. 3-4).
- Hard-exploration games stay near zero: Venture 0.0 for all three, Montezuma's Revenge 42.0 for PPO (Table 6, p. 12).
- Open question: does PPO's on-policy sample cost stay acceptable for one large transformer trained across many games, where each update needs fresh rollouts from every game?
- Open question: are ε, epochs and minibatch size as insensitive at transformer scale as Table 1 suggests for small MLPs?

## Related

- related [[serrano2024crossdomain]]: cross-domain RL transfer; the survey's [[serrano2024crossdomain#C3]] says on-policy learners (PPO is one) suit parameter and reward-shaping transfer.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
