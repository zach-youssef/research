---
id: eysenbach2022maximum
title: Maximum Entropy RL (Provably) Solves Some Robust RL Problems
authors: [Benjamin Eysenbach, Sergey Levine]
year: 2022
venue: ICLR 2022 conference paper (arXiv v2, 5 May 2022, read)
url: https://arxiv.org/abs/2103.06257
arxiv: 2103.06257
doi:
pdf_sha256: 87ff1cb614e9cdd9898129be4297f23741edf8d5aaa7bdcebeeecd04274aa5a7
tags: [reinforcement-learning, maximum-entropy, robustness, theory, sac, stochastic-policy, generalization]
status: read
read: 2026-11-26
---

# Maximum Entropy RL (Provably) Solves Some Robust RL Problems

## TL;DR

Proves that the MaxEnt RL objective lower-bounds a worst-case (robust RL) objective over perturbed
dynamics or rewards, with a robust set at least as large as the policy's entropy. Needs a pessimistic
reward for dynamics robustness; SAC with large entropy coefficient matches PR-MDP/NR-MDP on mass shifts.

## Problem

Stochastic MaxEnt policies are conjectured to be robust to disturbances, but no prior work formally
proves it or says which disturbances are covered; explicit robust RL methods add adversaries and
hyperparameters (Sec. 1-2, pp. 1-2). The paper characterizes the robust set of standard MaxEnt RL,
which is a statement about worst-case performance under perturbed dynamics or rewards, not about
average-case generalization to new environments (Sec. 2, pp. 2-3).

## Method

- MaxEnt objective J_MaxEnt(pi; p, r) = E[sum_t r + alpha * H_pi[a|s]] with alpha = 1 and no discounting
  (Eq. 1, Sec. 3, p. 3). Robust RL objective: max over pi of min over dynamics p~ in P and reward r~ in R
  of the expected return (Sec. 3, p. 3).
- Theorem 4.1 (reward robustness): for finite rewards and a policy with full support, MaxEnt RL on r equals
  the minimum expected return over reward functions r~ in the set where E_pi[sum_t log-integral of
  exp(r - r~)] <= epsilon (Eq. 2, Sec. 4.1, p. 4). The adversary may look at the policy's visited states.
- Theorem 4.2 (dynamics robustness): applying MaxEnt RL to the pessimistic reward
  r~(s,a,s') = (1/T) log r(s,a) + H[s'|s,a] (Eq. 3) lower-bounds the robust objective by
  exp(J_MaxEnt + log T) over dynamics within a divergence budget epsilon of the training dynamics
  (Eq. 4-5, Sec. 4.2, pp. 4-5).
- Lemma 4.3 gives epsilon exactly, with epsilon >= T * E[H_pi[a|s]] for a discrete action space, so a larger entropy
  (e.g. larger alpha) gives a larger robust set (p. 5). The proofs are in App. A (pp. 13-19).
- Experiments (Sec. 5, pp. 7-9): SAC (MaxEnt) against TD3 or SVG-0 (standard) on Pusher with a new obstacle,
  Metaworld button-box moves, puck perturbations, a peg-insertion with CMA-ES worst-case hole shift, and
  OpenAI Gym reward robustness; against Tessler et al.'s PR-MDP and NR-MDP on mass changes.

## Assumptions

- A1: The MDP is fully observed, episodes have a fixed horizon T, rewards are undiscounted (discount absorbed into dynamics) and alpha = 1 in the analysis (Sec. 3, p. 3).
- A2: Theorem 4.1 needs finite rewards and a policy with support on all actions in all states (Sec. 4.1, p. 4).
- A3: Theorem 4.2 needs positive rewards and finite dynamics entropy, and Lemma 4.3 assumes a discrete action space (Sec. 4.2, Sec. 4.3, pp. 4-5).
- A4: Robustness to dynamics holds only for the pessimistic reward r~ (Eq. 3), which depends on the dynamics entropy; it can be ignored only when every state has the same stochasticity (Sec. 4.2, p. 4, Sec. 4.4, p. 7).
- A5: The robust sets are defined by an expected divergence budget (Eq. 2, Eq. 5), not the H-infinity or KL constraint sets of prior robust-RL work (Sec. 4.4, p. 7).
- A6: The robustness is worst-case over a perturbation set for a fixed task and reward, not average-case generalization to new tasks or environments (Sec. 2, pp. 2-3).

## Claims

- C1: Theorem 4.1: MaxEnt RL on reward r achieves exactly the minimum expected return over the reward-perturbation set of Eq. 2 for the learned policy, so MaxEnt RL on a reward is robust to reward functions r~ that are not much smaller than r at the states and actions the policy visits [A1, A2] (Sec. 4.1, Eq. 2, p. 4).
- C2: Theorem 4.2: MaxEnt RL on the pessimistic reward of Eq. 3 maximizes a lower bound, exp(J_MaxEnt + log T), on the worst-case return over dynamics in the set P(pi) (Eq. 5), which is a lower bound on the robust RL objective (Corollary 4.2.1) [A1, A3, A4, A5] (Sec. 4.2, Eq. 3-5, pp. 4-5).
- C3: Standard MaxEnt RL on the original reward is not robust with respect to that reward in general: robustness to dynamics requires applying MaxEnt RL to the pessimistic reward, and the two objectives use different reward functions [A4] (Sec. 1, p. 2, Sec. 4.2, pp. 4-5).
- C4: Lemma 4.3 says the robust set size epsilon is at least T times the expected policy entropy (a policy with 10 bits of entropy gives epsilon >= 10); so the way to enlarge the robust set is a larger entropy coefficient, and entropy-constrained SAC variants lower-bound the set's size [A1, A3] (Sec. 4.3, p. 5).
- C5: For a fixed reward r, the reward-robust set for a higher temperature alpha is a subset of the one for a lower alpha; and if r0 must lie in the robust set, a high temperature admits only reward functions similar to or pointwise better than r0, while a small one admits a wide range. The authors note that being robust to everything is not always desirable [A1, A2] (App. A.8, Fig. 9, p. 19).
- C6: On benchmark mass-change tasks (HalfCheetah, Walker2d, Hopper), evaluated 30 episodes over 5 seeds, MaxEnt RL with a large entropy coefficient is at least competitive with, if not better than, PR-MDP and NR-MDP, and a DDPG baseline [A6] (Sec. 5, Fig. 3, p. 7).
- C7: Standard RL (TD3) learns one route and fails when an obstacle is added (Pusher) or the button box is moved (Metaworld), while the stochastic MaxEnt (SAC) policy uses many routes and degrades less, in the new-obstacle final distance to goal and the button-offset reward [A6] (Sec. 5, Fig. 1, Fig. 4a, Fig. 4b, p. 1, pp. 7-8).
- C8: MaxEnt RL is not standard RL plus action noise: in a 2D navigation task with an L-shaped test obstacle, MaxEnt keeps entropy low near costly regions and raises it later (return 13.6), while standard RL (9.7) and standard RL plus Gaussian noise of sigma 0.3 (-0.8) or 1 (-65.3) do worse [A6] (Sec. 5, Fig. 5, p. 8).
- C9: Under random mid-episode puck perturbations, all entropy coefficients do well without disturbance, but only the largest entropy coefficient tried (1e-3) stays robust to larger disturbances, supporting the theory that alpha sets the robust set size [A6] (Sec. 5, Fig. 6, p. 8).
- C10: In peg insertion with a worst-case hole perturbation found by CMA-ES, standard and MaxEnt RL both have success near 100% for small shifts, but only MaxEnt RL keeps solving the task for 1.5 cm and 2 cm shifts; neither succeeds for larger [A6] (Sec. 5, Fig. 7, p. 9).
- C11: On four Gym continuous-control tasks (alpha = 1/5), both SAC and standard (SVG-0) maximize expected reward, but only MaxEnt RL maximizes the worst-case reward r(s,a) - log pi(a|s), compared with fictitious play and standard RL [A6] (Sec. 5, Fig. 8, p. 9, App. B.2, p. 23).
- C12: In 5-armed robust-reward bandits, a "LowerBound + MaxEnt" procedure reaches normalized minimax reward 0.91 (optimum 1), against 0.56 for the pointwise-minimum policy and 0.60 for the uniform policy, so the bound is not tight but useful [A1] (App. A.11, Fig. 10, pp. 21-22).

## Implementation notes

- MaxEnt = SAC (TF-Agents implementation, defaults mostly), standard = same code with entropy coefficient 1e-5 (Pusher) or 1e-100 (Metaworld button; 1e1 for MaxEnt there); Pusher MaxEnt uses 1e-2 (source: App. B.1, p. 22).
- The Gym reward-robustness runs use alpha = 1/5 following Haarnoja et al. 2018a, five seeds (source: App. B.2, p. 23).
- Baselines PR-MDP, NR-MDP and DDPG use Tessler et al.'s open-source code; ablations (larger nets, dual critic, more noise) did not help them (source: Sec. 5, p. 7, App. C, p. 23).
- The analytic worst-case reward for MaxEnt policies is r(s,a) - log pi(a|s) (source: App. B.2, p. 23).
- Computing the pessimistic reward needs the dynamics' entropy; the paper gives no practical estimator (source: Sec. 4.4, p. 7).

## Limitations / open questions

- The robust set differs from standard H-infinity or KL sets; whether MaxEnt RL is robust for those is open, and the authors say it may not be (Sec. 4.4, p. 7).
- Building the pessimistic reward is hard unless dynamics entropy is constant; finding tractable estimators is future work (Sec. 4.4, p. 7).
- MaxEnt RL is not necessarily the ideal robust method; the entropy coefficient must still be chosen and the robust set is not always simple (Sec. 6, p. 9).
- Open question (reader's): App. A.8's closing sentence says a higher temperature gives robustness to a larger set, which reads as the reverse of its own preceding statements (and of C4's larger-alpha-larger-set); the paper does not reconcile them.
- Open question (reader's): the paper is about worst-case perturbations of one task, not new levels or new games, so how much of this transfers to zero-shot generalization is untested here.
- Open question (reader's): all tested domains are low-dimensional continuous control with SAC; nothing is shown for pixel observations or discrete-action games.

## Related

- related [[jiang2023importance]]: Jiang gets generalization from exploratory data collection with a deterministic test policy; here robustness comes from the stochastic policy itself at deployment, for worst-case perturbations of one task.
- related [[schulman2017proximal]]: PPO's entropy bonus is a MaxEnt-style regularizer, though the paper itself analyses SAC-style objectives.
- related [[farebrother2020generalization]]: tests a different regularizer (dropout, l2) for zero-shot generalization; this paper does not address environment generalization.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
