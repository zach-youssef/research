---
id: kim2020domain
title: Domain Adaptive Imitation Learning
authors: [Kuno Kim, Yihong Gu, Jiaming Song, Shengjia Zhao, Stefano Ermon]
year: 2020
venue: ICML 2020
url: https://arxiv.org/abs/1910.00105
arxiv: 1910.00105
doi:
pdf_sha256: 8e4db306f8fdfa7076d9f9c3b24914cd3d744c145762ddc98659f178e6638708
tags: [imitation-learning, cross-domain, cross-embodiment, unpaired-alignment, mdp-homomorphism, adversarial-training, zero-shot, theory, continuous-control]
status: read
read: 2026-11-22
---

# Domain Adaptive Imitation Learning

## TL;DR

DAIL learns state and action maps between a self and an expert domain from unpaired, unaligned
demos of shared alignment tasks (GAMA: adversarial transition matching plus behavioural cloning),
then zero-shot imitates new expert tasks; theory says when the maps are MDP reductions.

## Problem

Imitating across domains that differ in embodiment, viewpoint or dynamics (Sec. 1, p. 1). Most
earlier methods need paired, time-aligned demonstrations to get state correspondences and then an
RL step with environment interaction on the target task; such pairs are rarely available and RL
is expensive (p. 1, Fig. 1a, p. 2). Embodiment and viewpoint mismatch were also handled by
separate methods (App. A, p. 10).

## Method

- **Setting.** Deterministic MDPs; a domain is an MDP without reward, a task T a label for a
  reward. Given expert demonstrations of target task T in the expert domain y, find an optimal
  self-domain (x) policy without its reward, using an alignment task set D_{x,y} of
  demonstrations of N tasks in both domains, unpaired and unaligned (Sec. 2, p. 2).
- **MDP reductions.** (φ, ψ) maps x states and actions to y's such that optimal y pairs pull back
  only to optimal x pairs, every optimal y pair is hit, and dynamics commute on optimal pairs
  (Def. 1, Eq. 1-3, p. 3). MDPs are alignable if a reduction exists either way (Def. 2, p. 3);
  pairs sharing reductions are jointly alignable, and more alignment tasks shrink the set of
  shared reductions, like training data in supervised learning (Def. 3, Fig. 3, pp. 3-4).
- **Objectives.** The composite policy π̂_x = g ∘ π_y ∘ f (Fig. 1b, p. 2) is run in the self
  domain, giving translated transitions (f(s_x), â_y ~ π_y(f(s_x)), f(s′_x)) (Def. 4, Eq. 4, p. 4).
  Objective 1: π̂_x is optimal; objective 2: the distribution of those transitions matches the
  expert's own transitions σ_y (Eq. 5-6, p. 4); combined as −J(π̂_x) + λ d(σ, σ_y) (Eq. 7, p. 4).
- **GAMA.** Per alignment task, a discriminator separates real y transitions from translated x
  transitions; f, g are trained to fool it while a KL (behavioural cloning) term makes π̂_x match
  the self-domain demonstrations; π_{y,T_i} are fitted by BC on expert demos, and a learned x
  dynamics model carries gradients (Eq. 8, Alg. 1, p. 5).
- **Adaptation.** GAMA-PA: fit π_{y,T} on the new expert demos by BC and use g ∘ π_{y,T} ∘ f
  zero-shot. GAMA-DA: also learn f⁻¹, translate the demos into the self domain and BC on them
  (p. 5, Fig. 1c, p. 2, Sec. 6, p. 7).
- **Experiments.** Gym pendulum, cartpole, 2- and 3-link reacher, third-person reacher,
  3- and 4-link swimmer "snakes" (Sec. 6, p. 7, App. D, pp. 11-12). Six DAIL problems: dynamics
  (noise-injected reacher2), embodiment (reacher3 expert) and viewpoint (30° and 180° third-person
  reachers) with reach, push or letter-writing targets (Sec. 6.2, p. 8). Metrics: alignment
  complexity (alignment tasks needed) and adaptation complexity (target demos needed) (Sec. 6,
  p. 6). Baselines: CCA, UMA, IF, IfO, TPIL, SAIL, self-demonstrations (p. 7).

## Assumptions

- A1: An alignment task set exists: demonstrations of the same N tasks in both the self and the expert domain, unpaired and unaligned (Sec. 2, p. 2).
- A2: Demonstrations in both domains include actions; GAMA does not work from observations alone (Table 2, p. 10, Sec. 7, p. 8).
- A3: The theory assumes discrete, unichain MDPs with deterministic dynamics, plus dummy states and actions that are never optimal, as its Assumption 1 states (App. F, p. 13).
- A4: Self and expert MDPs are alignable, i.e. a reduction exists; experiments also use pairs the authors call only "weakly" alignable (Sec. 3, p. 3, Sec. 4, p. 4).
- A5: A target task's MDP pair shares the reductions of the alignment task pairs (lies in their equivalence class), so maps learned on the alignment tasks transfer to the target task (Sec. 3, pp. 3-4).
- A6: The converse direction of Theorem 2 needs an injective action map g; in experiments injectivity isn't enforced (Sec. 4, p. 4, Sec. 5, p. 4).
- A7: Demonstrations are from optimal or expert policies in both domains, and expert-domain policies are fitted from them by behavioural cloning (Sec. 5, p. 5, Eq. 8, p. 5).
- A8: A self-domain dynamics model is learned from the alignment demonstrations; the target task needs no environment interaction or reward (Alg. 1, p. 5, App. A, p. 10).
- A9: States and actions are low-dimensional joint-level vectors, except in the visual variants, which align in the latent space of a deep spatial autoencoder trained on alignment-task images (Sec. 6.2-6.3, p. 8, App. D, p. 11).
- A10: Alignment and target tasks are from one task family per problem (reaching for different goals, or reach to push or write), with target goals placed far from alignment goals (App. D, pp. 11-12).

## Claims

- C1: Theorem 1: if (φ, ψ) is a reduction from the self MDP to the expert MDP and π_y is optimal, then for any g with ψ ∘ g the identity on optimal expert actions, g ∘ π_y ∘ φ is optimal in the self MDP [A3, A4] (Sec. 3, p. 3, App. F.1, p. 13).
- C2: Theorem 2: if the self MDP reduces to the expert MDP, some f, g and covering optimal π_y satisfy objectives 1 and 2; conversely, any f with injective g and optimal covering π_y satisfying both implies alignability and f is a reduction's state map [A3, A4, A6] (Sec. 4, p. 4, App. F.2, pp. 13-19).
- C3: Mean ℓ2 loss of the learned state map against the ground-truth permutation over 5 seeds: GAMA 0.057 ± 0.017 (pen↔pen), 0.178 ± 0.051 (pen↔cart, pole dimensions) and 0.092 ± 0.043 (reach2↔reach2-tp), against best baselines 0.72 (CCA), 1.62 (IF) and 5.12 (IfO); 17.3× lower than the best baseline on average, UMA and random above 100 [A1, A2, A4, A7, A9] (Sec. 6.1, p. 7, Table 1, p. 6).
- C4: Qualitatively GAMA recovers the permutation for pen↔pen and maps the pendulum's angle and angular velocity onto the pole's for pen↔cart; on the weakly alignable snake4↔snake3 it matches two adjacent snake4 joints to snake3's two and keeps the wiggle's period, and on reacher2↔reacher3 it matches the first joint and the extent of contraction [A1, A2, A7, A9] (Sec. 6.1, p. 7, Fig. 4, p. 6, Fig. 7, p. 12).
- C5: The authors attribute the baselines' failure to DTW-based pairing (IF, IfO) on demos with diverse start states, up to 2× length differences and varying execution rates, and to the less flexible linear maps of CCA and UMA [A1, A9] (Sec. 6.1, p. 7).
- C6: Alignment complexity: GAMA's maps enable zero-shot imitation of target tasks, with clear gains over a policy pretrained on the self-domain alignment tasks; with 12 alignment tasks GAMA-PA reaches about 0.9-1.0 of scaled expert performance on the five problems tested that way while pretraining plateaus near 0.5-0.65, and on V-R2W with all tasks about 0.98 vs 0.88 (read from the figure); other baselines can't imitate zero-shot [A1, A2, A3, A4, A5, A7, A8, A9, A10] (Sec. 6.2, p. 8, Fig. 6, p. 8).
- C7: Adaptation complexity: GAMA's adapted demonstrations are about as useful as self-domain demonstrations, while CCA, UMA, IF and IfO mostly fail at DAIL, mainly because their alignment fails (5 runs) [A1, A2, A4, A5, A7, A9, A10] (Sec. 6.2, p. 8, Fig. 5, p. 7).
- C8: TPIL succeeds on V-R2R (30° viewpoint offset) but fails on V-R2W (180°) [A1, A9, A10] (Sec. 6.2, p. 8, Fig. 5, p. 7).
- C9: SAIL outperforms GAMA and even self-demonstrations on the dynamics problems where it is run, but uses the self-domain environment simulator, which GAMA doesn't [A1, A2, A7, A8, A9, A10] (Sec. 6.2, p. 8, Fig. 5, p. 7).
- C10: With 64×64×3 image states, aligning in a deep spatial autoencoder's latent space (GAMA-PA-vis, GAMA-DA-vis) gives alignment and adaptation complexity similar to GAMA on joint states, and better than baselines that use joint states [A1, A2, A4, A5, A7, A9, A10] (Sec. 6.3, p. 8, Fig. 5, p. 7, Fig. 6, p. 8).
- C11: In practice, enforcing injectivity of g was unnecessary in continuous domains [A6] (Sec. 5, p. 4).

## Implementation notes

- f, g, f⁻¹, the x dynamics model and the discriminators: MLPs with hidden sizes (200, 200); fitted policies π_{y,T_i} and π_{x,T}: (300, 200); Adam (β 0.9, 0.999), lr 1e-4 for the alignment maps and 1e-5 elsewhere, fixed across experiments (source: App. B, p. 10).
- Visual variants: the deep spatial autoencoder architecture of Finn et al. 2015 / Levine et al. 2016 (source: App. B, p. 10, Sec. 6.3, p. 8).
- λ in Eq. 7-8, the number of demonstrations per alignment task and training lengths are not stated (source: assumption: tune λ, start near 1).
- The state map acts only on non-goal dimensions; reacher states use joint coordinates instead of angles, with the end-effector-to-goal vector removed (source: App. D, p. 11).
- Baselines: IF and CCA use 5000 trajectories per domain, UMA 20 (pendulum, cartpole) or 50 (reacher); RL steps for baselines use DDPG on a proxy reward (source: App. C, pp. 10-11).
- Letter-writing reward: 100 on reaching a vertex, −1 otherwise (source: App. D, p. 12).
- No code link in the paper; videos at https://youtu.be/l0tc1JCN_1M (source: Fig. 4, p. 6).

## Limitations / open questions

- The theory needs exact alignability; extending it to MDPs that aren't perfectly alignable, imitation from observations without actions, and designing minimal alignment task sets are left open (Sec. 7, p. 8).
- Needs the same tasks demonstrated in both domains (proxy tasks), which GWIL ([[fickinger2022cross]]) and UDIL ([[franzmeyer2022matters]]) remove.
- Open question (reader): the theory is for discrete deterministic MDPs (A3), and all experiments are low-dimensional MuJoCo-style control; whether the alignment task set is cheap for games, where several games share sub-tasks (move to a target, avoid), is untested.
- Open question (reader): a small test for the cross-game embedding: two grid games with permuted action labels and mirrored layouts, a handful of shared reach tasks, and check whether GAMA's g recovers the action permutation.

## Related

- related [[gupta2017invariant]]: IF is a baseline; it needs paired, time-aligned proxy-task demonstrations (DTW pairing here), which GAMA avoids.
- related [[liu2018context]]: IfO is a baseline; its context translation also learns from paired demonstrations.
- related [[ganin2016domain]]: GAMA's discriminator on translated transitions is the same adversarial alignment idea, applied to MDP transitions.
- related [[zhang2020correspondence]]: later unpaired state and action correspondence with dynamics cycle-consistency, citing this line of work.
- related [[franzmeyer2022matters]]: UDIL builds on this paper's reduction result and drops the alignment tasks.
- related [[fickinger2022cross]]: GWIL also aligns domains without pairs but needs no alignment tasks, at the price of isometry ambiguity.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
