---
id: raychaudhuri2021crossdomain
title: Cross-domain Imitation from Observations
authors: [Dripta S. Raychaudhuri, Sujoy Paul, Jeroen van Baar, Amit K. Roy-Chowdhury]
year: 2021
venue: ICML 2021
url: https://arxiv.org/abs/2105.10037
arxiv: 2105.10037
doi:
pdf_sha256: d3e87c8ffb146f86a685911850900aae3a136ead64712bc8ca18ffa84ae7c1c6
tags: [imitation-learning, imitation-from-observation, cross-domain, cross-embodiment, unpaired-alignment, cycle-consistency, adversarial-training, mutual-information, continuous-control]
status: read
read: 2026-11-22
---

# Cross-domain Imitation from Observations

## TL;DR

xDIO learns an expert-to-agent state map from unpaired, unaligned, state-only proxy-task demos
(adversarial transition matching, cycle-consistency, a domain-confused latent, temporal-position
consistency), then imitates new tasks by BCO; 0.78-1.00 of expert on six MuJoCo pairs.

## Problem

Imitation usually needs demonstrations from the agent's own MDP; the expert may differ in
dynamics, viewpoint or morphology (Sec. 1, p. 1). Earlier cross-domain methods (IF, viewpoint
methods) need paired, time-aligned proxy-task demonstrations and an extra RL step; DAIL
([[kim2020domain]]) works from unpaired, unaligned demos but needs expert actions and an online
expert policy (Sec. 1, pp. 1-2, Table 1, p. 3). The paper's setting, xDIO, removes all four:
unpaired trajectories, states only, no online expert, no RL (Table 1, p. 3).

## Method

- **Setting.** A domain is (S, A, P, P0); an MDP for task T adds reward and discount. Expert
  domain E, agent domain A. Given state-only expert demos of an inference task T, learn the
  agent's policy for T, using M proxy tasks with state-only trajectories in both domains,
  unpaired and not time-aligned, and no access to the expert policy (Sec. 3, p. 3).
- **Local alignment.** A map ψ: S_E → S_A is trained per proxy task j against a discriminator
  on state transitions (s_t, s_t+1), so translated expert transitions match the agent's
  (Eq. 1, p. 4); a reverse map φ and a cycle loss ||φ∘ψ(s_E) − s_E||² plus the reverse
  (Eq. 2, p. 4) stop arbitrary permutations of states.
- **Domain-agnostic latent.** ψ = D_E ∘ E_E and φ = D_A ∘ E_A share a latent Z; minimising the
  mutual information between domain and latent transitions (a variational bound with a per-task
  discriminator q^j, Eq. 3-4, p. 4) aligns them, plus latent consistency
  ||E_A∘ψ(s_E) − E_E(s_E)||² and the reverse (Eq. 5, p. 4).
- **Global alignment.** Cycle-consistency alone can map start states to terminal ones. Each
  state gets a temporal position γ^(H−t) (1 at the terminal goal state, discounted back);
  per-domain estimators P_E^j, P_A^j are pre-trained by regression (Eq. 6, p. 5) and then a
  state's position must survive translation (Eq. 7, p. 5). The authors read these as value
  functions of a greedy optimal policy with reward 1 at terminal states (p. 5).
- **Inference task adaptation.** The inference task's expert states regularise training: cycle
  loss on them (Eq. 8) and a latent position predictor P_z matched to an estimator P_E^T trained
  on them (Eq. 9, Sec. 4.3, p. 5).
- **Objective.** Weighted sum of the five groups (Eq. 10, p. 5), all weights 1 (App. B, p. 12);
  Alg. 1 alternates proxy-task updates and inference adaptation (App. A, p. 12).
- **Imitation.** ψ translates the inference-task expert states to the agent domain; an inverse
  dynamics model fitted on random-exploration transitions (Eq. 11) labels them with actions, and
  the policy is trained by behavioural cloning, i.e. BCO (Sec. 4.5, p. 6). The map is
  independent of the imitation-from-observation algorithm (p. 6).
- **Tasks.** Seven Gym/MuJoCo environments with joint-level states; six pairs: D-R2R (friction
  doubled), V-R2R (third-person reacher, 180° planar offset), V-R2W (proxy reaching, inference
  letter tracing), M-R2R (3-link expert, 2-link agent), M-A2A (6-legged to 4-legged ant), M-A2C
  (ant to HalfCheetah) (Sec. 5.1, p. 6). Reacher proxies: 4 goals near the wall, inference 4 new
  goals in the corners; ant/cheetah: reach the centre of a 5 m circle from different starting
  arcs (App. C, pp. 12-13, Fig. 7-8).
- **Evaluation.** Reward normalised by expert performance, 10 runs; baselines CCA, IF (with DTW
  on the learned latent for pairing) and CycleGAN, all followed by the same BCO step (Sec. 5.2,
  pp. 6-7, App. B, p. 12).

## Assumptions

- A1: Proxy tasks exist that are demonstrated in both domains, with state-only, unpaired and not time-aligned trajectories (Sec. 3, p. 3).
- A2: States are low-dimensional joint-level vectors (angles, velocities, goal position), not images (Sec. 5.1, p. 6, App. C, pp. 12-13).
- A3: Proxy and inference tasks come from one task family per pair (reaching or moving to different goals, or reaching to letter tracing), with inference goals placed far from proxy goals (Sec. 5.1, p. 6, App. C, pp. 12-13).
- A4: Demonstrations are from trained expert policies (PPO for reacher, A3C for ant and cheetah) in both domains (Sec. 5.2, p. 7).
- A5: Trajectories end in a terminal goal state, so temporal position γ^(H−t) measures progress toward task completion (Sec. 4.2, p. 5).
- A6: The agent domain can be explored with random actions to fit the inverse dynamics model used by BCO (Sec. 4.5, p. 6).
- A7: The state map acts only on non-goal dimensions; goal coordinates are shared across domains (App. C, p. 12).
- A8: The state-only expert trajectories of the inference task are available during map training (inference task adaptation) (Sec. 4.3, p. 5).

## Claims

- C1: On inference tasks, xDIO's policy reaches expert-normalised reward 0.95 ± 0.03 (V-R2R), 0.93 ± 0.01 (V-R2W), 0.99 ± 0.02 (D-R2R), 0.96 ± 0.07 (M-R2R), 0.78 ± 0.08 (M-A2A) and 1.00 ± 0.00 (M-A2C), the best on every pair (10 runs) [A1, A2, A3, A4, A5, A6, A7, A8] (Sec. 5.3, p. 7, Table 2, p. 7).
- C2: The best baseline per pair reaches 0.32 (IF, V-R2R), 0.86 (CCA, V-R2W), 0.48 (IF, D-R2R), 0.61 (IF, M-R2R), 0.30 (CCA, M-A2A) and 0.80 (CycleGAN, M-A2C) [A1, A2, A3, A4, A6, A7] (Table 2, p. 7).
- C3: The authors attribute IF's failure to DTW pairing on demos with diverse start states, up to 1.5× length differences and varying execution rates, and CCA's and CycleGAN's to having nothing that preserves MDP task semantics [A1, A2] (Sec. 5.3, p. 7).
- C4: As the number of inference-task demonstrations grows (x-axes differ per pair: up to 100 demos for V-R2W, D-R2R and M-R2R, 5 to 200 for M-A2A, 1 to 200 for the rest), xDIO's transferred demos are about as effective as the agent's own expert demos (Self-demo upper bound) on all six pairs at larger counts, and above the baselines (read from the figure); at the smallest counts xDIO is only level with some baselines (e.g. V-R2W at 1 demo with CCA, M-A2A at 5 demos where all are near 0) [A1, A2, A3, A4, A6, A7, A8] (Sec. 5.3, p. 7, Fig. 4, p. 7).
- C5: More proxy tasks (1 to 4) give better alignment; with 4 xDIO reaches about 0.78 to 1.00 of expert performance depending on the pair (lowest: M-A2A at about 0.78, Table 2 and Fig. 5), and with fewer it still beats the baselines in most cases (read from the figure) [A1, A2, A3, A4, A6, A7, A8] (Sec. 5.3, p. 8, Fig. 5, p. 8).
- C6: Removing inference task adaptation lowers reward on every pair (to 0.81, 0.88, 0.74, 0.78, 0.46, 0.78), which the authors call a small drop, yet it still exceeds the baselines on almost all pairs [A1, A2, A3, A4, A6, A7] (Sec. 5.4, p. 8, Table 3, p. 8).
- C7: Removing the mutual-information term gives 0.60, 0.92, 0.76, 0.67, 0.28, 0.80, and removing temporal position preservation 0.64, 0.84, 0.70, 0.72, 0.36, 0.43 (V-R2R to M-A2C), which the authors read as showing that cycle-consistency alone doesn't give global alignment [A1, A2, A3, A4, A5, A6, A7, A8] (Sec. 5.4, pp. 8-9, Table 3, p. 8).
- C8: On M-R2R the learned map places the 2-link end effector so the reaching motion keeps its task semantics, unlike CCA, IF and CycleGAN (qualitative) [A1, A2, A7] (Sec. 5.3, p. 7, Fig. 6, p. 8).

## Implementation notes

- ψ, φ: MLP encoder and decoder each with hidden sizes [128, 64] (reacher) or [512, 256] (others); discriminators D_A^j, D_E^j, q^j: [128, 128] (reacher) or [512, 256, 128], spectral normalisation, least-squares GAN loss instead of the log loss (source: App. B, p. 12).
- Temporal position estimators [200, 128]; latent position predictor P_z [64, 64]; policy [64, 64]; inverse dynamics model [100, 100]; Adam lr 1e-4; λ1-λ5 = 1; one Titan Xp, PyTorch (source: App. B, p. 12).
- 200 expert trajectories per proxy task in each domain for learning the map; CCA embedding size is the smaller state dimension; CycleGAN uses the same architecture as the map, IF [128, 64] with leaky ReLU (source: App. B, p. 12).
- The discount γ for the temporal position target, mini-batch size, iteration counts and the size of the random-exploration dataset for the inverse model are not stated (source: assumption; guess: γ near 0.99, tune per environment).
- V-R2W reward: 100 for reaching a letter vertex, −1 otherwise; the goal moves to the next vertex of the letter C (source: App. C, p. 13).
- Code and videos: https://driptarc.github.io/xdio.html (source: Sec. 5, p. 6, not checked by the reader).

## Limitations / open questions

- Proxy tasks are still needed; future work is learning the correspondence from random trajectories instead (Sec. 6, p. 9).
- Table 2 and Table 3 report different spreads for xDIO on V-R2R (0.95 ± 0.03 against 0.95 ± 0.05) and V-R2W (± 0.01 against ± 0.00) for what reads as the same configuration; the paper doesn't explain it (reader's observation, p. 7, p. 8).
- Open question (reader): the inference-task expert states are used during map training (A8), so the zero-shot reading of the results is weaker than "map once, imitate anything"; the no-adaptation row (C6) is the cleaner transfer number.
- Open question (reader): all states are joint vectors with shared goal coordinates (A2, A7); for game states (objects, pixels) the map would need an encoder whose latent is domain-confused the same way, as in the MI term, which is untested here.

## Related

- related [[kim2020domain]]: DAIL is the closest prior work; xDIO drops its need for expert actions and an online expert, and uses its reacher task family.
- related [[gupta2017invariant]]: IF is a baseline here, paired by DTW on the learned latent since the data is unpaired.
- related [[zhang2020correspondence]]: also learns unpaired cross-domain state maps with adversarial and cycle-consistency losses, but uses actions and dynamics cycles.
- related [[dwibedi2019cycle]]: another cycle-consistency approach to temporal alignment, matching frames rather than learning a state map.
- related [[fickinger2022cross]]: GWIL removes proxy tasks altogether, the future work this paper names.
- related [[franzmeyer2022matters]]: UDIL also imitates across domains without paired data, learning task-relevant embeddings.
- related [[liu2018context]]: imitation from observation across viewpoints, needing paired demonstrations.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
