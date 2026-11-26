---
id: jiang2023importance
title: On the Importance of Exploration for Generalization in Reinforcement Learning
authors: [Yiding Jiang, J. Zico Kolter, Roberta Raileanu]
year: 2023
venue: NeurIPS 2023 (arXiv v1 preprint read)
url: https://arxiv.org/abs/2306.05483
arxiv: 2306.05483
doi:
pdf_sha256: 71f9533f05a51b9d4ad9c72177fe8d07ade28f9dad650866c1aee5b30b3fe6b1
tags: [reinforcement-learning, generalization, exploration, procgen, value-based, off-policy, on-policy, zero-shot]
status: read
read: 2026-11-26
---

# On the Importance of Exploration for Generalization in Reinforcement Learning

## TL;DR

Argues that how an agent explores its training environments changes zero-shot generalization in
contextual MDPs: a tabular toy shows UCB generalizes where epsilon-greedy does not, and EDE (UCB on
epistemic uncertainty from a QR-DQN ensemble, per-actor coefficients) tops Procgen median/IQM and beats Rainbow/PPO/DreamerV2 on Crafter (near LSTM-SPCNN).

## Problem

Deep value-based RL (DQN-style) generalizes much worse than policy-optimization methods (PPO and
descendants) on contextual MDPs such as Procgen, and most fixes treat generalization as pure
representation learning (regularization, augmentation) (Sec. 1, p. 1). The paper asks whether the
agent's exploration strategy, a RL-specific factor, also matters, and why value-based methods lag
(Sec. 1, Sec. 6, p. 1, p. 9).

## Method

- Setting: a contextual MDP (CMDP) with training MDPs drawn from a distribution q_M; the test
  return is the expected return over that distribution; generalization is measured as test return
  at a fixed interaction budget (Sec. 2, p. 3).
- Tabular study: a 5x5 gridworld, train start (x=0, y=0), test start (x=0, y=4), terminal reward 2 at (x=4, y=0), -0.04
  per step, 250-step cap. Compares Q-learning with epsilon-greedy ("Greedy"), Q-learning with UCB,
  each with one exploration hyperparameter swept over 10 values x 100 runs (Sec. 3, pp. 3-4).
  App. C adds policy gradient with entropy bonus, Sarsa(lambda) and a windy-world dynamics shift
  (App. C.1-C.4, pp. 20-23).
- EDE (Exploration via Distributional Ensemble): a shared feature extractor f with M = 5 QR-DQN
  heads, each trained on its own minibatch; epistemic variance is the disagreement between heads
  at each quantile (Eq. 1-2, Sec. 4, pp. 5-6). Actions come from UCB (Eq. 4) or Thompson sampling
  (Eq. 3). With K parallel actors, actor k uses coefficient phi * lambda^(1 + k*alpha/(K-1))
  ("temporally equalized exploration", TEE, Eq. 5, p. 6).
- Only one randomly chosen head per update back-propagates into f, so f is not trained M times
  faster than the heads (Alg. 3, App. E, pp. 25-26).
- Procgen: easy mode, 200 training levels, 25M steps, tested on the full level distribution, vs
  QR-DQN, PPO, IDAAC; ablations vs Bootstrapped DQN, UA-DQN, NoisyNet, epsilon-z-greedy
  (Sec. 5.1, pp. 7-8). Crafter: Rainbow code with only the exploration swapped (Sec. 5.2, p. 8).

## Assumptions

- A1: Train and test MDPs are drawn from one distribution over MDPs that share state and action spaces, and a single policy is optimal for all of them (Sec. 2, p. 3).
- A2: The tabular results use deterministic dynamics (except the windy variant), a tabular Q-table with no function approximation, and greedy (deterministic) action selection at test time (Sec. 3, p. 3, App. C.2, p. 21).
- A3: The deep-RL results are for discrete-action, pixel-observation value-based agents with a 5-head ensemble (Table 4, Table 5, pp. 32, 34).
- A4: Procgen results use easy mode, 200 training levels and a 25M-step budget, with EDE's hyperparameters tuned on bigfish only (Sec. 5.1, p. 7, Table 4, p. 32).
- A5: The argument that exploration helps assumes sub-optimal trajectories on training MDPs can pass through states that matter at test time (the paper says not all CMDPs have this structure) (App. B, p. 20).
- A6: In the tabular runs the test-time policy is greedy, with training-time exploration turned off (App. C.2, p. 21).
- A7: The Procgen runs collect experience from K = 64 parallel actors (Table 4, p. 32).

## Claims

- C1: In the 5x5 tabular CMDP all of Q-learning with epsilon-greedy and with UCB reach near-optimal training return, but on the test MDP (different start state) epsilon-greedy has the worst generalization and the highest variance while UCB reliably finds the optimal policy with negligible variance [A1, A2, A5] (Sec. 3, Fig. 2b, Fig. 2c, p. 4).
- C2: With each method's best hyperparameter, epsilon-greedy's test return depends strongly on epsilon and even epsilon = 0.9 is far worse than UCB; UCB and policy gradient with entropy bonus are fairly insensitive to their hyperparameters; policy gradient lands between epsilon-greedy and UCB [A1, A2] (App. C.1, Fig. 8, Fig. 9, pp. 20-21).
- C3: Policy gradient (softmax exploration) and Sarsa(lambda = 0.9) with epsilon-greedy, both on-policy whole-trajectory updates, beat one-step Q-learning with epsilon-greedy on the test MDP but remain worse than UCB; Sarsa(lambda) also has larger variance. The authors hypothesize that whole-trajectory updates pass the learning signal to rarely visited states faster, a benefit they treat as distinct from exploration [A1, A2] (App. C.3, Fig. 10, pp. 21-22).
- C4: In a windy-world variant where dynamics change at test time (40% chance of being blown two cells), the same pattern holds: better exploration gives better test return, though returns are lower and noisier for all methods [A1, A2] (App. C.4, Fig. 11, Fig. 12, pp. 22-23).
- C5: Using Q-learning's sample-complexity bound of Li et al., sample complexity is inversely proportional to the visitation probability of the hardest-to-reach state-action pair under the behaviour policy, so exploration that covers states more uniformly lowers Q-error across states, which the authors use to motivate why exploration helps in CMDPs [A1] (App. A, pp. 17-18).
- C6: On Procgen (easy, 25M steps, 5 seeds) EDE gets almost 3x QR-DQN's median and over 2x its IQM score, beats PPO and IDAAC on median and IQM, and is competitive with IDAAC on mean and optimality gap [A3, A4, A7] (Sec. 5.1, Fig. 4, p. 7).
- C7: In the Procgen exploration comparison, the epistemic-uncertainty UCB bonus and training on diverse data from TEE are both needed: QR-DQN+UCB+TEE (EDE) beats QR-DQN+UCB and QR-DQN+TEE; DQN+UCB and Bootstrapped DQN (which use total rather than epistemic uncertainty) are worse than DQN; UA-DQN is worse than QR-DQN+UCB; NoisyNet is the best of the other baselines but below EDE [A3, A4, A7] (Sec. 5.1, Fig. 5, pp. 7-8, App. G, p. 31).
- C8: On Crafter after 1M steps (10 runs) EDE scores 11.7 +/- 1.0 %, against Rainbow 4.3 +/- 0.2, PPO 4.6 +/- 0.3, DreamerV2 10.0 +/- 1.2 and LSTM-SPCNN 12.1 +/- 0.8; EDE uses Thompson sampling and no TEE there [A3] (Sec. 5.2, Table 1, p. 8, App. I, p. 33).
- C9: On a subset of Procgen games EDE is much more sample-efficient than the baselines, but on others it matches policy-optimization methods or fails to train; chaser, maze, heist and miner (long-horizon planning) stay much worse than policy optimization. EDE is better than or equal to QR-DQN in 11 of 16 games [A3, A4, A7] (App. F, Fig. 13, pp. 27-28, App. B, p. 20).
- C10: In a meta-analysis of 7 PPO-based algorithms, train and test performance are strongly rank-correlated on many Procgen games (almost slope 1, almost no generalization gap), while games with negative correlation are mostly saturated or too hard; so for many games better optimization on the training levels would be expected to improve test performance [A4] (App. G, Fig. 15a, Fig. 16a, Fig. 16b, pp. 29-31).
- C11: Single-environment sample efficiency does not explain the DQN-PPO gap (DQN is usually slightly more sample-efficient on single Procgen levels), and giving the Q-network a task ID does not improve training performance, so partial observability may not be the main problem as far as training is concerned; the authors conclude value overfitting is a real problem, mitigated by distributional RL, while exploration brings the largest gain [A3] (App. G, Fig. 15b, Fig. 15c, pp. 29-30).
- C12: EDE with 5 heads runs about 2.4x slower than QR-DQN, is insensitive to the number of heads M and to TEE's lambda and alpha, mildly sensitive to phi, and training the feature extractor from every head's gradient lowers performance (the ablations use 1 seed on a subset of games) [A3, A4, A7] (App. E, p. 25, App. J, Fig. 17, Fig. 18, pp. 34-35).

## Implementation notes

- Ensemble M = 5 heads, N = 200 quantiles, K = 64 actors, minibatch 512, buffer 1e6, gamma 0.99, 3-step returns, Adam lr 2.5e-4, phi = 30, lambda = 0.6, alpha = 7 for Procgen; phi, lambda, alpha were searched on bigfish over {10,20,30,50}, {0.5,0.6,0.7}, {6,7,8,9} (source: Table 4, App. I, p. 32).
- Architecture: IMPALA ResNet feature extractor, flattened to d = 2048, then 2 x 512 MLP heads, |A| = 15 (source: App. I, p. 32).
- Baselines use decaying epsilon(t) = min(0.1 + 0.9 exp(-(t-2000)/8000), 1) (source: App. I, p. 33).
- Crafter: Hafner's Rainbow code with C51 replaced by QR-DQN, minibatch 64, no prioritized replay, phi = 0.5 chosen from {0.5, 1, 2} (source: App. I, pp. 33-34).
- Tabular runs use base learning rate 0.05 decaying as 1/sqrt(t), gamma = 0.9, 1000 episodes x 100 trials (source: App. C.1, pp. 20-21).
- Compute: about 250 GPU-days on 2080 and V100 GPUs (source: App. K, p. 34).
- The authors' code is open-sourced at github.com/facebookresearch/ede; not consulted (source: p. 1).

## Limitations / open questions

- The authors say they do not claim exploration is the only route to better generalization; benefits are complementary to representation-learning methods (Sec. 3, p. 4).
- Not every CMDP has the structure in which exploring the training MDPs helps; the claim is that it should not hurt, supported empirically only by the Procgen results (App. B, p. 20).
- Value-based methods, EDE included, still fail on four long-horizon Procgen games, and the generalization gap is not closed (Fig. 13, p. 28, App. G, p. 29).
- EDE's cost (2.4x) comes from the ensemble; they expect cheaper uncertainty estimates to help (Sec. 7, pp. 9-10).
- The Procgen hyperparameters were tuned on one game, bigfish, while IDAAC's were tuned per game (App. F, p. 27).
- Open question (reader's): evidence on on-policy vs off-policy is confined to tabular toys (C3) and cross-algorithm Procgen comparisons; no controlled deep-RL experiment isolates it.
- Open question (reader's): nothing here concerns exploration when training data is a fixed human-play dataset, as in behaviour cloning; exploration only enters as the data-collection policy.

## Related

- related [[farebrother2020generalization]]: earlier DQN generalization study that finds regularization helps; this paper locates the problem in exploration and value overfitting instead.
- related [[schulman2017proximal]]: PPO is the on-policy baseline that EDE is compared with and that Sec. 1 calls stronger than value-based methods on Procgen.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
