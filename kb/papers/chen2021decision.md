---
id: chen2021decision
title: "Decision Transformer: Reinforcement Learning via Sequence Modeling"
authors: [Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch]
year: 2021
venue: NeurIPS 2021 conference paper (read as arXiv v2)
url: https://arxiv.org/abs/2106.01345
arxiv: 2106.01345
doi:
pdf_sha256: 6ff675a7b8286ca7bf8e01e4b90a41bf5c90f4c3c007e17fdb4d24e26fd05e28
tags: [reinforcement-learning, offline-rl, transformer, sequence-modeling, return-conditioning, atari, continuous-control, imitation-learning]
status: read
read: 2026-11-20
---

# Decision Transformer: Reinforcement Learning via Sequence Modeling

<!-- Read from arXiv v2 (2021-06-24), 21 PDF pages; pp. 13-17 are acknowledgements and references, pp. 18-21 are App. A (hyperparameters, graph shortest path) and App. B (Atari raw scores). -->

## TL;DR

Casts offline RL as conditional sequence modelling: a GPT over (return-to-go, state, action) tokens,
trained with a supervised action loss and prompted with a target return; without TD learning it is
comparable to CQL on 3 of 4 Atari games, best on most D4RL tasks, far above CQL and BC on Key-to-Door.

## Problem

Offline RL from fixed, suboptimal data usually relies on TD learning with bootstrapping, discounting
and value pessimism or policy constraints, which are unstable ("deadly triad") and propagate credit
slowly over long horizons (Sec. 1, p. 3; Sec. 6.1, p. 11). The paper asks whether generative trajectory
modelling with a plain transformer can *replace* conventional RL algorithms rather than just serve as
a component inside them (Sec. 1, p. 3).

## Method

- **Trajectory representation**: rewards are replaced by returns-to-go R̂_t = Σ_{t'≥t} r_{t'}, giving
  τ = (R̂_1, s_1, a_1, R̂_2, s_2, a_2, …) (Eq. 2, Sec. 3, p. 4). At test time the user gives a target
  return and the start state; after each step the target is decremented by the reward received
  (Sec. 3, p. 5; Alg. 1, p. 5).
- **Tokenisation and embedding**: the last K timesteps make 3K tokens (one per modality). Each
  modality gets its own linear embedding followed by LayerNorm; visual states (Atari) go through a
  convolutional encoder instead. A learned per-*timestep* embedding (not per-token) is added to all
  three tokens of a step (Sec. 3, p. 5; Alg. 1, p. 5).
- **Backbone**: a causally masked GPT (Sec. 2.2, p. 4); a linear head on the hidden state of the s_t
  token predicts a_t (Sec. 3, p. 5; Fig. 1, p. 1).
- **Objective**: purely supervised on offline trajectories — cross-entropy for discrete actions, MSE for
  continuous ones, averaged over timesteps; predicting states or returns was not found to help
  (Sec. 3, p. 5). No value function, no policy gradient, no Bellman backup.
- **Atari setup**: one model per game, trained on 1% of the DQN-replay dataset (500k of 50M
  transitions), 3 seeds, gamer-normalised scores, frame stacking, K = 30 (K = 50 for Pong)
  (Sec. 4.1, p. 6; Sec. 5.3, p. 9). Observation encoder: the DQN conv net plus a linear projection
  (App. A.1, p. 18).
- **Gym (D4RL)**: HalfCheetah, Hopper, Walker on medium, medium-replay and medium-expert data, plus a
  self-generated sparse-reward 2D Reacher (Sec. 4.2, p. 7).
- **Key-to-Door**: grid world in three phases (key room, empty distractor room, door room); binary
  reward only when the door is reached after picking up the key; trained on random-action
  trajectories, the whole episode as context (Sec. 5.4, p. 9).
- **Percentile BC (%BC)**: a diagnostic baseline that behaviour-clones only the top X% of timesteps
  by episode return (Sec. 5.1, p. 8).

## Assumptions

- A1: The task is an MDP with a per-step reward, and learning is offline from a fixed dataset of trajectories from arbitrary policies, with no further environment interaction during training (Sec. 2.1, p. 4)
- A2: Each timestep is tokenised as (return-to-go, state, action), where return-to-go is the undiscounted sum of future rewards (Sec. 3, Eq. 2, p. 4)
- A3: At test time a target return is given as a prompt and decremented by each observed reward, so the agent must observe rewards online during evaluation (Sec. 3, pp. 4-5, Alg. 1, p. 5)
- A4: Return targets are hand-picked per environment: about 1× or 5× the dataset's maximum return on Atari, and expert performance on Gym except 50% for HalfCheetah (App. A.1, p. 18, Table 8, p. 18, App. A.2.1, p. 18, Table 9, p. 19)
- A5: Conditioning is on a single deterministic return; the authors leave conditioning on return distributions for stochastic settings to future work (Sec. 7, p. 12)
- A6: The model sees the last K timesteps (3K tokens) with a learned per-timestep embedding: K = 30 on Atari (50 for Pong), 20 on HalfCheetah, Hopper and Walker, 5 on Reacher, and the whole episode on Key-to-Door (Sec. 3, p. 5, Table 8, p. 18, Table 9, p. 19, Sec. 5.4, p. 9)
- A7: Each modality has its own embedding: a linear layer on vector states, returns and actions, and a convolutional encoder (the DQN encoder on Atari) on visual states (Sec. 3, p. 5, App. A.1, p. 18)
- A8: Only actions are predicted, with cross-entropy for discrete and mean-squared error for continuous actions (Sec. 3, p. 5)
- A9: The Atari data is 1% of the DQN-replay dataset (500k of the 50M transitions seen by an online DQN agent) on Breakout, Qbert, Pong and Seaquest (Sec. 4.1, p. 6)
- A10: Atari observations use frame stacking (Sec. 5.3, p. 9)
- A11: Frame size, frame skip, sticky actions and the action set aren't stated, and presumably follow the DQN-replay dataset's preprocessing (reader's inference, Sec. 4.1, p. 6)
- A12: Atari scores are normalised so that 0 is a random policy and 100 a professional gamer, following Hafner et al. (Sec. 4.1, p. 6, Table 11, p. 20)
- A13: Baseline numbers come from several sources: CQL, REM and QR-DQN on Atari, and CQL on D4RL, are taken from their papers; BEAR, BRAC-v and AWR from the D4RL paper; Atari BC is DT without return conditioning, and Gym BC is an MLP run by the authors (Sec. 4.1, p. 6, Sec. 4.2, p. 7, App. A.2.2, p. 19)
- A14: The Gym data is D4RL medium, medium-replay and medium-expert for HalfCheetah, Hopper and Walker, plus a sparse-reward, goal-conditioned 2D Reacher with datasets generated the same way; scores are normalised so that 100 is an expert policy (Sec. 4.2, p. 7)
- A15: The Gym models have 3 layers, 1 head and 128 dimensions and are trained for 10⁵ AdamW steps; the Atari models are minGPT with 6 layers, 8 heads and 128 dimensions, trained for 5 epochs with Tanh instead of LayerNorm (App. A.1, p. 18, Table 8, p. 18, App. A.2.1, pp. 18-19, Table 9, p. 19)
- A16: The policy is limited by the behaviours in the data: the authors read DT as honing in on a high-return subset, and returns beyond the dataset's best occur only on some tasks (Sec. 5.1, p. 8, Sec. 5.2, p. 9)
- A17: Choosing the best %BC subset requires environment rollouts (Sec. 5.1, p. 8)
- A18: Key-to-Door is a three-phase grid task with a binary reward for reaching the door after picking up the key, trained on random-action trajectories (Sec. 5.4, p. 9)
- A19: The critic variant also predicts return tokens, including the first one (Sec. 5.5, p. 10)
- A20: The delayed-reward variant gives the trajectory's whole return at the final step and zero elsewhere (Sec. 5.6, p. 10)
- A21: The graph task uses one random 20-node graph (edge sparsity 0.1), 1,000 random walks of 10 steps, and a return prior favouring short paths, combined with the model's own return predictions; the other experiments don't use such a prior (App. A.3, pp. 19-20)
- A22: A premise of the method: self-attention can assign credit directly over long horizons, without bootstrapping or discounting (Sec. 1, p. 3)
- A23: Each environment gets its own model, trained from scratch on that environment's data; no cross-game or cross-domain training or transfer is tested (reader's inference, Sec. 4, p. 6, App. A, p. 18)
- A24: Reusing a model in another game or body would need a new state encoder and action head, since both depend on the environment's spaces, and a re-chosen return target, since return scales differ per game (reader's inference, Sec. 3, p. 5, Table 8, p. 18)

## Claims

- C1: On 1% DQN-replay Atari (gamer-normalised, 3 seeds), DT scores Breakout 267.5 ± 97.5, Qbert 15.4 ± 11.4, Pong 106.1 ± 8.1, Seaquest 2.5 ± 0.4, against CQL 211.1 / 104.2 / 111.9 / 1.7; DT is comparable to CQL on 3 of 4 games and, in the paper's words, outperforms or matches REM, QR-DQN and BC on all 4, though BC's Qbert mean is higher (17.3) [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A15, A23] (Table 1, Sec. 4.1, pp. 6-7)
- C2: On D4RL (3 seeds), DT averages 74.7 without Reacher vs CQL 63.9, BEAR 48.2, BRAC-v 36.9, AWR 34.3, BC 46.4, and 69.2 vs CQL 54.2 (BC 47.7) over all settings; it has the highest score on a majority of the tasks [A1, A2, A3, A4, A5, A6, A7, A8, A13, A14, A15, A23] (Table 2, Sec. 4.2, p. 7)
- C3: DT is not uniformly best on D4RL: e.g. CQL is higher on Hopper medium-expert (111.0 vs 107.6) and BRAC-v on HalfCheetah medium and medium-replay (46.3, 47.7 vs 42.6, 36.6) [A1, A2, A3, A4, A5, A6, A7, A8, A13, A14, A15, A23] (Table 2, p. 7)
- C4: With plentiful data (D4RL), percentile BC trained on the best subset can match or beat offline-RL methods, and DT (average 56.1) is competitive with the best %BC (10%BC average 56.7) over the medium and medium-replay sets; choosing the subset needs environment rollouts [A1, A2, A3, A4, A6, A7, A8, A13, A14, A15, A16, A17] (Table 3, Sec. 5.1, p. 8)
- C5: In the low-data Atari regime %BC is weak and DT beats every %BC variant on Breakout, Pong and Seaquest (e.g. Pong 106.1 vs at most 85.2), though 25%BC and 100%BC are slightly higher on Qbert (16.0, 17.3 vs 15.4) [A1, A2, A3, A4, A6, A7, A8, A9, A10, A11, A12, A15, A16, A17] (Table 4, Sec. 5.1, p. 8)
- C6: Desired target returns and achieved returns are highly correlated on every task; on Pong, HalfCheetah and Walker the achieved returns almost match the target, and on some Atari games such as Seaquest prompting above the dataset's best return yields higher returns (some extrapolation) [A2, A3, A5, A6, A7, A8, A9, A10, A11, A12, A14, A15, A16] (Fig. 4, Sec. 5.2, pp. 8-9)
- C7: Context length matters on Atari even with frame stacking: K = 1 drops Breakout from 267.5 to 73.9, Pong from 106.1 to 2.5 and Seaquest from 2.5 to 0.6 (Qbert 15.1 vs 13.6) [A1, A2, A3, A4, A6, A7, A8, A9, A10, A11, A12, A15] (Table 5, Sec. 5.3, p. 9)
- C8: On Key-to-Door, trained on random trajectories, DT succeeds 71.8% (1K trajectories) and 94.6% (10K) vs CQL 13.1% / 13.3%, BC 1.4% / 1.6%, %BC 69.9% / 95.1% and random 3.1% [A1, A2, A3, A5, A6, A8, A13, A17, A18, A22] (Table 6, Sec. 5.4, pp. 9-10)
- C9: A DT variant that also predicts return tokens updates its predicted success probability at key events, and its attention concentrates on the key pickup and door reaching [A2, A6, A18, A19, A22] (Fig. 5, Sec. 5.5, p. 10)
- C10: With delayed (sparse) rewards — all reward given at the last step — DT on Hopper scores 107.3 / 60.7 / 78.5 (medium-expert / medium / medium-replay), close to its dense scores 107.6 / 67.6 / 82.7, while CQL collapses to 9.0 / 5.2 / 2.0 [A1, A2, A3, A4, A5, A6, A7, A8, A13, A14, A15, A20, A22] (Table 7, Sec. 5.6, p. 10)
- C11: On Atari, an MLP with K = 1 as in prior work was found worse than the transformer; the remark is footnote 2, attached to the description of the BC baseline, so it may concern BC rather than DT [A9, A10, A11, A13] (Sec. 4.1, p. 6)
- C12: On a 20-node random graph, trained only on 1,000 random walks of 10 steps and sampled with a return prior favouring short paths, DT's generated paths closely match shortest-path lengths; 15.8% of generated paths stitch sub-optimal segments not present in the data [A1, A2, A21] (Fig. 6, App. A.3, pp. 19-20)
- C13: In Atari, Tanh instead of LayerNorm after the modality embeddings made no significant difference [A7, A9, A10, A11, A15] (App. A.1, p. 18)

## Implementation notes

- Atari model: 6 layers, 8 heads, 128-dim embedding, dropout 0.1, Adam lr 6e-4 with betas (0.9, 0.95), grad clip 1.0, weight decay 0.1, linear warmup then cosine decay, max 5 epochs, batch 128 (512 Pong) (source: Table 8, App. A.1, p. 18).
- Atari encoder: conv channels 32/64/64, kernels 8×8/4×4/3×3, strides 4/2/1, ReLU in the encoder and GeLU elsewhere (source: Table 8, p. 18).
- Atari target returns: 90 Breakout, 2500 Qbert, 20 Pong, 1450 Seaquest (≈1× or ≈5× the dataset max) (source: Table 8, p. 18).
- Gym model: Huggingface Transformers GPT, 3 layers, 1 head, 128-dim, ReLU, batch 64, K = 20 (5 for Reacher), dropout 0.1, lr 1e-4, grad clip 0.25, weight decay 1e-4, linear warmup over the first 1e5 steps, 1e5 AdamW steps; targets 6000 HalfCheetah, 3600 Hopper, 5000 Walker, 50 Reacher (source: Table 9, App. A.2.1, pp. 18-19).
- The Atari timestep embedding is per episode timestep, shared by the three tokens of a step; how timesteps beyond the training maximum are handled is not stated (source: Sec. 3, p. 5; handling beyond the training maximum: assumption, to check in the authors' code).
- Atari frame stack depth and preprocessing are not given in the paper; the DQN-replay convention is 4 stacked 84×84 grayscale frames (source: assumption, following the DQN-replay data the paper uses).
- Code: built on minGPT (Atari) and Huggingface Transformers (Gym); the paper links https://sites.google.com/berkeley.edu/decision-transformer (source: p. 1 footnote 1, App. A, p. 18).

## Limitations / open questions

- Evaluated only offline; online use is discussed as potential, not tested (Sec. 5.8, p. 11).
- Needs a target return chosen by hand per environment (Table 8, Table 9, pp. 18-19); "more possibilities exist for principled return-to-go conditioning" (App. A.1, p. 18).
- Deterministic returns are conditioned on; the authors suggest conditioning on return distributions for stochastic settings as future work (Sec. 7, p. 12).
- Types of errors transformers make in MDPs, and dataset biases or adversarial reward design, are flagged as underexplored (Sec. 7, p. 12).
- The paper reports Qbert DT as 15.4 ± 11.4 in Tables 1 and 4 but 15.1 ± 11.4 in Table 5 (pp. 7-9); tables label ± as "variance" while Sec. 4.1 says standard deviation (p. 6).
- Open question (survey lens): every Atari result is one model per game with a game-specific encoder and action head; DT says nothing about multi-game training, cross-game transfer or shared action spaces.
- Open question (survey lens): DT is supervised on logged trajectories; whether its return-conditioned policy helps as initialisation for PPO-style online fine-tuning is not studied here.

## Related

- uses [[vaswani2017attention]]: the causally masked GPT variant of the Transformer is the whole policy (Sec. 2.2, p. 4).
- related [[schulman2017proximal]]: DT replaces policy-gradient/actor-critic training with a supervised sequence loss, the opposite design choice to the survey's PPO plan.
- related [[serrano2024crossdomain]]: survey of cross-domain RL transfer; DT itself trains per environment and does not transfer.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
