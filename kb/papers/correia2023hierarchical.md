---
id: correia2023hierarchical
title: Hierarchical Decision Transformer
authors: [André Correia, Luís A. Alexandre]
year: 2023
venue: IROS 2023 (arXiv v1 read)
url: https://arxiv.org/abs/2209.10447
arxiv: 2209.10447
doi: 10.1109/IROS55552.2023.10342230
pdf_sha256: e00406edade1f042ac6a5a5b469d163a87ea54d0ffc8a8520459c10c4f8ac3e2
tags: [hierarchical-rl, decision-transformer, offline-rl, imitation-learning, sub-goals, sparse-rewards, long-horizon, d4rl, robomimic]
status: read
read: 2026-11-25
---

# Hierarchical Decision Transformer

## TL;DR

Replaces the Decision Transformer's return-to-go with sub-goal states: a high-level GPT predicts the next
sub-goal from past states, and a low-level GPT is conditioned on it. Sub-goals are picked offline from the
reward trace. Beats DT and BC on 8 of 10 Gym/D4RL/RoboMimic tasks, with no desired-return input.

## Problem

The Decision Transformer needs the user to specify a desired return, a hard-to-guess value
that the paper finds matters, and it struggles with long episodes and sparse rewards, where the
returns-to-go barely change (Sec. 1, p. 1; Sec. 3.3, p. 4). The paper wants a sequence model
trained from sub-optimal demonstrations that is guided by key states instead (Sec. 1, p. 1).

## Method

- Two Decision Transformers (GPT architecture): a high-level mechanism pi_phi and a low-level
  controller pi_theta (Sec. 4.1, Fig. 1, p. 2, p. 4).
- Sub-goal selection, done once on the dataset before training: for state s_i the candidate
  sub-goals are later states s_j of the same trajectory, scored by the accumulated reward from
  i+1 to j divided by the distance j-i, W(s_j) = sum_{k=i+1..j} r_k / (j-i); the highest-scoring
  state becomes the sub-goal (Sec. 4.2, p. 5). The division stops the choice from always being
  the end of the trajectory (Sec. 4.2, p. 5).
- Low-level controller: a goal-conditioned DT receiving sequences of states, sub-goals and
  actions, trained with the behaviour-cloning loss ||a - pi_theta(s, a, sg)|| (Sec. 4.3, p. 5).
- High-level mechanism: a state-conditioned DT receiving states and past sub-goals and
  predicting the next sub-goal state with the MSE loss ||sg - pi_phi(s, sg)|| (Sec. 4.4, p. 5).
- Both are trained simultaneously on sampled sub-trajectories (Alg. 1, p. 4); at test time the
  history of states, actions and the predicted sub-goals is kept as the input (Sec. 4.3, p. 5).
- Setup: 100k training iterations, batch 64, learning rate 1e-4, sequence length 20 tokens;
  validation on 100 episodes every 1000 iterations, with the table reporting the highest average return
  seen over training (Sec. 5, p. 5).
- Time split: Fig. 1 and Alg. 1 show a sub-goal at every timestep, so the high level runs at the same rate as the low level (reader's reading of Fig. 1, p. 2, Alg. 1, p. 4). Hierarchy here is of goals over states, not of time scales.

## Assumptions

- A1: Demonstration trajectories include per-step rewards from which sub-goals can be scored (Sec. 4.2, p. 5).
- A2: The sub-goal is a state (a vector from the same observation space) taken from a later point in the same trajectory (Sec. 4.2, p. 5).
- A3: Environments are low-dimensional state-vector control tasks (OpenAI Gym, D4RL, RoboMimic), not pixel tasks (Sec. 5, p. 5).
- A4: Demonstration data sets are the benchmarks' provided ones (medium, human, complete), sub-optimal and diverse (Sec. 5, Sec. 1, p. 5, p. 1).
- A5: Returns are reported as the maximum over validation checkpoints of 100 evaluation episodes (Sec. 5, p. 5).
- A6: Per-step rewards are available offline to score sub-goals even when the environment's reward is sparse (reader's inference, Sec. 4.2, p. 5).

## Claims

- C1: A DT without its desired-returns input fails: for example Half-Cheetah medium 47 (versus 5095 for DT with the returns), Hopper 300 (versus 2303), Kitchen 1.0 (versus 2.4), Walker 149 (versus 3619) [A3, A4, A5] (Sec. 5.1, Table 1, p. 6).
- C2: DT's result depends on the chosen desired return and degrades with an arbitrarily high one (10000), for example Hopper 2036 against 2557 at the middle value [A3, A4, A5] (Sec. 5.1, Table 1, p. 6).
- C3: Adding desired returns as an extra input to HDT does not help and lowers returns, by more than half on Hopper (the paper calls the drop slight): Half-Cheetah 5004 and 5002 versus 5205, Hopper 1213 and 1198 versus 3071, Kitchen 2.1 and 1.4 versus 2.6, Walker 3645 and 3516 versus 3879 [A1, A2, A3, A4, A5] (Sec. 5.2, Table 2, p. 6).
- C4: HDT has the highest maximum return on 8 of 10 tasks against DT and an MLP BC baseline, with large gaps on Hammer (3115 versus DT -68 and BC 50), Hopper (3071 versus 2557 and 1969), Maze 2D (188 versus 68 and 37) and Pen (3897 versus 3606 and 867); on Walker 2D the lead is small (3879 versus 3711 and 3723) [A1, A2, A3, A4, A5] (Sec. 5.3, Table 3, p. 7).
- C5: HDT does not win on Door (338 versus DT 459, BC 363) and Relocate (35, tied with DT, versus BC 232) [A1, A2, A3, A4, A5] (Sec. 5.3, Table 3, p. 7).
- C6: Maze 2D is a sparse-reward task where about 90% of transitions have no reward, and HDT "vastly" outperforms the baselines there; the gap between transformers and BC is larger on such long-horizon and sparse tasks [A1, A2, A3, A4, A5] (Sec. 5.3, Table 3, p. 7).
- C7: The sub-goal selection needs no task knowledge, so the method needs no user-specified returns [A1, A2, A6] (Sec. 4.2, Sec. 6, p. 5, p. 7).

## Implementation notes

- Evaluation: 100 episodes every 1000 iterations of 100k, with the best average reported; the number of seeds is not stated (source: Sec. 5, p. 5).
- DT desired returns: the maximum dataset return, half of it, and 10000 (source: Sec. 5, Sec. 5.1, p. 5, p. 6).
- GPT sizes (layers, heads, embedding), the sub-goal state representation for RoboMimic, and the "ablation across sub-goal selection methods" promised in the contributions are not in the text (source: assumption; Sec. 1 p. 2 lists ablations on selection methods and sub-goal types, but only Tables 1-3 appear in the PDF).
- BC baseline is an MLP with two fully connected layers (source: Sec. 5.3, p. 7).
- Code is "to be disclosed" (source: Sec. 1, p. 2); no repository known to the reader.

## Limitations / open questions

- Future work: more tasks and different architectures for the two models (Sec. 6, p. 7).
- The ablations on sub-goal selection methods and sub-goal types listed in the contributions are not shown in the 9-page PDF read (Sec. 1, p. 2).
- Reported numbers are the best over validation checkpoints, which flatters all methods and may favour noisier ones (reader's reading of Sec. 5, p. 5).
- Open question (reader): no pixel observations or Atari; sub-goals are full state vectors. For a frames-to-gamepad shmup policy, a sub-goal would be a latent (as in [[hafner2022director]]) rather than a raw state.
- Open question (reader): the sub-goal is a future state chosen by accumulated reward over distance. For a shmup, would scoring by score gain pick weapon upgrades or shop visits, which pay off late, over dodging, which pays off immediately? The 1/(j-i) factor favours near sub-goals.
- Open question (reader): the high level predicts one sub-goal per step from 20 tokens of context; a long-horizon strategist would need its own slow rate and memory tokens ([[bulatov2022recurrent]], [[cherepanov2026recurrent]]) instead.
- Open question (reader): a score or reward trace can be recovered offline from game RAM, so reward-based sub-goal selection could be used to label gameplay data without the agent ever reading RAM; the paper does not test it.

## Related

- extends [[chen2021decision]]: replaces return-to-go with predicted sub-goal states and adds a second DT that predicts them.
- related [[cherepanov2026recurrent]]: another Decision Transformer extension for long horizons, using memory instead of sub-goals.
- related [[lee2022multigame]]: multi-game Decision Transformer on Atari, flat.
- related [[hafner2022director]]: latent sub-goals learned by RL, with a time-scale split of K=8.
- related [[lynch2019play]]: sub-goals as future states of a window, trained from unlabeled play.
- related [[vezhnevets2017feudal]]: a Manager-Worker split trained by RL.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
