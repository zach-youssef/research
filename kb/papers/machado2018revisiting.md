---
id: machado2018revisiting
title: "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents"
authors: [Marlos C. Machado, Marc G. Bellemare, Erik Talvitie, Joel Veness, Matthew Hausknecht, Michael Bowling]
year: 2018
venue: JAIR 2018 journal article (read as arXiv v2)
url: https://arxiv.org/abs/1709.06009
arxiv: 1709.06009
doi: 10.1613/jair.5699
pdf_sha256: 43f1f5a743900642cf78eba901a256f18d1cdf3804f5842efcc077c495510c11
tags: [atari, ale, benchmark, evaluation-protocol, sticky-actions, game-modes, difficulties, stochasticity, transfer-learning, reinforcement-learning, dqn]
status: read
read: 2026-11-24
---

# Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents

## TL;DR

Survey and protocol paper for the ALE: recommends one evaluation protocol (score of the last 100
training episodes, sticky actions with probability 0.25, no lives signal) and benchmarks DQN and
Sarsa(lambda)+Blob-PROST on 60 games. It adds game modes/difficulties ("flavors") as a transfer testbed.

## Problem

ALE papers use many incompatible protocols (episode termination, hyperparameter selection,
training-data accounting, performance summaries, stochasticity), so reported results are hard to
compare (Sec. 3, pp. 5-7). The ALE is also deterministic, so agents can score by memorising action
sequences instead of learning closed-loop policies (Sec. 5, p. 10). The paper proposes best practices,
a stochasticity mechanism, a benchmark, and a list of open problems (Sec. 1, p. 2).

## Method

- **Setting.** The agent sees a 210x160 image and/or the 1024-bit RAM state; reward is the score
  change; episodes end at game over; common preprocessing is frame skipping, colour averaging or
  frame pooling, and frame stacking (Sec. 2.1, pp. 3-4). The emulator is deterministic (p. 3).
- **Recommended protocol.** (1) Terminate episodes only on the game-over signal, not on loss of a life
  (Sec. 3.1, p. 6). (2) Use a train/test split of games for hyperparameter selection (p. 6).
  (3) Measure training in frames, count skipped frames, and train through the end of the episode
  that crosses the budget (pp. 6-7). (4) Summarise learning by the average score of the last k
  training episodes, with no separate evaluation phase, reported at several points in training
  (Sec. 4.2, pp. 9-10); the benchmark uses k = 100 at 10, 50, 100 and 200M frames. (5) Use sticky
  actions (below).
- **Why not other summaries.** Evaluation after learning hides sample efficiency; evaluating the best
  policy ignores stability and is a biased estimate; area under the curve cannot see plummeting
  performance (Sec. 4.1, p. 8, Fig. 1, p. 9; Centipede is the example).
- **The Brute.** A deterministic-environment exploit: a partial history tree with maximum-likelihood
  transitions and a bottom-up lower bound on q*, followed epsilon-greedily with epsilon =
  min(0.05/log(n(h)+1), 1) (Sec. 5.1, p. 11, App. A, pp. 31-34). It is used to show what determinism allows.
- **Sticky actions.** With probability varsigma the emulator repeats the previous executed action
  instead of the agent's new one (Sec. 5.2, p. 12). Desiderata: minimally non-Markov, no change to
  difficulty for robust agents, easy to implement outside the emulator. Unlike random delays the
  agent can change its mind each step (Q*bert example, p. 13, Fig. 3). It composes with frame skipping.
- **Alternatives discussed.** Initial no-ops, random human start states, uniform action noise, random
  frame skips, asynchronous environments, each with pros and cons (Sec. 5.3, pp. 14-15).
- **Modes and difficulties.** Many Atari games have a default mode and difficulty selectable by console
  switches that change dynamics or add actions (e.g. Freeway, Fig. 8, p. 23). The new ALE exposes every
  single-player mode and difficulty; a mode-difficulty pair is a "flavor" (Sec. 7.4.1, pp. 22-23).
  Table 10 (App. D, p. 39) gives the number of modes and difficulties per game (single-player only).
- **Open problems.** Representation learning, planning/model learning, exploration, transfer, off-policy
  learning (Sec. 7, pp. 17-23).

## Assumptions

- A1: Agents learn from scratch per game, from the screen and/or RAM, with the full 18-action set, frame skip 5, and no game-specific information such as the lives signal (App. B, p. 34, Table 5, p. 34).
- A2: Benchmark numbers are for sticky actions with varsigma = 0.25 (0.0 only for the determinism comparison in Table 1), episodes capped at 18,000 frames (5 minutes) (Table 5, p. 34).
- A3: Benchmark scores are means over 24 trials for Sarsa(lambda)+Blob-PROST and the Brute, but only 5 for DQN, so significance tests on DQN have low power (Sec. 6.1, p. 16, p. 17).
- A4: Atari 2600 games share a screen/RAM interface, so the ALE's 60 games stand in for "general competency" (Sec. 1, pp. 1-2, Sec. 7.1, p. 18).
- A5: Modes and difficulties of one game share most of the underlying state representation, hence the authors expect negative transfer to be less likely there (reader's inference, Sec. 7.4.1, p. 22).

## Claims

- C1: Under the deterministic ALE with 50M frames the Brute beats DQN and Sarsa(lambda)+Blob-PROST on Asterix (6,909 vs 3,501 and 4,173) and Space Invaders (1,432 vs 673 and 723), but is far worse on Freeway (1.1 vs about 32) [A1, A3] (Sec. 5.1.1, p. 12, Table 1).
- C2: With sticky actions (varsigma = 0.25) the Brute collapses on all five training games (e.g. Asterix 6,909 to 308, Freeway 1.1 to 0.0), while DQN and Sarsa(lambda)+Blob-PROST lose little (e.g. Asterix DQN 3,501 to 3,123, Blob-PROST 4,173 to 3,411; Seaquest DQN 1,397 to 1,431) [A1, A2, A3] (Sec. 5.2.1, p. 13, Table 2, p. 14).
- C3: Sticky actions reduced DQN and Sarsa(lambda)+Blob-PROST performance, relative to earlier published results, in only three games: Breakout, Gopher and Pong [A1, A2, A3] (Sec. 6.2, p. 17).
- C4: Blob-PROST's score keeps rising over 200M frames in most games: scores at 200M are lower than at 100M in about 10% of games, significantly so in 3 (Carnival, Centipede, Wizard of Wor); only 22 of 60 games show a significant gain from 100M to 200M; its best score is at 200M in 50 of 60 games (Welch's t-test, p < 0.05, n = 24) [A1, A2, A3] (Sec. 6.2, pp. 16-17).
- C5: DQN's learning is more variable and gains less from extra data: its highest score is at 200M in 35 of 60 games, and 200M is significantly better than 100M in 18 of 60; no significant decrease was found, but with n = 5 the test may lack power [A1, A2, A3] (Sec. 6.2, pp. 16-17).
- C6: Performance drops appear algorithm-dependent rather than game-dependent: Centipede is the only game where both DQN and Blob-PROST plummet (Blob-PROST 15,600 at 100M to 1,189 at 200M); both make little progress on Asteroids, Pitfall! and Tennis [A1, A2, A3] (Sec. 6.2, p. 17, Table 8, p. 37).
- C7: At 10M frames DQN beats Blob-PROST in under 20% of games and the two are comparable at 100M frames, which the authors attribute to DQN's sample cost of learning a representation [A1, A2, A3] (Sec. 7.1, p. 18).
- C8: Episode termination on loss of a life, as in DQN training, is not recommended: it uses game-specific information and was found by prior work to be possibly detrimental [A1] (Sec. 3.1, p. 6).
- C9: Best-policy and after-learning evaluations, and area under the learning curve, can misrank agents: in Centipede Blob-PROST peaks high then plummets, DQN peaks lower but is steadier [A1] (Sec. 4.1, pp. 8-9, Fig. 1, p. 9).
- C10: Rewards are very sparse or negative under random exploration in Montezuma's Revenge, Private Eye, Pitfall! and Tennis: DQN sees non-zero rewards in Montezuma's Revenge only occasionally and does not learn to collect them consistently [A1, A2] (Sec. 7.1, p. 18, Fig. 4, p. 19, Sec. 7.3, p. 21).
- C11: Learned models are accurate only for a limited horizon in the ALE (a quad-tree model's rollout of Freeway drifts by t+30 to t+60) and the authors see no clear demonstration of planning with a learned model, crediting only Chiappa et al. (2017) with "the ability to plan with the internal state" [A1] (Sec. 7.2, pp. 19-20, Fig. 5, p. 20).
- C12: Prior transfer work in the ALE covers about a dozen mostly similar games and often needs an expert network; the authors point to game modes and difficulties as an easier transfer setup, because flavors of one game likely share state representations, so negative transfer should be less likely. The paper reports no transfer experiment [A4, A5] (Sec. 7.4, p. 22, Sec. 7.4.1, pp. 22-23).
- C13: The ALE offers numbers of single-player modes and difficulties per game ranging from 1 and 1 (e.g. Assault, Asterix) to, e.g., 33 modes for Asteroids and 16 for Double Dunk and Space Invaders; Freeway has 8 modes and 2 difficulties [A4] (Table 10, p. 39).

## Implementation notes

- ALE parameters: 18 actions always, max episode length 18,000 frames, frame skip 5, no lives signal, varsigma 0.25, 100 final episodes, milestones 10/50/100/200M, 24 trials (Blob-PROST, Brute) and 5 (DQN) (source: Table 5, p. 34, App. B.1).
- DQN: original Mnih et al. 2015 settings except frame skip 5 and final epsilon 0.01 (no evaluation phase): RMSProp lr 0.00025, momentum 0.95, gamma 0.99, replay 1M, batch 32, update frequency 4, 4-frame history, frame pooling (source: App. B.3, p. 35, Table 6, p. 35).
- Sarsa(lambda)+Blob-PROST: alpha 0.5 (divided by the largest number of active features seen), lambda 0.9, gamma 0.99, epsilon 0.01, eligibility threshold 0.01, colour averaging, 4x7 pixel grid, neighbourhood 6, 128 colours (source: App. B.4, pp. 35-36, Table 7, p. 36).
- Brute: gamma 1.0, epsilon = 0.005/log(n_i + 2) in B.2 versus min(0.05/log(n(h)+1), 1.0) in A.3; the two formulas differ in the paper (source: p. 34, p. 35).
- Raw learning-curve data was published with the paper; the URL is on p. 10, footnote 3 (source: p. 10). Code for the ALE modes/difficulties is the ALE GitHub repository (source: Acknowledgements, p. 24); the API for selecting a flavor is not described in the paper.
- Which modes make a game "single player" and how each differs is not described per game; only counts (Table 10) (source: p. 39).

## Limitations / open questions

- Table 8 has columns still marked "*" to be filled in (Journey Escape at 200M); the authors note the incomplete columns (Appendix C, p. 36, p. 37).
- DQN significance tests use n = 5 and may be underpowered (Sec. 6.2, p. 17).
- Score alone is not necessarily an indicator of AI progress: agents can loop on small rewards (Sec. 2.1, p. 3).
- The paper states that negative transfer across flavors is "less likely" without evidence; [[farebrother2020generalization]] tests the question for DQN (reader's note).
- Open (reader): the paper gives only counts of modes per game, not which perturbations each mode makes; for within-game generalisation experiments one has to inspect the modes.
- Open (reader): the protocol is single-game, per-game training; it says nothing about one policy across games (A1).

## Related

- related [[farebrother2020generalization]]: uses the ALE flavors this paper introduces as held-out variants for DQN generalisation.
- related [[delfosse2025intelligence]]: another within-game perturbation benchmark for Atari (HackAtari), with sharp drops for agents trained on the originals.
- related [[delfosse2024ocatari]]: object-centric Atari environments built on the same ALE.
- related [[lee2022multigame]]: a multi-game Atari agent, the cross-game setting this paper's single-game protocol does not cover.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
