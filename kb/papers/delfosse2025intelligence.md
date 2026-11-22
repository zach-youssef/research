---
id: delfosse2025intelligence
title: "Deep Reinforcement Learning Agents are not even close to Human Intelligence"
authors: [Quentin Delfosse, Jannis Blüml, Fabian Tatai, Théo Vincent, Bjarne Gregori, Elisabeth Dillies, Jan Peters, Constantin Rothkopf, Kristian Kersting]
year: 2025
venue: arXiv preprint
url: https://arxiv.org/abs/2505.21731
arxiv: 2505.21731
doi:
pdf_sha256: 173da6bf88bf7bcce1e4e6d7979696bb611dc59ed2ccaf5e3d9a72439722ba28
tags: [reinforcement-learning, hackatari, atari, benchmark, generalization, robustness, shortcut-learning, object-centric, human-study, ram-extraction]
status: read
read: 2026-11-22
---

# Deep Reinforcement Learning Agents are not even close to Human Intelligence

## TL;DR

HackAtari: 224+ RAM-edited variants of 33+ ALE games, mostly simplifications (Kangaroo without
enemies, Freeway with stopped cars, recolors). Agents trained on the originals drop sharply on
them, even IMPALA and object-centric ones, while 128 non-expert humans mostly improve.

## Problem

ALE agents are trained and evaluated on the same game, which hides shortcut learning (e.g. a Pong
agent that tracks the enemy paddle as a proxy for the ball). Earlier robustness tests made tasks
harder, so a drop could mean poor adaptation rather than a wrong policy. The paper tests agents
on variants humans find no harder, so that a drop points to a misaligned policy (Sec. 1, p. 2,
Sec. 3, p. 3). It defines "intelligence" narrowly, as adapting to structural simplifications
(footnote, p. 1).

## Method

- **Variants by RAM edits.** The ROMs' source is proprietary, so every variant reads and
  overwrites the emulator's RAM each step. Pong LazyEnemy: when the RAM cell for the ball's
  horizontal speed says the ball is moving toward the agent, the enemy's vertical position is
  overwritten with its previous value (Sec. 3, pp. 3-4, Fig. 1, p. 4). There are visual variants
  (recolored or invisible objects) and gameplay variants (object speed or presence, new gravity,
  layouts, levels). All are listed per game in App. G (pp. 38-49).
- **Genre coverage of the variant list** (App. G, pp. 38-49): maze games (MsPacman caged or
  edible ghosts, `maze_man`, levels 0-2; Amidar; Alien `last_egg`), paddle games (Pong drifts,
  `hidden_enemy` which removes the enemy from the OCAtari object list; Breakout drift and gravity),
  shooters (SpaceInvaders shields disabled or shifted; RiverRaid `restricted_firing`, `no_fuel`;
  StarGunner static enemies), platformers (Kangaroo `no_danger`, levels; DonkeyKong `no_barrel`),
  plus Freeway, Frostbite, Seaquest, Boxing, Bowling, Tennis, Skiing and others. Each variant
  perturbs one game, not a new game.
- **Agents.** All are trained only on the original game, mostly for 200M frames (v5, frameskip 4,
  sticky actions 0.25): DQN, MDQN, C51 (12 games), i-DQN, IMPALA, PPO, and object-centric PPO
  agents: OCCAM Binary Mask and Planes (40M frames), Semantic Vector (OC-NN, OCAtari positions,
  boxes and classes over 2 frames into an MLP) and SCoBots (decision trees). Most checkpoints come
  from public repositories; deep PPO and the OCNN variants were trained in-house with CleanRL
  (Sec. 4, p. 5); App. B attributes the Semantic Vector and SCoBots agents to their authors (p. 18).
- **Evaluation.** 30 episodes (3 seeds x 10), epsilon 0.001, up to 30 no-ops, minimal action set
  (App. D, p. 21, Table 3, p. 21). Metrics: expert-HNS (human and random scores from Badia et
  al. 2020a; random scores also measured in-house per variant), aggregated by IQM with 95%
  bootstrap CIs via rliable, plus a per-game performance change (Eq. 2, p. 17) that subtracts
  each condition's random score (Sec. 4, p. 5, App. A, pp. 16-17).
- **Human study.** Prolific, non-expert participants, one game each: 10-15 min of free training
  on the original, 15 min on the original (Eval1), then 15 min on one variant (Eval2), with a
  performance bonus of up to £3. Fifteen game-variant pairs were assigned round-robin. Scores
  are the mean over participants of each participant's IQM (Sec. 4, pp. 5-6, App. A, p. 17,
  App. F, pp. 28-29).

## Assumptions

- A1: Variants can only change what the game keeps in its RAM; there is no access to game code (Sec. 3, p. 3).
- A2: Agents are trained only on the unmodified game and evaluated zero-shot on variants, with no fine-tuning (App. D, p. 21, App. D.1, p. 23).
- A3: A variant counts as a simplification if humans keep or raise their score; this is checked by the human study for 15 variants only, the other Fig. 3 variants are simplifications by the authors' design (Sec. 4, p. 6, Table 4, p. 22, Table 7, p. 26).
- A4: Expert-HNS uses Badia et al.'s professional-tester human scores and random scores; the performance change subtracts each condition's own random score (App. A, pp. 16-17).
- A5: Human participants are non-experts who always play the variant after 25-30 minutes on the original, in that fixed order (p. 6, App. F, p. 28).
- A6: The object-centric agents get objects from OCAtari's RAM extraction or from masks built on it, not from a learned detector (App. B, p. 18, App. B.1, p. 19).

## Claims

- C1: HackAtari has more than 224 variants over more than 33 ALE games; because variants only edit RAM, they add negligible time over the emulator, so training and testing on them is as fast as on the originals [A1] (Sec. 3, p. 5).
- C2: On 32 variants of 17 games, every agent's expert-HNS IQM drops: DQN -1.3, MDQN -1.4, C51 -1.4, i-DQN -1.0, PPO -1.2, IMPALA -0.5, Binary Mask -0.7, Planes -0.6, OC-NN -0.6, SCoBots -0.2; IMPALA stays superhuman on average on the variants, and SCoBots has the smallest drop (20%) [A2, A3, A4] (Sec. 4, p. 6, Fig. 3, p. 6).
- C3: The authors read Fig. 3 as showing that the architecture or inference paradigm affects adaptation to variants more than the RL algorithm does [A2, A4] (Sec. 4, p. 6).
- C4: On the 15 human-study variants, human scores rise drastically on 11 games and notably on 2 more; they fall slightly on BankHeist (two police cars) and notably on MsPacman (set level 1, a maze change: 996.9 to 673.6), which the authors attribute to fatigue or to overfitted safe paths [A3, A5] (pp. 6-7, Fig. 4, p. 7, Table 7, p. 26).
- C5: On the same 15 variants, every deep agent drops severely on most; even IMPALA loses more than 50% on 10 of the 15 games [A2, A3, A4] (Sec. 4, p. 7, Fig. 4, p. 7).
- C6: The object-centric agents are insensitive to visual changes such as recolors; SCoBots drops less than 30% on 9 of 15 games, but all object-centric agents drop a lot on most gameplay variants, so object-centricity helps but isn't enough [A2, A3, A6] (Sec. 4, p. 7, Fig. 5, p. 8).
- C7: Kangaroo `no_danger` (monkeys and coconuts removed): deep agents fall from 6244-14269 to 0-81 and every object-centric agent scores 0 (Semantic Vector and SCoBots already score 0 on the original), while humans rise from 460 to 2431 [A2, A3, A5, A6] (Table 5, p. 23, Table 6, p. 25, Table 7, p. 26).
- C8: Pong `lazy_enemy`: DQN 19 to -15, PPO 18 to -15, i-DQN 19 to -18 and IMPALA 9 to 6; among object-centric agents, Planes (19 to 17) and SCoBots (16 to 17) hold, while the Semantic Vector agent falls from 19 to -20 [A2, A6] (Table 5, p. 23, Table 6, p. 25).
- C9: Recoloring breaks the pixel agents but most object-centric agents hold: Breakout with a red paddle and ball takes PPO from 391 to 4 while Binary Masks (259 to 256), Class Masks (222 to 216) and Planes (371 to 390) hold, though Object Masks drops from 216 to 99; RiverRaid `game_color_change01` takes PPO from 16186 to 368 (the mask agents stay near 7600-8000) [A2, A6] (Table 5, p. 23, Table 6, p. 25).
- C10: Shifting SpaceInvaders' shields 3 px, a change the authors say humans often don't notice, takes DQN from 4173 to 475 and MDQN from 6183 to 820 (IMPALA 11514 to 8252), while humans go from 238 to 287 [A2, A3, A5] (Sec. 3, p. 5, Table 5, p. 23, Table 7, p. 26).
- C11: Freeway `stop_all_cars`: DQN, C51, i-DQN and MDQN score 0-0.4 and PPO 7.6, but IMPALA gets 35 and SCoBots rises from 30 to 41; humans rise from 17.7 to 35.8 [A2, A3, A5, A6] (Table 5, p. 23, Table 6, p. 25, Table 7, p. 26).
- C12: Evaluating 12 agent configurations on 50 game configurations took about 18,000 runs; training about 120 PPO models took 4-6 h each, 500-700 A100 GPU hours in total [A2] (App. C, p. 20).

## Implementation notes

- Code: github.com/k4ntz/HackAtari; Gymnasium wrappers that select a variant by name, RAM viewing and editing tools to build new ones, SB3 and CleanRL compatible (source: p. 2, App. E, p. 27).
- PPO: CleanRL, 10 envs, batch 1280, minibatch 320, 4 epochs, lr 2.5e-4, gamma 0.99, GAE 0.95, clip 0.1, entropy 0.01, value 0.5, max grad norm 0.5, 2e8 steps, 84x84 grey, 4 stacked frames (source: App. B.1, Table 1, p. 19).
- PPO's encoder is stated inconsistently: "IMPALA-style" in App. B.1 (p. 19), Nature-CNN on p. 5 and in Fig. 5 (p. 8) (source: those pages; which one is right is unknown).
- Who trained which agent is stated inconsistently: p. 5 says deep PPO and OCNN variants were
  trained in-house; App. B says PPO, Binary Mask and Planes were trained by the authors, Semantic
  Vector and SCoBots Vector were provided by their authors (source: p. 5, App. B, p. 18).
- Checkpoints: DQN, MDQN, C51 from Gogianu et al. 2022 (atari-agents), i-DQN from Hugging Face TheoVincent/Atari_i-QN, IMPALA from Hugging Face cleanrl, OCCAM from VanillaWhey/OCAtariWrappers (source: App. B, p. 18).
- Evaluation stack: Python 3.11, Gym 0.26.2, ale-py 0.8.1; IMPALA needed envpool, on Python 3.9 (source: App. C, p. 20).
- Human interface "HackTari": Flask plus Socket.IO, Gym with HackAtari wrappers, on AWS, per-session CSV logs; £7 base pay (source: App. F.3-F.4, p. 29).

## Limitations / open questions

- The authors name three limitations: the ALE's stylized visuals may not transfer to complex domains; agents may overfit to HackAtari's own perturbations; and variants have no standardized difficulty scale (Sec. 7, p. 9).
- Counts disagree: 128 subjects (p. 5), "134 users" (p. 6), 160 completed and 128 kept after exclusions (App. F, pp. 28-29); 32 variants in Fig. 3's caption, 30 in the text (p. 6), 50 game configurations in App. C (p. 20).
- Reader's observation: humans always play the variant after 25-30 min of the original (A5), and Fig. 12 shows many games already rising from free training to Eval1, so part of the human gain on variants may be practice (Fig. 12, p. 35).
- Reader's hypothesis: some "simplifications" may remove much of the reward; e.g. on Seaquest `disable_enemies` C51, PPO, IMPALA and every object-centric agent score 0 (DQN 268, i-DQN 1134, MDQN 1392) while humans reach 12192, so part of the drop may reflect scarcer reward rather than a misread game (Table 5, p. 23, Table 6, p. 25, Table 7, p. 26).
- Reader's observation: the OCCAM mask agents trained for 40M frames, not 200M (App. B, p. 18), so their original-game scores aren't directly comparable.
- For the synthesis: variants perturb one game. They test robustness of objects and dynamics within a game, not transfer to an unseen game of the same genre.
- Testable cheaply (reader): Pong `lazy_enemy` and `hidden_enemy` with an object-list PPO agent, comparing absolute and relative object frames ([[derstroff2026relative]] reports such a test).

## Related

- uses [[delfosse2024ocatari]]: variants are built on OCAtari's RAM knowledge, and the Semantic Vector agent and SCoBots read OCAtari objects.
- uses [[schulman2017proximal]]: the in-house pixel and object-centric agents are PPO.
- related [[derstroff2026relative]]: evaluates ORCA on HackAtari variants and reports PPO-CNN dropping only 7% on its 10 variants, a different variant set from this paper's.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
