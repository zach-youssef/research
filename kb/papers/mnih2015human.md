---
id: mnih2015human
title: Human-level control through deep reinforcement learning
authors: [Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, Demis Hassabis]
year: 2015
venue: Nature 2015 journal article (read as the DeepMind-hosted PDF, 13 pages with Methods and Extended Data)
url: https://www.nature.com/articles/nature14236
arxiv:
doi: 10.1038/nature14236
pdf_sha256: 3ae04e292f2882483ad539e24cd532743165afce51d45df4e31bd1e3ed9753bb
tags: [atari, dqn, reinforcement-learning, evaluation-protocol, human-normalized-score, benchmark, experience-replay]
status: read
read: 2026-11-27
---

# Human-level control through deep reinforcement learning

<!-- Read from https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf, 13 PDF pages: main text pp. 1-4, references p. 5, pp. 6-7, Extended Data Figs 1-2 pp. 8-9, Extended Data Tables 1-4 pp. 10-13. All read. Tables 1-4 (pp. 10-13) have no text layer beyond their captions; their numbers were read from the page images. Supplementary Discussion and videos are online-only and not read.  Table N in references means Extended Data Table N (pages 10-13). -->

## TL;DR

DQN: one convolutional Q-network, trained per game with experience replay and a periodically frozen
target network from raw pixels, reaches over 75% of a professional human tester's score on 29 of 49
Atari 2600 games with one set of hyperparameters. This note focuses on its evaluation protocol
(preprocessing, frame skip, no-ops, episode cap, human and random reference scores).

## Problem

Deep networks had not been combined stably with RL from high-dimensional input: sequential
correlations, a data distribution that shifts with the policy, and a moving target make Q-learning
with a nonlinear approximator unstable (p. 1). Earlier Atari agents used hand-designed features or
game-specific prior knowledge (pp. 2-3).

## Method

- **Network** (Fig. 1, p. 2, p. 6): input 84x84x4; conv 32 filters 8x8 stride 4, conv 64 4x4
  stride 2, conv 64 3x3 stride 1, FC 512 ReLU, linear output with one unit per valid action (4 to 18
  depending on the game); all hidden layers ReLU. One forward pass gives all Q-values.
- **Preprocessing phi** (p. 6): from 210x160 128-colour frames, take the per-pixel maximum of
  the current and previous frame (removes flicker from sprites drawn on alternate frames), extract the
  Y (luminance) channel, rescale to 84x84; stack the m = 4 most recent such frames.
- **Frame skipping** (p. 6): the agent selects an action every k-th frame and the last
  action is repeated on skipped frames; k = 4 for all games (Table 1, p. 10: action
  repeat 4, update frequency 4).
- **Rewards and episodes** (p. 6): during training every positive reward becomes +1 and every
  negative one -1 (0 unchanged); in games with lives, the lives counter marks the end of an episode
  in training; the error term of the update is clipped to [-1, 1] (p. 7).
- **Training** (pp. 6-7, Alg. 1, p. 7, Table 1, p. 10): replay memory of the
  last 1M frames, uniform minibatches of 32, target network copied every 10,000 updates, RMSProp
  (lr 0.00025, momentum 0.95, min squared gradient 0.01), gamma 0.99, epsilon annealed linearly 1.0 to
  0.1 over the first 1M frames then 0.1, replay starts after 50,000 frames of uniform random play, 50M
  training frames per game (about 38 days of game experience).
- **Evaluation procedure** (p. 6): epsilon-greedy with epsilon = 0.05, 30 episodes of up to 5
  minutes each, each starting with a random number (up to 30) of "do nothing" actions
  (no-op max 30, Table 1, p. 10).
- **Human and random baselines** (p. 6): see Claims C8-C10 and Implementation notes.
- **Normalisation** (Fig. 3, p. 3, Table 2, p. 11): 100 x (agent -
  random) / (human - random), so human = 100% and random = 0%.

## Assumptions

- A1: The testbed is the Atari 2600 emulator (ALE), 210x160 colour frames at 60 Hz, 49 games with published comparison results; the agent sees only pixels, the game score, the number of legal actions and the lives counter (p. 2, p. 6).
- A2: The preprocessing is the max-over-two-frames, luminance, 84x84, 4-frame stack of `phi`, with frame skip k = 4 (p. 6).
- A3: Training rewards are clipped to {-1, 0, +1}; reported scores are the unclipped game scores (p. 6, Table 2, p. 11).
- A4: One agent is trained per game, with the same architecture, algorithm and hyperparameters in all games; the hyperparameters were chosen by an informal search on Pong, Breakout, Seaquest, Space Invaders and Beam Rider (p. 6, Table 1, p. 10).
- A5: Evaluation uses epsilon = 0.05, up to 30 random no-op actions at the start, 30 episodes of at most 5 minutes (p. 6, Table 1, p. 10).
- A6: The emulator is the ALE as it was in 2015: deterministic apart from the random starts, with no sticky actions, which the paper never mentions (reader's inference, p. 6).
- A7: The human reference is one professional games tester under the controlled conditions of C9; the random reference is the policy of C10 (p. 6).

## Claims

- C1: DQN reaches more than 75% of the professional human tester's normalised score on 29 of the 49 games and exceeds the best prior learning methods on 43 of them, with the same network, algorithm and hyperparameters in every game [A1, A2, A3, A4, A5, A7] (pp. 2-3, Fig. 3, p. 3, Table 2, p. 11)
- C2: Training clips rewards to +1 for positive, -1 for negative, 0 unchanged, so the same learning rate suits all games; the authors note this can hurt a game where reward magnitudes matter [A1, A2] (p. 6)
- C3: Frames are preprocessed as the pixel-wise maximum of the current and previous frame, then luminance, then 84x84, with a stack of the last m = 4 (m = 3 or 5 are said to work too) [A1, A2] (p. 6)
- C4: Frame skipping with k = 4 is used in all games: the agent picks an action on every 4th frame and repeats it; the action repeat and the update frequency are both 4 [A1, A2] (p. 6, Table 1, p. 10)
- C5: The action set is the game-specific set of valid actions, 4 to 18 per game, taken from the emulator, so the network output size varies by game [A1] (Fig. 1, p. 2, p. 6)
- C6: During training the lives counter ends an episode when a life is lost in games that have lives; the paper does not say whether evaluation episodes do the same [A1] (p. 6)
- C7: Evaluation plays 30 episodes of at most 5 minutes each with epsilon = 0.05, each episode starting after a random number of no-ops of up to 30 ("no-op max 30"); the reported error bars are the standard deviation over those 30 episodes, one trained agent per game [A1, A4, A5] (p. 6, Table 1, p. 10, Fig. 3, p. 3)
- C8: The normalised score is 100 x (DQN - random) / (human - random), with unclipped game scores; Table 2 lists random and human scores for the 49 games [A1, A7] (Fig. 3, p. 3, Table 2, p. 11)
- C9: The human score is that of a professional games tester using the same emulator, with no pausing, saving or reloading, audio off, taken as the average reward over about 20 episodes of at most 5 minutes after about 2 hours of practice per game [A1, A7] (p. 6)
- C10: The random baseline picks a uniformly random action at 10 Hz (every sixth frame) and repeats it in between; picking an action every frame (60 Hz) had a minimal effect, changing the normalised DQN score by more than 5% in only six games (Boxing, Breakout, Crazy Climber, Demon Attack, Krull, Robotank), and DQN still beat the expert human in all of them by a large margin [A1, A7] (p. 6)
- C11: Training uses 50M frames per game, a 1M-frame replay memory, epsilon annealed from 1.0 to 0.1 over the first million frames, and 50,000 frames of random play before learning [A1, A2, A3, A4] (p. 6, Table 1, p. 10)
- C12: Ablations on five games (Breakout, Enduro, River Raid, Seaquest, Space Invaders; 10M frames, three learning rates, best validation score over evaluations every 250,000 frames on 135,000 validation frames) show replay and the separate target network each help, and DQN beats a linear approximator (e.g. Breakout 316.8 vs 3.00) [A1, A2, A3] (Table 3, p. 12, Table 4, p. 13)
- C13: Those ablation episodes were not truncated at 5 minutes, which gave higher Enduro scores than the main results, and trained 10M instead of 50M frames [A1] (Table 3, p. 12, Table 4, p. 13)

## Implementation notes

- Per-game random and human reference scores (random/human) for the 49 games, read from Table 2, p. 11 (image; text layer absent): Alien 227.8/6875; Amidar 5.8/1676; Assault 222.4/1496; Asterix 210/8503; Asteroids 719.1/13157; Atlantis 12850/29028; Bank Heist 14.2/734.4; Battle Zone 2360/37800; Beam Rider 363.9/5775; Bowling 23.1/154.8; Boxing 0.1/4.3; Breakout 1.7/31.8; Centipede 2091/11963; Chopper Command 811/9882; Crazy Climber 10781/35411; Demon Attack 152.1/3401; Double Dunk -18.6/-15.5; Enduro 0/309.6; Fishing Derby -91.7/5.5; Freeway 0/29.6; Frostbite 65.2/4335; Gopher 257.6/2321; Gravitar 173/2672; H.E.R.O. 1027/25763; Ice Hockey -11.2/0.9; James Bond 29/406.7; Kangaroo 52/3035; Krull 1598/2395; Kung-Fu Master 258.5/22736; Montezuma's Revenge 0/4367; Ms. Pac-Man 307.3/15693; Name This Game 2292/4076; Pong -20.7/9.3; Private Eye 24.9/69571; Q*bert 163.9/13455; River Raid 1339/13513; Road Runner 11.5/7845; Robotank 2.2/11.9; Seaquest 68.4/20182; Space Invaders 148/1652; Star Gunner 664/10250; Tennis -23.8/-8.9; Time Pilot 3568/5925; Tutankham 11.4/167.6; Up and Down 533.4/9082; Venture 0/1188; Video Pinball 16257/17298; Wizard of Wor 563.5/4757; Zaxxon 32.5/9173 (source: Table 2, p. 11). These human scores differ from the "Average Human" column of [[badia2020agent57]] (App. H.4, p. 27) beyond rounding (by more than 0.5) on 33 of the 49 shared games (for example Alien 6875 here, 7127.7 there; Boxing 4.3 and 12.1; Assault 1496 and 742; Asteroids 13157 and 47388.7; Seaquest 20182 and 42054.7). The random scores agree up to rounding (for example Centipede 2091 and 2090.9) except Video Pinball: 16257 here and 0.00 there, which changes that game's normalisation denominator from about 1041 to 17667.9 (source: comparison of the two tables by script).
- Episode cap in emulator frames: 5 minutes at 60 Hz is 18,000 frames, which is 4,500 agent steps at frame skip 4 (source: assumption, arithmetic on p. 6; [[machado2018revisiting]] also uses 18,000 frames).
- Whether evaluation ends an episode at a life loss is not stated; the training rule is C6 (source: assumption: not stated for evaluation).
- Source code: https://sites.google.com/a/deepmind.com/dqn, "for non-commercial uses only" (source: p. 6). The link was not opened.
- The action set is "all legal actions of the game" (4-18) rather than the fixed 18 of the later multi-game papers; [[lee2022multigame]] uses 18 for all games (source: Fig. 1, p. 2, p. 6).
- DQN scores in Table 2 are the average of 30 evaluation episodes of the 50M-frame agent; the paper does not say whether the checkpoint is the last or the best (source: assumption: not stated).

## Limitations / open questions

- The paper says games needing temporally extended planning, such as Montezuma's Revenge, remain a major challenge for DQN (p. 4).
- Uniform replay sampling does not weight transitions by importance; prioritised sweeping is suggested as future work (p. 7, p. 4).
- Hyperparameters came from an informal search on five games, not a systematic one, owing to cost (p. 6, Table 1, p. 10).
- Open question: which emulator settings (sticky actions, frame-skip randomisation, life-loss termination at evaluation) were on is not stated beyond what Methods says; later work found these matter ([[machado2018revisiting]]).
- Open question: the origin of the human scores of Agent57's table (a different set) is not given in either paper.

## Related

- related [[machado2018revisiting]]: recommends sticky actions and other protocol choices that differ from this paper's no-op-start protocol.
- related [[badia2020agent57]]: its 57-game table has different human scores and a 30-minute cap; it cites this paper for DQN.
- related [[lee2022multigame]]: multi-game DT; its DQN-Replay data and HNS protocol follow this lineage.
- related [[cheng2025jointly]]: JOWA reports HNS and DQN-normalised scores on Atari.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
