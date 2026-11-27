---
id: badia2020agent57
title: "Agent57: Outperforming the Atari Human Benchmark"
authors: [Adrià Puigdomènech Badia, Bilal Piot, Steven Kapturowski, Pablo Sprechmann, Alex Vitvitskyi, Daniel Guo, Charles Blundell]
year: 2020
venue: ICML 2020 conference paper (read as arXiv v1, 2020-03-30)
url: https://arxiv.org/abs/2003.13350
arxiv: 2003.13350
doi:
pdf_sha256: 62de571fd053eb0238aa5aca076a8e85f5335dd4d58e1ea786e6b35726b43b2b
tags: [atari, reinforcement-learning, exploration, meta-controller, evaluation-protocol, human-normalized-score, benchmark, distributed-rl]
status: read
read: 2026-11-27
---

# Agent57: Outperforming the Atari Human Benchmark

<!-- Read from arXiv v1 (30 PDF pages): main text pp. 1-8, references pp. 9-11, appendices A-H pp. 12-30 (all read; the learning curves in App. H.5, p. 29, only glanced at). The note stresses the evaluation protocol: Table 2 (p. 23), Sec. 4 (p. 5), the 57-game score table (p. 27). The venue year follows ICML 2020; v1 was read. -->

## TL;DR

Agent57 extends NGU with separate extrinsic and intrinsic Q-networks, a sliding-window UCB bandit that picks
the exploration rate and discount per episode, and a longer backprop window; it is the first agent above the
human baseline on all 57 Atari games (capped mean HNS 100%). Its appendix gives the 57-game human and random scores
and the preprocessing used.

## Problem

No agent beat the standard human benchmark on all 57 Atari games: MuZero and R2D2 pass it on 51 and 52, and NGU
(exploration-driven) on 51, failing on different games (Sec. 1, pp. 1-3, Table 1, p. 6). Hard exploration (Montezuma's
Revenge, Pitfall!, Private Eye, Venture) and long-term credit assignment (Skiing, Solaris) are the remaining
failure modes (Sec. 1, pp. 1-2).

## Method

- **NGU base** (Sec. 2, pp. 3-4): intrinsic reward r^i = r^episodic * clip(alpha, 1, L=5) from episodic
  novelty and Random Network Distillation; N = 32 mixes of (beta_j, gamma_j) with total reward r^e + beta_j r^i;
  a distributed R2D2-style learner with 256 actors and prioritised sequence replay.
- **Parameterisation** (Sec. 3.1, p. 4, Fig. 9-10, pp. 21-22): Q(x, a, j) = h(h^-1(Q^e) + beta_j h^-1(Q^i)) with two
  separate networks (conv torso, LSTM 512, dueling head) trained with their own transformed Retrace losses on the
  same replay sequences (App. B-C, pp. 13-16).
- **Meta-controller** (Sec. 3.2, pp. 4-5, App. D, pp. 17-18): per actor, a sliding-window UCB bandit with epsilon-greedy
  arm choice picks an arm j = (beta_j, gamma_j) per episode, rewarded by the undiscounted extrinsic return; the evaluator uses
  one too and reports the greedy arm (App. E, p. 20). gamma ranges 0.99 to 0.9999 (App. G.1, p. 23).
- **Backprop window** (Sec. 4.3, p. 7): trace length 160 (replay period 80) instead of 80 (40) (App. E, p. 19).
- **Evaluation** (Sec. 4, p. 5, App. E, pp. 19-20): see claims C6-C10.

## Assumptions

- A1: Atari 2600 through ALE, all 57 games, full 18-action set, 1 stacked grayscale frame, action repeat 4, frame max-pooling over the last 3 and 4 frames, random no-op starts up to 30, no sticky actions, no zero-discount on life loss, 30-minute episode cap (App. G.2, Table 2, p. 23).
- A2: Scores are undiscounted extrinsic episode returns from a separate evaluator process that does not feed the replay buffer (Sec. 4, p. 5, App. E, pp. 19-20).
- A3: The human and random reference scores are those of the 57-game table in App. H.4; the paper does not say where the human scores come from (reader's inference, App. H.4, p. 27).
- A4: Training uses 256 actors at about 260 env steps per second each and a single GPU learner at about 5 updates per second on batches of 64 sequences of length 160, for up to about 9x10^10 frames (App. E, p. 20, Fig. 1, p. 1).
- A5: Hyperparameters are those of NGU except the meta-controller and the discount range; the bandit window and epsilon were tuned over small ranges (App. G.3-G.4, pp. 23-24, Tables 3-4).

## Claims

- C1: Agent57 is above the human baseline on all 57 games, with capped mean HNS 100.00, mean HNS 4766.25 and median 1933.49; R2D2 (bandit) has 96.93 capped mean, 54 games above human, mean 5461.66, median 2357.92; NGU 95.07, 51, 3421.80, 1359.78; R2D2 (Retrace) 94.20, 52, 3518.36, 1457.63; R2D2 94.33, 52, 4622.09, 1935.86; MuZero 89.92, 51, 5661.84, 2381.51 [A1, A2, A3, A4] (Table 1, p. 6)
- C2: Agent57's low percentiles of HNS are 5th: 116.67, 10th: 184.35, 20th: 324.78, 30th: 614.65, 40th: 1091.07, the best of the six agents at the 5th, 10th and 20th (R2D2 bandit is best at the 30th and 40th) [A1, A2, A3, A4] (Table 1, p. 6)
- C3: On the 10-game challenging set (Beam Rider, Freeway, Montezuma's Revenge, Pitfall!, Pong, Private Eye, Skiing, Solaris, Surround, Venture) each of the three changes to NGU (separate networks, meta-controller with longer trace) raises final CHNS, and all are needed for the consistent 100% [A1, A2, A4] (Sec. 4.1, p. 6, Fig. 4, p. 6)
- C4: Agent57 passes the human benchmark on 51 games within the first 5 billion frames and takes 78 billion on Skiing, where the human score is -4336.9, random -17098.1 and optimal -3272 [A1, A2, A4] (Sec. 4.1, p. 6, Fig. 3, p. 5)
- C5: A longer backprop window (160 against 80) helps both R2D2 and Agent57 on the challenging set, most on Solaris [A1, A2, A4] (Sec. 4.3, p. 7, Fig. 6, p. 7, Fig. 12, p. 26)
- C6: Preprocessing is one grayscale frame (no stacking), action repeat 4, max-pooling over frames 3 and 4, random no-ops range 30, sticky actions off, life loss does not zero the discount, full 18-action set, 30-minute maximum episode [A1] (App. G.2, Table 2, p. 23)
- C7: The final score is the maximum over training of the seed-averaged curve of the windowed mean (last 50 episodes) of the evaluator's undiscounted returns, averaged over 3 seeds, and over 6 seeds for Agent57 (MuZero's numbers come from its paper); the maximum is taken after averaging over seeds [A1, A2] (Sec. 4, p. 5)
- C8: The evaluator acts epsilon-greedily with epsilon 0.01 (Table 3: evaluation epsilon 0.01) and alternates every 5 episodes between a training mode (bandit-chosen arm) and an evaluation mode (greedy arm), reporting the evaluation mode's 5-episode average [A1, A2, A5] (App. E, pp. 19-20, App. G.3, Table 3, p. 24)
- C9: The metrics are HNS = (agent - random) / (human - random) and the capped version CHNS = max(min(HNS, 1), 0), which the authors consider a better descriptor of general performance because it stresses the games below the human level; the paper also reports mean, median, the 5th-40th percentiles and the number of games above human [A3] (Sec. 4, p. 5, Table 1, p. 6)
- C10: The paper gives average human and random scores for all 57 games (App. H.4, p. 27) and all methods' per-game scores with standard deviations over seeds (pp. 27-28); no confidence intervals are reported on aggregates [A1, A3] (App. H.4, p. 27, p. 28)
- C11: The separate-network parameterisation fixes NGU's failure when intrinsic reward scale grows, shown on a 15x15 "random coin" task after 150M frames, and on Surround, where NGU scores like random and the new parameterisation is near-optimal [A1, A2] (Sec. 4.2, pp. 6-7, Fig. 5, p. 7)
- C12: The adaptive meta-controller lets very high discounts (gamma = 0.9999) be included without hurting other games, and enhances R2D2's final CHNS by nearly 20% on the challenging set; fixing gamma = 0.9999 for all games was unstable [A1, A2, A4] (Sec. 4.4, pp. 7-8, Fig. 7, p. 8)
- C13: The choice of identity or h-transform mix for combining extrinsic and intrinsic values makes no visible difference in performance [A1, A2] (App. H.3, p. 26, Fig. 13, p. 26)
- C14: MuZero has the highest uncapped mean and median but the lowest capped mean, because it fails on games such as Venture while scoring very high on others such as Beam Rider (27469% HNS) [A1, A3] (Sec. 4.1, pp. 5-6, Table 1, p. 6)

## Implementation notes

- 57-game table of average human and random scores (human/random), from App. H.4, p. 27 (text layer present, checked): alien 7127.70/227.80; amidar 1719.50/5.80; assault 742.00/222.40; asterix 8503.30/210.00; asteroids 47388.70/719.10; atlantis 29028.10/12850.00; bank heist 753.10/14.20; battle zone 37187.50/2360.00; beam rider 16926.50/363.90; berzerk 2630.40/123.70; bowling 160.70/23.10; boxing 12.10/0.10; breakout 30.50/1.70; centipede 12017.00/2090.90; chopper command 7387.80/811.00; crazy climber 35829.40/10780.50; defender 18688.90/2874.50; demon attack 1971.00/152.10; double dunk -16.40/-18.60; enduro 860.50/0.00; fishing derby -38.70/-91.70; freeway 29.60/0.00; frostbite 4334.70/65.20; gopher 2412.50/257.60; gravitar 3351.40/173.00; hero 30826.40/1027.00; ice hockey 0.90/-11.20; jamesbond 302.80/29.00; kangaroo 3035.00/52.00; krull 2665.50/1598.00; kung fu master 22736.30/258.50; montezuma revenge 4753.30/0.00; ms pacman 6951.60/307.30; name this game 8049.00/2292.30; phoenix 7242.60/761.40; pitfall 6463.70/-229.40; pong 14.60/-20.70; private eye 69571.30/24.90; qbert 13455.00/163.90; riverraid 17118.00/1338.50; road runner 7845.00/11.50; robotank 11.90/2.20; seaquest 42054.70/68.40; skiing -4336.90/-17098.10; solaris 12326.70/1236.30; space invaders 1668.70/148.00; star gunner 10250.00/664.00; surround 6.50/-10.00; tennis -8.30/-23.80; time pilot 5229.20/3568.00; tutankham 167.60/11.40; up n down 11693.20/533.40; venture 1187.50/0.00; video pinball 17667.90/0.00; wizard of wor 4756.50/563.50; yars revenge 54576.90/3092.90; zaxxon 9173.30/32.50 (source: App. H.4, p. 27).
- These human scores differ from [[mnih2015human]]'s Extended Data Table 2 (p. 11) beyond rounding (by more than 0.5) on 33 of the 49 shared games (for example Alien 7127.70 here, 6875 there; Boxing 12.10 and 4.3; Enduro 860.50 and 309.6; Assault 742.00 and 1496). The random scores agree up to rounding except Video Pinball: 0.00 here, 16257 there. The paper doesn't say where its human scores come from (source: comparison of the two tables).
- 30 minutes is 108,000 emulator frames at 60 Hz, or 27,000 agent steps at action repeat 4 (source: assumption, arithmetic on App. G.2, Table 2, p. 23; matches the 108K frames of [[cheng2025jointly]]'s protocol per its note).
- "Frames max pooled: 3 and 4" means the pixel-wise maximum of the third and fourth frame of each repeated block (source: assumption; Table 2, p. 23 gives no more).
- Meta-controller: window 90 episodes, UCB beta 1, epsilon_UCB 0.5 in Table 3, p. 24, but Sec. 4 (p. 5) says window 160 and epsilon 0.5 for actors and window 3600 and epsilon 0.01 (the sentence doesn't name the second user; the reader reads it as the evaluator) (source: Sec. 4, p. 5 against App. G.3, Table 3, p. 24; inconsistent in the paper).
- Evaluation epsilon: Table 3, p. 24 lists 0.01 (C8), but App. E, p. 20 says the evaluator "acts in the same way as the actors", whose epsilon_l is 0.4^(1+8l/(L-1)), which is above 0.01 for about half of the 256 actors; the paper is inconsistent and C8 follows Table 3 (source: App. E, pp. 19-20 against Table 3, p. 24).
- Observation size and reward handling: Table 2, p. 23 gives no resolution and Sec. 4, p. 5 defers to NGU's "standard preprocessing of Atari frames", so the frame size is not stated here (84x84 is the usual value, source: assumption). No reward clipping is mentioned; Table 3, p. 24 lists the value transform sign(x)(sqrt(|x|+1)-1)+0.001x ("R2D2 reward transformation") (source: Table 3, p. 24).
- Per-actor exploration epsilon_l = 0.4^(1 + 8 l / (L-1)) over L = 256 actors; the target network copies every 1500 learner steps; Adam lr 1e-4 (source: App. E, pp. 19-20, Table 3, p. 24).
- N = 32 mixes: beta_j = 0 for j = 0, 0.3 for j = N-1, else 0.3 sigmoid(10 (2j-(N-2))/(N-2)); gamma_j from 0.9999 down to 0.99 (source: App. G.1, p. 23).
- Code is not released with the paper; the paper gives videos at a project site (source: App. H.6, p. 30).

## Limitations / open questions

- The authors say 100% CHNS does not mark the end of Atari: data efficiency (it needs tens of billions of frames) and general performance remain open, and Skiing needs better credit assignment (Sec. 5, p. 8).
- The agent uses a great amount of computation (Sec. 5, p. 8).
- Open question: the human reference source is not stated, and differs from Mnih et al.'s; harness presets must pick one table.
- Open question: whether life loss ends an evaluation episode is not stated; "zero discount on life loss: false" only concerns the discount (App. G.2, Table 2, p. 23).
- Open question: "30 min" is stated without saying whether it counts emulator frames before or after the action repeat.
- Open question: the maximum over training of the windowed mean is an optimistic final-score rule (reader's own remark).

## Related

- uses [[mnih2015human]]: cites it for DQN and the human-level claim; this paper's 57-game human scores differ from its 49-game table.
- related [[machado2018revisiting]]: recommends sticky actions and other settings that this paper's Table 2 turns off.
- related [[schrittwieser2020mastering]]: MuZero, the strongest baseline in Table 1; its numbers are taken from that paper.
- related [[lee2022multigame]]: HNS and IQM on a 41-game subset of this 57-game suite.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
