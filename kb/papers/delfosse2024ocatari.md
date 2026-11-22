---
id: delfosse2024ocatari
title: "OCAtari: Object-Centric Atari 2600 Reinforcement Learning Environments"
authors: [Quentin Delfosse, Jannis Blüml, Bjarne Gregori, Sebastian Sztwiertnia, Kristian Kersting]
year: 2024
venue: Reinforcement Learning Journal 2024 article (read as arXiv v2)
url: https://arxiv.org/abs/2306.08649
arxiv: 2306.08649
doi:
pdf_sha256: 711af4e99f641b51ac42881b9a54d84f4c78588570d2939be9f0221202d17075
tags: [reinforcement-learning, object-centric, atari, benchmark, state-representation, ram-extraction, object-detection, dataset]
status: read
read: 2026-11-22
---

# OCAtari: Object-Centric Atari 2600 Reinforcement Learning Environments

## TL;DR

OCAtari wraps the ALE and returns per-frame object lists (class, x, y, w, h) decoded from RAM by
hand-built per-game decoders (REM, 42-44 games) or color filters (VEM, 44-46 games); REM agrees
with VEM at about 90% mean F1, runs about 50x faster, and PPO on its positions matches pixel PPO.

## Problem

Object-centric RL lacks an Atari benchmark: object discovery methods are trained and judged on
ALE frames without ground-truth objects, and raw RAM (used as an RL state, or AtariARI's labelled
RAM bytes) is entangled, partly irrelevant, and often not enough to describe objects: offsets,
grid anchors, categorical bytes (Sec. 1, p. 2, Sec. 2.2, p. 4, Sec. 3.3, p. 7). Publications on
object-centric RL in Atari are negligible next to Atari RL as a whole (Fig. 1, p. 2).

## Method

- **Objects.** Each object has a category (e.g. "Pacman"), position x, y, size w, h, RGB, and
  optionally orientation or a value (oxygen bar, score). HUD objects (score, lives) are flagged
  and can be filtered by a boolean option. Static background (MsPacman's maze, Pong's walls) is
  not an object; OCAtari therefore returns both frames and object lists (Sec. 2.1-2.2, pp. 3-4).
  Per-game object lists are in App. G (pp. 18-47) and App. I (Table 52, p. 49).
- **VEM (vision extraction).** Hand-written per-game color filters plus priors on object
  positions (Pong: paddles and scores share an orange, but paddles always lie between the white
  borders). It only sees what is drawn: no blinking, overlapping or invisible objects, and no
  tracking, since each frame is extracted independently (Sec. 2.2, p. 4).
- **REM (RAM extraction).** Per-game decoders from the 128 RAM bytes to the same object list
  as VEM. Found by having human, random or DQN agents play while VEM tracks objects, correlating
  object properties with each byte, and editing bytes to watch the rendered effect; it computes
  positions from anchors plus offsets and reads presence flags (Sec. 2.2, p. 4, Sec. 3.3, p. 7).
  RAM state persists, so REM tracks objects through blinking and occlusion (Sec. 2.2, p. 4).
- **Package.** An ALE-compatible Gymnasium wrapper (step, render, seed, plus get_ram and
  set_ram), with an optional buffer of the last 4 84x84 grey frames (Sec. 2.2, p. 4).
- **ODA dataset.** Frames (210x160 RGB) with both REM and VEM object lists and HUD separated,
  30% of games from a random agent and 70% from a DQN agent; the RAM is edited to start agents
  in later levels they can't reach (App. A, p. 15, App. F, p. 17).
- **Evaluation.** REM's object lists are scored against VEM's as ground truth over 500 frames
  from random, DQN and (where available) C51 agents: per-category precision, recall and F1
  (correct if within 5 px of the center) and IoU (Sec. 3.1, p. 5). PPO (stable-baselines3,
  2x64 tanh MLP, 3 seeds, 20M frames, no tuning) is trained on x, y of each REM object in the
  last two frames, on 8 games (Sec. 3, p. 5, Sec. 3.2, p. 6, App. E, p. 17).

## Assumptions

- A1: The game is an Atari 2600 title whose full state lives in 128 bytes of emulator RAM that REM can read and write (Sec. 2.2, p. 4).
- A2: Each supported game has a hand-built extractor: REM decoders found by correlating RAM bytes with VEM detections and editing bytes, VEM filters tuned to the game's fixed colors and object positions (Sec. 2.2, p. 4).
- A3: Objects are drawn with a small set of fixed per-object RGB colors, so color filtering separates them (Sec. 2.2, p. 4).
- A4: VEM's detections are treated as ground truth when scoring REM; VEM misses blinking, occluded and overlapping objects and picks up particle effects (Sec. 3.1, p. 5, App. H, p. 48).
- A5: The object vocabulary per game is fixed in advance by the authors (which objects, which categories); static background such as mazes is excluded (Sec. 2.1, p. 3, Sec. 5, p. 8, App. H, p. 48).
- A6: Evaluation frames come from fixed agents (random, DQN, C51) playing 500 frames per game, so states those agents don't reach are scored only via ODA's RAM-edited starts (Sec. 3.1, p. 5, App. A, p. 15).
- A7: The object-centric RL agents see only object x, y positions in the last two frames, not frames (Sec. 3.2, p. 6).

## Claims

- C1: REM detects the objects of its supported games with high agreement with VEM: mean precision, recall, F1 and IoU of 90.5, 93.5, 91.3, 91.0 with a random agent and 88.9, 92.3, 89.7, 90.7 with a DQN agent, over the games of Table 6 [A1, A2, A4, A6] (Table 1, p. 5, Table 6, p. 18).
- C2: Per game, agreement varies widely: DQN-agent F1 is 63.4 on BattleZone, 64.1 on Gopher, 65.0 on Pacman, 67.1 on Alien and 67.3 on DemonAttack, but 99-100 on Bowling, Breakout and MontezumaRevenge [A1, A2, A4, A6] (Table 6, p. 18).
- C3: Lower precision mostly reflects objects REM reports that VEM doesn't see (blinking ghosts in MsPacman, objects not rendered for frames after a death, invisible objects in DemonAttack and YarsRevenge) and particle effects VEM picks up, rather than REM errors; rendering jitter lowers IoU for small objects such as the ball in Pong and Tennis [A4, A5] (Sec. 3.1, pp. 5-6, App. H, p. 48).
- C4: On 7 games, REM's F1 (93.9 average) exceeds the object-discovery methods SPACE (31.5) and SPOC (77.3), e.g. Tennis 99.3 vs 3.6 and 40.2; SPOC is higher only on MsPacman (90.5 vs 87.4) [A1, A2, A4] (Table 2, p. 6).
- C5: REM is on average about 50x faster than VEM: the time for 10^4 steps is around 1 (log-scale axis, unit unstated) per game with REM vs about 12-200 with VEM, over 23 games [A1, A2, A3] (Sec. 3.1, p. 6, Fig. 7, p. 50).
- C6: PPO trained on REM object positions (3 seeds, 20M frames) performs on par with or better than pixel-based PPO on Asterix, Bowling, Boxing, Freeway, Kangaroo, Pong, Seaquest and Tennis, by human-normalized score (the Fig. 4 caption adds "and humans") [A1, A2, A5, A7] (Sec. 3.2, p. 6, Fig. 4, p. 6).
- C7: OCAtari supports 46 games with VEM and 44 with REM at the time of writing, 28 more than AtariARI, which lacks information needed to play 8 of its games (6 of them in Table 3) (e.g. Seaquest's oxygen bar and divers, SpaceInvaders' shields) [A2] (Sec. 2.2, p. 4, Sec. 3.3, p. 7, Table 3, p. 7, Table 53, p. 50).
- C8: Generating ODA's dataset for one game takes about 3 minutes on an AMD Ryzen 7 with 64 GB RAM and no GPU [A1, A2] (App. E, p. 17).

## Implementation notes

- Game coverage is stated inconsistently: the text says 46 VEM and 44 REM games (p. 4), Table 3 sums to 44 VEM and 42(43) REM (p. 7); the repository README is the stated authority for the current list (source: App. D, p. 16).
- The REM-vs-VEM detection threshold is 5 px from the object center; IoU follows Rezatofighi et al. 2019 (source: Sec. 3.1, p. 5).
- How SPACE's and SPOC's F1 in Table 2 were scored (against VEM, or their own labels) isn't stated (source: assumption that the numbers come from the SPOC paper, Delfosse et al. 2022, which the text cites for them).
- Environments: Gymnasium Atari v4 and v5, frameskip 5, repeat-action probability 0.25, Deterministic and NoFrameskip variants where needed; seeds 0 and 42 (source: App. D, p. 16, App. E, p. 17).
- PPO: 8 actors, minibatch 32x8, horizon 128, 3 epochs, Adam 2.5e-4 x alpha, gamma 0.99, GAE 0.95, clip 0.1 x alpha, c1 1, c2 0.01, "alpha linearly increases from 0 to 1" per the paper (source: Table 4, p. 17), likely meaning annealed 1 to 0 as in PPO (source: assumption, from schulman2017proximal); the pixel baseline uses stable-baselines3 CnnPolicy with 4 stacked frames (source: App. E, p. 17).
- The speed metric's unit (seconds per 10^4 steps) and hardware aren't given for Fig. 7 (source: assumption that it ran on the App. E CPU machine).
- Code: github.com/k4ntz/OC_Atari, MIT licence; docs at oc-atari.readthedocs.io (source: p. 1, p. 4, p. 5).

## Limitations / open questions

- Static elements such as the mazes in Pacman and MsPacman are hard-coded in the ROM, not RAM, so REM can't extract them, or only partly; combining frames and object lists to recover them is left open (Sec. 5, p. 8).
- REM's F1 and IoU are measured against VEM, itself an imperfect hand-written detector, so the numbers measure agreement, not accuracy against a human label (Sec. 3.1, p. 5).
- REM deliberately ignores particle effects and keeps unrendered objects, so it can report objects a pixel-based detector should not see (App. H, p. 48).
- Reader's observation: in Fig. 4 the Bowling and Seaquest bars of both agents sit below the dashed human line (100 HNS), so the caption's "and humans" doesn't hold per game (Fig. 4, p. 6).
- Open question (reader): the per-game cost of building a REM or VEM extractor (human reverse-engineering effort) isn't quantified; only dataset generation time is.
- Open question (reader): no game outside the 2600 library is covered; RAM-based extraction doesn't carry to games whose state isn't small, fixed and readable.

## Related

- uses [[schulman2017proximal]]: the object-centric and pixel agents are trained with PPO.
- related [[derstroff2026relative]]: ORCA/ORBiT reads its object sets from OCAtari's RAM extraction.
- related [[kirillov2023segment]]: a learned, game-agnostic segmenter, the pixel-based alternative to hand-built extractors.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
