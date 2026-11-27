---
id: pleines2025memorygym
title: "Memory Gym: Towards Endless Tasks to Benchmark Memory Capabilities of Agents"
authors: [Marco Pleines, Matthias Pallasch, Frank Zimmer, Mike Preuss]
year: 2025
venue: JMLR 2025 journal article
url: https://arxiv.org/abs/2309.17207
arxiv: 2309.17207
doi:
pdf_sha256: d416d608230bf8f827e061c65a15a0ea62957ad3affbc22bdadf35108c28401d
tags: [reinforcement-learning, benchmark, memory, partial-observability, endless-tasks, transformer-xl, gru, ppo, curriculum]
status: read
read: 2026-11-25
---

# Memory Gym: Towards Endless Tasks to Benchmark Memory Capabilities of Agents

<!-- Read from arXiv v6 (2025-01-21), the JMLR 26 (2025) version, 40 PDF pages: main text pp. 1-25, appendices A-F pp. 26-34, references pp. 35-40 (all read). Extends the ICLR 2023 Memory Gym paper, which was not read. -->

## TL;DR

Turns Memory Gym's three 2D pixel tasks into endless ones that grow harder as the agent succeeds, and
adds an open-source PPO + Transformer-XL baseline; TrXL beats GRU on finite tasks only with an
observation-reconstruction loss, but GRU beats TrXL by large margins on every endless task.

## Problem

Memory benchmarks end in terminal states, which caps the memory they demand; once agents solve them,
they are compared on sample efficiency rather than on how much they can remember and for how long
(Sec. 1.1, p. 2, Sec. 2.2, p. 5). Transformer memory baselines for RL are also rarely open source
(Sec. 1.2, p. 3).

## Method

- **Environments** (Sec. 3.2, pp. 7-10): 84×84 RGB observations, multi-discrete actions,
  procedurally generated levels.
  - *Endless Mortar Mayhem* (Fig. 1, pp. 7-8): show one command, execute it, show the next, then
    execute the whole list so far; each new command is shown only once.
  - *Endless Mystery Path* (Fig. 2, pp. 8-9): an invisible path generated left to right, the agent
    fixed horizontally with past tiles rendered behind it; falling off restarts it from the origin.
    A 20-step budget per tile and early termination keep bad episodes short; +0.1 per new tile.
  - *Endless Searing Spotlights* (Fig. 3, pp. 9-10): a dark arena with spotlights that cost health
    (10 HP); the agent must track its hidden position. Each coin (+0.25) is visible for 6 frames and
    must be collected within 160 steps; spotlights spawn at a steady rate.
- **Automatic curriculum** (Sec. 3.1, p. 6): difficulty grows within an episode as the agent
  succeeds, like "I packed my bag", so no task scheduler is needed.
- **Baseline** (Sec. 4, pp. 10-14, Fig. 4-5): PPO with an Atari CNN encoder, a memory encoder
  (TrXL or GRU), policy and value heads, an optional observation-reconstruction head (transposed
  CNN, BCE loss) and an optional ground-truth estimation head (MSE) used as a diagnostic. Loss Eq. 6.
- **TrXL as episodic memory** (Sec. 4.3, pp. 12-14): the per-layer inputs of each step are cached
  in an episodic memory; each layer attends over a sliding window of length L, so the maximum
  context is L_max = N × (L − 1) + 1 for N layers (Eq. 7); no gradient flows into the cache.
  Absolute positional encoding over the whole episode (2048 for endless tasks), pre-LN and
  identity-map reordering after GTrXL; GTrXL gating is supported but was not better.

## Assumptions

- A1: PPO with GAE: 32 workers × 512 steps = 16384 samples per batch, 3 epochs, 8 minibatches, γ 0.995, λ 0.95, clip 0.1, no advantage normalization; hyperparameters were tuned with optuna, mostly on the finite tasks, and are not claimed optimal (Table 4, p. 28, App. B, pp. 27-29).
- A2: GRU: one layer, hidden size 512, training sequence length 512 (or the maximum episode length). TrXL: 3 layers, dimension 384, 4 heads, window 256 (or max), absolute positional encoding (Table 4, p. 28, App. B, p. 27).
- A3: Each experiment uses 5 seeds; each data point averages 50 episode seeds played 3 times; aggregate plots pool 15 runs across the three tasks (Sec. 5.2, p. 15).
- A4: Rewards are +0.1 per executed command or new path tile and +0.25 per coin in the endless tasks, 0 for failure (Table 3, p. 27); Endless Mystery Path's reward only initially appears dense, because after a fall the agent must retrace its steps to earn more, so rewards get sparser the longer it takes to catch up (Sec. 3.2.2, p. 9).
- A5: Only GRU and TrXL are tuned; MRA, HCAM and GTrXL implementations were not available, HELM was too slow, and the GTrXL and LSTM results are untuned (Sec. 5.1, pp. 14-15, App. D, p. 31).
- A6: Finite tasks train for about 400-500M steps and endless tasks for about 800M (reader's inference, Fig. 6-7, pp. 16-18).
- A7: Memory Gym's own setting: 84×84 RGB observations, multi-discrete actions, procedurally generated levels, and the default reset parameters (Sec. 3.2, p. 7, Table 2, p. 26, Table 3, p. 27).
- A8: The Table 1 ablation starts from the "Old" hyperparameters (Table 4), replaces one setting at a time with the current one, and runs finite Searing Spotlights with full observability, episode length 512 and health 10; 5 repetitions, with the data taken directly from training (top data point over the last 100 episodes), not the standard protocol (Sec. 5.7, p. 23, Table 1, p. 24).

## Claims

- C1: In finite Mortar Mayhem only TrXL with observation reconstruction completes all 10 commands; GRU reaches 4.2-6.2 commands and TrXL without reconstruction 1.3-5 [A1, A2, A3, A6, A7] (Fig. 6, pp. 15-16)
- C2: Aggregated over the finite tasks, TrXL with reconstruction completes 99% of the tasks on average and 56% without it; GRU reaches 83%, and 82% with reconstruction but more sample-efficiently [A1, A2, A3, A6, A7] (Fig. 6, p. 16, Sec. 5.3, p. 17)
- C3: In finite Mystery Path all agents are effective and TrXL is more sample-efficient, but the two are about equal in wall time because TrXL is slower [A1, A2, A3, A7] (Sec. 5.3, p. 17, Fig. 12, App. E, pp. 32-33)
- C4: In Endless Mortar Mayhem GRU executes 84-120 commands after about 300M steps (best run 140; best mean of 5 runs 115 ± 15) while both TrXL variants average 17 [A1, A2, A3, A6, A7] (Fig. 7, pp. 17-18, Sec. 5.4, p. 17)
- C5: In Endless Mystery Path the best TrXL run visits 21 tiles and the best GRU 29; final means are 26 (GRU) and 20 (TrXL) [A1, A2, A3, A6, A7] (Fig. 7, p. 18, Sec. 5.4, p. 17)
- C6: In Endless Searing Spotlights (both with reconstruction) TrXL collects at most 7.5 coins and GRU's best is 29.8; final means are 23.9 (GRU) and 6.7 (TrXL) [A1, A2, A3, A6, A7] (Fig. 7, p. 18, Sec. 5.4, pp. 17-19)
- C7: Normalized by GRU's maximum mean score per task, GRU peaks at 0.98 and TrXL at 0.39 over the endless tasks, and the standard deviations stop overlapping after about 250M steps [A1, A2, A3, A6, A7] (Fig. 7, p. 18, Sec. 5.4, p. 19)
- C8: Endless tasks reveal more memory than the finite ones: episodes last 274 steps for 10 finite Mortar Mayhem commands but over 3000 (GRU) and 400 (TrXL) in the endless version; about 100 vs over 600 and about 400 in Mystery Path; under 50 vs 691 and 240 in Searing Spotlights [A1, A2, A3, A6, A7] (Sec. 5.5, p. 19)
- C9: TrXL's weakness is not capacity: varying layers (2-4), dimension (256-512) and window (256-512) does not close the gap, a GRU cut to 2.7M parameters still wins clearly against TrXL's 2.8M, and TrXL's 766-step maximum context does not limit any endless task [A1, A2, A3, A7] (Fig. 8, Sec. 5.6.1, pp. 20-21)
- C10: Extra learning signal helps TrXL on Endless Mortar Mayhem: a ground-truth head (next target position) lifts it from 16 to up to 36 commands (Sec. 5.6.2) and a higher final learning rate to 25; augmenting the first layer's query with an absolute positional encoding gives only 18, testing the hypothesis that the query otherwise lacks temporal information, which the authors say only moderately improves the agent (Sec. 5.6.3); all combined reach at most 47, still below GRU [A1, A2, A3, A7] (Fig. 8, p. 20, Sec. 5.6.2-5.6.3, p. 21)
- C11: TrXL with absolute positional encodings of length 1024 or 2048 completes only 1-2 commands in finite Mortar Mayhem (2 runs each); on the endless task, learned or relative encodings (each combined with the adjusted learning rate) bring no notable gain over the adjusted-learning-rate run alone (Fig. 9), with relative-encoding runs ranging from 16 to 56 commands [A1, A2, A7] (Sec. 5.6.5, pp. 22-23, Fig. 9, p. 23)
- C12: The authors hypothesize, without testing it, that cached hidden states from episodes longer than the 512-step rollout go stale over PPO epochs and hurt TrXL [A1, A7] (Sec. 5.6.4, p. 22)
- C13: Recurrent agents are not vulnerable to spotlight perturbations as earlier assumed: with full observability, turning off advantage normalization gives GRU a 100% success rate, and normalized advantages produce gradient norms over 12 times larger [A8] (Table 1, p. 24, Sec. 5.7, pp. 23-24)
- C14: Memory-less PPO, 4- or 16-frame stacking and a time-step input fail where commands or cues are hidden, confirming the tasks need memory; only GRU and TrXL are effective on Mortar Mayhem Grid [A1, A7] (Fig. 10, App. C, pp. 29-30)
- C15: Untuned GTrXL and LSTM usually underperform on the finite tasks, which suggests out-of-the-box use needs hyperparameter tuning (Sec. 5.1, p. 15): Mortar Mayhem 2.1-2.8 (GTrXL) and 2.2 (LSTM) commands; in Mystery Path LSTM reaches 90%, above GTrXL with bias 0 (88%) and bias 2 (86%), the latter challenging Parisotto et al.'s assertion that a larger bias helps; in Searing Spotlights LSTM with reconstruction is only slightly less efficient than GRU while GTrXL lags TrXL with reconstruction. The authors avoid conclusive statements because both baselines are untuned [A5, A7] (Sec. 5.1, p. 15, Fig. 11, App. D, pp. 31-32)
- C16: GRU is also cheaper: about 57 h per Endless Mortar Mayhem run vs about 77 h for compiled TrXL on one A100 with 32 CPU cores; PyTorch compilation cut TrXL's time by about 30% [A1, A2, A7] (Table 5, App. E, p. 32)
- C17: The environments run at about 5,800-14,000 steps per second with a constant action, slower than only Procgen among related benchmarks, measured on one machine (AMD Ryzen 7 2700X, 1000 episodes) [] (Table 7, App. F, p. 34)

## Implementation notes

- Code, data and website: marcometer.github.io/jmlr_2024.github.io/; the TrXL baseline is contributed to CleanRL (source: p. 1, Sec. 1.2, p. 3, Sec. 5.2, p. 15).
- Learning rate 2.75e-4 → 1e-5 and entropy 1e-4 → 1e-6, linear over the first 10,000 PPO updates (163,840,000 steps); value coefficient 0.5, reconstruction coefficient 0.1, max gradient norm 0.25, AdamW, Xavier init for TrXL (source: Table 4, p. 28, App. B, p. 27).
- Endless task parameters: Mortar Mayhem commands shown for 3 steps with a 1-step delay, executed in 6 steps with an 18-step delay; Mystery Path stamina 20; Searing Spotlights 3 initial spotlights, spawn interval 50, coin shown for 6 steps (source: Table 3, p. 27).
- Compute: a 25,000 GPU-hour budget; batch size chosen for 32 CPU cores and a 40 GB A100 (source: Sec. 5.1, p. 15, App. B, p. 29).
- Shmup lens: the memory length used is the sequence length: GRU trains on 512-step truncated sequences yet holds over 3000-step episodes in Endless Mortar Mayhem; TrXL sees at most 766 steps (3 layers × window 256) with no gradient into its cache. At 60 Hz, 766 steps is about 13 s of frames, or about 3.4 to 3.8 min at one step per action chunk (16 steps in the NitroGen paper, 18 in the released checkpoint's config) (source: assumption, from Eq. 7, p. 13, Sec. 5.5, p. 19, Table 4, p. 28).
- Shmup lens: memory is nearly decoupled from credit assignment because rewards are paid immediately on success (A4; Endless Mystery Path's only initially look dense); [[ni2023decoupling]]'s audit (not this paper's Table 3) puts finite Mortar Mayhem's credit assignment at most 25 steps. The endless tasks therefore test memory capacity and duration, not delayed credit assignment of the kind weapon or rank choices need (source: assumption, from Table 3, p. 27, Sec. 3.2.2, p. 9).
- Shmup lens: Endless Searing Spotlights (dodge threats, collect timed pickups, keep a hidden position, health in the HUD) is the closest of the three to a shmup, and Endless Mortar Mayhem is the closest to accumulating facts shown once (like an item or economy state); both suit a short probe of a frames-in policy before a game harness exists (source: assumption).
- Shmup lens: the ground-truth estimation head is the paper's analogue of a RAM probe. Training it jointly leaks privileged information (Sec. 4.1, p. 11), but the authors also suggest fitting it after training to query what memory holds, which matches the offline-only RAM rule (source: Sec. 4.1, p. 11).

## Limitations / open questions

- Hyperparameters were tuned mostly on finite tasks, so the endless comparison may favour GRU's tuned settings; the authors do not claim optimality (App. B, p. 29).
- The cause of TrXL's endless-task weakness is unresolved (stale cache, query lacking time information, positional encoding) (Sec. 5.6, pp. 20-23, Sec. 6, p. 25).
- Only GRU and TrXL are tuned; SSMs are named as future work (Sec. 6, p. 25).
- Open question: GRU here contradicts the transformer advantage in Passive T-Maze (ni2023decoupling) and GTrXL's LSTM comparison; is the difference PPO with a cached, gradient-free TrXL memory, or the endless tasks' growing demands?
- Horizon mismatch explained: ni2023decoupling's 135-step memory horizon is for the ICLR 2023 five-command Mortar Mayhem (Ni et al. App. B.2, p. 20), while this paper's 274 steps are for 10 commands (Sec. 5.5, p. 19).
- Open question: R2I (samsami2024mastering) wasn't evaluated on Memory Gym; would an SSM world model match GRU on the endless tasks?
- Open question: can an endless shmup scenario be built the same way, such as a stage whose rank or pattern state grows with survival, with RAM used only offline to measure what the policy's state encodes?

## Related

- uses [[schulman2017proximal]]: all baselines are trained with PPO's clipped surrogate and clipped value loss.
- uses [[parisotto2020stabilizing]]: TrXL takes GTrXL's pre-LN and identity-map reordering; untuned GTrXL is tested in App. D, whose bias-0 vs bias-2 result challenges the earlier assertion by Parisotto et al. (p. 31), and GRU's endless-task win diverges from GTrXL's LSTM results.
- uses [[vaswani2017attention]]: multi-head attention and the absolute positional encoding of the TrXL baseline.
- related [[ni2023decoupling]]: audits finite Mortar Mayhem as long memory with short credit assignment and finds transformers stronger in memory, unlike the endless results here.
- related [[samsami2024mastering]]: an SSM-based memory agent evaluated on POPGym, BSuite and Memory Maze, the backbone this paper names as future work.
- related [[magne2026nitrogen]]: a short-context game policy; Memory Gym's endless tasks are a cheap test of how long such a policy's memory lasts.

## Experiments

<!-- kb:begin experiments -->
- [2026-11-27-memory-token-toys](../../experiments/2026-11-27-memory-token-toys/plan.md) (proposed): tests C4, C10
<!-- kb:end experiments -->
