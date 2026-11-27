---
id: agarwal2021deep
title: Deep Reinforcement Learning at the Edge of the Statistical Precipice
authors: [Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare]
year: 2021
venue: NeurIPS 2021 (Outstanding Paper; arXiv v4 read)
url: https://arxiv.org/abs/2108.13264
arxiv: 2108.13264
doi:
pdf_sha256: a6a5abb49e58a9e4a0be8556c8351ce8c3b52e70216bffc9fa80eac0823ec5ed
tags: [evaluation, statistics, atari, benchmark, iqm, bootstrap]
status: read
read: 2026-11-27
---

# Deep Reinforcement Learning at the Edge of the Statistical Precipice

## TL;DR

Shows that point estimates of mean or median normalised score from 3-10 runs are unreliable on
Atari 100k, and recommends stratified-bootstrap CIs, performance profiles, IQM and optimality gap
(the rliable library). It gives no emulator settings beyond sticky actions and no reference-score table.

## Problem

Deep RL papers compare point estimates (mean or median over tasks) from a handful of runs, and
ignore the statistical uncertainty that implies; the compute cost of ALE makes more runs
infeasible (Sec. 1, p. 2). Median is a biased and high-variance summary, mean is dominated by
outlier tasks, and some papers use non-standard protocols such as the maximum during training
(Sec. 1, Sec. 3, pp. 2-5). The paper wants reporting tools that work with 3-10 runs per task.

## Method

- Setting: M tasks, N independent runs per task, each run gives a normalised score x_{m,n};
  normalisation is a linear rescale between two reference points, for Atari a random agent
  (0) and an average human (1) (Sec. 2, p. 3). Aggregates are Mean and Median of the per-task
  run means; "true" values are the N to infinity limits (Sec. 2, p. 3).
- Case study: Atari 100k with DER, OTR, DrQ, CURL, SPR (26 games), 100 runs per game, each run
  scored as the mean over 100 evaluation episodes after training; subsample 3-100 runs to
  study variability (Sec. 3, p. 4).
- Stratified bootstrap CIs: resample runs with replacement independently per task, N per task,
  M tasks, recompute the aggregate, take percentiles (Sec. 4.1, p. 6; App. A.5, p. 22).
- Performance profiles: run-score distribution, the fraction of all M*N runs above score tau,
  with pointwise stratified-bootstrap bands (Eq. 1, Sec. 4.2, pp. 6-7).
- Aggregates: IQM (25% trimmed mean, mean of the middle floor(NM/2) runs), optimality gap
  (mean shortfall below threshold gamma = 1) and average probability of improvement via the
  Mann-Whitney U statistic (Sec. 4.3, pp. 7-8; Eq. A.2, App. A.7, p. 25).
- Re-evaluation of ALE 200M (55 games), DM Control (6 tasks) and Procgen (16 tasks) with these
  tools (Sec. 5, pp. 8-10).

## Assumptions

- A1: Per-run scores are scalar normalised scores, obtained by linear rescaling between two
  reference points per task (Sec. 2, p. 3).
- A2: The N runs per task are independent and identically distributed realisations of the task's
  score; fixing seeds does not control all randomness (Sec. 2, footnote 3, p. 2).
- A3: The regime is few runs per task, 3-10 (Sec. 1, p. 2).
- A4: Atari 100k means 26 games with 100k agent steps (2-3 hours of game play), each run scored
  on 100 evaluation episodes after training (Sec. 3, p. 4).
- A5: The ALE 200M results use 55 games with sticky actions, as in Machado et al. (Fig. 9, p. 8).
- A6: Run-level scores of the compared algorithms were produced by their authors or by Dopamine
  under their own published hyperparameters (App. A.1-A.2, pp. 18-19).

## Claims

- C1: On Atari 100k, the median score computed with the run counts the papers report (5 for DER, OTR, DrQ; 10 for SPR; 20 for CURL) varies enough that the sampling distribution of the median overlaps between algorithms: the reported point estimates can severely over- or underestimate the expected median, and DER may be better than OTR, unlike the reported numbers [A2, A3, A4] (Sec. 3, Fig. 2, pp. 4-5).
- C2: The sample median is a biased estimator in the few-run regime: SPR's median differs by +0.03 between 5 and 100 runs, about 36% of SPR's +0.08 mean improvement over DrQ(eps), and the sign and size of the bias depend on the algorithm [A3, A4] (Sec. 3, Fig. 3, p. 5).
- C3: 95% CIs on the median need about 50-100 runs per game on Atari 100k to be small [A2, A4] (Sec. 3, Fig. 2, pp. 4-5).
- C4: Detecting a lift of l% on the median needs 25 runs for l = 25, 100 runs for l = 10, and even 100 runs are insufficient at l = 0 (deviations of 20% possible); IQM needs fewer runs [A2, A4] (Sec. 3, Fig. 4, p. 5, p. 6).
- C5: Maximum-based protocols (best evaluation during training, best over runs or over hyperparameter configurations) give scores incomparable with end-of-training scores; applying CURL's protocol to DER gives scores far above CURL's reported ones, and much of SUNRISE's gain over DER is explained by its protocol [A4] (Sec. 3, Fig. 5, p. 6, App. A.4, pp. 20-21).
- C6: Percentile stratified-bootstrap CIs give good coverage for median and IQM with as few as 10 runs; with 3 runs they underestimate the true 95% CI; percentile CIs beat basic and BCa variants in this regime, and CI widths for IQM are much smaller than for median [A2, A3, A4] (Sec. 4.1, Fig. 6, p. 6, App. A.5, p. 22).
- C7: Bootstrap CIs on per-game mean scores need about 20-30 runs per game for near-nominal coverage, versus 5-10 runs for stratified CIs on aggregates [A2, A4] (App. A.3, p. 20, Fig. A.18-A.19, p. 21).
- C8: IQM is the 25% trimmed mean over all runs on all tasks (the middle 50% of the NM runs); it is less sensitive to outliers than mean, uses magnitudes unlike median, and has smaller CIs than median and smaller bias than median on Atari 100k (IQM bias about 0.006 versus 0.05 for SPR) [A1, A4] (Sec. 4.3, pp. 7-8, Fig. A.17, p. 20).
- C9: Optimality gap is the mean shortfall below gamma = 1 (human level for HNS), and probability of improvement P(X>Y) averages per-task Mann-Whitney probabilities; the paper states no single metric suffices and that ranking changes with the metric [A1] (Sec. 4.3, p. 8, App. A.7, pp. 24-25).
- C10: A run-score distribution (Eq. 1) is an unbiased estimator, and one outlier run changes it by at most 1/(MN) at any tau; average-score distributions are a biased estimate with larger variance [A2] (Sec. 4.2, Eq. 1, p. 7).
- C11: Default bootstrap sizes are 50,000 re-samples for aggregate metrics and 2,000 for pointwise bands and probability of improvement [A2] (App. A.5, p. 22, Fig. A.28, p. 25).
- C12: On ALE 200M (55 games, sticky actions), intervals overlap for M-IQN vs Rainbow and C51 vs DQN (Adam), and the order between algorithms changes between Median, IQM and optimality gap (IQN vs Rainbow depends on tau) [A5, A6] (Sec. 5, Fig. 9, Fig. 10, pp. 8-9).
- C13: ALE 200M results use 5 runs per game, except 3 for M-IQN and 11 for DreamerV2, with Dopamine run data for most agents [A5, A6] (Fig. 9, p. 8, App. A.1, p. 18).
- C14: Human-normalised score is used for Atari 100k and ALE; record-normalised score is not used for ALE because ALE results are 30 minutes of game play rather than hours of play [A1] (App. A.6, p. 23).
- C15: Fixing Python, NumPy and JAX seeds does not make two sets of 100 DER runs on Atari 100k correlate strongly (Pearson coefficient below about 0.3 per game), because of GPU non-determinism [A2] (Fig. A.13, App. A.2, p. 18).
- C16: On DM Control 100k/500k (6 tasks), most algorithms do not consistently rank above the ones they claimed to improve on, with large CI overlaps in mean normalised score [A2, A3] (Sec. 5, Fig. 11, p. 9).
- C17: For Procgen (16 tasks, 10 runs), PPO-normalised means are heavy-tailed and outlier driven; min-max normalisation is recommended, and some claimed improvements are only 50-70% likely [A2, A3] (Sec. 5, Fig. 12, pp. 9-10).
- C18: The Atari 100k case study cost 18,200 runs (100 per game, 26 games, 6 algorithms plus 100 more for DER), 3-5 hours each on a P100, 2,400-3,600 GPU days; 5 runs on 50+ ALE games at the standard protocol is stated to take over 1,000 GPU days [A4] (App. A.2, p. 19, Sec. 1, p. 2).
- C19: Individual runs are released (Atari 100k at gs://rl-benchmark-data/atari_100k, ALE at gs://rl-benchmark-data/ALE) with the rliable library; DER and OTR were reimplemented in Dopamine [] (App. A.1-A.2, p. 18, Sec. 6, p. 10).

## Implementation notes

- The emulator settings behind the ALE and Atari 100k numbers (frame skip, max-pool, no-op
  starts, life-loss handling, reward clipping, action set, observation size) are not stated;
  they are those of the source papers and of Dopamine (source: assumption).
- For ALE 200M only "sticky actions [69]" is stated as a setting (source: Fig. 9, p. 8,
  Fig. A.22, p. 23).
- ALE agents are evaluated for "30 minutes of game-play"; 108K frames at 60 Hz is the reader's
  conversion (source: App. A.6, p. 23, assumption for the conversion).
- Footnote 9: Dopamine Rainbow omits double, dueling and noisy networks, and Hessel et al.'s
  Rainbow results were a single run without sticky actions (source: Sec. 5, p. 8).
- The human and random reference scores per game are not given; only "random agent = 0,
  average human = 1" and a citation to Mnih et al. (source: Sec. 2, p. 3). The 26 Atari 100k
  and 55 ALE game lists are not given either; the games appear only in per-game plots (source:
  Fig. A.14, p. 19, reader's reading). The standard table is in Mnih et al. 2015 and Badia et
  al. 2020 (source: assumption).
- Evaluation epsilon matters: Dopamine's standard greedy evaluation is 0.001, DrQ's codebase used
  0.05 and DrQ(eps) restores 0.001 (training epsilon decayed to 0.01 not 0.1) (source: App. A.2,
  p. 18).
- Atari 100k DER uses n-step 10 instead of 20; configs are in Dopamine labs/atari_100k (source:
  App. A.2, p. 18).
- Procedure for a harness: per game collect N final-score runs (each the mean over evaluation
  episodes), normalise, then stratified bootstrap over runs, percentile 95% CIs, IQM, median,
  mean, optimality gap; rliable implements this (source: Sec. 4, pp. 6-8, App. A.5, p. 22).
- Licence of the released runs and rliable: not stated in the paper (source: assumption: check the
  repository before use).

## Limitations / open questions

- The paper says finding the best aggregate metric is "still an open question" and depends on
  the normalised score distribution (Sec. 4.3, p. 8).
- Bootstrap CIs still underestimate the true 95% CI with 3 runs (Fig. 6, p. 6).
- Open question: the Atari 100k analysis is on one family of algorithms (Rainbow-based); do the
  run counts carry over to large offline-trained policies evaluated on 16 rollouts?
- Open question: the paper's stratification is over runs, so with 1 evaluation run per game
  (as in MGDT-style evaluations) it offers only the over-tasks bootstrap of Fig. A.22.

## Related

- related [[machado2018revisiting]]: source of the sticky-action ALE protocol used for its 200M results.
- related [[agarwal2020optimistic]]: same first author; its "best evaluation during training" scoring is among the maximum-based protocols discussed here (Sec. 3, p. 5).
- related [[lee2022multigame]]: reports IQM and median human-normalised scores.
- related [[cheng2025jointly]]: reports HNS IQM and median.
- related [[pan2026mhwa]]: reports IQM HNS with RLiable CIs.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->

