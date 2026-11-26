---
id: kirk2023survey
title: A Survey of Zero-shot Generalisation in Deep Reinforcement Learning
authors: [Robert Kirk, Amy Zhang, Edward Grefenstette, Tim Rocktäschel]
year: 2023
venue: JAIR 76 (2023) 201-264, journal survey (arXiv v6, 19 Jan 2023)
url: https://arxiv.org/abs/2111.09794
arxiv: 2111.09794
doi: 10.1613/jair.1.14174
pdf_sha256: 55cf137243bca83d183f78068fe9ed2c920ff7b7707e517e6cbae70f2473d395
tags: [reinforcement-learning, generalization, zero-shot, survey, benchmark, taxonomy, offline-rl, procgen, exploration]
status: read
read: 2026-11-26
---

# A Survey of Zero-shot Generalisation in Deep Reinforcement Learning

## TL;DR

Formalises zero-shot generalisation (ZSG) in RL as a class of contextual-MDP problems, then
taxonomises 55 benchmarks (PCG vs controllable, four variation types) and methods (similarity,
handling differences, RL-specific). Says little on on/off-policy or exploration; flags offline-RL ZSG.

## Problem

RL research mostly evaluates on the training environment (Atari, MuJoCo), which does not match
deployment, and the scattered work on generalisation lacks a shared formalism, terminology and
categorisation of benchmarks and methods (Sec. 1-2, pp. 1-5). The survey supplies these and critiques
the field. Page numbers below are PDF pages (journal pages 201-264 are PDF pages 1-64).

## Method

- Scope: single-agent, empirical work, zero-shot policy transfer (ZSPT) with no data from test instances;
  multi-agent, theory, domain adaptation and most meta-RL are out of scope (Sec. 1, pp. 3-4).
- Formalism (Sec. 3, pp. 5-15): a contextual MDP (Def. 1, p. 8) whose context (seed, id, parameters)
  fixes reward, transitions and initial state and does not change within an episode; training and testing
  context subsets (Def. 2); ZSPT as training on C_train and maximizing expected return on C_test with a
  possibly non-Markovian policy (Def. 4, pp. 9-10); a controllable-context variant (Def. 5, p. 10);
  GenGap = train minus test return (Eq. 2, p. 10).
- Benchmarks (Sec. 4): environments listed in Table 1 (p. 17) by style, context type (PCG, continuous,
  discrete cardinal or ordinal) and variation (state, dynamics, observation, reward); evaluation protocols
  from training/test context sets (PCG protocols A/B/C, Fig. 2, p. 20; controllable ones with
  interpolation, combinatorial interpolation, extrapolation, Fig. 3, p. 21).
- Methods (Sec. 5, Fig. 4, p. 26, Tables 2-3, pp. 36-37): 5.1 increase training/test similarity (data
  augmentation and domain randomisation, environment generation, optimisation objectives incl. robust
  RL and PLR); 5.2 handle differences (inductive biases, regularisation, learning invariances, online
  adaptation); 5.3 RL-specific problems and better optimisation without overfitting.
- Discussion and future work (Sec. 6, pp. 38-44) and takeaways (Sec. 7, pp. 44-45). Appendix A (pp. 45-47)
  defines other structured MDPs; references are pp. 48-64.

## Assumptions

- A1: The survey covers zero-shot transfer: a fixed policy, possibly recurrent, is evaluated on test contexts with no extra training data or episodes from them (Sec. 1, p. 3, Sec. 3.4, p. 10, Sec. 3.7, p. 15).
- A2: The formalism assumes the action space is fixed between training and testing and that contexts do not change within an episode (Sec. 3.3, p. 9, Sec. 6.1, p. 39).
- A3: Task-family variation is assumed to satisfy the Principle of Unchanged Optimality (a single policy can be optimal for every context), usually by conditioning the policy on a goal or reward specification (Sec. 4.1.1, p. 19, Sec. 6.4, p. 42).
- A4: The survey describes work up to its mid-2022 submission (submitted 08/2022) and covers only single-agent empirical papers (reader's inference, p. 1, Sec. 1, pp. 3-4).

## Claims

- C1: ZSG in RL is a class of problems, not one problem: each is fixed by a CMDP, training and testing context sets and whether the context is controllable; no unifying assumption covers all of them, and a No-Free-Lunch-style argument means improving generalisation in some settings can hurt it in others [A1, A2] (Sec. 1, p. 3, Sec. 3.4, pp. 9-10).
- C2: The generalisation gap (train minus test return, Eq. 2) is not a good sole metric, since a random policy has a gap near 0 and rewards may not be comparable; the authors recommend problem-specific benchmarks and overall test performance, using the gap as an auxiliary metric (tie-breaking, or when assurance matters more) [A1] (Sec. 3.4, Eq. 2, pp. 10-11).
- C3: Of the 55 environments in Table 1, 21 (38%) use PCG; state variation is the most common (42, 76%), then observation (29, 53%), reward (20, 36%) and dynamics (19, 35%); only one environment varies dynamics alone and none varies only reward; gridworlds (14) and continuous control (13) dominate [A1] (Sec. 4.1.1, Table 1, pp. 17-19).
- C4: For purely PCG environments the only meaningful evaluation choice is the number of training contexts (a single context, a small set as in Procgen, or all but held-out seeds); protocol C (hold out seeds) tests mostly optimisation rather than ZSG beyond avoiding memorisation, but the authors argue it should be standard RL evaluation [A1] (Sec. 4.2, Fig. 2, pp. 19-20).
- C5: Controllable environments allow interpolation, combinatorial interpolation (tests systematicity) and extrapolation (tests productivity) per factor; protocols form a partial order of difficulty, more training contexts and variety make a protocol easier, and extrapolation along cardinal axes is hardest [A1] (Sec. 4.2, Fig. 3, pp. 21-22, Sec. 4.3, p. 25).
- C6: Purely PCG benchmarks cannot isolate specific factors of variation; the authors recommend future environments combine PCG with fine-grained controllable factors (as Obstacle Tower and MiniHack partly do) [A1] (Sec. 4.3, pp. 23-24, Sec. 7, p. 44).
- C7: With difficult protocols (extrapolation) standard RL is unlikely to generalise; methods need prior knowledge, transfer, inductive biases or online adaptation, and authors should state which inductive bias they add and when it may hurt [A1, A3] (Sec. 4.3, p. 25, Sec. 5.4, p. 38).
- C8: Methods are categorised as (5.1) increasing training-test similarity, (5.2) handling train-test differences and (5.3) RL-specific problems and optimisation without overfitting; most surveyed methods change the loss or algorithm, with architectural changes and model-based RL under-explored [A1] (Sec. 5, Fig. 4, p. 26, Table 2, Table 3, pp. 36-37, Sec. 6.6, pp. 43-44).
- C9: The survey reports that Cobbe et al. found that data augmentation, dropout, batch norm, L2 weight decay and policy entropy each improve CoinRun generalisation, with the combination only slightly better than the individual ones, implying they address similar causes [A1] (Sec. 5.2.2, p. 31).
- C10: The survey lists exploration-exploitation, non-stationarity of the data distribution and bootstrapping/TD learning as RL-specific factors that "likely" interact with generalisation, in the authors' words often negatively, and calls for work on them, but it reviews no method that improves ZSG through exploration during training; in-episode meta-exploration (HyperX, BOReL) appears under online adaptation [A1] (Sec. 5.3.1, p. 35, Sec. 6.6, p. 43, Sec. 5.4, p. 38, Sec. 5.2.4, p. 34).
- C11: Reported RL-specific findings include: iterated policy re-initialisation distillation counters non-stationarity (Igl et al.); PPG and DAAC (PPO variants) show value functions can be optimised more strongly than policies without overfitting; MuZero Reanalyse improves Procgen over SOTA model-free methods at lower sample cost but not as much on Meta-World reward variation [A1] (Sec. 5.3.1-5.3.2, p. 35).
- C12: Citing Bengio et al., the survey notes temporal-difference methods tend to have lower-interference training, which correlates with worse generalisation [A1] (Sec. 6.5, p. 43).
- C13: Offline-RL ZSG is under-explored: offline benchmarks do not measure generalisation to contexts unseen in the dataset; the authors suggest training on offline data from 200 Procgen levels and evaluating on the full level distribution, or an offline-pretrain, online-fine-tune compromise [A1] (Sec. 6.2, pp. 40-41, Sec. 7, p. 44).
- C14: Inverse RL from human demonstrations raises two generalisation problems, of the learned reward function and of the trained policy [A1] (Sec. 6.2, p. 41).
- C15: Reward-function variation is the least-studied variation type and stronger types (dynamics, reward) are the ones to prioritise; they need goal conditioning and care with the Principle of Unchanged Optimality [A1, A3] (Sec. 6.4, p. 42, Sec. 7, p. 45).
- C16: Evaluating context efficiency (performance versus number of training contexts) is recommended alongside sample efficiency, since new samples and contexts are costly in real deployments [A1] (Sec. 6.2, p. 40, Sec. 7, p. 44).
- C17: Online adaptation within one episode (context encoders, meta-RL such as RL2 and VariBAD) is a permitted zero-shot approach, since the policy class allows non-Markovian policies, and is under-explored [A1] (Sec. 3.4, p. 10, Sec. 5.2.4, pp. 33-34, Sec. 6.6, p. 43).

## Implementation notes

- This is a survey: no algorithm to implement. The benchmark table (Table 1, p. 17) and method tables (Tables 2-3, pp. 36-37) are the reusable parts; they are not copied here (source: Sec. 4-5).
- Procgen's standard protocol is 200 training levels, 25M steps, test on the full level distribution (source: Sec. 3.3-3.4, pp. 8-10).
- Survey cutoff is 2022: later work (e.g. [[jiang2023importance]], [[mediratta2024generalization]]) is not covered (source: p. 1).

## Limitations / open questions

- The authors scope out multi-agent RL, theory and non-zero-shot settings, and say their environment list may be incomplete (Sec. 1, pp. 3-4, footnote 3, p. 18).
- Method categories overlap and Tables 2-3 classify by main motivation only (Sec. 5.4, p. 37).
- Reader's observation: the survey has no section on on-policy vs off-policy training, discusses no behaviour cloning, and its Bengio et al. citation (C12) is its only TD-versus-other comparison; so it gives no evidence about whether on-policy vs off-policy training, policy entropy or exploration changes zero-shot generalisation beyond the points in C9-C12; for that see the primary papers it cites (Raileanu and Fergus 2021, Cobbe et al. 2019).
- Open question (reader's): behaviour cloning as a pretraining or generalisation baseline is not discussed at all; the offline-RL ZSG gap it flags (C13) is what [[mediratta2024generalization]] later measures.

## Related

- related [[jiang2023importance]]: exploration as a generalisation lever, which the survey names as an under-studied RL-specific factor (Sec. 5.3.1, p. 35).
- related [[mediratta2024generalization]]: builds the offline-RL ZSG benchmark the survey recommends (Sec. 6.2, p. 40).
- related [[farebrother2020generalization]]: the survey cites it among works showing poor generalisation of DQN to slightly different environment instances (Sec. 1, p. 2).
- related [[schulman2017proximal]]: PPO is the base of many surveyed methods (UCB-DrAC, PPG, DAAC) (Sec. 5.1.1, p. 27, Sec. 5.3.2, p. 35).

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
