---
id: bacon2017option
title: The Option-Critic Architecture
authors: [Pierre-Luc Bacon, Jean Harb, Doina Precup]
year: 2017
venue: AAAI 2017 (read the arXiv v2 version)
url: https://arxiv.org/abs/1609.05140
arxiv: 1609.05140
doi:
pdf_sha256: d8089c6012c607f3a9e8b560290dea629c91b174e246e18864ac6615467e208f
tags: [options, temporal-abstraction, hierarchical-rl, policy-gradient, termination-functions, atari, actor-critic]
status: read
read: 2026-11-25
---

# The Option-Critic Architecture

## TL;DR

Derives policy-gradient theorems for options (intra-option policy and termination function), so options and
the policy over them are learned end to end from reward with only the option count fixed. In 4 Atari games 8
options beat DQN in 3; options specialise (up/down in Seaquest).

## Problem

Options (temporally extended actions with initiation set, intra-option policy and termination function) are
well understood for planning, but creating them autonomously is hard. Most work finds subgoals
and then learns policies to reach them, which is combinatorial and costly, and prior option learners were
slower than flat learning on a single task (Introduction, p. 1).

## Method

- **Options** are triples (I_w, pi_w, beta_w); here all options are available everywhere (I = S) (Preliminaries, p. 1-2).
  Execution is call-and-return: the policy over options pi_Omega picks w, the intra-option policy runs until
  beta_w says stop, then a new option is chosen (Learning Options, p. 2; Fig. 1, p. 3).
- Intra-option policies pi_{w,theta} and terminations beta_{w,vartheta} are differentiable parametrised functions (p. 2).
- **Intra-Option Policy Gradient Theorem** (Theorem 1, p. 3): the gradient of the expected discounted return
  w.r.t. theta is the sum over state-option pairs weighted by their discounted occupancy of dpi/dtheta times
  Q_U(s, w, a) (proof App., p. 8).
- **Termination Gradient Theorem** (Theorem 2, Eq. 5, p. 3): the gradient w.r.t. vartheta is minus the occupancy-weighted
  dbeta/dvartheta times the advantage A_Omega(s', w) = Q_Omega(s', w) - V_Omega(s'). If the current option is worse than
  the best, the advantage is negative and termination probability rises (p. 3); an interpretation of interrupting options.
- **Architecture/algorithm** (Alg. 1, p. 4): actor = intra-option policies and terminations, critic = Q_U and A_Omega,
  two timescales (fast critic, slower actor). Tabular intra-option Q-learning for the critic; pi_Omega is
  epsilon-soft over Q_Omega. Q_U is not learned separately but estimated from Q_Omega (p. 4).
- **Deep version for the ALE** (Fig. 6, p. 5-6): the DQN conv stack (32 8x8 s4, 64 4x4 s2, 64 3x3 s1) into a shared dense layer of
  512, with heads for pi_Omega, beta_w (sigmoid, one per option) and linear-softmax pi_w per option. The critic trains with
  intra-option Q-learning and experience replay (RMSProp); the option policies and terminations update on-line (p. 5-6).
- **Regularisers** (p. 6): a margin xi = 0.01 added to the advantage in the termination gradient so options don't shrink to
  one step; an entropy regulariser against deterministic intra-option policies; a baseline Q_Omega in the policy gradient.

## Assumptions

- A1: Every option is available in every state (I_w = S) (p. 2, p. 7).
- A2: Options run call-and-return: no switching before beta says stop, and the policy over options picks only at termination (p. 2).
- A3: Learning optimises the single task's discounted return; option semantics come only from reward, with the number of options fixed in advance (p. 2, p. 1).
- A4: Gradients use the biased, discount-ignoring on-policy estimator rather than the unbiased one (p. 7).
- A5: Intra-option policies are stochastic and differentiable; terminations are stochastic and differentiable (p. 3).
- A6: The Atari agent sees raw frames (last 4) and is scored on game score (pp. 5-6).

## Claims

- C1: Theorem 1 gives the gradient of the discounted return with respect to the intra-option policy parameters, and Theorem 2 gives the one for the termination parameters; both hold for options available everywhere [A1, A2, A5] (Eq. 5, p. 3, pp. 8-9).
- C2: In the four-rooms task with the goal moved after 1000 episodes, option-critic with 4 or 8 options recovers faster than primitive actor-critic (AC-PG) and SARSA(0), and its options are learned from scratch at a rate comparable to primitive methods [A1, A2, A3, A4, A5] (Fig. 2, p. 4, p. 4).
- C3: With 4 or 8 options in four-rooms, terminations are more likely near doorways, although subgoals were not specified [A1, A2, A3, A4, A5] (Fig. 3, p. 5).
- C4: In Pinball with 2, 3 or 4 options (Fourier-basis critic), a near-optimal option set is learned by about 40 episodes in all settings; 4 options reach the highest return (above about 7,500), 2 options the lowest, and the options specialise [A1, A2, A3, A4, A5] (Fig. 4, Fig. 5, p. 5).
- C5: In Asterisk, Ms. Pacman, Seaquest and Zaxxon, the same parameters (8 options, termination regularisation 0.01, entropy regularisation 0.01, baseline in the policy gradient) learn options end to end with the features, and the authors report that in Asterisk, Seaquest and Zaxxon the result surpasses the DQN reference over primitive actions [A1, A2, A3, A4, A5, A6] (Fig. 8, p. 6, p. 7).
- C6: Without regularisation the termination gradient shrinks options over time, and policy-gradient intra-option policies become deterministic; a baseline in the gradient, in Seaquest, makes the option action distributions less deterministic [A1, A2, A3, A4, A5, A6] (Fig. 7, p. 6).
- C7: In Seaquest with 2 options, each learned option specialises into an up or down movement sequence, with transitions between them, as in Fig. 9 [A1, A2, A3, A4, A5, A6] (Fig. 9, p. 6, p. 7).
- C8: The biggest remaining limitation is that all options apply everywhere; with function approximation, initiation sets are left to future work, and the unbiased discounted gradient is not recommended for its sample complexity [A1, A4] (p. 7).

## Implementation notes

- Tabular four-rooms: Boltzmann policies (temperature 0.001), sigmoid terminations, zero-initialised weights, discount 0.99, reward +1 at the goal, 1/3 chance of a random move (source: Experiments, p. 4).
- Pinball: Fourier basis of order 3, learning rates 0.01 critic and 0.001 for intra and termination gradients, epsilon 0.01 over options, discount 0.99, step cap 10,000 (source: Pinball Domain, p. 5).
- Atari: learning rate 0.00025 for the intra-option and termination gradients, RMSProp for the critic, epsilon 0.05 at test, margin xi 0.01, 8 options (source: p. 5-6, Fig. 8 caption, p. 7). Replay size, batch size, frame skip and target-network details are not given (source: assumption: follow Mnih et al. 2013, cited p. 6).
- Harb's thesis (2016, cited p. 6) has the deep-network details; not read.
- Reference code is not cited in the paper (source: assumption).

## Limitations / open questions

- Authors: the all-options-everywhere assumption; learning initiation sets needs an ergodicity condition left to future work (Discussion, p. 7).
- Authors: policy gradient in deep networks tends to deterministic policies; it needs entropy regularisation (p. 6).
- Reader, the skill lens: options here are flat, learned from reward on one task, and unlabelled by any shared codebook; number of options is 2-8, and nothing is transferred across games (each Atari game trains its own, p. 7). Termination is a learned per-option, per-state probability; it is the cleanest worked example of how a skill ends.
- Open question: what does a fast reactive low level lose when the termination is a state-dependent sigmoid (a decision every frame at 60 Hz)? The paper's frame-level options do work on Atari (reader's inference from Fig. 9, p. 7: options seem to last tens of frames).
- Open question: DQN comparison is a reference line from Mnih et al. 2013, not a re-run with matched parameters (Fig. 8 caption, p. 7).

## Related

- related [[kujanpaa2023hierarchical]]: cites option-critic as the option alternative to subgoal-based hierarchy; HIPS uses subgoals, not options (Sec. 2, p. 2 of that paper).
- related [[mete2024quest]]: discrete skill tokens learned from demonstrations, with no explicit termination.
- related [[chen2021decision]]: another Atari offline agent; only a loose link, no options.
- related [[vezhnevets2017feudal]]: Feudal networks, a manager-worker hierarchy with a slower manager.
- related [[hafner2022director]]: Director, a manager-worker hierarchy with a slower manager.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
