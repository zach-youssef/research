---
id: lynch2019play
title: Learning Latent Plans from Play
authors: [Corey Lynch, Mohi Khansari, Ted Xiao, Vikash Kumar, Jonathan Tompson, Sergey Levine, Pierre Sermanet]
year: 2019
venue: CoRL 2019 (arXiv v2 read)
url: https://arxiv.org/abs/1903.01973
arxiv: 1903.01973
doi:
pdf_sha256: 4a617a68cc9ed89ebbdff8fdda000f0bb744128bd5e91dca1b2c5594631835b0
tags: [imitation-learning, play-data, latent-plans, goal-conditioned, cvae, robotics, self-supervised, hierarchical, multimodality]
status: read
read: 2026-11-25
---

# Learning Latent Plans from Play

## TL;DR

Play-LMP trains a conditional seq2seq VAE on unlabeled teleoperated play: a plan encoder reads a
window, a plan proposal sees only (current, goal), and a decoder policy follows the sampled latent plan.
One goal-image-conditioned policy gets 85.5% (state) / 69.4% (pixels) over 18 tasks vs 70.3% / 66.5% for BC.

## Problem

Task-specific demonstrations and reward functions are costly per skill, and a robot generalist
should reach any reachable goal state, a continuum of tasks indexed by (current state, goal
state) (Sec. 1, p. 1, p. 2). Random exploration covers little, and expert demonstrations cover
a narrow band and suffer distribution shift (Sec. 1, p. 2). A goal-conditioned policy trained
by plain supervision on play faces multimodality: many behaviours connect the same (s_c, s_g)
(Sec. 3.1, p. 5).

## Method

- Play data: unsegmented, unlabeled logs of (observation, action) from a human teleoperating
  the robot with their own curiosity, in a simulated playground with an 8-DOF arm, a door, a
  drawer, 3 buttons and a block (Sec. 3, Fig. 3, p. 4; App. A.3.1, p. 13).
- Play-GCBC (baseline): sample a window of length kappa from play, use its final observation as
  the goal, and train a goal-conditioned recurrent policy by maximum likelihood on the logged
  actions (Eq. 1, Sec. 3.1, p. 4, p. 5; Alg. 1, p. 12).
- Play-LMP: a conditional seq2seq VAE with three parts (Sec. 3.2, p. 5):
  - plan recognition V_enc: a bidirectional sequence encoder over the whole window, giving a
    diagonal-Gaussian posterior over a latent plan z;
  - plan proposal CG_enc: a feedforward conditional prior from the window's first and last states;
  - plan- and goal-conditioned policy pi_LMP: a stochastic RNN that gets (s_t, s_g, z) and
    outputs a distribution over the next action.
  Loss L_LMP = action reconstruction (Eq. 3) + beta * KL(posterior || proposal) (Eq. 2, Eq. 4,
  Sec. 3.2, p. 6; Alg. 2, p. 12). beta < 1 avoids posterior collapse (Sec. 3.2, p. 6).
- Test time: encode the current and goal observations, sample one z from the plan proposal,
  decode actions closed loop at 30 Hz, and re-sample a plan every kappa = 32 steps, about 1 Hz;
  the recognition encoder is discarded (Sec. 3.2, Fig. 1b, p. 6, footnote 1, p. 6).
- Perception: per-channel encoders (a CNN with spatial softmax for 299x299 RGB, raw normalized
  proprioception), trained end to end by the same losses; only the visual part of the goal is
  used as goal state (App. A.2, pp. 12-13, Fig. 9).
- Actions: 8-D end-effector position, orientation and gripper, quantized to 256 bins each,
  output as a mixture of discretized logistics (App. A.2, p. 12).
- Time split: low level at 30 Hz, plan latent at about 1 Hz (once per 32 steps); the latent
  interface is a continuous Gaussian code z (Sec. 3.2, p. 6). Training is purely supervised
  and self-supervised, with no reward or RL.

## Assumptions

- A1: Tasks are specified by a single goal state (an image), and test tasks come from the same single environment as the play data (App. A.5, p. 17).
- A2: Play data is not heavily imbalanced towards one object interaction (App. A.5, p. 17).
- A3: The agent observes a fixed first-person RGB view plus proprioception, or ground-truth object states in the state experiments (Sec. 4, App. A.2, p. 7, p. 12).
- A4: Test tasks are drawn near the play and demonstration distribution, with expert demonstrations of the tasks supplying the BC baselines (App. A.5, Sec. 4, p. 17, p. 7).
- A5: The environment is a simulated 8-DOF arm with continuous teleoperated controls collected at 30 Hz (Sec. 3, Sec. 3.2, p. 4, p. 6).
- A6: The latent plan and the plan proposal are unimodal Gaussians (App. A.5, p. 17).
- A7: Success on each task is checked by a hand-written success function, used for evaluation only (App. A.3.1, p. 13).

## Claims

- C1: In the state experiments, a single Play-LMP policy trained on unlabeled play reaches 85.5% average success over the 18 tasks, versus 70.3% for 18 per-task BC policies, 66.2% for multi-task BC and 77.9% for Play-GCBC; in pixel experiments the figures are 69.4% ± 10.8 (LMP), 58.7% ± 11.6 (GCBC) and 66.5% ± 12.1 (BC) over 3 seeds [A1, A2, A3, A4, A5, A6, A7] (Sec. 4, Fig. 6a, p. 7).
- C2: Per task, Play-LMP improves over Play-GCBC by up to about 25 points on some tasks and is worse on a few (for example grasp lift, push blue), and over BC by up to 50 or more points [A1, A2, A3, A4, A5, A6, A7] (Sec. 4, Fig. 7, pp. 7-8).
- C3: With only 30 minutes of play data, Play-LMP (71.8%, state) is still above the 18 BC policies trained on 90 minutes of expert demonstrations (70.3%) [A1, A2, A3, A5, A6, A7] (Sec. 4, Fig. 8, p. 8).
- C4: For the same collection time, play data covers 4.2 times more quantized interaction-space regions than the 18 tasks' expert data and 14.4 times more than random exploration, and coverage keeps growing about linearly up to 7 hours [A5] (Sec. 1, Fig. 2c, Fig. 2d, p. 3).
- C5: Policies trained on play (LMP and GCBC) degrade much less than BC when the initial end-effector position is perturbed by 0 to 0.4 m [A1, A2, A3, A5, A6, A7] (Sec. 4, Fig. 6b, p. 7, App. A.4.1, Fig. 12, p. 15).
- C6: The latent plan space organizes by function without task labels: t-SNE of 512 random play windows plus the validation demonstrations puts drawer, button, grasping and sweeping sequences in separate regions [A3, A5] (Sec. 4, Fig. 4, p. 4, p. 8).
- C7: Play-LMP policies show emergent retrying: after a failed grasp or slide attempt they replan and try again, which the paper observes qualitatively and not in BC trained on expert demonstrations only [A1, A5] (App. A.4.2, Fig. 13, Fig. 14, p. 16).
- C8: Splitting the problem into plan inference (a latent variable posterior and proposal) and plan execution gives systematic gains over Play-GCBC, which has no explicit plan inference [A1, A2, A3, A4, A5, A6, A7] (Sec. 4, Fig. 7, p. 7, p. 8).

## Implementation notes

- Window length: kappa is sampled from U(kappa_low, kappa_high) in training; the paper states average planning horizon 32 for replanning but not the two bounds (source: Alg. 1, p. 12; Sec. 3.2, p. 6; bounds not given, assumption).
- Network sizes: image CNN with 8x8 stride 4, 4x4 stride 2 and 3x3 stride 1 convolutions then spatial softmax, 64 visual features plus 8 proprioceptive; plan latent 256 dims; recognition net two stacked 2048-unit bidirectional RNN layers; proposal net four 2048 FC layers; policy two 2048 RNN layers (source: Fig. 9, p. 13).
- BC, GCBC and LMP policies share one architecture: an RNN with 2 hidden layers of 2048 units over quantized actions (source: App. A.1.2, p. 12).
- beta, learning rate and optimizer are not given (source: assumption; none stated in Sec. 3.2, p. 6 or App. A.1-A.2, pp. 11-12).
- Dataset sizes: the paper says about 7 hours of play for pixel models (state models up to 180 minutes), but App. A.3.4 says about 3 hours of play were collected, while App. A.3.2 repeats "7 hours of unscripted continuous play"; the reader found these inconsistent (source: Sec. 4, p. 7; App. A.3.2, p. 13 versus App. A.3.4, p. 15).
- Evaluation: 20 rollouts per task, 100 expert demonstrations per task for BC, 10 each for validation and test (source: Fig. 8 caption, p. 8; App. A.3.4, p. 15).

## Limitations / open questions

- Tasks must be describable by a single goal state, which cannot say how to do something (App. A.5, p. 17).
- Generalization is within one environment and to tasks near the training pairs; out-of-distribution (living room to kitchen) is left open (App. A.5, p. 17).
- The play data is assumed balanced across object interactions (App. A.5, p. 17).
- Unimodal Gaussian latent plans may limit expressiveness; discrete codes are suggested (App. A.5, p. 17).
- Open question (reader): in a shmup there is no "play" corpus labeled by goal, but web gameplay video (NitroGen's data) is play-like and unlabeled by task. Goal states would be future frames; do future frames of a 1 Hz window carry strategy such as shop purchases, or only dodging style?
- Open question (reader): the plan z is sampled once per 32 steps (about 1 s). Strategy over minutes would need a slower z or a hierarchy over z, which this single-level latent does not provide.
- Open question (reader): only the plan proposal is used at test time; could the plan latent be fed by memory tokens ([[bulatov2022recurrent]]) instead of one (current, goal) pair?
- Open question (reader): the paper compares with BC and uses no RL at all; whether the same latent can serve as a manager's action space for RL, as in Director's code space ([[hafner2022director]]), is untested here.

## Related

- related [[hafner2022director]]: a manager acting in a learned discrete goal-code space trained by RL; here the high-level latent is learned from demonstrations (play) with a VAE.
- related [[vezhnevets2017feudal]]: a Manager emitting latent goals that a Worker follows, trained by RL rather than from play.
- related [[chen2021decision]]: another way to condition a sequence policy on an outcome (return) instead of a goal state.
- related [[magne2026nitrogen]]: behaviour cloning from unlabeled web gameplay video, the analogue of play data for games.
- related [[baker2022video]]: pseudo-labelled video data for behaviour cloning of long-horizon behaviour.
- related [[schmidt2024actions]]: latent actions learned from unlabelled frames, like Play-LMP's latent plans learned without task labels (there per frame pair, here per 32-step window).
- related [[ye2025latent]]: latent actions from unlabelled video used to pretrain a policy, an analogue of NitroGen-style video data.
- related [[bruce2024genie]]: latent actions inferred from unlabelled game video.
- related [[ni2023decoupling]]: separates memory from credit assignment, framing the long-horizon open questions above.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
