---
id: devin2016modular
title: Learning Modular Neural Network Policies for Multi-Task and Multi-Robot Transfer
authors: [Coline Devin, Abhishek Gupta, Trevor Darrell, Pieter Abbeel, Sergey Levine]
year: 2016
venue: arXiv preprint
url: https://arxiv.org/abs/1609.07088
arxiv: 1609.07088
doi:
pdf_sha256: 71cfe030c617720ceb7df5b3a337ab0c4dd1d9e5217f5dd21a3bcb55f9779285
tags: [reinforcement-learning, transfer-learning, multi-task, modular-networks, cross-embodiment, zero-shot]
status: read
read: 2026-11-21
---

# Learning Modular Neural Network Policies for Multi-Task and Multi-Robot Transfer

## TL;DR

Splits a policy net into a task module (task observations to a small latent) feeding a robot module
(latent plus robot state to torques), trained jointly over a robot-task grid. Composing modules for a
held-out robot-task pair transfers zero-shot in 2 of 3 simulated tests and speeds up learning in the 3rd.

## Problem

Deep RL policies have no structure that says which learned parts belong to the robot and which
to the task, so experience can't be reused for a new robot or task, and fine-tuning works poorly
in robotic RL without direct supervision in the target (Sec. I, p. 1). Prior transfer work
addresses algorithms (shaping, task mappings, sub-task reuse); this paper addresses the policy
representation (Sec. II, p. 2). Pretraining cannot provide zero-shot generalization, and progressive
networks and evolutionary recombination of modules across Atari games did not demonstrate it (Sec. II,
p. 2).

## Method

- **Universe of worlds** (Sec. III, pp. 2-3): discrete degrees of variation (DoVs), here robot
  r, one of R robots, and task k, one of K tasks; a world is one combination. Each task is assumed achievable by
  every robot (p. 3).
- **Decomposition assumptions** (Sec. III-A, p. 3): the observation splits into intrinsic o_R
  (joint angles, velocities, sensors) and extrinsic o_T (images, object positions, end-effector
  position); the cost splits as c_R(x_R, u) + c_T(x_T), actions affecting only the robot term.
- **Composition** (Eq. 1, Sec. III-B, p. 3): the Gaussian policy mean is φ_rk(o) = f_r(g_k(o_T),
  o_R), with one parameter set per robot module f_r and per task module g_k. The task module goes
  first because the task decides the plan and the robot the control output (p. 4). The interface
  g_k(o_T) is an unsupervised latent, with no fixed meaning (p. 4). With more DoVs, modules form a
  DAG (p. 4).
- **Zero-shot composition** (Sec. III-B, p. 4): for an unseen (r_test, k_test), use f_{r_test}
  and g_{k_test}, trained in different worlds; no mapping between robots or tasks is learned.
- **Training** (Sec. III-C, p. 4): all training worlds are trained synchronously, with weights
  tied across worlds and the summed per-world loss; gradients reach θ_k through f_r by
  backpropagation. Each module must appear in at least one training world (Fig. 3, p. 5).
- **Regularization** (Sec. III-D, pp. 4-5): with few robots and tasks, modules overfit (e.g. a
  robot module splits into per-task "receptors"), so the interface is kept to few hidden units and
  dropout is applied.
- **RL algorithm** (Sec. IV-A, p. 5): BADMM guided policy search: LQR-based local time-varying
  linear-Gaussian controllers supervise the network by Euclidean regression. Policy gradient or
  actor-critic could be used instead (pp. 4-5).
- **Architecture** (Sec. IV-B, p. 5, Fig. 4): vision tasks use two conv layers plus spatial
  softmax as the task module and the following fully connected layers (fed the joint state) as the
  robot module; non-vision tasks replace the convs with FC layers. The task module also receives
  the end-effector position, which all robots have.
- **Experiments** (Sec. IV-C to IV-E, pp. 6-7), MuJoCo, torque control, one world held out each:
  colored-block reaching from 80×64×3 images (3-link, 3-link with other link lengths, 4-link;
  4 tasks; 11 of 12 trained); manipulation from target coordinates (3/4/5-link; horizontal
  drawer, vertical drawer, block push; 8 of 9 trained); visually distinct reach, push and peg
  insert from images (the reaching robots; 8 of 9 trained).

## Assumptions

- A1: The worlds vary along a few predefined, discrete degrees of variation, here robot identity and task identity, and each world's robot and task are known (Sec. III, pp. 2-3).
- A2: Every task is achievable by every robot in the universe (Sec. III, p. 3).
- A3: The observation and the state split into a robot-intrinsic part (joint angles, velocities, sensors) and a task-extrinsic part (images, object positions, end-effector position) (Sec. III-A, p. 3).
- A4: The cost splits into a robot term c_R(x_R, u) and a task term c_T(x_T), and actions enter only the robot term (Sec. III-A, p. 3).
- A5: The task plan depends on the task and the control output on the robot, so the task module feeds the robot module (Sec. III-B, pp. 3-4).
- A6: Every robot module and every task module is trained in at least one training world; a held-out pair is made of a robot and a task each trained with other partners (Sec. III-B, p. 4, Sec. III-C, p. 4, Fig. 3, p. 5).
- A7: All training worlds are trained jointly and synchronously, with module weights tied across worlds and the per-world losses summed (Sec. III-C, p. 4, Sec. V, p. 8).
- A8: The module interface has few hidden units and dropout is applied, to keep modules from specialising to training worlds (Sec. III-D, pp. 4-5).
- A9: Every robot has an end-effector whose position is fed to the task module (Sec. IV-B, p. 5).
- A10: Policies are trained with BADMM guided policy search: LQR-based time-varying linear-Gaussian controllers from fixed initial states supervise the network by regression (Sec. IV-A, p. 5).
- A11: The robots are simulated serial-link arms with 3 to 5 links and differing link lengths, controlled by joint torques (Sec. IV-C, p. 6, Sec. IV-D, p. 6).
- A12: All experiments run in MuJoCo simulation (Sec. IV, p. 5).
- A13: The task's goal is visible in the task observation: the target's colour or appearance in an 80x64x3 image from the simulated camera, or the target coordinates (Sec. IV-C, p. 6, Sec. IV-D, p. 6, Sec. IV-E, p. 7).
- A14: The universe is small: 3 robots and 3 or 4 tasks, with one held-out world (Sec. IV-C, p. 6, Sec. IV-D, pp. 6-7, Sec. IV-E, p. 7).
- A15: Zero-shot performance is measured from four test positions by the end-effector or object distance to the target over the last 5 timesteps, without seeds or variances (Table I, p. 6, Table II, p. 7).

## Claims

- C1: Composing the 4-link robot module with the reach-black task module, a combination never trained, reaches the black block zero-shot: mean end-effector distance over the last 5 timesteps 0.08-0.28 at four test positions, vs 1.16-1.35 for a random network and 1.65-2.41 for the 4-link module with a different color's task module [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15] (Table I, Sec. IV-C, p. 6).
- C2: The authors read C1 as showing that visual recognition learned by a task module transfers across robots [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15] (Sec. IV-C, p. 6).
- C3: On the held-out 3-link horizontal drawer pull, composing the modules does not succeed zero-shot, but it is an excellent initialization: without reward shaping it learns faster than training from scratch with shaping, while from scratch without shaping makes no progress, and composing with the wrong (vertical drawer) task module doesn't transfer [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14] (Fig. 7, Sec. IV-D, p. 7).
- C4: On the held-out 4-link block push from images, composing the 4-link robot module with the push task module works zero-shot: distance to target averaged over the last five timesteps 0.19-0.48 at four test positions, vs 0.94-1.79 random and 0.95-1.32 with the reach task module, which performs on par with random [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15] (Table II, Fig. 9, Sec. IV-E, p. 7).
- C5: The authors conclude that the modules decompose perception and the kinematic goal (task module) from robot-specific feedback control (robot module) [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15] (Sec. IV-E, p. 7).
- C6: With only a few robots and tasks (e.g. 3 and 3), overfitting of the modules was found to be a problem, hence the small interface and dropout [A1, A6, A7, A14] (Sec. III-D, p. 4).
- C7: The authors argue that as more worlds use a module, it becomes more invariant to the other DoVs, task modules more robot-agnostic and robot modules more task-agnostic, and expect generalization to grow with more robots and tasks; this is not measured [A1, A3, A6, A7] (Sec. III-B, p. 4, Sec. V, p. 8).
- C8: The authors state theirs is the first method to decompose policy networks into interchangeable modules that transfer zero-shot in new combinations [] (Sec. I, p. 2).
- C9: The abstract claims zero-shot generalization for both visual and non-visual tasks; in the experiments zero-shot transfer succeeds only in the two vision experiments, while the one non-visual experiment (target coordinates, drawer pulling) needs further training [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15] (p. 1, Sec. IV-C, Sec. IV-D, Sec. IV-E, pp. 6-7).

## Implementation notes

- Interface width, dropout rate, FC layer sizes, learning rate and GPS settings (number of local controllers, BADMM penalty schedule, iterations) aren't given (source: assumption that the GPS setup of Levine et al. 2016 [4] was followed, as Sec. IV-B says for the architecture).
- Inputs for colored-block reaching: an 80×64×3 image plus joint angles and velocities, 15366 values for the 3-link arms and 15368 for the 4-link (source: Sec. IV-C, p. 6).
- Policy covariance is learned and observation-independent (source: Sec. III-C, p. 4).
- End-effector position goes into the task module, robot joint state into the first FC layer of the robot module (source: Sec. IV-B, p. 5, Fig. 4, p. 5).
- Table II's caption says "the distance between the drawer and its target position" for a block-pushing task; the reader takes it to mean block-to-goal distance (source: assumption).
- Videos and details at sites.google.com/site/modularpolicynetworks (source: Sec. IV, p. 5; not checked by the reader). The published version is ICRA 2017, pp. 2169-2176 (source: the reference list of Hejna et al. 2020, p. 10 of arXiv 2003.01709); this note describes arXiv v1.

## Limitations / open questions

- All robot-task combinations are trained simultaneously, which in practice may need several
  robots learning at once; sequential/lifelong training is left to future work (Sec. V, p. 8).
- Only 2 DoVs, 3 robots and 3-4 tasks per universe, simulation only, one held-out world per
  experiment, four test positions each, with no seeds or variances reported (Sec. IV,
  pp. 6-7, Table I, Table II).
- Assumes every task is achievable by every robot and that costs and observations split into
  robot and task parts (Sec. III, p. 3, Sec. III-A, p. 3).
- Open question (survey lens): here the shared part is the task module over task observations
  and the robot module owns the action space; a game-agnostic policy over a learned state
  embedding with per-game action heads has the same shape, with the head as f_r. Does the
  interface stay robot- (game-) agnostic without the small bottleneck and dropout at many-game scale?
- Open question: a new robot here still needs training in some world with an existing task; a new
  game would likewise need some data to fit its head, so "zero-shot" is only for new pairings.
- Open question: zero-shot fails when the held-out task is physically far from the trained ones
  (C3); how many tasks per robot module are needed before that stops?

## Related

- related [[serrano2024crossdomain]]: the survey files it as [78] under parameter transfer (Sec. V-A4, p. 11 there).
- related [[reed2022generalist]]: Gato shares one set of weights across embodiments and games through
  tokenisation instead of composing per-embodiment modules.
- related [[lee2022multigame]]: also shares one network across games (41 Atari games) instead of per-game modules.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
