---
id: chen2018hardware
title: Hardware Conditioned Policies for Multi-Robot Transfer Learning
authors: [Tao Chen, Adithyavairavan Murali, Abhinav Gupta]
year: 2018
venue: NeurIPS 2018 conference paper (read as arXiv v2)
url: https://arxiv.org/abs/1811.09864
arxiv: 1811.09864
doi:
pdf_sha256: a28687009941823c2ce5c61d7d28fbfff59d184781c04237365e0c3baaf9495e
tags: [reinforcement-learning, cross-embodiment, morphology-transfer, transfer-learning, zero-shot, hardware-embedding, ddpg, her, ppo, continuous-control, robotics]
status: read
read: 2026-11-21
---

# Hardware Conditioned Policies for Multi-Robot Transfer Learning

## TL;DR

Conditions one torque-control policy on a hardware vector v_h appended to the state: explicit
(HCP-E, joint-to-joint poses) or learned per robot (HCP-I). Zero-shot on held-out arm types HCP-E
gets 88-93% (reacher, fixed-hole peg), 66% on unseen 7-DoF, 4-76% with random holes (DDPG+HER <3%).

## Problem

Deep RL policies are tied to the robot they were trained on, and learning one per robot is too
costly; hardware differs across labs in DoF, kinematic design and dynamics, so data can't be
pooled, especially under torque control (Sec. 1, p. 1). Dynamics randomisation buys invariance
to dynamics but not to DoF or kinematic structure (Sec. 1, p. 1). Devin et al.'s modular
networks split a policy into robot and task modules, which needs a module per robot (Sec. 2,
p. 2). The paper instead learns one policy that is told what hardware it runs on (Sec. 1, p. 2).

## Method

- **Policy input.** π(ŝ_t) with ŝ_t = s_t ⊕ v_h (concatenation); usable with any RL algorithm.
  PPO for dense reward, DDPG+HER for sparse reward (Sec. 4.3, p. 4; Alg. 1, p. 5; Alg. 2-3, p. 13).
  Each episode samples a robot from a pre-generated pool with varied kinematics and dynamics.
- **HCP-E (explicit).** For n revolute joints, P_i is the relative pose between consecutive joint
  frames (base to J0, J_i to J_i+1, J_n-1 to end effector): a 3-d position difference plus a
  3-d Euler rotation; v_h concatenates them, following the URDF convention (Sec. 4.1, p. 4).
  It encodes kinematics only, since dynamics are rarely measurable; kinematics are usually
  available from the URDF of a new robot (Sec. 4, p. 3).
- **HCP-I (implicit).** v_h is a free vector per training robot, randomly initialised and
  updated by gradient descent on the RL loss with the policy weights (Sec. 4.2, p. 4). A new
  robot's v_h is unknown, so it is estimated by fine-tuning (Sec. 5.2, p. 8).
- **Manipulators.** Nine Sawyer-based MuJoCo arm types A-I: four 5-DoF, four 6-DoF and one
  7-DoF, made by removing joints of the 7-DoF arm; each varied in link lengths, damping,
  friction, armature and link mass (Sec. 5.1, p. 5; Table 2, App. B.1.1, p. 14). Tasks: reacher
  and peg insertion with 3-D goals, sparse ±1 reward within 0.02 m plus an action penalty
  (Sec. 5.1, pp. 5-6; App. B.1, p. 15). States are joint angles and velocities, zero-padded to
  7 joints; the policy always outputs 7 torques and only the first n are used (Sec. 5.1, p. 5).
- **Evaluation.** Leave-one-type-out: train on 8 types (140 variants each), test zero-shot on
  1000 robots of the held-out type (10 trials of 100) with new link lengths and dynamics;
  baseline is DDPG+HER on pooled data without v_h (Sec. 5.1.1-5.1.2, p. 6; Table 1, p. 7).
- **Hopper.** 2D hopper with varied link lengths, masses, damping, friction and armature
  (Table 5, p. 16); HCP-I vs HCP-E, HCP-E plus true dynamics (HCP-E+Dyn) and plain PPO
  (Sec. 5.2, p. 8).

## Assumptions

- A1: The environment is a fully observable MDP with continuous states and continuous actions (Sec. 3, p. 3).
- A2: The manipulators are serial arms with n ≤ 7 revolute joints from one family: nine Sawyer-based types A-I made by removing joints of the 7-DoF arm, varied in link lengths, damping, friction, armature and link mass (Sec. 4.1, p. 4, Sec. 5.1, p. 5, Table 2, App. B.1.1, p. 14).
- A3: Control is direct joint torque without gravity compensation in simulation; the state is joint angles and velocities zero-padded to 7 joints, and the policy always outputs 7 torques of which the first n are used (Sec. 5.1, p. 5, p. 7).
- A4: Robots with fewer than 7 joints fill the first n state and action slots in chain order and pad the rest with zeros; the 5- and 6-DoF training types are made by removing joints from the 7-DoF arm (e.g. H removes J5), so a slot doesn't keep its joint's role across types, and the policy has to tell the joints apart from v_h (reader's inference, Sec. 4.1, p. 4, Sec. 5.1, p. 5, p. 7, App. B.1.1, p. 14).
- A5: For HCP-E the robot's kinematic chain is known (e.g. from its URDF) and encoded as the concatenated relative poses of consecutive joint frames, zero-padded to 7 joints; dynamics are not encoded (Sec. 4, p. 3, Sec. 4.1, p. 4, Sec. 5.1, pp. 5-6).
- A6: The tasks don't depend heavily on robot dynamics, so policies trained over varied dynamics are robust to them without a dynamics input (Sec. 4.1, p. 3, App. C.1, p. 16, Sec. 6, p. 9).
- A7: The policy is a single MLP conditioned by concatenating v_h to the state, trained on a pool of many robots with one robot sampled per episode: 140 variants per training type for the arms, 1000 hoppers (Sec. 4.3, p. 4, Alg. 1, p. 5, Table 3, Table 4, p. 15).
- A8: Test robots' link lengths and dynamics are sampled from the same ranges as the training robots' (Table 2) (reader's inference, Sec. 5.1.2, p. 6, App. B.1.1, p. 14).
- A9: Arm tasks are goal-conditioned with a sparse ±1 reward within 0.02 m of the goal plus an action penalty, trained with DDPG+HER, which relabels goals in hindsight (Sec. 5.1, p. 6, Alg. 3, p. 13, App. B.1, p. 15).
- A10: Goals are 3-D positions of a point of interest (end effector or peg bottom) in a world frame shared by all robots (Sec. 5.1, p. 5, App. B.1, p. 15).
- A11: Training and evaluation are in simulation (MuJoCo for the arms), except the real Sawyer reacher test (Sec. 5.1, p. 5, Sec. 5.1.2, p. 7).
- A12: The hopper experiments use one 2D hopper topology varied in link lengths, masses, damping, friction and armature, with the Gym dense reward and PPO (Sec. 5.2, p. 8, App. B.2, Table 5, p. 16).
- A13: HCP-I learns a free v_h per training robot by gradient descent on the RL loss; a new robot's v_h is unknown and has to be estimated from interaction by fine-tuning (Sec. 4.2, p. 4, Sec. 5.2, p. 8).
- A14: The real-robot test is on the Sawyer itself, the robot the simulated family is derived from, with gravity compensation on (Sec. 5.1, p. 5, Sec. 5.1.2, p. 7).
- A15: The hardware vector v_h captures the differences between robots that matter for the task, and the policy's behaviour varies smoothly enough with v_h to handle robots between or near the training ones (reader's inference, Sec. 4, p. 3, Sec. 5.2, p. 9, Sec. 6, p. 9).
- A16: Train and test robots do the same task with the same goal distribution and reward; only the hardware changes (reader's inference, Sec. 5.1, pp. 5-6).

## Claims

- C1: Trained on types A-G and I and tested on 100 new robots of those types, HCP-E's success rate rises to about 0.95 (reacher), 0.9 (peg, fixed hole) and 0.7 (peg, random hole) within 2.5e4 episodes while pooled DDPG+HER stays near 0 on all three (values read off Fig. 3; 5 seeds, ±1 std); the authors put the 70% ceiling down to holes outside some robots' workspace, mostly 5-DoF ones [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A15, A16] (Fig. 3, p. 6, Sec. 5.1.1, p. 6).
- C2: Zero-shot on a held-out type, reacher with random goals: HCP-E 92.50 ± 1.96% on type H (trained A-G and I) and 88.00 ± 2.00% on type E (trained A-D and F-I), against DDPG+HER 0.20 ± 0.40% and 2.70 ± 2.19% [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A15, A16] (Table 1, p. 7).
- C3: Zero-shot, peg insertion with a fixed hole: HCP-E 92.20 ± 2.75% on H, 87.60 ± 2.01% on E, and 65.60 ± 3.77% on the 7-DoF type I when trained only on 5- and 6-DoF types (A-H); DDPG+HER gets 0.00, 0.80 ± 0.60 and 0.10 ± 0.30% [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A15, A16] (Table 1, p. 7, Sec. 5.1.2, p. 7).
- C4: Zero-shot, peg insertion with random holes, success is much lower and depends on the held-out type: HCP-E 4.10 ± 1.50% on H, 76.10 ± 3.96% on E, 23.50 ± 4.22% on I, against DDPG+HER at 0.00-0.20% [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A15, A16] (Table 1, p. 7).
- C5: In the low-success random-hole cases (H and I held out), HCP-E still ends with the peg near the hole on most test robots: read off Fig. 5a, median final distance about 0.07 m for HCP-E vs about 0.5 m and 0.4 m for DDPG+HER [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A15, A16] (Fig. 5a, p. 8, Sec. 5.1.2, p. 8).
- C6: Fine-tuning the pretrained HCP-E on the held-out type learns much faster than training from scratch: read off Fig. 5, on H about 0.8 success after 5000 episodes vs about 0.2-0.3 from scratch, and on I about 0.9 after 2500 episodes vs about 0.2 [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A15, A16] (Fig. 5b, Fig. 5c, p. 8).
- C7: On a real Sawyer reacher (20 random goals), the simulation-trained HCP-E policy reaches 75% success zero-shot, while pooled DDPG+HER doesn't get close (median distance about 0.34 m, read off Fig. 4) and a policy trained on the Sawyer CAD model with randomised dynamics ends around 0.05 m (read off); simulation had no gravity compensation but the real robot used it, and HCP-E's motion was jerky [A1, A2, A3, A4, A5, A6, A7, A9, A10, A14, A15, A16] (Fig. 4, p. 7, Sec. 5.1.2, p. 7).
- C8: The fixed-hole peg policy trained on A-G and I (Exp. V) transfers zero-shot to a new 5-DoF type J at 86.30 ± 4.41% (1000 robots) and to a 7-DoF Fetch arm at 10 of 10 trials [A1, A2, A3, A4, A5, A6, A7, A9, A10, A11, A15, A16] (Fig. 9, p. 15).
- C9: On held-out link lengths and dynamics of the training types themselves, HCP-E gets 81-99% on reacher and fixed-hole peg (Exp. I, III, V, VII, IX) but less on random-hole peg (Exp. XI, XIII, XV), down to 6.70 ± 2.53% on type D (Exp. XI), 16.00 ± 3.26% on D (Exp. XV) and 45.70 ± 4.86% on F (Exp. XI), with 54.10-89.60% in Exp. XIII; the authors attribute the 6.70% case to D lacking joints J4 and J5 [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A15, A16] (Table 7, Table 8, Table 9, Table 10, Table 11, Table 12, Table 13, Table 14, pp. 19-20).
- C10: On 1000 training hoppers, HCP-I's reward rises to about 5200 by 2e8 steps, on par with HCP-E+Dyn (about 5200) and about twice HCP-E and PPO (both about 2300-2500), all read off Fig. 6a; the authors conclude the learned v_h captures kinematics and dynamics [A1, A7, A11, A12, A13, A15] (Fig. 6a, p. 9, Sec. 5.2, p. 8).
- C11: Reusing HCP-I's hidden layers on 100 new hoppers (v_h re-estimated) learns faster than from scratch: read off Fig. 6b, about 5000 within roughly 0.3e8 steps vs roughly 1.0-1.5e8 from scratch, with both near 5300-5800 by 2e8 [A1, A7, A11, A12, A13, A15, A16] (Fig. 6b, p. 9, Sec. 5.2, p. 8).
- C12: With only torso mass varied over 1000 hoppers and a 2-d v_h, the learned embeddings order smoothly by torso mass [A1, A7, A11, A12, A13] (Fig. 6c, p. 9, Sec. 5.2, p. 9).
- C13: On 7-DoF arms with varied link lengths and dynamics, HCP-E, HCP-E+Dyn and HCP-I all beat DDPG+HER by a wide margin: read off Fig. 12, reacher about 0.9-0.95 vs under 0.1, peg about 0.85-0.9 (HCP-E), 0.8 (HCP-E+Dyn) and 0.6 (HCP-I) vs about 0.2 [A1, A2, A3, A5, A6, A7, A8, A9, A10, A11, A13, A15, A16] (Fig. 12, p. 18, App. C.3, p. 17).
- C14: When only dynamics vary (7-DoF type I, 100 training robots), DDPG+HER on joint angles and velocities alone reaches about 100% on reacher and fixed-hole peg; adding true dynamics or a random vector of the same size slows learning, which is the paper's case for leaving dynamics out of HCP-E on manipulation [A1, A2, A3, A8, A9, A10, A11, A16] (Fig. 10, p. 16, App. C.1, p. 16).
- C15: A DDPG+HER peg policy trained with damping in [0.01, 2) succeeds on 100% of 100 test robots with damping up to [20, 30), 85% in [30, 40) and 47% in [40, 50), which the authors link to torque limits [A1, A2, A3, A9, A10, A11, A16] (Table 6, Fig. 11, p. 17, App. C.2, p. 17).

## Implementation notes

- Manipulator DDPG+HER: actor and critic learning rate 1e-4, critic weight decay 0.001, hidden layers 128-256-256 with SELU, Adam, γ 0.99, batch 128, 50 warm-up episodes, replay buffer 1e6, 100 training iterations per episode, soft target update τ 0.01, k = 4 future goals, action penalty β 0.1, 200-step episodes, 50 Hz control; actions enter Q at the second hidden layer (source: Table 3, App. B.1.2, pp. 14-15).
- Reward r = sgn(ε − ‖POI − g‖) − β‖a‖², ε = 0.02 m; uniform observation noise in [−0.02, 0.02] on joint angles (rad) and velocities (rad/s) (source: App. B.1, p. 15).
- Hopper PPO: 1000 training hoppers, 8 actors, 2048-step episodes, learning rate 1e-4, hidden 128-128 SELU, Adam, γ 0.99, GAE λ 0.95, clip 0.2, batch 512, 5 epochs per rollout, value coefficient 0.5, entropy coefficient 0.015, v_h dimension 32 (source: Table 4, App. B.2, pp. 15-16).
- Half of the sampled damping values are below 1 (underdamped) and half at or above 1 (source: App. B.1.1, p. 14).
- HCP-E's v_h length for a 7-DoF arm is 8 poses × 6 = 48, zero-padded for fewer joints (source: Sec. 4.1, p. 4, Sec. 5.1, p. 6; the count is the reader's arithmetic).
- The Euler angle convention, the v_h initialisation distribution and its learning rate for HCP-I, and the v_h dimension for the manipulator HCP-I runs are not stated (source: assumption; the authors' code on the project page would settle them).
- On new hoppers only the hidden layers are reused (source: Fig. 6b caption, p. 9).

## Limitations / open questions

- HCP-E encodes kinematics only; the authors note dynamics are usually unavailable or inaccurate, and leave other hardware aspects such as shape and actuation to future work (Sec. 4, p. 3).
- HCP-I can't run zero-shot on a new robot: v_h must be estimated by fine-tuning; a regression from limited interaction to v_h is suggested but not shown, nor are few-shot results (Sec. 5.2, pp. 8-9).
- Random-hole peg zero-shot success is low for some held-out types (C4); the real-robot result is 20 trials on one robot and task (C7).
- Open question: transfer is within one family of Sawyer-derived serial arms with matching joint semantics; how far does a flat concatenated v_h extend to bodies with different topologies, where the per-limb policies of [[pathak2019assembling]] or [[huang2020modular]] apply?
- Open question (survey lens): HCP-I is the closest analogue to conditioning a game policy on a per-game learned embedding, and it inherits the same issue: an unseen game has no embedding until some interaction data is used to fit one.

## Related

- related [[devin2016modular]]: the paper's "most relevant" prior work; robot and task modules per robot, where HCP uses one network conditioned on v_h.
- related [[pathak2019assembling]]: generalises across bodies by sharing per-limb modules over a changing graph instead of conditioning a flat policy on a hardware vector.
- related [[huang2020modular]]: shared per-limb modules with message passing, a later alternative to a hardware-vector input.
- related [[gupta2022metamorph]]: also feeds morphology descriptors to the policy, per limb token in a transformer rather than one vector.
- related [[gupta2017invariant]]: cited as domain adaptation via invariant feature spaces, which needs paired data from the target robot.
- uses [[schulman2017proximal]]: PPO trains the hopper policies and HCP-I's embeddings.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
