---
id: zakka2021xirl
title: "XIRL: Cross-embodiment Inverse Reinforcement Learning"
authors: [Kevin Zakka, Andy Zeng, Pete Florence, Jonathan Tompson, Jeannette Bohg, Debidatta Dwibedi]
year: 2021
venue: CoRL 2021 conference paper (oral)
url: https://arxiv.org/abs/2106.03911
arxiv: 2106.03911
doi:
pdf_sha256: 683b42c1b58738f7a18222963da9409901be4f02ce358209e26e4d18b259a940
tags: [inverse-reinforcement-learning, imitation-learning, reward-learning, self-supervised, cross-embodiment, video, temporal-alignment]
status: read
read: 2026-11-21
---

# XIRL: Cross-embodiment Inverse Reinforcement Learning

<!-- Read from arXiv v3 (2021-12-13), 22 PDF pages, all read: main paper pp. 1-8, references pp. 9-10 (skimmed), appendices A-G pp. 11-22. -->

## TL;DR

XIRL trains a TCC image encoder on action-free demo videos from several embodiments of one task and
rewards SAC with the negative distance to the mean goal-frame embedding; on held-out X-MAGICAL agents
it learns faster than TCN, LIFS and goal-classifier rewards, and human videos teach a simulated arm.

## Problem

Learning a task from third-person videos of other agents (humans, other robots) faces an
embodiment gap: different shapes, end-effectors and dynamics lead to different strategies and
trajectories, the demonstrations carry no actions, and frame-to-frame correspondences between
demonstrator and learner are hard to label (Sec. 1, p. 1; Sec. 3.1, p. 3). Earlier learned
rewards from self-supervised embeddings (TCN, LIFS) align to a reference trajectory by time
index, which fails when embodiments work at different speeds or orderings, and RLV needs
labelled expert-policy state pairs (Sec. 2, pp. 2-3). XIRL wants a reward that is invariant to
the embodiment, learned without labels or correspondences.

## Method

- Data: D = union of per-agent sets D_i of observation-only videos of successful demonstrations
  of one task T; embodiment ids are not used, so all agents' videos are merged into D_all
  (Sec. 3.1, p. 3; Sec. 3.2, p. 4).
- Representation (Sec. 3.2, p. 4): an image encoder φ trained with TCC ([[dwibedi2019cycle]]):
  for a random frame t of video i, the soft nearest neighbour in video j, then the expected
  cycle-back index μ; the loss is the MSE (μ - t)², summed over frames and video pairs. The
  final configuration is the MSE regression loss, not variance-aware, with normalised time
  indices, L2 distance and softmax temperature 0.1 (Table 6, p. 21).
- Encoder: ImageNet-initialised ResNet-18 with the head replaced by a 32-d embedding layer, on
  224×224 images (Sec. 4.3, p. 6); 64-d for Puck Pushing and X-REAL (Table 6, p. 21).
- Reward (Sec. 3.3, p. 5): goal embedding g = mean of φ(last frame) over all demonstration
  videos; r(s) = -(1/κ)·||φ(s) - g||², with κ the average distance from the demos' first-frame
  embeddings to g. It is dense, needs no reference trajectory, and lets trajectories of any
  length map to a common notion of progress.
- RL (Sec. 3.4, p. 5; Sec. 4.3, p. 6): the frozen encoder defines the reward of an MDP for any
  agent, including unseen ones; no environment reward is used. The policy is SAC on the
  environment's state vector (3 stacked 16-d states, 48-d), while the encoder sees rendered
  top-down RGB frames (Sec. 4.1, p. 6; App. B.1, p. 12; App. D.3, p. 17).
- X-MAGICAL benchmark (Sec. 4.1, pp. 5-6; App. B, p. 12): a 2D PyMunk sweeping task (push 3
  debris into a goal zone; reward = fraction of debris in the zone at episode end) with four
  embodiments, short-, medium-, long-stick and a two-armed gripper, 1000 SAC-oracle demos each;
  horizon 50 for long-stick, 100 otherwise; 8 Hz control.
- Baselines (Sec. 4.2, p. 6; App. D.1.1, p. 16): frozen ImageNet ResNet-18 features, a goal
  frame classifier (last frames positive), LIFS (contrastive plus autoencoding, time-aligned
  pairs) and single-view TCN, the latter two using the same goal-distance reward as XIRL.
- Settings: same-embodiment (the learner's embodiment is in D), cross-embodiment (leave one
  agent out, train on the other three), and real-to-sim (human videos to a simulated Sawyer on
  RLV's State Pusher); 50 eval episodes, mean over 5 seeds, ±0.5 std shading (Sec. 5, p. 7).
- X-REAL (App. C, pp. 12-14): 93 real videos of 9 end-effectors (hands and hand-held tools)
  moving five pens into a mug and then a cup, recorded at 1920×1080, 30 fps.

## Assumptions

- A1: Demonstrations are observation-only videos (no actions) of agents successfully performing one task T; X-MAGICAL demos are rollouts of an oracle SAC policy trained on the ground-truth reward, with unsuccessful ones discarded (Sec. 3.1, p. 3, App. B.3, p. 12).
- A2: Demonstrations of different embodiments share semantic temporal correspondences: however differently the agents solve the task, a common set of states or frames corresponds in time (Sec. 3.2, p. 4).
- A3: Task progress runs in one order through every demonstration, so cycle-consistent frame matching recovers progress; the paper criticises time-index heuristics for non-monotonic tasks but doesn't test TCC on them (reader's inference, Sec. 2, p. 3, Sec. 3.2, p. 4).
- A4: The goal is one visual state: the mean embedding of all demos' last frames, with demos ending at or near task completion; the reward is the negative squared distance to it, scaled by the mean first-frame distance κ (Sec. 3.3, p. 5, App. D.1.1, p. 16).
- A5: Task progress shows in task objects shared by all embodiments (debris and goal zone), so an encoder can ignore the agent's body, which differs across embodiments and appears in every frame (reader's inference, Sec. 3.2, p. 4, App. B.1, p. 12).
- A6: Demonstrators and learner are seen from the same fixed camera: an allocentric top-down RGB view of the whole workspace in X-MAGICAL, a tripod-mounted GoPro in X-REAL (App. B.1, p. 12, App. C.1, p. 14).
- A7: In X-MAGICAL, demonstrators and learner act in the same environment, with the same objects, goal zone and randomised initial-state distribution; only the agent changes (Sec. 4.1, pp. 5-6, App. B.2, p. 12).
- A8: The X-MAGICAL embodiments form one narrow family of 2D agents: three sticks that differ only in length and a circular gripper, all with force/torque control of forward/backward translation and rotation, plus an open/close action for the gripper (Sec. 4.1, pp. 5-6, App. B.1, p. 12).
- A9: The training embodiments cover the held-out embodiment's appearance and strategy well enough for the reward to transfer; each cross-embodiment test leaves out one of the four agents and trains on the other three (reader's inference, Sec. 5.2, p. 7).
- A10: The learner can bring the scene to the demos' goal state, even if it gets there differently (reader's inference, Sec. 3.3, p. 5).
- A11: Demonstration data are plentiful in X-MAGICAL: 1000 demos per embodiment, with random debris positions (Sec. 4.1, p. 6, App. B.3, p. 12).
- A12: The encoder is an ImageNet-pretrained ResNet-18 with a 32-d embedding (64-d for Puck Pushing and X-REAL), on 224x224 frames with crop, colour-jitter, grayscale and blur augmentation, trained with the TCC MSE regression loss on 40 sampled frames per video (30 for X-REAL) (Sec. 4.3, p. 6, App. D.1.2, pp. 16-17, Table 6, p. 21).
- A13: The encoder is frozen during RL, and its reward replaces the environment reward entirely, so the learned reward has to generalise to states the successful-only demos never show (Sec. 3.4, p. 5).
- A14: The policy gets a low-dimensional privileged state, not pixels: positions, orientation and debris distances stacked over 3 steps (48-d), or 15-d for Puck Pushing; pixels are only used to compute the reward (Sec. 4.1, p. 6, App. B.1, p. 12, App. D.3, p. 17).
- A15: Policies are learned with SAC on continuous actions at 8 Hz, with horizons of 50 (long-stick) or 100 steps and per-embodiment budgets of 75k-500k steps (200k for the Sawyer arm) (Sec. 4.3, p. 6, App. B.1-B.2, p. 12, App. D.2, p. 17, Table 7, p. 22).
- A16: The main task is one long-horizon 2D PyMunk sweeping task: push three debris into a fixed pink zone, with the agent spawned below the debris; one encoder is trained per task (Sec. 4.1, p. 5, App. B.2, p. 12).
- A17: The real-to-sim test uses RLV's State Pusher dataset of human videos and its simulated Sawyer arm; XIRL (sim only) uses 5 teleoperated simulated trajectories, and the RLV curve is copied from [6], which used a different SAC implementation. Drawer Opening, shown only in the reward plots, is another RLV task, and the paper doesn't say what its encoder was trained on (Sec. 5.3, p. 7, Fig. 5, p. 8, Sec. 5.4, p. 7, Fig. 7, p. 8).
- A18: Evaluation is the success rate (fraction of debris in the zone for Sweeping) of the mean policy action over 50 episodes every 5000 steps, plotted as the mean over 5 seeds with ±0.5 std shading (Sec. 5, p. 7, App. D.3.1, p. 18).
- A19: Policies are trained and evaluated in simulation only; no policy is learned on a real robot (Sec. 6, p. 8).
- A20: X-REAL has 93 videos of 9 end-effectors (a human hand and tools) doing one multi-step task, moving five pens into one cup and then another; the encoder is trained on all embodiments and shown on validation demos (App. C, pp. 13-14, Table 2, p. 13).
- A21: In the real-to-sim test, the real human videos and the simulator renders show the same task objects from a similar viewpoint, so an encoder trained only on real frames still measures progress on simulated frames despite the visual gap and the change of body (reader's inference, Sec. 5.3, p. 7, Fig. 5, p. 8).

## Claims

- C1: In the same-embodiment setting on X-MAGICAL sweeping with SAC, the authors find the XIRL reward more sample-efficient than the other learned rewards (ImageNet, goal classifier, TCN, LIFS); in the curves TCN roughly matches it on long-stick [A1, A2, A3, A4, A5, A6, A7, A8, A10, A11, A12, A13, A14, A15, A16, A18, A19] (Fig. 3, Sec. 5.1, pp. 6-7).
- C2: With the learner's embodiment held out of the demonstrations, XIRL generalises to the new agent better than TCN, LIFS and goal-classifier rewards, except on long-stick, where TCN and LIFS also reach about 1.0; on short-stick and gripper it ends near 0.7 success while the baselines stay below about 0.5 (curves read from the figure) [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A18, A19] (Fig. 4, Sec. 5.2, p. 7).
- C3: On RLV's State Pusher, XIRL trained only on real human videos, without paired frames, learns faster than RLV (whose curve is copied from [6], with a different SAC implementation) and than the sparse environment reward; XIRL trained on 5 simulated teleoperated trajectories is fastest [A1, A2, A3, A4, A5, A10, A12, A13, A14, A15, A17, A18, A19, A21] (Fig. 5, Sec. 5.3, pp. 7-8).
- C4: The XIRL reward tracks the ground-truth reward on successful and unsuccessful demos of Sweeping, State Pusher and Drawer Opening, and gives partial reward to a Drawer Opening collision trajectory that gets zero environment reward (qualitative) [A1, A2, A3, A4, A5, A6, A8, A9, A12, A16, A17, A21] (Fig. 6, Fig. 7, Sec. 5.4, pp. 7-8).
- C5: Against the ground-truth environment reward, XIRL is as or more sample-efficient in both same- and cross-embodiment settings, which the authors attribute to its denser progress signal (in the same-embodiment gripper plot XIRL trails ground truth for most of training) [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A18, A19] (Fig. 13, App. E.1, p. 18).
- C6: SimCLR pretrained on the X-MAGICAL demos performs poorly as a goal-distance reward; ImageNet SimCLR does much better on long-stick but both are poor on medium-stick, where XIRL is far ahead; the authors conclude both pretraining on cross-embodiment demos and TCC's inductive bias are needed [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A18, A19] (Fig. 14, Fig. 15, App. E.2, pp. 18-19).
- C7: X-MAGICAL embodiments solve the task with different strategies, state coverage and speed: mean demo length ranges from 48.0 ± 28.5 frames (long-stick) to 110.8 ± 39.8 (gripper) [A1, A8, A11, A16] (Table 1, p. 5, Fig. 2, Sec. 4.1, pp. 4-6).
- C8: On X-REAL (9 embodiments, demo lengths 10.8 ± 1.8 s to 38.1 ± 9.2 s), XIRL rewards separate successful from failed validation demos, e.g. a human demo that leaves the pens in the first cup gets about half the successful demo's final reward (qualitative) [A1, A2, A3, A4, A5, A6, A12, A20] (Table 2, Fig. 12, App. C, pp. 13-15).
- C9: In t-SNE, XIRL embeddings of different demos overlap and their distance to the goal follows task progress, while the goal classifier's embedding has much less structure (qualitative) [A1, A2, A3, A4, A6, A8, A11, A12, A16] (Fig. 16, App. F.1, pp. 19-20).
- C10: Cost on one RTX 2080 Ti: 25-30 min per representation run; one SAC seed takes about 0.5 h (long-stick, 75k steps) to 4 h (short-stick and gripper, 500k steps), with up to 5 seeds in parallel [A12, A15, A18] (App. D.1, App. D.2, pp. 16-17).

## Implementation notes

- Encoder learning rate conflicts: 1e-5 in the text, initial 1e-6 decaying to 0 in Table 6, whose range considered is 1e-5 to 1e-3 (source: Sec. 4.3, p. 6, Table 6, p. 21); which was used is unresolved (source: assumption).
- Encoder training: Adam β1 = 0.99, β2 = 0.999, weight decay 1e-5, 8k iterations, batch size 4, uniform frame sampler with 40 frames per video (X-MAGICAL, Puck Pushing) or 30 (X-REAL), embeddings not normalised, fixed temperature 0.1 (source: Table 6, p. 21, App. D.1.3, p. 17).
- The temperature 0.1 is not in the Sec. 3.2 equations, which use plain squared distances (source: Sec. 3.2, p. 4, Table 6, p. 21).
- Augmentation (Albumentations): random resized crop (area 0.6-1.0, aspect 0.75-1.33, 224×224), colour jitter (0.4, 0.4, 0.1, 0.1, p = 0.8), grayscale (p = 0.2), Gaussian blur (kernel 13, σ 1-2, p = 0.2), /255; the same sample for all frames of one video (source: App. D.1.2, pp. 16-17).
- Whether φ sees one frame or a stack of 3 frames is unclear: Sec. 3.2 says it ingests an image, Fig. 8 shows a 3-frame image stack (source: Sec. 3.2, p. 4, App. B.1, p. 12).
- In the cross-embodiment runs, g and κ are taken from the three training agents' demos only (source: assumption; Sec. 3.3 defines them over D_all).
- SAC: clipped double Q, actor and critic MLPs with 2 hidden layers of 1024 ReLU units (the text says 3-layer MLPs), orthogonal init, Adam lr 1e-4, γ 0.99, minibatch 1024, buffer 1e6, 5000 random seed steps, one update per step, critic target EMA 0.005, actor and target updates every 2 steps, log-std bounds [-5, 2], learnable entropy temperature from 0.1 (source: App. D.2.1, p. 17, Table 7, p. 22).
- Training steps: 75k long-stick, 225k medium-stick, 500k short-stick and gripper, 200k Sawyer (source: Table 7, p. 22).
- Demonstrations come from SAC oracle policies trained on the ground-truth reward, with unsuccessful rollouts discarded (source: App. B.3, p. 12).
- Baselines' hyperparameters, e.g. TCN positive/negative windows 1 and 4, LIFS temperature 0.1 with early stopping (source: Table 3, Table 4, Table 5, App. D.1.1, p. 16).
- Code and datasets are said to be at https://x-irl.github.io (source: abstract, p. 1); not inspected.

## Limitations / open questions

- No policy has been learned on a real robot yet; the real-data results are reward plots and
  real-to-sim transfer (Sec. 6, p. 8; App. C.2, p. 14).
- The demos contain only successful trajectories, so the encoder must generalise to states the
  learner reaches while exploring (Sec. 3.4, p. 5).
- One encoder and one goal embedding per task; the reward is distance to a single mean goal
  state (Sec. 3.3, p. 5).
- The RL policy observes ground-truth state vectors; the learned visual representation is used
  only for the reward (App. D.3, p. 17).
- The baseline curve for RLV comes from another SAC implementation (Fig. 5 caption, p. 8).
- Table 6 lists ranges searched (loss type, distance, embedding size, frame count), but the paper
  reports no per-choice ablation results (Table 6, p. 21).
- Open question (reader): games often have no single goal frame (scores, endless or looping
  play); would a progress-aware embedding still give a useful reward without a fixed goal state?
- Open question (reader): the paper says the reward is agnostic to the RL algorithm (Sec. 4.3,
  p. 6) but only tests SAC; would a dense, possibly exploitable distance reward behave under PPO
  with a pixel transformer policy, and does the policy hack the embedding?
- Open question (reader): XIRL transfers across embodiments of one task; transfer across games
  (different tasks and visuals) is untested, and the SimCLR result (C6) suggests the in-domain
  demos and TCC objective both matter.

## Related

- uses [[dwibedi2019cycle]]: TCC (MSE cycle-back regression variant) trains the embodiment-invariant encoder.
- related [[serrano2024crossdomain]]: reviews XIRL as [53], a reward-shaping transfer method usable by on-policy learners ([[serrano2024crossdomain#C3]]).
- related [[schulman2017proximal]]: XIRL uses SAC, but its reward is meant to plug into any RL algorithm such as PPO.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
