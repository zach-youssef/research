---
id: hansen2024tdmpc2
title: "TD-MPC2: Scalable, Robust World Models for Continuous Control"
authors: [Nicklas Hansen, Hao Su, Xiaolong Wang]
year: 2024
venue: ICLR 2024 (arXiv v2 read)
url: https://arxiv.org/abs/2310.16828
arxiv: 2310.16828
doi:
pdf_sha256: 99832c555b66665ad1e26cd48e75ea9dda4f3750175b993e7ed5b8dfe399b9e0
tags: [world-model, model-based-rl, decoder-free, latent-planning, mpc, multi-task, continuous-control, scaling]
status: read
read: 2026-11-26
---

# TD-MPC2: Scalable, Robust World Models for Continuous Control

## TL;DR

Decoder-free latent world model (MLP encoder, dynamics, reward, value, policy prior) trained by
joint-embedding prediction (stop-gradient, no EMA encoder), reward and TD losses, planned with MPPI
in latent space. One hyperparameter set covers 104 continuous tasks; one 317M agent does 80.

## Problem

Generalist embodied agents need algorithms that consume large mixed-quality (not near-expert)
datasets and don't need per-task tuning; prior TD-MPC got worse, not better, when scaled up, and
sequence-model generalists assume expert demonstrations and discretised actions (Sec. 1, Fig. 1,
pp. 1-2; Sec. 6, p. 10). The paper revisits TD-MPC for robustness and multitask scaling.

## Method

- Five MLP components with a learnable task embedding e: encoder h (state to latent z), latent
  dynamics d, reward R, terminal value Q, and a policy prior p (Eq. 2, Fig. 3, p. 3).
- Model loss (Eq. 3, p. 4), summed over a horizon H=3 with weight lambda^t: joint-embedding
  prediction, the paper's own label for the term in Eq. 3 (squared error between d's prediction and
  the stop-gradient encoding of the next state by the same online encoder, with no EMA target
  encoder; only the Q target is an EMA), so the encoder is not trained by a decoder, reward cross-entropy and TD value cross-entropy (rewards and
  values are regressed as discrete classes in log space). Observations are never reconstructed
  (Sec. 3.1, pp. 3-4).
- Policy prior: maximum-entropy actor trained through the model (Eq. 4, p. 4). Q is an ensemble of 5
  (up to 8 in the 317M model) with 1% dropout; targets use the minimum of two random members (Sec. 3.1, App. A, p. 4, p. 16).
- SimNorm: the latent is split into groups and a softmax is applied per group (Eq. 5, p. 4;
  App. H, pp. 28-29).
- Planning (Sec. 3.2, Eq. 6, pp. 4-5): MPPI over Gaussian action sequences rolled out in latent
  space, with the Q value at the horizon as bootstrap, some samples from the policy prior and a
  warm start from the previous step; only the first action is executed.
- Multitask (Sec. 3.3, p. 5): task embeddings (norm at most 1) condition all components;
  observations and actions are zero-padded to the largest dimensions and invalid action dimensions
  are masked in the policy prior and the planner, so tasks with different embodiments share one model.

## Assumptions

- A1: Actions are continuous; discrete action spaces are not handled and MPPI planning is for continuous actions (Sec. 5, p. 9, App. I, p. 29).
- A2: A scalar reward is available during training, and tasks are defined by it; the learned model predicts it (Sec. 5, p. 9, Eq. 3, p. 4).
- A3: Single-task results use online interaction with a simulator, with a replay buffer; multitask results train offline on replay buffers of 240 single-task agents (545M transitions, 80 tasks) (Sec. 3.1, Sec. 4.1, p. 4, p. 7).
- A4: Episodes have fixed length with no termination conditions, and success counts only if the final step is successful (App. C, Table 6, p. 20).
- A5: The task id is known to the agent and conditions every component through a learned embedding (Sec. 3.3, p. 5).
- A6: Observations are mostly low-dimensional state vectors; the image-based results use a 4-layer conv encoder on 64x64 images (Sec. 4.1, pp. 8-9, App. H, p. 29).
- A7: Baselines use per-task hyperparameters where tuned (SAC, TD-MPC); TD-MPC2 uses UTD 1 and batch 256, DreamerV3 uses UTD 512 (Sec. 4, p. 6).

## Claims

- C1: A single hyperparameter set gives TD-MPC2 higher data efficiency and asymptotic performance than SAC, DreamerV3 and TD-MPC across 104 continuous-control tasks from DMControl (39), Meta-World (50), ManiSkill2 (5) and MyoSuite (10), including Humanoid/Dog (action dim up to 38) and MyoSuite (39) [A1, A2, A3, A4, A6, A7] (Fig. 1, Fig. 4, p. 1, pp. 5-6, App. C, pp. 18-19).
- C2: TD-MPC (the predecessor) gets worse as it grows from 1M to about 48M parameters on the 80-task set, while TD-MPC2's normalized score rises 16.0 (1M), 49.5 (5M), 57.1 (19M), 68.0 (48M), 70.6 (317M); the authors say it looked roughly linear in log parameters and had not saturated at 317M [A1, A2, A3, A5, A6] (Fig. 1, Fig. 7, p. 1, p. 7).
- C3: The multitask training cost on one RTX 3090 is 3.7 GPU-days for 1M parameters and 33 GPU-days for 317M [A3, A6] (Table 1, p. 7).
- C4: Fine-tuning a 19M agent pretrained on 70 tasks to 10 held-out tasks for 20k environment steps scores 47.0 against 24.0 from scratch (the 19M from-scratch baseline is no better than a 5M one; gains are task dependent, with little or no gain on some tasks such as Hopper Hop and Walker Run in Fig. 17); each new task's embedding is initialised from a hand-picked similar training task [A1, A2, A3, A5, A6] (Fig. 8, Fig. 17, Table 7, pp. 7-8, p. 24).
- C5: Learned task embeddings place semantically similar tasks (Door Open and Door Close) close together; the authors read the similarity as following task dynamics (embodiment, objects) more than the objective (walk, run) [A1, A3, A5, A6] (Fig. 7, p. 7, Fig. 19, p. 25).
- C6: In ablations on the 80-task 19M model, planning plus the policy prior scores 54.2 against 53.7 planning only and 42.2 policy only; LayerNorm plus SimNorm 54.2 vs SimNorm only 51.0 vs none 46.8; discrete regression 54.2 vs continuous 49.6; 2/5/10 Q-functions 53.5/54.2/57.0; normalized task embeddings 54.2 vs 46.6 [A1, A2, A3, A5, A6] (Fig. 9, p. 8, Fig. 18, p. 25).
- C7: TD-MPC's gradient norm grows by many orders of magnitude on tasks like Walker Stand, Dog Trot and Quadruped Run (the log axis reaches about 1e12 for Dog Trot), and it diverges on some tasks, while TD-MPC2's stays small (one seed per task shown) [A1, A2, A6] (Fig. 21, App. G, p. 26, Fig. 12, p. 21).
- C8: With a 64x64 image input, a 4-layer conv encoder, and unchanged hyperparameters, TD-MPC2 is described as comparable to the two best baselines, DrQ-v2 and DreamerV3, on 10 DMControl tasks (CURL and SAC are also plotted; no claim of superiority) [A1, A2, A6] (Fig. 10, pp. 8-9, App. H, p. 29).
- C9: The latent model is decoder-free and reward-aware: it is trained for outcome (reward, value) and the joint-embedding prediction term only, and the authors argue this lets modest model sizes cover many tasks where reconstruction-based models spend capacity on observations; they note it may suit large visual variation better than reconstruction, a point they support only by citing prior work [A2] (Sec. 3.1, p. 3, Sec. 5, p. 9).
- C10: On ManiSkill2 Pick YCB (all 74 objects, 14M steps), TD-MPC2 reaches up to about 60% success (the mean curve fluctuates between about 40% and 60%) while the other methods fail to learn within the budget; SAC and DreamerV3 are described as prone to numerical instabilities on Dog tasks and significantly less data-efficient on Humanoid [A1, A2, A3, A4] (Fig. 5, Fig. 6, Fig. 14, p. 6, p. 23).
- C11: A planner-level uncertainty penalty (an ensemble-based penalty scaled by the mean value, test-time only, not used elsewhere) raises the 80-task offline score of a 19M model from 56.5 to 62.0 at c=0.01 and lowers it to 44.1 at c=0.1 [A1, A2, A3, A5, A6] (Table 10, App. J, p. 30).
- C12: The authors say DreamerV3 has delivered strong results on discrete-action domains (Atari, Minecraft) while TD-MPC2 is significantly better on difficult continuous control; extending TD-MPC2 to discrete actions is an open problem, and they suggest MCTS as in MuZero without testing it [A1] (Sec. 5, p. 9, App. I, p. 29).

## Implementation notes

- Planning: horizon 3, 6 iterations (8 if action dim is at least 20), population 512, 24 policy-prior samples, 64 elites, std in [0.05, 2], temperature 0.5, no momentum (source: Table 8, p. 27).
- Optimisation: Adam, LR 3e-4 (encoder 1e-4), batch 256, UTD 1, gradient clip 20, joint-embedding coefficient 20, reward/value coefficients 0.1, lambda 0.5, Q momentum 0.99, replay buffer 1M uniform (source: Table 8, p. 27).
- Architecture for the 5M default: encoder dim 256, MLP dim 512, latent 512, task embedding 96, 101 reward/value bins, SimNorm V=8, tau=1 (source: Table 8, p. 27; App. H, p. 26).
- Model sizes 1M/5M/19M/48M/317M: encoder dims 256/256/1024/1792/4096, latent 128/512/768/768/1376, encoder layers 2/2/3/4/5; multitask runs use a 4x larger batch (1024) (source: Table 9, p. 28; Sec. 4.1, p. 7).
- Discount: gamma = clip((T/5 - 1)/(T/5), 0.95, 0.995) from the expected episode length T after action repeat; seed steps max(5T, 1000) (source: Eq. 7, Eq. 8, p. 28).
- Action repeat 2 (1 for MyoSuite), episode lengths 1000/200/200/100 (source: Table 6, p. 20).
- Code, 300+ checkpoints and datasets: tdmpc2.com (source: Sec. 1, pp. 1-2); the commit is not given in the paper (source: assumption: not checked).
- Wall-clock for single-task runs and planning latency are not reported (only Table 1's multitask GPU-days) (source: assumption).

## Limitations / open questions

- Authors: discrete actions are unsupported; gains of multitask pretraining likely need orders of magnitude more tasks; fine-tuning is only a preliminary result and is task dependent; reward misspecification and data concentration are risks (Sec. 5, p. 9; App. E, p. 24).
- Authors: image-based experiments use 64x64 inputs to cut cost; larger inputs are only cited from prior work (App. H, p. 29). Reader's note: every environment is a simulator with a dense or sparse reward.
- Open question: a shmup's joystick and buttons are discrete; would a continuous relaxation or MCTS planner keep the latent model useful?
- Open question: the latent is shaped by reward and TD targets, so it is task-supervised, not an unsupervised cross-game embedding; whether it transfers across games with different rewards is untested (the 80 tasks share physics engines and similar embodiments).
- Open question: planning costs H=3, 512 samples, 6 iterations per action, so each decision needs hundreds of model rollouts; a transformer policy would need a distilled (policy-prior-only) mode to run at emulator speeds (policy-only ablation scores 42.2 vs 54.2, Fig. 9).

## Related

- related [[hafner2024mastering]]: the DreamerV3 baseline; reconstruction-based latent model plus actor-critic, strong on discrete actions where TD-MPC2 is weaker.
- related [[srinivas2020curl]]: the CURL baseline in the visual-RL comparison.
- related [[reed2022generalist]]: Gato, the sequence-model generalist the paper contrasts with (expert demonstrations, discretised actions; Sec. 6, p. 10).
- related [[lee2022multigame]]: Multi-Game Decision Transformer, cited with Gato as expert-data-dependent scaling (Sec. 6, p. 10).
- related [[assran2025vjepa]]: also plans by MPC in a learned latent space, but trains its encoder with no rewards and on video; TD-MPC2's Eq. 3 labels its latent term "joint-embedding prediction", with the reward and value losses alongside it (reader's inference: these keep the latent from collapsing).

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
