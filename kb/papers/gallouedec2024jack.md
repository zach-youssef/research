---
id: gallouedec2024jack
title: "Jack of All Trades, Master of Some, a Multi-Purpose Transformer Agent"
authors: [Quentin Gallouédec, Edward Beeching, Clément Romac, Emmanuel Dellandréa]
year: 2024
venue: ARLET 2024 workshop (arXiv v3, 10 Jul 2024)
url: https://arxiv.org/abs/2402.09844
arxiv: 2402.09844
doi:
pdf_sha256: d1d863581c2c973363a511622ffbeb4dd4938eaae2a09848ec3690b06142768c
tags: [generalist-agent, transformer, imitation-learning, multi-task, observation-prediction, atari, metaworld, mujoco, babyai, open-source]
status: read
read: 2026-11-24
---

# Jack of All Trades, Master of Some, a Multi-Purpose Transformer Agent

## TL;DR

JAT is a 12-layer behavior-cloning transformer trained on Atari 57, BabyAI, Meta-World, MuJoCo, text
and captioning with one set of weights, released openly with its data. An auxiliary next-observation
MSE loss (weight kappa) helped on a 6-task subset: 99.1 +/- 0.4% at kappa = 0.005 vs 94.5 +/- 1.1%.

## Problem

Gato (Reed et al. 2022) is the only prior model covering RL, vision and language together, and it is
closed. The paper asks how to design a model, a loss balance and a training strategy that cover very
different modalities and tasks without favouring one (Sec. 1, p. 1). It also tests the open question
left by Decision Transformer and Gato of whether predicting observations helps action learning
(Sec. 3.1.3, p. 5, Sec. 4.3, p. 8).

## Method

- Each RL timestep gets one observation embedding (reward appended or added) and one action
  embedding, interleaved: [phi(s0, 0), phi(a0), phi(s1, r1), phi(a1), ...] (Sec. 3.1.1, Fig. 1, p. 4).
  This widens the attention window in timesteps, 19x vs Gato for Atari and over 25x for Meta-World
  (Sec. 1, p. 2).
- Encoders: continuous vectors are padded to 377 and passed through a linear layer; discrete
  observations through a lookup table; images are resized to 84x84 (bicubic), normalized, padded to
  4 channels and encoded by three blocks of conv, instance norm and attention (Sec. 3.1.1, p. 4).
  Text uses the GPT-2 BPE tokenizer (50,257 tokens); captioning images go through ViT-style 16x16
  patches at 224x224 (Sec. 3.1.1, p. 5).
- Backbone: EleutherAI GPT-Neo with Longformer-like attention, global window 512 tokens and local
  window 256, 12 layers, 12 heads, hidden 768, FFN 8192, causal (Sec. 3.1.2, p. 5).
- Outputs: a transposed CNN decodes image embeddings (pixel prediction), a linear layer decodes
  continuous vectors, both with MSE; discrete values (Atari and BabyAI actions, text) use a shared
  projection with cross-entropy. The sequence loss averages the per-element losses (Sec. 3.1.3, p. 5).
- Observation-prediction objective: L = kappa * L_obs + (1 - kappa) * L_act (Eq. 1, Sec. 4.3, p. 8).
  Observations are predicted in their native form (Atari frames as pixels via the transposed CNN),
  not as discrete tokens, and there is no separate tokenizer or world-model stage.
- Training: 250,000 steps, 8 V100 GPUs, about 9 days, batch 20 x 2 accumulation steps x 8 GPUs = 320,
  AdamW (0.9, 0.999, eps 1e-8), learning rate 5e-5 decayed linearly to 0; each batch comes from one
  dataset (Sec. 3.3.1, p. 6).
- Heuristic weights: sample weight 10.0 for Oscar, Conceptual Captions and Wikipedia, 1.0 otherwise;
  loss weight 10.0 for MuJoCo tasks except Pendulum 20.0 and Double Pendulum 50.0, and 50.0 for
  Meta-World (Sec. 3.3.2, p. 6).
- Data: expert PPO agents (Sample Factory) per Atari game (2B steps each, roughly 500k interactions
  per game in the dataset), per Meta-World task and per MuJoCo task; BabyAI bot demonstrations
  (100,000 episodes over 39 settings); 10,000 episodes per Meta-World and MuJoCo task; plus Oscar,
  Conceptual Captions, OK-VQA and Wikipedia (App. B.1, p. 22, App. B.1.1, p. 27).
- Evaluation: no prompt; 10 episodes per task during training, 100 for the final checkpoint; scores are
  normalized by the dataset's expert score and aggregated by domain (Sec. 4.2, p. 7, App. A, p. 15).

## Assumptions

- A1: The model is trained by plain behavior cloning on expert-agent trajectories, with no return conditioning (Sec. 2.1, p. 2, App. B.1, p. 22).
- A2: Atari observations are 84x84 images fed to the model as pixels, and Atari actions are discrete (Sec. 3.1.1, p. 4).
- A3: The observation-prediction ablation uses a subset of 6 tasks (Freeway, Pong, ButtonPressWall, WindowClose, Ant, DoubleInvertedPendulum), so only 2 of the 6 have image observations (Sec. 4.3, p. 8).
- A4: The ablation's models share the main training setup (architecture, optimizer, data), varying only kappa (reader's inference, Sec. 4.3, p. 8).
- A5: Scores are normalized by the dataset's own expert score and random-agent score, not by Gato's, so comparisons with Gato are indicative only (Sec. 4.2, p. 8).
- A6: Reward is encoded in the observation embedding, which the authors rely on to identify the task without a prompt (Sec. 3.1.1, p. 4, App. D, p. 29).
- A7: Expert data comes from PPO agents that exceed human level on 43 of 57 Atari games and fall below it on 14; training is dominated by those experts' quality (App. B.1, p. 22).
- A8: Task and loss weights are chosen heuristically, not tuned (Sec. 3.3.2, p. 6).
- A9: Evaluation counts are 100 episodes per task for final results; the paper does not state the number of training seeds (Sec. 4.2, p. 7, Fig. 6, p. 9).

## Claims

- C1: On 6 tasks, adding the auxiliary observation loss at kappa = 0.005 gives a near-optimal expert-normalized score of 99.1 +/- 0.4%, vs 94.5 +/- 1.1% when observations are not predicted [A1, A2, A3, A4, A8, A9] (Sec. 4.3, p. 8, Fig. 6, p. 9)
- C2: The observation-loss weight matters: kappa = 0.5, the highest tested, hinders learning, small kappa values behave like the base score, and the best value is near 0.005; the paper tested kappa in {0, 0.0005, 0.005, 0.05, 0.5} (read off Fig. 6) [A1, A3, A4, A8, A9] (Sec. 4.3, p. 8, Fig. 6, p. 9)
- C3: The authors conclude that joint observation prediction improves sequential decision-making performance, which they present as addressing an open question of Decision Transformer and Gato, provided the loss is balanced [A1, A3, A4, A9] (Sec. 3.1.3, p. 5, Sec. 4.3, p. 8, Sec. 5, p. 9)
- C4: One timestep costs two embeddings (observation, action), giving a 19x longer attention window in timesteps than Gato on Atari and more than 25x on Meta-World [A1, A2] (Sec. 1, p. 2, Sec. 3.1.1, p. 4)
- C5: JAT is more than 6 times smaller than Gato (1.18B) and uses a much lower training budget: 250k steps, 8 V100, about 9 days [A1, A5] (Sec. 1, p. 2, Sec. 3.3.1, p. 6, Sec. 4.2, p. 8)
- C6: On Atari 57, JAT's final agent reaches 14.1% of expert score, 37.6% of human performance and above human in 21 games; the paper's reported average normalized score is 31.1% vs Gato's 30.9% [A1, A2, A5, A7] (Sec. 4.2, pp. 7-8, Fig. 5, p. 7, Table 1, p. 16)
- C7: On BabyAI JAT scores 99.0% on the aggregate used and 86.2% average (Gato 93.2%, with 39 tasks vs Gato's 46); only Move Two Across S8N9 is below 50% [A1, A5, A6] (Sec. 4.2, pp. 7-8, Table 2, p. 18)
- C8: On Meta-World JAT scores 65.5% (average 62.8%, Gato 87.0%) with some tasks such as Basketball not learned at all, and on MuJoCo 84.8% (average 73.6%, versus Gato's 63.6% on DMC, a different suite) [A1, A5, A6, A8] (Sec. 4.2, p. 8, Table 3, p. 19, Table 4, p. 21)
- C9: BabyAI reaches 90% after only 30,000 steps; the authors hypothesize that semantic similarity of tasks gives positive transfer but give no proof [A1, A6] (Sec. 4.2, p. 8)
- C10: Task indeterminacy is a minor problem on 10 Meta-World tasks even without reward in the observation: a JAT trained without reward reaches 97.6 +/- 0.7% IQM because the task can generally be inferred from the initial conditions, and adding reward has a significant but small effect (probability that full JAT beats it 51.8 +/- 2.5%) [A1, A6, A9] (App. D, p. 29, Fig. 17, p. 29)
- C11: Text completion and image captioning are only qualitative demonstrations; outputs are often syntactically fine but off-topic or wrong [A1] (Sec. 4.1, pp. 6-7, Fig. 2, p. 6, Fig. 3, p. 7)
- C12: The model, the dataset (with expert policies) and the training code are released openly on Hugging Face [] (p. 1, Sec. 3.2, p. 5)
- C13: Expert data quality limits results: Asterix's expert (3,699.6) lags far behind R2D2's 999,153.3, and the authors expect better per-task experts to improve the distilled generalist [A1, A7] (Sec. 5, p. 9)
- C14: The final agent's average expert-normalized IQM over the tasks studied is 65.8% [A1, A5, A6] (Sec. 4.2, p. 7)

## Implementation notes

- kappa used for the released/main JAT model is not stated; the ablation says 0.005 is the sweet spot (source: assumption; Sec. 4.3, p. 8 states only the ablation).
- Parameter count 192,701,771 (F32) from the Hugging Face model metadata, not the paper (source: https://huggingface.co/jat-project/jat, API metadata queried 2026-11-24).
- Code license Apache-2.0 for the huggingface/jat GitHub repo; dataset jat-project/jat-dataset tagged apache-2.0 on the hub; the model card metadata carries no license field (source: GitHub and Hugging Face APIs, 2026-11-24).
- Seeds, number of ablation runs, and whether kappa-ablation models were trained on the 6 tasks only or the full mix are not stated (source: assumption; Sec. 4.3, p. 8 says "trained the model on a subset of 6 dataset tasks").
- Consumer GPU (RTX 3080, 10-12 GB): fp32 weights are about 0.77 GB, and a short 512-token context makes inference light; full fine-tuning with AdamW needs roughly 4x the weights in optimizer and gradient state, about 3 GB plus activations (source: assumption, arithmetic from the parameter count).
- The Atari loss on observations is pixel MSE through a transposed CNN; whether the reward is also predicted is not stated (source: Sec. 3.1.3, p. 5; remainder is assumption).

## Limitations / open questions

- The authors say task and loss weights are heuristic and that task sampling should account for difficulty; BC is rudimentary and expert quality should improve (Sec. 3.3.2, p. 6, Sec. 5, p. 9).
- Some Meta-World tasks are not learned at all; the authors guess task indeterminacy but do not confirm it (Sec. 4.2, p. 8).
- Comparison with Gato rests on different expert and random scores and on a 39-task subset of Gato's 46, so the paper asks for caution (Sec. 4.2, p. 8).
- Reader: the observation-loss ablation covers 6 tasks of which 2 are Atari, reports no seed count, and the 94.5 baseline is quoted without saying which aggregate (the plots show median, IQM and mean). It is a weaker test than a full multi-domain run, and it does not say whether pixel prediction specifically (rather than vector prediction) helps.
- Reader: no held-out-game or fine-tuning experiment appears in the paper; transfer to unseen games is untested.
- Reader: the paper does not report the actual JAT parameter count.

## Related

- extends [[reed2022generalist]]: an open, smaller Gato-style generalist across RL, vision and language.
- related [[lee2022multigame]]: addresses the gap Multi-Game DT left open (future-observation prediction, Sec. 3.1, p. 4 there) but does not build on it; JAT frames it as an open question of DT and Gato, uses BC without return conditioning, and covers Atari 57 plus control and BabyAI.
- related [[cheng2025jointly]]: the other Atari follow-up on observation prediction; JOWA predicts VQ tokens and adds a CQL critic.
- related [[hafner2024mastering]]: a world-model alternative to JAT's auxiliary prediction (model-based RL with one hyperparameter set across domains).

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
