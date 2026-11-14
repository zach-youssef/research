---
id: serrano2024crossdomain
title: "Knowledge Transfer for Cross-Domain Reinforcement Learning: A Systematic Review"
authors: [Sergio A. Serrano, Jose Martinez-Carranza, L. Enrique Sucar]
year: 2024
venue: IEEE Access 2024 journal article (survey)
url: https://arxiv.org/abs/2404.17687
arxiv: 2404.17687
doi: 10.1109/ACCESS.2024.3435558
pdf_sha256: ab03f1574d67d81ced5bbe0a6eb57053f46fdbe77568e10f7d7b07b9f293aebc
tags: [reinforcement-learning, transfer-learning, cross-domain, imitation-learning, survey]
status: read
read: 2026-11-14
---

# Knowledge Transfer for Cross-Domain Reinforcement Learning: A Systematic Review

<!-- Read from arXiv v2 (2024-11-20), which states it is the IEEE Access version. -->

## TL;DR

Survey of 53 methods that transfer knowledge between single-agent RL tasks whose state and/or
action spaces differ, sorted by knowledge origin, source count and transfer form (Table II) and
compared on data needs and similarity assumptions (Table III); lists open problems (Sec. VI).

## Problem

RL needs much interaction data; transfer from a related task can cut it, but a common
assumption in transfer is that tasks share state and action spaces and differ only in dynamics
or reward (Sec. I, pp. 1-2). When representations differ (robots with different sensors and actuators,
different board games), a method must both select which source to use, to avoid negative
transfer, and adapt the knowledge so it becomes domain-invariant yet still task-relevant
(Sec. I, p. 2). Earlier surveys cover same- and cross-domain transfer together ([10], [20], [21])
or, closest to this one, cross-embodiment policy transfer by type of robot gap ([37]); this one
organises cross-domain methods by transfer approach and data requirements (Sec. II-C, pp. 3-4).

## Method

- Scope: single-agent RL, model-free or model-based, transferring learned or expert knowledge
  between problems with different representations (Fig. 1, p. 1).
- Definitions (Sec. III, p. 4): a domain is D = ⟨S, A⟩; a task adds a reinforcement signal
  (reward or demonstrations) (Def. 3.1-3.2). A space mismatch is a different size or "different
  phenomena" (Def. 3.3). Cross-domain transfer requires a mismatch in S or A (Def. 3.4); tasks
  differing only in transition or reward dynamics are out of scope, and tasks with a shared
  observation space count only if their state-action spaces differ (e.g. [15], p. 4).
- Evaluation metrics, from [10]: jumpstart, asymptotic performance, total reward, transfer
  ratio (R_TL − R_RL)/R_RL, time to threshold (Sec. II-B, Fig. 3, pp. 3-4).
- Taxonomy, three levels (Sec. III-A, Table II, pp. 4-5): origin (imitation learning from an
  expert vs transfer learning from learned knowledge); type of source (state-action vs
  state-only demonstrations; single vs multiple sources); transfer approach (single source:
  demonstrations, policy, reward shaping, parameters, value function, bias, data; multiple
  sources: combination or selection).
- Comparison dimensions (Sec. III-C, pp. 6-8, Table III p. 9), ordered most to least critical:
  knowledge quality as a (source; target) tuple (expert, imperfect, environment access with
  trained/learning/random policy, simulator, trained policy, model, oracle parameters, proxy-task
  data) with a data-pairing level (task overlap o, task pairing t, demo pairing d, alignment a);
  inter-domain mapping (which spaces, hand-coded h, learned l or chosen from candidates c);
  similarity assumptions (ali aligned by shared tasks, lin linear shared basis, sameS/A/O,
  shareS, app same application, rel shared latent dynamics structure, none); allowed learner
  (any, nn, pg, vf, qf, ac, mb, ms method-specific, bc, opt); source selection (man, auto,
  multi); knowledge transferred; evaluation tasks; comparisons.
- Sections IV-V (pp. 8-14) describe each method; Sec. VI (pp. 14-16) discusses open questions;
  Tables IV-V (pp. 15-16) list RL libraries and simulators.
- Methods most relevant to learning a shared representation from observations: [15] trains an
  image encoder and prototypes self-supervised on random-policy observations from several
  domains (predict the next observation's cluster), then RL in an unseen domain uses the
  encoder's output plus a prototype-based auxiliary reward (Sec. V-B1, p. 13); [53] (XIRL)
  fine-tunes an ImageNet ResNet for temporal alignment over different robots' demonstrations
  and rewards distance to the mean goal embedding (Sec. IV-B, p. 10); [52] learns a latent space
  maximising mutual information with the source state and adversarially aligns target state
  sequences into it (p. 10); [61] trains a shared latent policy with per-domain encoder-decoder
  pairs on proxy tasks (Sec. V-A1, p. 10); [80] reuses option sub-networks between
  domain-specific input and output layers trained from scratch (Sec. V-A4, p. 12).

## Claims

- C1: The review covers 53 cross-domain methods: 8 imitation-learning (2 state-action, 6 state-only), 37 single-source transfer (5 demonstrations, 9 policy, 4 reward shaping, 6 parameters, 8 value function, 3 bias, 2 data) and 8 multi-source (4 combination, 4 selection) (Table II, p. 5).
- C2: About 43% of the reviewed works (23 of 53) transfer knowledge in a form only an off-policy algorithm can use in the target domain (policies, demonstrations, source interaction samples) (Sec. VI-E, p. 16).
- C3: For on-policy learners (preferred when online performance or low computational cost matters), transfer by reward shaping ([51], [53], [55], [71]-[74]) or by model parameters ([15], [75]-[80]) can improve sample efficiency without generating off-policy data (Sec. VI-E, p. 16).
- C4: Over time the inter-domain mapping has needed less human supervision: hand-coded mappings, then domain-agnostic alignment (manifold alignment), then mappings learned from transition and reward dynamics; recent work avoids paired demonstrations, selects sources automatically and learns from state-only demonstrations (Sec. VII, pp. 16-17).
- C5: Methods that use target-domain information to decide how to transfer ([12], [14], [18], [19], [87]) mitigate negative transfer, but only by picking the best source from a library; the authors know of no method that detects that no available source will help (Sec. VI-A, p. 14).
- C6: Most reviewed methods assume source and target are similar in some way (shared S, A or O, or a shared latent structure), and for latent-structure assumptions it is unclear whether a new setting meets them; formalising these conditions is open (Sec. VI-B, pp. 14-15).
- C7: Cross-domain transfer is still too data-costly for most real-world uses; only a few of the reviewed works show results on real robots (Sec. VI-C, p. 15).
- C8: Single-source transfer can accelerate learning when domain expertise identifies how tasks relate, but such methods cannot assess inter-task similarity, so they may fail when tasks are not related in the way the method exploits (Sec. III-B, p. 5).
- C9: Multi-source methods (combination or selection) remain prone to task interference when combining conflicting knowledge and to negative transfer when selecting an unrelated source (Sec. III-B, pp. 5-6).
- C10: Imitation-learning performance depends strongly on the variety of the training data, and methods commonly assume first-person demonstrations unless a perspective mapping exists (Sec. III-B, p. 5).
- C11: Evaluation tasks in the reviewed works are MuJoCo, DeepMind Control, keepaway soccer, classic control, robot platforms, point-mass mazes, X-MAGICAL, board games, blocks world and deformable objects (Sec. III-C7, p. 8, Table III, p. 9).
- C12: Two reproducibility problems persist in transfer-RL papers: code repositories that disappear after publication (with implementation details omitted because code was linked), and unreported details of the policy that sampled the data (Sec. VI-F, p. 16).
- C13: Combining strategies for inter-domain mismatch with hierarchical models is proposed to significantly extend the settings where knowledge can be reused, notably long-horizon tasks; decoupling high- and low-level behaviour as in [74], fine-tuning only the low-level policies, might reduce the data needed for real-world tasks (Sec. VI-C, p. 15).
- C14: Using uncertainty estimates (e.g. Gaussian processes, statistical significance) on inter-MDP similarity is proposed to improve source selection, and comparing a partly trained target policy against the best source policy to decide to learn from scratch (Sec. VI-A, p. 14).

## Implementation notes

- Lens for these notes: an unsupervised cross-domain embedding of a game's state from N frames of its visual output, a large transformer policy trained on it with PPO or actor-critic, and zero-shot play on unseen games of the same genre (source: assumption, the user's design idea of 2026-11-14, not the paper's)
- Under Def. 3.3-3.4 this is cross-domain transfer: games share the observation space (RGB frames) but their latent states are "different phenomena"; it is the sameO setting, plus sameA if every game is driven through one controller layout (source: Def. 3.3-3.4, p. 4; the controller point is an assumption).
- In the taxonomy it is multiple-source combination, knowledge transferred as parameters (the shared encoder and policy) (source: Sec. III-A, Table II, p. 5); the closest reviewed method is [15], (en_r; en_l), sameO, any learner, multi, par+shape, tested only on DeepMind Control (source: Table III, p. 9; Sec. V-B1, p. 13).
- Data requirement: [15] needs only random-policy observations from source domains and no pairing, which matches unsupervised pretraining from gameplay video. Of the 25 learned (l) mappings in Table III, 9 need paired or overlapping data (o, t, d or a: [48], [50], [59]-[61], [67], [68], [72], [73]); the rest use unpaired proxy data ([14], [18], [56], [66], [87]), unpaired state-only expert demonstrations plus target interaction ([52], [53], [55], which gameplay video could supply), a source policy, simulator or environment plus target interaction ([57], which also needs unpaired proxy data and a settable simulator, [58], [64], [65], [89], [90]), or imperfect target demonstrations of unknown quality instead of target interaction ([69], [92]; [92] needs no environment on either side) (source: Table III, p. 9, legend p. 6; the gameplay-video reading is an assumption).
- Embedding N frames to estimate the state is the POMDP setting of Sec. II-A (agent estimates state from observations); the review gives no guidance on N (source: Sec. II-A, Fig. 2, p. 3; the frame-stacking reading is an assumption).
- PPO is on-policy, so per C3 parameter and reward-shaping transfer suit it, while the 23 of 53 methods that transfer policies, demonstrations or source samples need an off-policy learner (source: Sec. VI-E, p. 16; that PPO is on-policy is general knowledge, the review cites PPO [149] only as a standard baseline on p. 15).
- Methods whose allowed target learner PPO fits: ac [19], [65] (opt, ac), [68]; pg [12], [13], [58] (target side); rows marked any (e.g. [15], [53]) are learner-agnostic (source: Table III, p. 9; PPO being a policy-gradient actor-critic method is general knowledge, not stated in the review).
- Report transfer with the five metrics of [10] (jumpstart, asymptotic, total reward, transfer ratio, time to threshold), learning from scratch on the held-out game as the baseline (source: Sec. II-B, Fig. 3, pp. 3-4).
- ALE (Atari 2600), Gymnasium and Malmo (Minecraft) are the game simulators listed; CleanRL and Stable Baselines3 are PyTorch RL libraries listed (source: Tables IV-V, pp. 15-16).
- Record the data-collecting policy and its training details for reproducibility (source: Sec. VI-F, p. 16).

## Limitations / open questions

- No method reviewed detects that no source will help, so a generalist policy can still suffer negative transfer on an unrelated game (Sec. VI-A, p. 14).
- Similarity assumptions such as a shared high-level or latent structure are hard to check for a setting other than the one a method was evaluated on; in the reader's reading, "same genre" is the review's app assumption (same application, e.g. turn-taking games) plus rel, and is of this hard-to-check kind (Sec. III-C3, p. 7, Sec. VI-B, pp. 14-15).
- The review excludes tasks that differ only in dynamics or reward; in the reader's reading, generalisation across levels of one game with a shared representation is out of scope; multi-task, lifelong/continual, hierarchical and curriculum RL and transfer benchmarks are covered only by pointing to other surveys [40]-[47] (Def. 3.4, p. 4, Sec. II-C, pp. 3-4).
- Reader's observation, not stated by the paper: none of the 53 methods is evaluated on video games (Atari appears only as a simulator in Table V), and none is a large multi-game generalist or transformer policy; the evidence is robotics, control and board games (Sec. III-C7, p. 8, Table III, p. 9).
- Long-horizon tasks are hard for standard RL; combining inter-domain mapping with hierarchical models is proposed as a direction, citing [74] (Sec. VI-C, p. 15).
- Open question: does a self-supervised embedding like [15]'s, trained on random-policy frames, capture game-relevant state (score, enemies, player) across visually different games, or only low-level visual statistics?
- Open question: how should actions be shared across games with different control schemes; learned action embeddings ([79], [68]) or a fixed controller layout?
- Open question: is PPO on a large transformer practical to tune, given the review's note that in online RL tuning hyperparameters can take many iterations (Sec. VI-D, p. 15)?

## Related

- none

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
