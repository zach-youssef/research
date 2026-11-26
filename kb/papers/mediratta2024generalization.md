---
id: mediratta2024generalization
title: The Generalization Gap in Offline Reinforcement Learning
authors: [Ishita Mediratta, Qingfei You, Minqi Jiang, Roberta Raileanu]
year: 2024
venue: ICLR 2024 conference paper (arXiv v2, 15 Mar 2024, read)
url: https://arxiv.org/abs/2312.05742
arxiv: 2312.05742
doi:
pdf_sha256: 980e9f7b149f1bed31711da5abcd4378df15247454f2cd631ace8f8eea24a91f
tags: [reinforcement-learning, offline-rl, behavioral-cloning, generalization, procgen, benchmark, zero-shot, decision-transformer]
status: read
read: 2026-11-26
---

# The Generalization Gap in Offline Reinforcement Learning

## TL;DR

Procgen (16 games) and WebShop offline datasets for testing zero-shot generalization to new levels or
instructions. Behavioral cloning beats BCQ, CQL, IQL, DT and BCT on unseen levels, all lag online PPO,
and dataset diversity (levels), not size, shrinks the gap.

## Problem

Offline RL and sequence-model methods are benchmarked on singleton environments, trained and tested
on the same environment, so their generalization to new environments (different initial states,
transitions, rewards or instructions) is untested (Sec. 1, p. 1). The paper builds datasets and a
protocol to measure it, and compares BC, offline RL and sequence modelling against online RL.

## Method

- Contextual MDP framing: a context c (Procgen level, WebShop instruction) fixes the MDP; contexts
  are split into train, validation (hyperparameters and model selection, online) and test sets
  (Sec. 2, pp. 2-3). Procgen: levels [0,200) train, [200,250) validation, [250, inf) test; WebShop:
  398 / 54 / 500 instructions (Sec. 2-3, pp. 2-3, App. E, p. 22).
- Datasets: Procgen 1M expert, 1M mixed expert-suboptimal, 10M expert, 25M suboptimal; WebShop human
  (452 trajectories) and suboptimal sets of 100 to 10K trajectories (Sec. 3.1, p. 3). The expert is
  the final checkpoint of a PPO policy trained 25M steps on easy mode with the Raileanu-Fergus
  hyperparameters; "suboptimal" uses a checkpoint at 50% of the expert's return, mixed 50/50 with
  the expert (about 75% of expert return) (App. E.1, App. F.1, pp. 22-24).
- Methods: BC (cross-entropy over actions; ResNet or BERT), BCQ, CQL, IQL, BCT (causal transformer
  on state-action history) and DT (adds return-to-go, conditioned on the maximum possible), all about
  1M parameters with a ResNet encoder; WebShop only BC, BCQ and CQL (context length rules out DT/BCT,
  and IQL is hard with variable action spaces) (Sec. 3.2, pp. 3-4).
- Metric: min-max normalized IQM return over 16 games (3 or 5 seeds), 100 test episodes (Sec. 3.2,
  Fig. 2, p. 4).
- Experiments: expert data (Sec. 4.1), mixed data (4.2), single-level training and testing (4.3),
  number of training levels at 1M transitions (4.4), data size 1M to 10M at 200 levels (4.5),
  WebShop (4.6, 4.7) (pp. 4-10).

## Assumptions

- A1: Contexts (levels or instructions) are split into disjoint train and test sets from the same distribution, and test performance is measured by online rollouts (Sec. 2, pp. 2-3).
- A2: The data are fixed offline datasets collected by PPO policies (or human demonstrators in WebShop), with no online interaction during training; online interaction is used only for validation and model selection (Sec. 2-3, p. 3, App. E, p. 22).
- A3: The Procgen datasets are from easy mode, 200 training levels, discrete actions (up to 15), 64x64 RGB observations (App. E.1, p. 22, App. A, p. 20).
- A4: All methods use similar-size (about 1M parameter) networks and per-dataset hyperparameter sweeps on validation levels (Sec. 3.2, p. 4, App. F.2, pp. 24-26).

## Claims

- C1: On the 1M Procgen expert dataset, BC outperforms BCT, DT, BCQ, CQL and IQL on both train and test levels by a significant margin, and all offline methods are below online PPO on both train and test (IQM, 16 games) [A1, A2, A3, A4] (Sec. 4.1, Fig. 2, pp. 4-5).
- C2: On expert data DT and BCT match BC on training levels more closely but are similar to or slightly worse than offline RL on test levels, so the gap between BC and the transformer policies is small in training and large at test [A1, A2, A3, A4] (Sec. 4.1, p. 5).
- C3: With the 1M mixed expert-suboptimal dataset BC still beats all other offline methods on train and test; contrary to singleton-environment results, offline RL does not overtake BC with suboptimal data, and all methods show a similar generalization gap on average [A1, A2, A3, A4] (Sec. 4.2, Fig. 3, pp. 5-6).
- C4: Trained and tested on a single level, all methods do well: on expert data offline methods reach about PPO's return on most games and BC does best; on mixed data offline RL is comparable to or better than BC on most games (9 of 16 for level 1), as in prior work, with Chaser (highly stochastic) a weak spot. So the BC advantage appears specifically when training on many environments and testing on new ones [A2, A3, A4] (Sec. 4.3, Fig. 4, pp. 6-7, App. G, p. 26).
- C5: With dataset size fixed at 1M transitions, increasing the number of training levels from 200 to 100k raises test return and shrinks the generalization gap for BC, BCQ and DT while training return barely changes; the paper reads data diversity as the key lever for generalization [A1, A2, A3, A4] (Sec. 4.4, Fig. 5, pp. 7-8, App. H, p. 27).
- C6: With 200 levels fixed, growing the dataset from 1M to 5M to 10M expert transitions slightly raises train and test return but leaves the generalization gap about constant; on the suboptimal data (datasets of about 1M, 5M and 10M transitions sampled uniformly from the 25M-transition PPO training log) performance stays poor on train and test for all methods [A1, A2, A3, A4] (Sec. 4.5, Fig. 6, Fig. 7, pp. 8-9, App. J, p. 27).
- C7: On WebShop with human demonstrations, BC has the highest average score on train and test instructions, BCQ the slightly higher train success rate, and BC the highest test success rate; with longer training all methods overfit (train rises, test falls) [A1, A2, A4] (Sec. 4.6, Fig. 8, pp. 9-10).
- C8: On WebShop with data from a BC-pretrained imitation policy, 100 to 10K episodes, train and test improve markedly up to about 500 episodes then plateau, and BC outperforms BCQ and CQL on all dataset sizes [A1, A2, A4] (Sec. 4.7, Fig. 9, p. 10).
- C9: The paper's conclusion: existing offline methods, developed without generalization in mind, generalize significantly worse than online RL (PPO) to new environments and do not improve with dataset size without diversity; the authors' suggested directions include combining offline RL with online-RL generalization techniques (augmentation, regularization, representation learning) [A1, A2, A3] (Sec. 6, p. 12).
- C10: The authors say that in Procgen and other CMDPs better training performance usually also gives better test performance, because existing algorithms underfit rather than overfit, so both optimization and generalization are challenges (this citing Jiang et al. 2023) [A1, A3] (Sec. 4.2, p. 6).

## Implementation notes

- PPO behaviour policy hyperparameters: gamma 0.999, GAE lambda 0.95, rollout 256, 3 epochs, 8 minibatches, clip 0.2, 64 envs, Adam lr 5e-4, entropy bonus 0.01, value coef 0.5 (source: Table 2, App. F.1, p. 24).
- Offline sweeps: BC batch {64,128,256,512} and lr {5e-3, 1e-4, 5e-4, 6e-5}; CQL alpha {0.5,1,4,8}; IQL temperature {3,7,10}, expectile {0.7,0.8,0.9}; BCQ threshold {0.3,0.5,0.7}; DT/BCT context {5,10,30,50}, DT rtg multiplier {1,5}; best values per dataset in Table 3 (source: App. F.2.1, pp. 24-25).
- WebShop: BC with batch size 1, lr 5e-5; BERT encoder; the IL checkpoint from Yao et al. (score 59.9, success 29.1%) generates the suboptimal data (source: Table 4, App. F.1-F.2.2, pp. 24, 26).
- Code and datasets: github.com/facebookresearch/gen_dgrl, CC-by-NC; not consulted (source: p. 1, App. A, p. 20).

## Limitations / open questions

- Main Procgen datasets are only 1M transitions (far below PPO's 25M steps), chosen to be affordable; all datasets are discrete-action, so recent continuous-action offline RL methods are not covered (App. A, p. 20).
- WebShop lacks cross-product-category generalization, and transformer policies (DT, BCT) could not be run there for context-length reasons (App. A, p. 20, Sec. 3.2, p. 4).
- The paper offers no theory; it cites offline RL's quadratic-in-horizon error bound and says its results are consistent with it (App. A, p. 20).
- Open question (reader's): the PPO expert is itself a trained policy with an entropy bonus; the paper does not vary the behaviour policy's stochasticity, so it says nothing separate about action entropy in the data.
- Open question (reader's): Procgen has discrete actions and 64x64 images, so transfer to higher-resolution shmup play is untested.

## Related

- related [[farebrother2020generalization]]: both measure zero-shot generalization of deep agents, there for online DQN on ALE variants, here for offline methods on Procgen levels.
- related [[schulman2017proximal]]: PPO is both the behaviour policy that generates the datasets and the online baseline above all offline methods.
- related [[chen2021decision]]: DT and BCT are two of the benchmarked methods; here they trail plain BC at test time.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
