---
id: hafner2024mastering
title: "Mastering Diverse Domains through World Models"
authors: [Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap]
year: 2024
venue: arXiv preprint
url: https://arxiv.org/abs/2301.04104
arxiv: 2301.04104
doi:
pdf_sha256: d0385798e8bada8e81b915c1743be81d9ce8776f12ed937e09351995e099ce37
tags: [reinforcement-learning, world-model, model-based-rl, actor-critic, minecraft, atari, robustness, scaling]
status: read
read: 2026-11-20
---

# Mastering Diverse Domains through World Models

<!-- Read from arXiv v2 (2024-04-17 revision), 40 PDF pages; pp. 12-16 are references, pp. 17-22 methods
appendices (baselines, implementation, benchmarks, model sizes, hyperparameters, Minecraft) and pp. 23-40
supplementary figures and per-task tables, all read (per-task tables checked for the summary rows). First
submitted 2023-01; the id and year follow the version read (v2, 2024). -->

## TL;DR

DreamerV3: an RSSM world model (CNN encoder, block GRU, discrete latents) plus actor-critic trained in
15-16-step imagination, one hyperparameter set for 150+ tasks in 8 domains, one A100 per run; first to
get Minecraft diamonds from scratch (no human data); gains rest mostly on the reconstruction loss.

## Problem

RL algorithms need per-domain tuning (continuous vs discrete actions, image vs vector inputs,
sparse vs dense rewards, reward scales), which is costly and blocks applying RL to new problems
or expensive models. Learned world models are appealing but have been hard to make robust. The
paper asks for one algorithm with fixed hyperparameters that beats specialised methods across
domains (p. 2).

## Method

- **World model (RSSM, Eq. 1, Fig. 3, p. 4).** Sequence model h_t = f(h_{t-1}, z_{t-1}, a_{t-1});
  encoder q(z_t | h_t, x_t); dynamics predictor p(ẑ_t | h_t); reward, continue and decoder heads
  on the model state s_t = {h_t, z_t}. z_t is a vector of softmax categoricals sampled with
  straight-through gradients (p. 4).
- **Observation embedding (p. 18).** Images: stride-2 convolutions down to 6x6 or 4x4, flattened
  (no patch tokens); decoder is transposed convolutions with a sigmoid output. Vector inputs are
  symlog-transformed and use 3-layer MLPs. Minecraft frames are 64x64x3 plus inventory,
  max-inventory, equipped-item and health/hunger/breath vectors (p. 22).
- **Sequence model (p. 18, Table 3).** A GRU with block-diagonal recurrent weights (8 blocks);
  recurrent units = 8d. Input each step is a linear embedding of z_t, a_t and the recurrent state.
  Not a transformer; context is the recurrent state, trained on batches of 16 x 64 steps.
- **World-model loss (Eq. 2-3, pp. 4-5).** β_pred=1 (decoder + reward + continue, symlog
  squared error / twohot), β_dyn=1 and β_rep=0.1 KL terms with stop-gradients (KL balancing),
  each clipped below 1 nat (free bits); all categoricals mix in 1% uniform (unimix).
- **Actor-critic in imagination (p. 5-6, Eq. 5-7).** From replayed states the model and actor
  roll out a short horizon; the critic is a distributional λ-return predictor (λ=0.95,
  γ=0.997, i.e. 333-step discount horizon), also trained on replay trajectories (β_repval=0.3)
  and regularised toward an EMA of itself. The actor uses Reinforce for discrete and continuous
  actions, fixed entropy scale η=3e-4, and returns divided by max(1, S), S = EMA of the
  95th minus 5th return percentile (decay 0.99). No planning at decision time.
- **Scale-robust regression (Eq. 8-12, p. 7, p. 18).** symlog(x) = sign(x) ln(|x|+1) for inputs
  and decoder targets; reward and critic predict a softmax over exponentially spaced bins
  B = symexp([-20, ..., +20]) trained on twohot targets; their output weights start at zero.
- **Optimiser and replay (p. 18, Table 4, p. 21).** AGC clipping at 30% of the weight norm,
  LaProp (ε=1e-20, β1=0.9, β2=0.99); lr 4e-5, RMSNorm + SiLU and 5e6 replay capacity (Table 4);
  uniform replay with an online queue, storing and refreshing latent states. Training amount is
  set by the replay ratio.
- **Fixed hyperparameters (Table 4, p. 21).** Same values on all benchmarks, no annealing,
  prioritized replay, weight decay or dropout. Per benchmark only model size, replay ratio, env
  instances and action repeat differ (Table 2, p. 19).
- **Model sizes (Table 3, p. 20).** 12M-400M parameters, set by hidden size d (256-1536):
  recurrent units 8d, base CNN channels d/16, codes per latent d/16; layer and latent counts fixed.
- **Minecraft setup (p. 10, p. 22).** MineRL v0.4.4 with a flat categorical action space and
  abstract crafting actions, the block-breaking setting of prior work, jump held 200 ms; episodes
  end on death or after 36000 steps (30 min at 20 Hz); +1 per first-time milestone of 12 items,
  ±0.01 per lost or restored heart.

## Claims

- C1: One configuration with fixed hyperparameters outperforms specialised methods across over 150 tasks in 8 domains and substantially outperforms a fixed-hyperparameter PPO in all of them (p. 1, Fig. 1, p. 8, Table 4)
- C2: Dreamer is the first algorithm to collect diamonds in Minecraft from scratch without human data or curricula, and every trained agent (10 seeds) found a diamond within 100M env steps, while IMPALA, Rainbow and PPO reach the iron pickaxe but none finds a diamond (p. 1, p. 8, Fig. 5, p. 10, p. 19)
- C3: At 100M steps the Minecraft return is 9.1 for Dreamer vs 7.1 IMPALA, 6.3 Rainbow and 5.1 PPO, but diamonds are obtained in only 0.4% of episodes at the budget, which the authors leave as an open challenge (p. 24, Table 5, Fig. 9)
- C4: Minecraft Dreamer used 1 GPU for about 9 days (8.9 A100 days) without human data, against VPT's 720 GPUs for 9 days with contractor demonstrations and RL fine-tuning (p. 11, p. 19, Table 2)
- C5: On Atari (57 games, 200M frames, sticky actions) Dreamer reaches a human-gamer median of 830% and mean of 3381%, vs MuZero 693% and 3054% and PPO 180% and 892%, using a fraction of MuZero's compute (p. 8, p. 26, Table 6)
- C6: On ProcGen (16 games, hard, unlimited levels, 50M frames) the normalized mean is 66.01 vs PPG 64.89 and PPO 42.80, and the fixed-hyperparameter PPO matches the tuned official PPO (41.16) (p. 8, p. 19, p. 28, Table 7)
- C7: On DMLab (30 tasks) Dreamer at 100M steps scores 71.4% human mean capped vs IMPALA 66.3% at 1B and 31.0% at 100M, which the authors call a data-efficiency gain of over 1000% (p. 8, p. 19, p. 30, Table 8)
- C8: On Atari100k Dreamer reaches a 125% gamer mean and 49% median in 0.1 A100 days, above IRIS (105%), TWM (96%), SPR (62%) and SimPLe (33%) on the mean (TWM's median is 51%); EfficientZero (190%) is set aside because it resets levels early and changed the standard setup (p. 9, p. 32, Table 9, Table 10)
- C9: Dreamer sets the state of the art on proprioceptive control (500K steps, over D4PG, DMPO, MPO), visual control (1M steps, over DrQ-v2 and CURL) and BSuite (over Boot DQN, especially on reward-scale robustness and memory) (p. 9, p. 10, p. 34, p. 36, p. 37, Fig. 16)
- C10: In the 14-task ablation every robustness technique helps on average, the KL objective (balancing plus free bits) most, then return normalization, then symexp twohot regression, though each technique matters only on a subset of tasks (p. 9, Fig. 6, p. 10, p. 39, Fig. 17)
- C11: Performance rests predominantly on the task-agnostic reconstruction objective: stopping reconstruction gradients collapses it, while stopping reward and value gradients costs less, which the authors suggest could enable pretraining on unsupervised data (p. 9, Fig. 6, p. 10, p. 40, Fig. 18)
- C12: Scores increase monotonically with model size from 12M to 400M parameters on Crafter and one DMLab task (Goals), and larger models also need fewer environment steps (p. 2, p. 9, Fig. 6, p. 10)
- C13: Higher replay ratios (1 to 64) predictably increase performance, so compute can be traded for data efficiency (p. 9, Fig. 6, p. 19)
- C14: Percentile-based return normalization with a floor of 1 was found stable across domains, where advantage normalization, return-std normalization and constrained entropy were not (p. 6)
- C15: The world model predicts 45 frames open-loop from 5 context frames on DMLab, quadruped and Minecraft (p. 4, Fig. 4, p. 23, Fig. 7)
- C16: On the two control suites the 12M model reaches the same performance as the default 200M model (p. 19)
- C17: Uniform replay is used for simplicity although prioritized replay was also found to improve Dreamer (p. 18)

## Implementation notes

- Imagination horizon H = 15 (source: Table 4, p. 21)
- p. 5 calls the prediction horizon T = 16; read as 15 imagined steps plus the start state (source: assumption)
- Number of latents per z_t is fixed across model sizes but not stated (source: Table 3, p. 20)
- 32 categorical latents per z_t, as in DreamerV2, not checked against the authors' code (source: assumption)
- Per-benchmark settings, Minecraft: 100M steps, action repeat 1, 64 env instances on remote CPU workers, replay ratio 32, 200M model, 8.9 GPU days (source: Table 2, p. 19)
- Gradient steps: replay ratio / (batch steps x action repeat), e.g. Atari replay ratio 32, repeat 4, batch 16x64 gives one update per 128 env steps (source: p. 18)
- Seeds: 5 per benchmark, 1 for ProcGen, 10 for BSuite and Minecraft (source: p. 19)
- Expected value of the twohot softmax: sum positive and negative bins separately from small to large for numerical stability (source: p. 18)
- PPO baseline: Acme, fixed hyperparameters for all domains (lr 3e-4, clip 0.2, entropy 0.01, γ 0.997, λ 0.95, IMPALA network) (source: Table 1, p. 17)
- Minecraft image input 64x64x3 (source: p. 22)
- Other visual domains also use 64x64 images; the paper doesn't state their resolution (source: assumption)

## Limitations / open questions

- No cross-domain transfer: every task is trained from scratch with the same hyperparameters;
  one world model across domains and learning from internet video are named future work (p. 11).
- Minecraft success is about discovery across training (100% of agents), not reliability per
  episode (0.4%), and uses MineRL's categorical actions with abstract crafting and the
  block-breaking simplification, not VPT's raw keyboard and mouse (p. 10, p. 11, p. 24).
- Open question (reader): memory is a recurrent state trained on 64-step sequences with a 15-step
  imagination horizon (Table 4); BSuite shows it beats prior methods on the memory category (p. 37,
  Fig. 16), but memory over game-length horizons isn't studied.
- Reader's note (survey lens): the reconstruction-driven latent (C11) is a candidate unsupervised state
  embedding, but it is learned per task, from 64x64 frames, as a flat CNN feature, not tokens.
- Reader's note: results use 5 seeds per benchmark (1 for ProcGen, 10 for BSuite and Minecraft),
  and baselines come from published curves at different budgets (p. 19).
- Reader's note: p. 8 says Dreamer at 100M exceeds IMPALA and R2D2+ at 1B, but Table 8 lists
  R2D2+ only at 10B (85.4%, above Dreamer's 71.4%), so C7 keeps only the IMPALA comparison (p. 30).

## Related

- related [[schulman2017proximal]]: PPO is the main fixed-hyperparameter baseline, which Dreamer outperforms in every domain.
- related [[baker2022video]]: VPT gets Minecraft diamonds from human-video BC plus RL on 720 GPUs; Dreamer does it from sparse reward on one GPU with simplified actions.
- related [[serrano2024crossdomain]]: Dreamer fixes hyperparameters across domains but learns each task from scratch, without the cross-domain transfer that survey covers.
- related [[he2021masked]]: an alternative unsupervised visual objective; Dreamer's own representation also rests mostly on reconstruction.
- related [[vaswani2017attention]]: Dreamer's sequence model is a block GRU, not a transformer; transformer world models (IRIS, TWM) are its Atari100k baselines.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
