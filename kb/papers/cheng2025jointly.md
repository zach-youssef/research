---
id: cheng2025jointly
title: "Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining"
authors: [Jie Cheng, Ruixi Qiao, Yingwei Ma, Binhua Li, Gang Xiong, Qinghai Miao, Yongbin Li, Yisheng Lv]
year: 2025
venue: ICLR 2025 conference paper (arXiv v4, 29 Jan 2026)
url: https://arxiv.org/abs/2410.00564
arxiv: 2410.00564
doi:
pdf_sha256: 2de5781c1bb2f389d662d0ea82989e63498cebe34e841f53bd82147e1e1b394d
tags: [reinforcement-learning, offline-rl, world-model, model-based-rl, atari, multi-game, scaling, transfer-learning, vq-vae, planning]
status: read
read: 2026-11-27
---

# Scaling Offline Model-Based RL via Jointly-Optimized World-Action Model Pretraining

## TL;DR

JOWA trains one transformer on VQ-VAE image tokens of 15 Atari games with a next-token world loss
(weight 0.1) plus a CQL/C51 Q-head, and plans with beam search. JOWA-150M reaches 78.9% IQM HNS from
10% data, and fine-tunes on 5 held-out games from 5k transitions to 0.647 DQN-normalized IQM.

## Problem

Scaling offline RL has relied on supervised, return-conditioned sequence models that need expert data
(Multi-Game DT, Elastic DT), or on TD methods (Scaled-QL) that need large data and many gradient steps
to adapt to new games because the Q-heads are reset (Sec. 1, pp. 1-2). Model-based work such as
TD-MPC2 uses low-dimensional states and generalizes poorly across heterogeneous tasks (Sec. 1, p. 2).
JOWA asks whether an image-observation world model can scale offline RL and help generalization to
unseen games.

## Method

- One transformer backbone (minGPT-based, GPT-2 style) over interleaved observation tokens and
  action tokens, trajectory segments of L = 8 timesteps, with learnable task embeddings added for
  observation and action tokens (Sec. 4.1.1, p. 4, Sec. 5.1, p. 7, App. E.1.1, p. 19).
- Tokenizer: VQ-VAE (IRIS-style code) represents each 84x84 grayscale frame by K = 36 discrete tokens,
  vocabulary 2048, token embedding 512. Loss: equally weighted L1 reconstruction, commitment loss and
  perceptual loss; frozen after stage 1 (Sec. 4.1.1, p. 4, App. E.1.1, pp. 18-19, Tables 8-9, p. 19).
- World part (backbone plus heads p_o, p_r, p_d): predicts the next frame's tokens autoregressively
  token by token, the reward (clipped to the ternary sign {-1, 0, 1}) and termination, all as
  classification with cross-entropy: L_world = mean over timesteps of [mean over K tokens of -log p_o
  - log p_r - log p_d] (Eq. 4-7, pp. 4-5).
- Action part: Q-head h_Q with distributional C51 TD loss (51 atoms over [-10, 30], gamma 0.99, target
  update every 1000 steps) plus CQL regularizer: L_action = alpha[log sum exp Q - Q(o, a)] + KL
  (T* Z^- || Z), alpha = 0.1 (Eq. 8, p. 5, Table 13, p. 20).
- Joint objective: L = beta L_world + L_action, beta = 0.1; both losses back-propagate into the
  transformer (Eq. 9, p. 5).
- Planning at inference: beam search through the learned model. Objective (Eq. 11, p. 6): first H
  rewards from the reward predictor plus gamma^H max Q-hat at the leaf, with actions restricted to the
  top-K Q-values; width K, horizon H = 2, K = 2 to 4 per game (Eq. 10-11, p. 6, Table 17, p. 21).
  Cost: K^2 (H-1) + K world-model calls but only H sequential forward passes (Sec. 4.2, p. 6).
- Theorem 4.1 (informal): with L_r-Lipschitz reward model, L_Q-Lipschitz Q-hat and errors eps_s,
  eps_r, eps_Q, the search value's error is bounded as in Eq. 12, and no worse than Q-hat's error under
  the condition in Eq. 13 (Sec. 4.2, p. 6, proof App. C, pp. 16-17).
- Training: stage 1 for M1 = 250k steps (VQ-VAE and world part), stage 2 for M2 = 1.5M steps (joint),
  total 1.75M steps, batch 512, AdamW, lr 1e-4 then 5e-5; about 12 days on A100 GPUs (count not given)
  (Sec. 4.3, p. 6, Sec. 3.3, p. 3, Table 13, p. 20).
- Data and games: DQN-Replay (Agarwal et al. 2020); 20 Atari games chosen to match the difficulty mix
  of the full set; 15 for pretraining and 5 held out (Gravitar, MsPacman, Pong, Robotank,
  YarsRevenge); data from 2 of 5 training runs, 10% of transitions drawn at random, 10M transitions
  per game, about 6B tokens (Sec. 5.1, p. 7, App. D, p. 18, Table 7).
- Model sizes: JOWA-40M (4 layers, hidden 512, 8 heads), 70M (6, 768, 12), 150M (12, 768, 12); Q-head
  3-layer MLP with 768, 1024, 1792 hidden units; the observation, reward and termination predictors
  are 2-layer MLPs (App. E.1.1, p. 19, Tables 10-12).
- Fine-tuning: 5k uniformly sampled transitions per held-out game from the last 20% of DQN-Replay
  (expert-level, about 4 trajectories), 50k gradient steps, batch 32, lr 5e-5; JOWA samples 6-step
  segments and plans with H = 2, K = 2 to synthesize the last 2 steps in the second stage; the
  reported number is the snapshot with the best score during fine-tuning (App. E.2, p. 20).
  JOWA-150M scratch trains 500k steps (Sec. 5.5, p. 9).
- Evaluation: 16 rollouts per game, no sticky actions, epsilon-greedy 0.001, 4 frame skip, 4 stacked
  frames, 108K frame cap; HNS and IQM aggregates (Sec. 5.1, p. 7, App. E.3, p. 21, Table 16).
- Baselines, all with batch 512 and 1.75M steps: MTBC (JOWA without world part, 34/65/120M), MGDT
  (40M, 200M), EDT-200M, Scaled-QL 80M (ResNet-101), FICC-85M (Sec. 5.2, p. 7, App. E.1, pp. 19-20).

## Assumptions

- A1: Only Atari with 18 discrete actions and 84x84 grayscale frames, sticky actions off, is used; all 20 games come from DQN-Replay (Sec. 3.3, pp. 3-4, Sec. 5.1, p. 7, Table 16, p. 21).
- A2: Pretraining uses a 10% random subsample of 2 of 5 DQN-Replay runs (10M transitions per game, mixed-quality), not the full dataset (Sec. 5.1, p. 7).
- A3: Pretraining covers 15 games; the 5 held-out games are fine-tuned separately, one model per game, from 5k expert-level transitions (Sec. 5.5, p. 8, App. E.2, p. 20).
- A4: Rewards are mapped to {-1, 0, 1} by sign, so the world model sees clipped rewards, but Zaxxon's rare large reward distorts raw-score comparisons (Sec. 4.1.2, p. 5, App. F, p. 23).
- A5: The VQ-VAE is trained on the same pretraining frames, then frozen; its tokens (36 per frame, vocabulary 2048) are the world model's targets (Sec. 4.3, p. 6, Tables 8-9, p. 19).
- A6: Planning needs both the learned dynamics and the Q-head to be fairly accurate (Theorem 4.1 conditions), and uses horizon 2 only (Sec. 4.2, p. 6, Table 17, p. 21).
- A7: Compute is large: roughly 12 days on A100 GPUs, and the paper could not run the full Atari set, which would take about 2 months (Sec. 3.3, p. 3, Sec. 6, p. 10).
- A8: The ablation study trains from scratch on only 6 games (Assault, Carnival, Centipede, NameThisGame, Phoenix, SpaceInvaders) with 10% data, 1M gradient steps and 40M parameters (Sec. 5.6, p. 9).
- A9: The paper reports one training run per configuration; seed counts are not stated (reader's inference, Sec. 5.3, p. 7).
- A11: Table 6 does not say whether the ablation models were evaluated with planning; the full model can plan and "No L_world" has no world model to plan with (reader's inference, Sec. 5.1, p. 7, Sec. 5.6, p. 9).
- A10: Baselines were retrained or reproduced by the authors under a common 1.75M-step, batch-512 budget (Sec. 5.2, p. 7).

## Claims

- C1: On 15 pretraining Atari games with 10% data, JOWA-150M gets 78.9% IQM HNS and 0.456 median HNS, above MGDT-200M (49.8%), EDT-200M (50.2%), FICC-85M (43.3%), Scaled-QL-80M (42.0%) and MTBC-120M (32.9%); 6 games are superhuman; the paper reports a 71.4% average improvement over the baselines [A1, A2, A9, A10] (Table 2, p. 8, Sec. 5.3, p. 7, Sec. 1, p. 2)
- C2: JOWA's IQM HNS rises with size, 37.1%, 47.6%, 78.9% for 40M, 70M, 150M (with planning), a steeper curve than MGDT (33% at 40M, 49.8% at 200M) or MTBC (28% to 32.9%, 34M to 120M; the last two read off Fig. 2) [A1, A2, A9, A10] (Fig. 2, p. 8, Table 4, p. 9)
- C3: Raw per-game scores do not always follow the size trend, which the authors hypothesize is an inherent problem of offline RL itself; Zaxxon's jump at 150M is a nonlinear-reward effect, since with rewards clipped to [-1, 1] JOWA-70M and 150M score 0.08 and 0.44 [A1, A4] (Sec. 6, p. 10, App. F, p. 23)
- C4: After fine-tuning on 5 held-out games from 5k expert-level transitions, JOWA-150M gets 0.647 mean/IQM DQN-normalized score (median 0.615), against FICC-85M 0.543 (mean), MGDT 0.422, EDT 0.430, Scaled-QL 0.360, MTBC 0.164 and JOWA-150M trained from scratch on the same data 0.196; the paper reports a 69.9% average improvement over baselines [A1, A3, A9, A10] (Table 3, p. 8, Table 18, p. 22, Sec. 5.5, p. 9)
- C5: The fine-tuned score also scales with size (IQM DNS 0.498, 0.603, 0.647 for 40M, 70M, 150M; the median does not follow it: 0.512, 0.715, 0.615) [A1, A3, A9] (Table 3, p. 8)
- C6: Fine-tuning quality tracks data quality: JOWA-150M mean DNS 0.647 with expert, 0.516 with suboptimal, 0.422 with highly suboptimal 5k transitions [A1, A3] (Table 21, p. 23)
- C7: Ablating the joint objective (6 games, 40M, 1M steps) hurts: IQM HNS 1.123 for the full loss vs 0.637 without CQL, 0.659 without L_world (a model-free critic), 0.307 with the action loss detached from the backbone, 0.126 with MSE instead of distributional TD [A1, A2, A6, A8, A9, A11] (Table 6, p. 9, Table 20, p. 23)
- C8: Detaching L_action's gradient (separate world model and critic) and using MSE TD loss both fail to train usable multi-game agents; with MSE the CQL term over-optimizes, giving extremely high Q for in-domain pairs and low Q for out-of-domain actions, for every alpha tried (0.01, 0.05, 0.1) [A1, A8] (Sec. 5.6, p. 10)
- C9: The world loss acts as a regularizer that the authors say stabilizes TD learning at scale; the paper presents the no-L_world gap and the scaling as evidence that offline model-based RL is more efficient than model-free [A1, A2, A6, A8, A9, A11] (Sec. 4.1.3, p. 5, Sec. 5.4, p. 8, Sec. 5.6, p. 10)
- C10: Planning raises IQM HNS on 15 pretrained games: 150M 50.8% to 78.9%, 70M 45.1% to 47.6%, 40M 27.1% to 37.1%; the model scales with size even without planning [A1, A2, A6, A9] (Table 4, p. 9, Sec. 5.6, p. 9)
- C11: On 7 games where JOWA-150M is weak without planning, the parallel beam search reaches IQM HNS 23.7% (mean 37.8%) vs MCTS 13.4% (22.1%) and no planning 3%, and is about 10x faster than MCTS (1.26 vs 0.12 FPS; 10.8 FPS without planning, Python implementations); MCTS was tuned by grid search and is sensitive to temperature and depth [A1, A6, A9] (Table 5, p. 9, App. F, p. 22)
- C12: Synthetic rollouts in pretraining (6 games, 40M, 1M steps, 31 days, COMBO-style loss) gave negative gains: IQM HNS 0.464 vs 1.123; the authors speculate error accumulation and over-conservatism [A1, A8] (Sec. 5.6, p. 10, Table 6, p. 9)
- C13: Learnable task embeddings help: without them JOWA-40M gets mean HNS 0.964 vs 1.183 and median 0.721 vs 1.078 (6 games), which the text reports as 21.9% and 35.7% improvements (absolute HNS points) [A1, A8] (App. F, p. 23, Table 20, p. 23)
- C14: Theorem 4.1: the search-based Q-value's error is bounded as in Eq. 12, below Q-hat's error under the condition in Eq. 13 (reward and transition errors small enough relative to the Q error); the proof assumes the optimal search policy lies in the top-K-restricted policy class [A6] (Sec. 4.2, p. 6, Eq. 12-13, App. B, pp. 15-16, App. C, pp. 16-17)
- C15: Pretraining uses about 6B tokens (K = 36, L = 8), 1.75M steps in two stages, about 12 days on A100s; the full Atari 2600 suite would take about 2 months, so was not tried [A5, A7] (Sec. 5.1, p. 7, Sec. 3.3, p. 3, Sec. 6, p. 10)
- C16: Training, evaluation code and model weights are released; Table 1 marks train, eval and checkpoints as open for JOWA; Multi-Game DT lacks only training code, FICC only released world-model pretraining code, Scaled-QL released preliminary code (footnotes p. 3); the abstract and conclusion say "will be released" while the intro says "We release" [] (p. 1, Table 1, p. 3)
- C17: The paper states planning is on in evaluation: "During evaluation, we enable planning for JOWA" (Sec. 5.1, p. 7), with horizon H = 2 for every game and beam width K = 2 when the valid action space is under 10, else 3 or 4, per game in Table 17 (15 training games only, no entries for the held-out games). The tables that report JOWA at 0.789 IQM HNS (Table 2) and 0.647 DNS (Table 3) don't repeat it, and for the fine-tuned models Sec. 5.1 is the only statement: App. E.2 gives planning only for the training batches (horizon and width 2) and no evaluation widths for held-out games, so "planning on in the held-out evaluation" is the Sec. 5.1 sentence applied, not a per-table statement; the beam width of the 18-action held-out games (Gravitar, Robotank, YarsRevenge) is therefore unstated within 3 or 4, while MsPacman (9 actions) and Pong (6) get 2 by the width rule [A1, A3, A6] (Sec. 5.1, p. 7, App. E.3, Table 17, p. 21, App. E.2, p. 20)
- C18: The per-game raw scores behind Table 3 are in Table 18 (5 held-out games, raw / DQN reference): Gravitar 173.0 / 473.0, MsPacman 307.3 / 3085.6, Pong -20.7 / 19.5, Robotank 2.2 / 63.9, YarsRevenge 3092.9 / 18089.9 (random / DQN). JOWA-150M scores 273.3, 2016.7, 17.7, 25.0, 17506.2; MGDT-200M 250.0, 1290.3, 9.7, 16.0, 10886.3; EDT 253.3, 1210.7, 11.3, 15.5, 11276.9; MTBC-120M 35.7, 905.0, 5.8, 6.8, 7987.5; Scaled-QL-80M 137.5, 1040.2, 13.7, 19.7, 10838.5; scratch JOWA-150M 83.3, 786.7, 8.8, 11.0, 6507.0 (Gravitar, MsPacman, Pong, Robotank, YarsRevenge in order). Per-game raw scores for JOWA-40M and JOWA-70M, and per-game scores for any held-out evaluation without planning, aren't in the paper [A1, A3, A9, A10] (Table 18, p. 22)
- C19: The planning ablation covers training games only: Table 4 (15 games, IQM HNS) and Table 5 and 19 (7 games, JOWA-150M); no held-out-game score without planning is reported, and Table 4's JOWA-70M "with planning" IQM 47.6% differs from the 0.485 IQM printed for JOWA-70M in Table 2 (the text uses 47.6%) [A1, A2, A9] (Table 4, p. 9, Table 2, p. 8, Sec. 5.3, p. 7, Table 19, p. 22)
- C20: Reference tables: HNS is (score - random)/(human - random), cited to Mnih et al. (2015), with the Random and Human columns printed in Table 2 (15 training games, e.g. Assault 222.4 / 742.0, Atlantis 12850 / 29028.1, Seaquest 68.4 / 42054.7) and DNS "following Lee et al. (2022)" uses the Random and DQN columns printed in Table 18 (e.g. Gravitar 173.0 / 473.0, Pong -20.7 / 19.5). The paper gives no source for the printed values themselves, and its DQN column is not described as the best DQN score in the dataset (MGDT's definition) [A1, A3] (Sec. 5.1, p. 7, Table 2, p. 8, Table 18, p. 22, Table 3, p. 8)
- C21: The aggregates in C4's comparison are of different kinds: Table 3 gives MGDT-200M mean 0.422, median 0.354 and IQM 0.377 DNS, so the 0.422 is the mean (JOWA-150M 0.647 is mean and IQM, median 0.615). MGDT-40M's 33% in C2 is not stated in the text or a table: only MGDT-200M 49.8% (Table 2) is, and the 40M value is read off Fig. 2, as are MTBC's 34M and 65M [A1, A2, A3, A9] (Table 3, p. 8, Fig. 2, p. 8, Table 2, p. 8)

## Implementation notes

- Observation-prediction targets are VQ-VAE codes, not pixels; the observation head is a 2-layer MLP over the backbone's output at each token position, trained with cross-entropy per token and averaged over K = 36 tokens (source: Eq. 7, p. 5, App. E.1.1, p. 19).
- Weight on the world loss: beta = 0.1 against L_action with alpha = 0.1; no beta sweep is reported (source: Eq. 9, p. 5, Table 13, p. 20).
- Number of A100 GPUs, wall-clock for fine-tuning, and seed counts are not stated (source: assumption; unreported).
- Code: github.com/CJReinforce/JOWA (GPL-3.0 by the GitHub license field; README asks for Python 3.8, recommends FlashAttention, built on IRIS, minGPT and EDT code) (source: GitHub API and README, queried 2026-11-24).
- Weights: the README points to checkpoints/JOWA in the repo and a Google Drive folder; the paper gives no weights URL (source: README, 2026-11-24).
- Weights licence (web read 2026-11-27): the README says only "Download the model weights from here (checkpoints/JOWA) or google drive"; it states no licence for the weights, and its checklist has "Release model weights" ticked. The repository's GitHub licence field is GPL-3.0 (last push 2024-12-01), so treat the weights as covered by the repository's GPL-3.0 at most; no separate model licence or terms for the Drive copy were found (source: README and GitHub API, 2026-11-27; the "at most" reading is an assumption).
- Terminal condition: Table 16 (p. 21, Atari evaluation settings) lists "Terminal condition: Game Over", next to no sticky actions, grayscale, 84x84, 4 stacked frames, frame skip 4, 108K max frames and evaluation noise 0.001; read as: an evaluation episode does not end on a life loss (the reading is the reader's inference) (source: Table 16, p. 21, checked against the page text).
- Consumer GPU (RTX 3080, 10-12 GB): all three sizes (40M, 70M, 150M) plus the VQ-VAE should fit for inference and fine-tuning at batch 32 and context 296 tokens; 150M weights are about 0.6 GB fp32 and AdamW training state about 2.4 GB (arithmetic, not a measurement). Beam planning at about 1.26 FPS in the paper's Python code makes evaluation slow. FlashAttention needs an Ampere or newer GPU, which the 3080 is (source: assumption, arithmetic from the parameter counts; hardware speed not reported).
- Mac smoke test (exploratory, experiment 2026-11-24-jowa-mac-smoke, upstream commit 8cb02c5): the GitHub repo's `checkpoints/JOWA/*.pt` are Git LFS pointers and LFS is disabled for the repo (batch API 403), so the weights come only from the README's Google Drive folder; the sha256 and size of all three files match the oids in the repo's LFS pointers: JOWA_40M.pt 76,485,274 B (c1c3d824...), JOWA_150M.pt 511,212,618 B (3f607766...), tokenizer.pt 75,162,490 B (4f624acf...) (source: experiments/2026-11-24-jowa-mac-smoke/results.json, GitHub LFS API, 2026-11-24).
- Inference-time parameter counts: model 17.4M ("40M") and 109.0M ("150M") without the unused `head_q_target` copy (raw checkpoint totals 19.1M and 127.8M); tokenizer 4.0M without the training-only LPIPS-VGG16 slices (raw 18.8M); the tokenizer constructor downloads VGG weights (about 0.5 GB) unless patched (source: experiments/2026-11-24-jowa-mac-smoke/diagnostics.json, results.json, src/jowa_patch.py).
- Runs on Apple silicon with torch 2.2.2 on Python 3.8 (CPU and MPS, float32) after runtime patches: gym 0.19/atari-py replaced by gymnasium ALE v5 (frameskip 1, no sticky actions, minimal action set), `attn.mask` keys absent from checkpoints without FlashAttention, empty head slices assert on MPS, CPU-created `atoms_support` not moved by `.to()` (source: experiments/2026-11-24-jowa-mac-smoke/src/jowa_patch.py).
- Inference time per agent step, as means of per-cell medians pooled over Assault and Seaquest (different beam widths, MPS planning episodes capped at 30 steps and CPU at 60): greedy 29 ms (40M) and 55 ms (150M) on MPS, 31 and 124 ms on CPU; beam-search planning (repo's per-game width, horizon 2) 0.95 s (40M) and 2.0 s (150M) on MPS, 0.86 s and 3.2 s on CPU. MPS vs CPU for planning depends on the game: 40M Assault (width 2) 934 ms MPS vs 474 ms CPU, Seaquest (width 4) 975 vs 1253 ms (source: experiments/2026-11-24-jowa-mac-smoke/results.md per-game table; exploratory, other jobs running on the Mac).
- All 32 capped episodes (Assault, Seaquest, 2 seeds, with and without planning) ran with finite Q values; "legal actions only" holds by construction (upstream masks and asserts). Port fidelity is untested: the tokenizer does not reconstruct the port's Assault frames (reconstruction MSE about equal to the frame variance, also with a near-identical alternative grayscale conversion, so that does not test a different input; the port uses the same `getScreenGrayscale` call as upstream) while Seaquest reconstructs near-exactly, and JOWA-150M greedy play on Seaquest repeats one action in 3 of 4 runs; causes unexplained. Returns of capped episodes are not comparable with the paper (source: experiments/2026-11-24-jowa-mac-smoke/results.md, diagnostics.json).
- arXiv abstract page (latest) says outperforming baselines "by 31.6%"; the PDF v4 abstract and intro say 71.4% (source: arXiv listing vs PDF read, 2026-11-24); this note follows the PDF.
- The IQM HNS values for the smaller MTBC (34M, 65M) and MGDT-40M in Fig. 2 are read from the chart (source: Fig. 2, p. 8, approximate).

## Limitations / open questions

- Reader: the paper never reports held-out scores without planning, and does not state the planning settings of the fine-tuned evaluation beyond Sec. 5.1's general sentence (C17, C19).
- Could not train on the full Atari 2600 set with complete datasets (about 2 months) (Sec. 6, p. 10).
- Synthetic data in pretraining was harmful and untested at scale; how to use it is left open (Sec. 6, p. 10).
- Raw scores in some games decrease with size, which the authors attribute to offline RL (Sec. 6, p. 10).
- Reader: Table 20 shows No L_world beating the full model on Assault (1650 vs 1423.5) and Centipede (8146 vs 5018.9) and tying Carnival (5560); the aggregate gap comes from NameThisGame, Phoenix and SpaceInvaders. Planning is on by default in evaluation (Sec. 5.1, p. 7), and a model without L_world cannot plan, so the 1.123 vs 0.659 gap may mix the regularizer and planning effects (planning alone: 27.1% to 37.1% IQM at 40M, Table 4, p. 9).
- Reader: there is no ablation of next-observation prediction alone; "No L_world" removes the observation, reward and termination predictions together and turns the model into a model-free critic.
- Reader: fine-tuning reports the best snapshot during fine-tuning, which is selected on the evaluation score, and baselines used different augmentation (random crop and rotation) from JOWA (App. E.2, p. 20).
- Reader: the ablation set (6 games) is small and trained at 40M and 1M steps, and no seeds are reported.
- Reader: all evidence is Atari-only; cross-domain transfer is not tested.

## Related

- extends [[lee2022multigame]]: same Atari multi-game pretraining and held-out-game fine-tuning setting, with reduced data (10%), a model-based Q-learning objective, and a world loss that Multi-Game DT left out (Sec. 3.1, p. 4 there).
- related [[gallouedec2024jack]]: the other follow-up with observation prediction; JAT uses pixel MSE with a small weight and BC, JOWA uses VQ tokens with CE and CQL.
- related [[hafner2024mastering]]: a model-based RL agent with a learned world model (online, per-task).
- related [[hafner2025scalable]]: another world-model agent trained offline from data.
- related [[gao2025adaworld]]: world-model pretraining for transfer to new environments.

## Experiments

<!-- kb:begin experiments -->
- [2026-11-24-jowa-mac-smoke](../../experiments/2026-11-24-jowa-mac-smoke/results.md): C16 inconclusive
<!-- kb:end experiments -->
