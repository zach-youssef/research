---
id: ni2023decoupling
title: When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment
authors: [Tianwei Ni, Michel Ma, Benjamin Eysenbach, Pierre-Luc Bacon]
year: 2023
venue: NeurIPS 2023 conference paper
url: https://arxiv.org/abs/2307.03864
arxiv: 2307.03864
doi:
pdf_sha256: be3c4f1676330e17004b6d0f0a7c4e8821c7d88ad0d0093abdd8621596f35291
tags: [reinforcement-learning, transformer, memory, credit-assignment, partial-observability, benchmark-design, pomdp, theory]
status: read
read: 2026-11-25
---

# When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment

<!-- Read from arXiv v4 (2023-11-03), 24 PDF pages: main text pp. 1-10, references pp. 11-14, appendices A-D pp. 15-24 (all read). -->

## TL;DR

Defines memory length (reward, transition, policy, value) and forward-view credit-assignment length
for POMDPs, shows many memory benchmarks entangle the two, and finds GPT-2 agents solve pure-memory
Passive T-Maze up to 1500 steps but, like LSTMs, fail long-term credit assignment (Active T-Maze ≥ 250).

## Problem

Transformer-based RL does well, but it is unclear whether because of better memory (representing
the history) or better temporal credit assignment (learning which past actions earned a reward)
(Sec. 1, p. 1). Common benchmarks (Atari, MuJoCo) need little of either, others such as Key-to-Door
mix them, and "long-term" has no rigorous definition (Sec. 1, p. 2).

## Method

- **Definitions** (Sec. 2, pp. 2-4): a context-based policy sees the last l_ctx observations and
  actions. Reward memory length m_reward and transition memory length m_transit are the smallest n
  such that the expected reward (next observation) given the last n observations equals that given
  the full history (Def. 1.A-1.B, p. 3). Policy memory length l_mem(π) is the shortest suffix that
  the action depends on; value memory length the shortest suffix that gives Q^π (Def. 1.C-1.D, p. 3).
- **Theorem 1** (Eq. 1, p. 3, proof App. A, pp. 15-16): for optimal policies with the shortest
  policy memory, l_mem(π*) ≤ l_value(π*) ≤ max(m_reward, m_transit) =: m^M.
- **Credit-assignment length** (Def. 2.A-2.B, Eq. 2, p. 4): the forward view; c(h; π) is the
  smallest n at which a greedy action's n-step reward sum strictly beats every non-greedy action's;
  c(π) is the worst case over visited histories, and the task's c^M the minimum over optimal
  policies. Sparse terminal reward gives c = T. It relates to the number of TD(0) back-ups needed.
- **Benchmark audit** (Table 1, p. 5, App. B, pp. 17-20): estimates T, l_mem(π*), m^M and c^M by
  hand for prior tasks; e.g. Memory Gym's Mortar Mayhem (T 135, c^M ≤ 25), Numpad (T 500, short c),
  Memory Maze 15² (T 4000, c^M ≤ 225), Key-to-Door (T 90, c^M = T), PyBullet-P/V (short).
- **Toy tasks** (Sec. 3.2, p. 6, Fig. 1): Passive T-Maze (oracle at the start, corridor L = T − 1,
  small per-step penalties for not moving right, terminal +1 for the right goal): memory length T,
  credit assignment 1. Active T-Maze (oracle one step left of the start; the agent must first go
  left): memory and credit assignment both T. Pixel analogues: Passive Visual Match (memory) and
  Key-to-Door (credit assignment) (Fig. 2, p. 7).
- **Agents** (Sec. 5, p. 7, App. C.1, pp. 21-23): model-free, from scratch, observation and
  previous-action embedders, a sequence model shared by actor and critic, and actor-critic heads
  (Fig. 10). Sequence model: LSTM (hidden 128) or GPT-2 (Hugging Face, causal, sinusoidal
  positions, dropout 0.1), 1, 2 or 4 layers. Context length = full episode T. DDQN for T-Mazes,
  SAC-Discrete for pixel tasks, TD3 for PyBullet (context 64). To share the encoder, critic
  parameters are frozen in the actor loss (Eq. 40, p. 22).

## Assumptions

- A1: In the T-Maze and pixel tasks, agents are model-free (DDQN or SAC-Discrete), trained online from scratch, with a sequence model over the full episode history (context length T) shared by actor and critic (Sec. 5, p. 7, App. C.1, pp. 21-22).
- A2: The sequence models are small: LSTM with hidden size 128 or GPT-2 with 1, 2 or 4 layers and heads, GPT-2 observation embedding 64 (state) or 100 (pixels); the LSTM's state embedder is 32 for observations and 16 for actions (Sec. 5, p. 7, App. C.1, pp. 22-23, Table 3, p. 23, Table 4, p. 23).
- A3: The T-Maze tasks have no discount within the task definition, a deterministic transition, dense per-step penalties and a terminal reward of +1 for the right goal; exploration needs little search; the agents train with γ = 0.99 (Sec. 3.2, p. 6, Table 5, p. 23).
- A4: DDQN uses ε-greedy exploration decaying linearly to 1/T, so the chance of an all-greedy episode stays near 1/e as T grows (App. C.1, p. 23, Eq. 41).
- A5: Memory and credit-assignment lengths of prior benchmarks are estimated by hand from task definitions, not computed (Table 1, p. 5, Sec. 6, p. 10).
- A6: Results average 10 seeds, with 95% confidence intervals; final performance is the evaluation average over the last 5% of interactions (Fig. 3, p. 8, App. C.3, p. 24).
- A7: The credit-assignment length counts only the delay until a greedy action's n-step reward overtakes the others, ignoring reward variance; memory length ignores memory capacity (bits) and observation noise (Sec. 6, p. 10).
- A8: Training runs to convergence or at most 10M steps (Sec. 5, p. 7), with 8000, 4000, 10000 and 4000 episodes for Passive T-Maze, Active T-Maze, Passive Visual Match and Key-to-Door (Table 6, p. 23).
- A9: The PyBullet occlusion runs use TD3 and follow POMDP-baselines with a context length of 64 and single-layer sequence models, not the full episode (App. C.1, p. 22, Sec. 5, p. 7).

## Claims

- C1: Theorem 1 bounds the policy memory length of an optimal policy with the shortest policy memory by its value memory length, which is in turn at most max(m_reward, m_transit) [] (Eq. 1, Sec. 2.1, p. 3, App. A, pp. 15-16)
- C2: Many tasks designed for memory also need long-term credit assignment, and some credit-assignment tasks also need long memory; some POMDP benchmarks need only short-term memory and credit assignment [A5] (Table 1, p. 5, Sec. 3.1, pp. 5-6)
- C3: Passive T-Maze has memory length T and credit-assignment length 1, Active T-Maze has both T, so the two lengths are independent quantities [A3] (Sec. 3.2, p. 6)
- C4: Transformer (GPT-2) agents consistently solve Passive T-Maze at memory lengths 50 to 1500, which the authors believe may be the longest memory length an RL agent has solved; LSTM agents falter from 250 and fall below the optimal Markovian policy's return of 0.5 [A1, A2, A3, A4, A6, A8] (Fig. 3, p. 8, Sec. 5.1, pp. 7-8)
- C5: Single-layer models perform best in Passive T-Maze for both architectures, and the transformer is far more stable and sample-efficient than the LSTM at memory lengths 500, 750, 1000 and 1500 [A1, A2, A3, A4, A6, A8] (Fig. 4, p. 8, Sec. 5.1, p. 8, App. C.1, p. 22)
- C6: In Passive Visual Match, transformers beat LSTMs only slightly in success rate beyond memory length 120 and both stay near the optimal Markovian policy's 1/3, but transformers earn much higher returns by collecting more apples, i.e. they help short-term credit assignment [A1, A2, A6, A8] (Fig. 3, Fig. 5, pp. 8-9, Sec. 5.1, p. 8)
- C7: At memory length 1000 in Passive Visual Match about 30% of LSTM runs hit NaN gradients and are left out of the plots [A1, A2] (p. 8)
- C8: In Active T-Maze both transformer and LSTM agents fail once credit-assignment (and memory) length reaches 250, although transformers have the memory (Passive T-Maze) and reach the junction; the authors conclude credit assignment is the bottleneck [A1, A2, A3, A4, A6, A8] (Fig. 6, p. 9, Sec. 5.2, pp. 8-9)
- C9: Key-to-Door shows the same pattern: both architectures have low success at credit-assignment length 250, while transformers do better than LSTMs around length 100 in both tasks [A1, A2, A6, A8] (Fig. 6, p. 9, Sec. 5.2, p. 9)
- C10: More transformer layers and heads greatly help credit assignment up to length 100 (Active T-Maze) and 120 (Key-to-Door) but not longer, and 4 layers are similar to or worse than 2; stacking LSTM layers barely changes results [A1, A2, A3, A4, A6, A8] (Fig. 7, p. 9, Sec. 5.2, p. 9, Fig. 12, App. C.2, pp. 23-24)
- C11: On PyBullet occlusion tasks (short-term memory, dense rewards, 1.5M steps), transformer agents are less sample-efficient than LSTM agents in most tasks, consistent with POPGym's GRU-vs-transformer PPO results [A6, A9] (Fig. 8, p. 10, Sec. 5.3, p. 10)
- C12: The authors conclude transformers improve long-term memory but not long-term credit assignment, and generally have poor sample efficiency; they are not a universal fix for RL [A1, A2, A6] (Sec. 7, p. 10, p. 1)
- C13: The worst-case cost of computing c^M is O(|S||A|T²) for a finite MDP and of m^M O((|O|+|A|)^T |O||A|T) for a finite POMDP, so computing them in general remains open [A7] (Sec. 6, p. 10)
- C14: The credit-assignment length of optimal policies need not be unique: in a 7-state example two optimal Markov policies have c = 3 and c = 2 [] (Fig. 9, App. A, pp. 16-17)
- C15: Sharing the history encoder between actor and critic adds gradient terms to the actor-critic update, and freezing the critic's parameters in the actor loss removes them; in preliminary PyBullet runs this matched also freezing the encoder in the actor's action input [A1, A9] (Eq. 35-40, App. C.1, pp. 21-22)
- C16: Transformer GPU memory grows roughly with the square of the context length (at most 4 GB at context 1500) and LSTM's roughly linearly; Passive T-Maze at 1500 took about 6 (transformer) and 4 (LSTM) days on one A100 [A1, A2] (App. C.4, p. 24)

## Implementation notes

- Code: github.com/twni2016/Memory-RL, built on POMDP-baselines (github.com/twni2016/pomdp-baselines), with the o_t shortcut to the heads removed and one sequence model shared by actor and critic (source: p. 1, App. C.1, p. 21).
- RL hyperparameters for all runs: MLP hidden (256, 256), γ 0.99, target update rate 0.005, replay 1e6, learning rate 3e-4, batch 64; DDQN ε linear from 1.0 to 1/T over 0.1 × num_episodes; SAC-Discrete entropy temperature 0.1; update frequency 0.25 updates per environment step (source: Table 5, p. 23, App. C.4, p. 24).
- Best settings: (N, H) = (1, 1) for Passive T-Maze and Passive Visual Match, (2, 2) for Active T-Maze and Key-to-Door; LSTM 1 layer except 2 for Key-to-Door (source: App. C.1, p. 22).
- The training input is t observations and t actions, the first action zero-padded; embeddings are concatenated per step, rewards are not inputs (source: App. C.1, p. 22, Fig. 10, p. 21).
- Shmup lens: the memory length these agents exploit is the whole episode, because the context is the episode (T up to 1500 steps); a frames-in policy can't keep a full-game context, so the paper's 1500-step result says what attention can recall, not what a bounded context or memory tokens would (source: assumption, from Table 6, p. 23).
- Shmup lens: to test a shmup policy's memory without the credit-assignment confound, a task should be "passive" in the paper's sense, the remembered fact shown to the agent rather than earned, with dense rewards after it; weapon choice or rank management that pays off much later is the "active" case where this paper finds both architectures fail (source: assumption, from Sec. 3.2, p. 6).
- Shmup lens: RAM can measure the m^M and c^M of a shmup scenario offline (which RAM variables the reward depends on and how far back they were set), since the definitions are about conditional independence of reward and transition given history suffixes (source: assumption, after Def. 1.A-2.B, pp. 3-4).

## Limitations / open questions

- Programs that compute memory and credit-assignment lengths for a given task don't exist yet and are expensive (Sec. 6, p. 10).
- c^M ignores reward noise, and m^M ignores memory capacity and observation noise (Sec. 6, p. 10).
- Only GPT-2 at small sizes, model-free online RL; other sequence architectures and history abstraction are future work (Sec. 6, p. 10).
- Return-redistribution transformers and offline return-conditioned agents (Decision Transformer) are out of scope (Sec. 4, p. 7).
- Open question: credit assignment here is TD learning from rewards; for a behaviour-cloned policy (NitroGen-style) credit assignment is absent during pretraining, so does the memory result transfer to BC while the credit-assignment failure only matters in RL fine-tuning?
- Open question: Memory Gym's Mortar Mayhem is listed with memory up to its horizon of 135 and credit assignment at most 25, i.e. a mostly-memory task at short horizon; the endless versions in the Memory Gym JMLR paper ([[pleines2025memorygym]]) aren't analysed here.

## Related

- uses [[vaswani2017attention]]: the transformer agents are GPT-2 decoders with sinusoidal positions.
- related [[parisotto2020stabilizing]]: GTrXL is a prior transformer memory agent; its Numpad task is audited in Table 1 as long memory with short credit assignment.
- related [[chen2021decision]]: return-conditioned offline transformers are named as a separate line, outside this paper's TD-learning scope (Sec. 4, p. 7).
- related [[hafner2024mastering]]: model-based RL is not studied here; DreamerV3 keeps its memory in a GRU state, the recurrent alternative to the transformer agents tested.
- related [[cherepanov2026recurrent]]: RATE carries a T-Maze cue over up to 9,600 steps with memory tokens and a bounded context, where this paper's agents keep the whole episode in context; RATE is offline and return-conditioned, outside this paper's TD scope (Sec. 4, p. 7), and its success declines at the longest lengths.
- related [[magne2026nitrogen]]: a short-context game policy; this paper's definitions say what a longer context would and wouldn't fix.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
