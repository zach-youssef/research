---
id: parisotto2020stabilizing
title: Stabilizing Transformers for Reinforcement Learning
authors: [Emilio Parisotto, H. Francis Song, Jack W. Rae, Razvan Pascanu, Caglar Gulcehre, Siddhant M. Jayakumar, Max Jaderberg, Raphaël Lopez Kaufman, Aidan Clark, Seb Noury, Matthew M. Botvinick, Nicolas Heess, Raia Hadsell]
year: 2020
venue: ICML 2020 conference paper
url: https://arxiv.org/abs/1910.06764
arxiv: 1910.06764
doi:
pdf_sha256: 98b64dec5b94f8b122a136cc0bdc6536958daf28e881b5f9b794375385386636
tags: [reinforcement-learning, transformer, memory, partial-observability, gating, transformer-xl, dmlab, v-mpo]
status: read
read: 2026-11-25
---

# Stabilizing Transformers for Reinforcement Learning

<!-- Read from arXiv v1 (2019-10-13, printed "Preprint"), 20 PDF pages: main text pp. 1-10, references pp. 11-14, appendices A-D pp. 15-20 (all read). The ICML 2020 camera-ready version was not read and may differ in numbers or experiments. -->

## TL;DR

GTrXL makes Transformer-XL trainable as an RL agent's memory by moving layer norm onto the submodule
inputs and replacing residual adds with GRU-style gates initialised near identity; with V-MPO it
scores 117.6 on DMLab-30 (LSTM 99.3, plain TrXL 5.0, MERLIN@100B 115.2) and beats LSTMs on memory tasks.

## Problem

Transformers model long sequences well in supervised learning, but in RL the canonical transformer
is hard to optimize and often performs like a random policy; earlier work found transformers could
not even solve simple bandits or tabular MDPs (Sec. 1, p. 2). RL agents in partially observable
settings therefore mostly use LSTMs, and more expressive external memories (DNC, MERLIN) are complex
and little adopted (Sec. 1, p. 2).

## Method

- **Base: Transformer-XL** (Sec. 2, pp. 2-3, Eq. 1-4, App. C.2, p. 19): per-timestep observation
  embeddings pass through L layers of multi-head attention and a position-wise MLP; each layer also
  attends to a T-step memory tensor M of earlier hidden states, held constant (StopGrad) during
  backprop, with Dai et al.'s relative position encodings.
- **Identity map reordering, TrXL-I** (Sec. 3.1, p. 4, Fig. 1 centre, Eq. 23-26, App. C.3, p. 20):
  layer norm is applied only to each submodule's input, so there is an untransformed path from the
  first layer's input to the last layer's output; a ReLU is applied to each submodule output before
  the residual add. Hypothesis: with near-zero submodule outputs at initialisation, the network
  starts as a Markovian policy and value function, so reactive behaviour is learnt before memory
  ("learn how to walk before it can learn how to remember", p. 4).
- **Gating, GTrXL** (Sec. 3.2, pp. 4-5, Eq. 5-8): the two residual connections become gating layers
  g(x, y) (x the stream, y the submodule output). Variants: input, output, highway, sigmoid-tanh
  and GRU-type gating (an untied GRU applied over depth, p. 4). **Gated identity initialisation**
  (p. 5): a bias b_g > 0 makes gates start near the identity map (b_g = 2 for GRU gating, 1 for the
  others, pp. 9-10).
- **Training** (Sec. 4, pp. 5-6, App. B, pp. 17-18): V-MPO (on-policy MPO with a learned V(s)) in an
  actor-learner setup on 16-core TPUs. Batch 128, unroll length 95, discount 0.99; 6-8 hyperparameter
  samples per model, shared across models.
- **Architecture** (Sec. 4, p. 5, Table 9, p. 18): 12 layers, 8 heads of dimension 64, embedding 256,
  memory 512, so an upper-bound receptive field of 6144 timesteps (12 × 512). LSTM baseline: 3
  layers of 256. Image encoder: a ResNet from Hessel et al. (App. A, p. 16); a 256-unit MLP head for
  policy logits, Gaussian parameters or values (p. 17).
- **Environments** (Sec. 4, App. A, pp. 15-17): DMLab-30 (72×96 RGB, reduced 15-action set, action
  repeat 4, a 64-d LSTM for language levels); Numpad (proprioceptive continuous control; activate an
  unknown sequence of up to N² pads, reward +1 per new correct pad, a wrong pad resets, 500-step
  episodes); Memory Maze (find an apple, get teleported, return using landmarks; 8 continuous and 1
  discrete action, hybrid policy).
- **Memory/reactive split** of DMLab-30 (App. D, p. 20, Table 13): 8 "memory" levels (rooms_select_nonmatching_object,
  rooms_watermaze, the small and large variants of explore_obstructed_goals and
  explore_goal_locations, and explore_object_rewards_few and _many: five families, eight levels)
  and 22 "reactive" ones (including psychlab_arbitrary_visuomotor_mapping); the split is the authors'.

## Assumptions

- A1: Agents are trained online from scratch with V-MPO (on-policy, learned state-value function, KL-constrained updates), batch 128, unroll length 95, discount 0.99 (Sec. 4, pp. 5-6, Table 8, p. 17).
- A2: The transformers are 12-layer Transformer-XL stacks with 8 heads of dimension 64, embedding 256 and memory 512, with gradients stopped at the memory; the LSTM baseline is 3 layers of 256 (Sec. 4, p. 5, Eq. 3, p. 3, Table 9, p. 18).
- A3: Gating layers are initialised near the identity map (b_g = 2 for GRU-type gating, 1 for the others) unless stated otherwise (Sec. 4.3.4, pp. 9-10).
- A4: DMLab-30 runs at action repeat 4 on 72×96 RGB frames with a reduced 15-action set and scores are human-normalized (human 100, random 0) per level, then averaged (Sec. 4.1, p. 6, App. A, pp. 15-16, Table 5).
- A5: The memory/reactive split of DMLab-30 is the authors' partition (8 memory, 22 reactive levels) (App. D, p. 20, Table 13).
- A6: Each model gets 6-8 hyperparameter samples (shared across models, drawn from Table 8 ranges); the LSTM on DMLab-30 uses a narrower ε_α range found better in preliminary runs; unfinished runs were dropped for compute reasons, not performance (App. B, p. 17, Table 9, p. 18).
- A7: Numpad observations are proprioceptive state plus which pads are active, the previous action and reward, encoded by a 2-layer tanh MLP; the sequence is unknown and contiguous pads are Moore neighbours (Sec. 4.2, p. 6, App. A, p. 15).
- A8: Memory Maze uses 72×96 RGB frames with a hybrid continuous-discrete action space (8 continuous, 1 discrete) and action repeat 4 (Sec. 4.3.2, p. 8, App. A, pp. 15-17, Table 7, Table 8).
- A9: MERLIN's score comes from its authors (personal communication), trained for 100B steps with IMPALA and auxiliary losses, so it isn't directly comparable to the 10B-step V-MPO runs (Sec. 4.1, p. 6, Fig. 2, p. 5).
- A10: The memory length the agent can attend to (memory 512 per layer, at most 6144 steps across 12 layers) spans any episode in the tested environments, while gradients flow only within the 95-step unroll (reader's inference, Sec. 4, p. 5, Table 8, p. 17).
- A11: Numpad compares LSTM and GTrXL (GRU) with 5 hyperparameter settings per model, not the 6-8 of DMLab-30 (Fig. 3, p. 7, Table 10, p. 18).
- A12: DMLab's Arbitrary Visuomotor Mapping runs with its own reduced action set (look left/right small, look down/up, no-op; Table 6), not the 15-action set of Table 5 (App. A, p. 16, Table 6, p. 16).

## Claims

- C1: The canonical transformer is hard to optimize in RL and often learns no better than a random policy: TrXL scores 5.0 ± 0.2 human-normalized on DMLab-30 [A1, A2, A4, A6] (Sec. 1, p. 2, Table 1, p. 7)
- C2: On DMLab-30 at 10B steps GTrXL (GRU) reaches a mean human-normalized score of 117.6 ± 0.3 (100-capped 89.1 ± 0.2), above LSTM 99.3 ± 1.0 (84.0 ± 0.4) and the MERLIN@100B result 115.2 uncapped, but slightly below MERLIN on the 100-capped score (89.1 vs 89.4) [A1, A2, A3, A4, A6, A9] (Table 1, p. 7, Sec. 4.1, p. 6)
- C3: GTrXL's gain over LSTM on DMLab-30 comes mostly from the memory levels, while on reactive levels it matches or slightly beats LSTM [A1, A2, A3, A4, A5, A6, A10] (Fig. 2, p. 5, Sec. 4.1, p. 6)
- C4: Identity map reordering alone helps: TrXL-I reaches 107.0 ± 1.2 (100-capped 87.4 ± 0.3), above LSTM but below GTrXL (GRU), and less robust [A1, A2, A4, A6] (Table 1, p. 7, Sec. 4.3.1, p. 7)
- C5: Among gating variants on DMLab-30, GRU gating is best (117.6) ahead of output 112.8 ± 0.8, sigmoid-tanh 101.0 ± 1.3, highway 90.9 ± 12.9 and input 51.2 ± 13.2; input gating does worse than TrXL-I, which the authors read as support for the identity-map hypothesis, and highway and sigmoid-tanh are more hyperparameter-sensitive [A1, A2, A3, A4, A6] (Table 1, p. 7, Sec. 4.3.1, p. 7, Fig. 4, p. 8)
- C6: On Numpad, LSTM does badly at all pad sizes and worse as size grows from 2 to 4, while GTrXL does much better and almost instantly solves it, though it also drops at 4×4 because of the combinatorial growth; LSTM trained twice as long still doesn't reach GTrXL on 2×2 and 4×4 [A1, A2, A3, A7, A11] (Fig. 3, p. 7, Sec. 4.2, p. 6)
- C7: On Memory Maze, over 25 sampled hyperparameter settings and seeds, GTrXL (GRU) learns in fewer steps than LSTM and is the most robust; it is the only model reaching human performance within 2B steps, with 10 runs at mean return 8 or more [A1, A2, A3, A8] (Fig. 5, p. 8, Sec. 4.3.2, pp. 8-9)
- C8: Over 25 settings within 2B steps, training loss diverged for 16% of TrXL-I runs and 12% of GTrXL (Output) runs, and for none of the LSTM, TrXL and GTrXL (GRU) runs [A1, A2, A3, A8] (Table 3, p. 8)
- C9: The advantage isn't only parameter count: a Thin GTrXL (GRU) with 4 heads and embedding 128 (22.4M parameters, fewer than TrXL's 28.6M) scores 111.5 ± 0.6 on DMLab-30, matching every model within variance except GTrXL (GRU) (66.4M), including GTrXL (Output) with over 10M more parameters [A1, A3, A4, A6] (Table 2, p. 8, Sec. 4.3.3, p. 9, Fig. 6, p. 9)
- C10: Gated identity initialisation greatly speeds learning: on Memory Maze a 4-layer GTrXL (GRU) with b_g = 2 is far ahead of b_g = 0 over 10 hyperparameter samples at 1B, 2B and 4B steps, which the authors read as preconditioning the transformer to be near-Markovian [A1, A8] (Fig. 7, p. 9, Sec. 4.3.4, pp. 9-10)
- C11: With a reduced action set as in MERLIN, GTrXL reliably reaches human-level performance on DMLab's Arbitrary Visuomotor Mapping task, which MERLIN's paper used to show its memory [A1, A2, A3, A12] (Fig. 9, p. 16, App. A, p. 15)
- C12: The authors conclude GTrXL matches LSTM's robustness to hyperparameters and seeds while performing better, and can act as a drop-in replacement for LSTMs in RL [A1, A2, A3, A4, A6, A8] (Sec. 6, p. 10, Sec. 1, p. 2)

## Implementation notes

- GRU-type gate over depth: r = σ(W_r y + U_r x), z = σ(W_z y + U_z x − b_g), ĥ = tanh(W_g y + U_g (r ⊙ x)), g(x, y) = (1 − z) ⊙ x + z ⊙ ĥ, with x the residual stream and y the ReLU'd submodule output (source: Sec. 3.2, p. 4, Eq. 6, Eq. 8, p. 4).
- The attention layer normalises the concatenation [StopGrad(M), E] before attention, i.e. memory and current segment share the input layer norm (source: Eq. 5, p. 4).
- V-MPO hyperparameters per environment, including initial η and α, ε ranges sampled log-uniformly, target update period 10, pixel-control cost 2e-3 on DMLab-30 (source: Table 8, p. 17).
- How the memory is carried between unrolls (whether actors keep the last 512 hidden states across unroll boundaries and episode resets) isn't spelled out; Transformer-XL's practice is to carry the previous segments' states, and masking at episode starts is a guess (source: assumption, after Dai et al. 2019).
- The 64-d language LSTM's output is presumably combined with the image embedding before the transformer; how isn't stated (source: assumption).
- Shmup lens: the memory GTrXL attends to is the last 512 hidden states per layer, which at a shmup's 60 Hz and one step per frame is about 8.5 s (up to about 100 s if each layer reaches 512 further back, the 6144-step bound); with action chunks as steps it would be about 2.3 to 2.6 min (512 × 16 or 18 / 60 s; the chunk is 16 in the NitroGen paper and 18 in the released checkpoint's config) per layer (source: assumption, arithmetic on Sec. 4, p. 5).
- Shmup lens: credit assignment here is still V-MPO with a learned V(s), discount 0.99 and 95-step unrolls; memory is a forward-only (StopGrad) cache, so the architecture separates what the agent can read from how far gradients and returns reach, but the paper doesn't measure these separately (source: assumption, from Eq. 3, p. 3 and Table 8, p. 17).

## Limitations / open questions

- GTrXL's performance also drops on 4×4 Numpad because the task's combinatorics grow (Fig. 3, p. 7).
- MERLIN's comparison isn't like-for-like (100B steps, IMPALA, auxiliary losses, scores by personal communication) (Sec. 4.1, p. 6).
- Scaling to larger, deeper transformers and more varied training environments is left to future work (Sec. 4, p. 5, Sec. 6, p. 10).
- Open question: the memory split is a partition of DMLab-30 levels, not a controlled sweep of memory length; Numpad is the only horizon sweep, and only by pad count, so how long a span GTrXL actually uses is not measured.
- Open question: for a shmup policy, does the near-Markovian start (C10) matter when the policy is pretrained by behaviour cloning, as NitroGen is, rather than learnt online from scratch?
- Open question: the tasks test recall of places and pad sequences rather than long-term strategy state (weapon level, economy, rank); a shmup memory test needs a task where an early decision changes a reward many seconds later.

## Related

- uses [[vaswani2017attention]]: GTrXL is a modified Transformer (Transformer-XL with relative positions and segment memory).
- related [[zambaldi2018relational]]: self-attention over entities in the input rather than over time, cited (in its ICLR 2019 form) as another use of attention in RL (Sec. 5, p. 10).
- related [[baker2022video]]: VPT's policy uses Transformer-XL-style memory, the architecture GTrXL stabilises for RL.
- related [[magne2026nitrogen]]: a reactive game policy with no long memory; GTrXL is one candidate for adding it.
- related [[hafner2024mastering]]: DreamerV3 keeps memory in a GRU recurrent state, the recurrent alternative this paper compares transformers with.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
