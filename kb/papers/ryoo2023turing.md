---
id: ryoo2023turing
title: Token Turing Machines
authors: [Michael S. Ryoo, Keerthana Gopalakrishnan, Kumara Kahatapitiya, Ted Xiao, Kanishka Rao, Austin Stone, Yao Lu, Julian Ibarz, Anurag Arnab]
year: 2023
venue: CVPR 2023
url: https://arxiv.org/abs/2211.09119
arxiv: 2211.09119
doi:
pdf_sha256: 9dd7e3d6f4b7d69d396e29905175c0bebfe314cd3e487232290c9702375cf0c9
tags: [memory, memory-tokens, recurrent-transformer, neural-turing-machine, token-summarization, video, online-inference, robotics, behavioral-cloning, long-horizon]
status: read
read: 2026-11-25
---

# Token Turing Machines

<!-- Read from arXiv v2 (2023-04-13, CVPR 2023 camera-ready), 12 PDF pages: pp. 1-8 main text with App. A.1-A.2 on p. 8, pp. 9-12 references. -->

## TL;DR

A recurrent transformer with 96 memory tokens: each step reads memory plus input into 16 tokens,
processes them and writes memory back by token summarisation, at constant per-step cost. It beats
causal/recurrent transformers on Charades online detection (26.34 mAP) and lifts robot success ~80% → 89.26%.

## Problem

Streaming and sequential decision tasks (online activity detection, robot policies) need to summarise
past observations at bounded compute per step; re-running a transformer over the whole history grows in
cost with its length, and memory-augmented transformers (Transformer-XL, MeMViT) still grow linearly
(Sec. 1, p. 1, Sec. 2, p. 2). Neural Turing Machines had explicit memory but were hard to train
(Sec. 2, p. 2).

## Method

- **Token summarisation** S_k (Sec. 3.1.1, Eq. 1-3, p. 3): reduce p tokens V to k tokens as weighted
  sums, z_i = softmax(α_i(V)) · V, with weights from an MLP over V (default) or a learned query
  (latent-query attention), following TokenLearner and Perceiver.
- **Read** (Sec. 3.1.2, Eq. 4, p. 4): Z^t = S_r([M^t ‖ I^t]), r tokens from the concatenated m memory
  tokens and n input tokens; a learnable positional embedding added before each read lets it address
  by location as well as content and tells memory from input.
- **Process** (Sec. 3.1.3, p. 4): a Transformer (or MLPMixer) maps Z^t to r output tokens O^t; a linear
  head gives the per-step prediction Y^t = W_o O^t.
- **Write** (Sec. 3.1.4, Eq. 5, p. 4): M^(t+1) = S_m([M^t ‖ O^t ‖ I^t]); memory tokens not re-selected are
  erased. Read, Process, Write and Output form an RNN (Eq. 6-9, p. 4).
- **Tokens per step** (Sec. 4.1.2, p. 5): Charades uses ViViT-B over 32-frame segments, 3,136 tokens
  per step (14 × 14 × 16), or 16 after spatial pooling; memory m = 96, reads r = 16 (or 32), a 4-block
  processing unit with hidden size 512, trained on 6-step windows (192 frames). Robot: n = 48 input
  tokens from EfficientNet, 8-layer Transformer, 8 steps (Sec. 4.2.2, p. 8).
- **Temporal redundancy** (Sec. 3.1, p. 3): inputs that the memory already holds are redundant, so
  read summarises memory and input jointly and passes only r tokens to the processing unit.

## Assumptions

- A1: Per-step inputs are tokens from a pretrained backbone: ViViT over 32-frame segments for video, EfficientNet outputs as input tokens for the robot, with FiLM language conditioning stated for the BC-Zero baseline and assumed for TTM (reader's inference, Sec. 4.2.2, pp. 7-8).
- A2: Training unrolls fixed windows: 6 steps (Charades, loss on the last step only), 4 steps (AVA), 8 steps (robot) (Sec. 4.1.2, p. 5, Sec. 4.1.5, p. 7, Sec. 4.2.2, p. 8, App. A.1, p. 8).
- A3: Memory m = 96 tokens and r = 16 reads unless stated (Sec. 4.1.2, p. 5, Sec. 4.2.2, p. 8).
- A4: Inference is online: predictions for a frame use no future frames (Sec. 4.1, p. 5).
- A5: Charades videos average 30 s with ~6.8 overlapping activities each, and evaluation samples 25 frames per validation video ('charades_v1_localize') (Sec. 4.1.1, p. 5).
- A6: The robot policy is behaviour-cloned on 89,000 teleoperated episodes over 551 instructions in one mock kitchen and evaluated in the same kitchen; inputs are the head-camera image, previous actions and the instruction (Sec. 4.2.1, p. 7, Sec. 4.3, p. 8).
- A7: Reported GFLOPs are per step and exclude the backbone (Sec. 4.1.3, p. 6, Table 2, p. 6).

## Claims

- C1: On Charades online temporal activity detection, TTM raises ViViT-B from 23.18 to 26.34 mAP and ViViT-L from 26.01 to 28.79, above the other online methods listed (I3D 17.22, X3D 18.87) [A1, A2, A3, A4, A5] (Table 1, p. 5).
- C2: At n = 16 input tokens, TTM-Transformer uses 0.228 GFLOPs per step for 26.24 mAP vs causal Transformer 0.523 (25.85) and recurrent Transformer 0.410 (25.97), about half the compute for higher accuracy [A1, A2, A3, A4, A5, A7] (Table 2, Sec. 4.1.3, p. 6).
- C3: Other temporal models gain nothing from more input tokens (temporal Transformer 25.53 mAP at 3,360 tokens and 112.8 GFLOPs), which the authors attribute to overfitting, while TTM with n = 3,136 reaches 26.34 at 0.842 GFLOPs; an LSTM gets 23.96 [A1, A2, A3, A4, A5, A7] (Table 2, Sec. 4.1.3, p. 6).
- C4: Processing units: Transformer 26.34 mAP (0.842 GFLOPs), MLPMixer 26.14 (0.704), plain MLP 23.34 (0.689) [A1, A2, A3, A4, A5, A7] (Table 3, p. 6).
- C5: Summarisation functions: latent query 26.75 mAP (8.537 GFLOPs), MLP 26.34 (0.842), parameter-free pooling 25.75 (0.206) [A1, A2, A3, A4, A5, A7] (Table 4, p. 6).
- C6: Memory updates: TTM's summarised write 26.34 mAP vs concatenating every input token (Memorizing-Transformer style) 20.97, NTM-style erase-and-add 25.86, and TTM with memory zeroed after each step (same compute) 22.65 [A1, A2, A3, A4, A5, A7] (Table 5, p. 6).
- C7: On AVA v2.2 (Kinetics-400 pretraining), TTM raises ViViT-B from 25.2 mAP to 27.9 (per video), 31.3 (per box, 1 layer) and 31.5 (per box, 4 layers) at 0.8, 1.0 and 2.0 added GFLOPs, a larger gain than MeMViT's +2.3 over MViT [A1, A2, A3, A7] (Table 6, Sec. 4.1.5, p. 7).
- C8: On the real robot, TTM averages 89.26% success vs 79.80% (BC-Zero ResNet), 80.08% (BC-Zero EfficientNet) and 79.26% for TTM with zeroed memory, with per-type gains largest on Pick and Upright but TTM lower on Knock than every baseline [A1, A2, A3, A6] (Fig. 5, Fig. 6, Sec. 4.3, p. 8).
- C9: The authors saw no training instabilities with TTM in their experiments, unlike the NTM's reputation [A2] (Sec. 3.2, p. 4).
- C10: Per-step compute is constant in the length of the history, since each step touches only m memory tokens and the current n inputs [] (Sec. 1, p. 1, Sec. 3.1.2, p. 4).

## Implementation notes

- Charades: ViViT initialised from JFT then Kinetics-400 (Base) or Kinetics-600 (Large); batch 32, Adam lr 1e-4 (backbone × 0.1), cosine schedule, 100 epochs on 32 TPUv3; colour/scale jitter, RandAugment, mixup; sigmoid cross-entropy with label smoothing 0.1 (source: App. A.1, p. 8).
- AVA: bounding-box proposals from SlowFast; TTM computes the feature mask pooled per box (source: Sec. 4.1.5, p. 7).
- Robot training follows SayCan's setup (source: App. A.2, p. 8).
- Code: github.com/google-research/scenic/tree/main/scenic/projects/token_turing, no commit given (source: p. 1).
- The initial memory M^0 is not stated; zeros match the paper's zeroed-memory ablation (source: assumption, check the authors' code).
- How the r output tokens are pooled before the linear head is not stated (source: assumption: mean pooling).
- Gradients flow through the 6-step unroll (backpropagation through time); truncation beyond the window is implied by the fixed windows but not stated (source: assumption).
- Typos in the paper that the note corrects silently otherwise: Table 5's caption gives n = 3176 and Figs. 2-3 give n = 3076 input tokens per step, where the text uses 3,136; Eq. 5 writes S_n for the write summariser, which the note writes S_m (the text says it selects the memory size); Eq. 4 writes X^t for the input I^t (source: Table 5, p. 6, Figs. 2-3, pp. 3-4, Eqs. 4-5, p. 4).
- For a shmup policy (the KB's lens): one TTM step per 18-frame action chunk, with the frame tokens of the chunk as I^t and the action chunk decoded from O^t, keeps per-chunk cost fixed however long the run (source: assumption).

## Limitations / open questions

- Discussion point, not an admitted limitation: the paper says it targets vision tasks needing long token sequences (18,816 tokens for 6 Charades steps), comparable to Long Range Arena lengths (Sec. 5, p. 8).
- Open question: training windows are short (6 steps, 192 frames); the paper doesn't report how far beyond the training window memory stays useful.
- Open question: long-term shmup state (weapon system level, in-game economy, rank) builds up over minutes; would 6- to 8-step windows teach TTM to keep it, or does it need longer BPTT or auxiliary losses?
- Open question: write-by-reselection erases anything not reselected each step; can rarely changing facts (current weapon) survive many steps of busy input, or do they need reserved memory slots?
- Open question: memory tokens could be probed offline against game RAM (weapon, rank, lives, money) to check what they encode, while the agent itself sees only frames; RAM-derived score can also serve as reward.
- Open question: would memory written in one game transfer to another of the same genre, given the memory has no fixed semantics per slot?

## Related

- related [[vaswani2017attention]]: the processing unit is a standard Transformer.
- related [[magne2026nitrogen]]: a frames-in, gamepad-out policy without memory; TTM is one way to add bounded memory to such a policy.
- related [[bulatov2022recurrent]]: RMT also carries memory tokens between segments, but writes them as transformer outputs rather than by summarisation.
- related [[cherepanov2026recurrent]]: RMT-style memory tokens in an offline-RL policy, the closest setting to a game policy.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
