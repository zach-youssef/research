---
id: mu2023learning
title: Learning to Compress Prompts with Gist Tokens
authors: [Jesse Mu, Xiang Lisa Li, Noah Goodman]
year: 2023
venue: NeurIPS 2023
url: https://arxiv.org/abs/2304.08467
arxiv: 2304.08467
doi:
pdf_sha256: 50417a80cb655866ad97542a04f36d66810e5df00ef676d481a9b6b63e14fe08
tags: [memory-tokens, prompt-compression, gist-tokens, attention-mask, instruction-tuning, kv-cache]
status: read
read: 2026-11-25
---

# Learning to Compress Prompts with Gist Tokens

## TL;DR

Inserts k gist tokens after a prompt and masks attention so later tokens see only them, during
ordinary instruction finetuning. Compresses prompts up to 26x on LLaMA-7B and FLAN-T5-XXL with
quality near the uncompressed model, up to 40% fewer FLOPs and 4-7% lower wall time.

## Problem

Long prompts that are re-encoded for every query waste context and compute; finetuning or
distilling per task needs retraining for each task, and prefix-tuning needs gradient descent per
prompt (Sec. 1-2, pp. 1-2). Gisting wants a single model that predicts a compressed prefix G(t)
for any new prompt zero-shot (Sec. 2, p. 2).

## Method

- Setting: instruction data (t, x, y) with task prompt t, optional input x, output y; the goal is
  a compressed G(t) such that p_G(y | G(t), x) mimics p_LM(y | t, x) (Sec. 2, p. 2).
- The objective is a meta-distillation KL, Eq. 2, where G is the LM itself, which also makes it
  context distillation across a distribution of tasks (Sec. 2.1, Eq. 1-2, pp. 2-3).
- Implementation: one new gist token g is added to the vocabulary; input is (t, g_1..g_k, x);
  tokens after the gist tokens cannot attend to tokens before them (they can attend to the gist
  tokens), forcing the prompt into the gist prefix (Sec. 3, p. 3, Fig. 2a).
- Encoder-decoder: the encoder also stops x from attending to t and g from attending to x;
  decoder cross-attention cannot see t (Sec. 3, pp. 3-4, Fig. 2b).
- Training is plain instruction finetuning with the modified mask, no extra loss or cost
  (Sec. 3, p. 3). Gist activations are the key/value activations at the gist positions, i.e. a
  Transformer prefix, and can be cached (Sec. 2, p. 2).
- Each gist model is trained once on Alpaca+ with a fixed k; it does not accumulate or pass state
  across successive prompts: there is one compression step per prompt, not a recurrence
  (Sec. 3-4, pp. 3-4; reader's reading).

## Assumptions

- A1: Compression targets short instruction prompts of about 20-26 tokens on average, with up to 768 tokens in training and evaluation windows (Sec. 4.1, p. 4, App. B, p. 13).
- A2: Training data is Alpaca+ (130,321 examples, 104,664 unique prompts) whose outputs were sampled from GPT-3 variants text-davinci-001/003 (Sec. 4.1, p. 4).
- A3: The model is finetuned fully (LLaMA-7B, FLAN-T5-XXL 11B), not frozen, so it can adapt to the gist mask (Sec. 4.2, p. 4, App. B, p. 13).
- A4: Quality is judged by ROUGE-L, ChatGPT-3.5 pairwise win rate against the uncompressed positive control, and a human study (Sec. 4.3, p. 5).
- A5: Gist tokens are consumed at the next stage as a cached key-value prefix, one compression step per prompt (Sec. 2-3, pp. 2-3).

## Claims

- C1: Gist models with a single gist token reach 99.2% (LLaMA-7B) and 93.2% (FLAN-T5-XXL) of the positive control's ROUGE-L, normalised between negative and positive controls, on Seen prompts; 91.0% and 88.6% on Unseen; 75.4% and 80.9% on the Human split [A1, A2, A3, A4, A5] (Table 1, p. 6).
- C2: ChatGPT win rate against the positive control for 1-token gist models: Seen 48.6% (LLaMA) / 50.8% (FLAN-T5); Unseen 49.7% / 46.2%; Human 45.8% / 42.5% [A1, A2, A3, A4, A5] (Table 1, p. 6, Sec. 5, p. 5).
- C3: The number of gist tokens k (1, 2, 5, 10) makes little difference; too many hurt in some cases (LLaMA-7B with 10 tokens), so single-token models are used later; compression factors reach 26x on the Human split [A1, A2, A3, A4, A5] (Fig. 3, p. 6, Sec. 5, p. 5).
- C4: Discrete TF-IDF keyword compression at the same rate is far worse, only slightly above the negative control (e.g. LLaMA Human ROUGE-L 16.5 vs 14.4 negative, 27.0 positive) [A1, A2, A4] (Table 1, p. 6, Sec. 5, p. 5).
- C5: Human evaluation on 100 Human-split prompts: gist win rate over the positive control averages 52.3% (LLaMA-7B) and 40.6% (FLAN-T5-XXL), with ChatGPT's 48.0% and 42.0% in agreement; inter-annotator Cohen's kappa is low (.24 and .33) [A1, A2, A3, A4] (Table 2, p. 7, Sec. 5, pp. 5-7).
- C6: Gist outputs exactly match the positive control about 50% of the time on Seen tasks, 20-25% on Unseen and about 10% on Human [A1, A2, A4] (Fig. A.3, p. 19, Sec. 5, p. 7).
- C7: Compute: with a gist cache vs no caching, 40% FLOPs saving (LLaMA 915 to 552 GFLOPs) and 6.8% (LLaMA) / 4.2% (FLAN-T5) wall-time saving; vs caching the full instruction's KV, LLaMA gains only 0.11% FLOPs and about 1% wall time [A1, A5] (Table 3, Sec. 6, p. 8).
- C8: Storage: one KV-cache token of LLaMA-7B costs 1.05 MB in fp32, so gisting 26 prompt tokens into 1 lets about 26x more prompts be cached in the same storage [A1, A5] (Sec. 6, p. 8).
- C9: Failure cases: gist models can lose specific details (e.g. copying phrases from the instruction) and show runaway generations absent in the positive control; compression loses some nuance [A1, A2, A3] (Sec. 5.1, p. 7, Sec. 8, p. 9).
- C10: Encoder-decoder gist models generalise somewhat worse out of distribution than decoder-only; the authors suggest (unverified) that masking hurts the encoder's bidirectional flow [A3] (Sec. 5, p. 7).

## Implementation notes

- Gist mask: the PyTorch listing (about 10 lines) masks attention from tokens after the last gist token to tokens before the first gist token (source: Sec. 3, p. 4, App. A, p. 14).
- LLaMA-7B: 3000 steps, batch 128, lr 2e-5, warm-up 0.03, bf16, AdamW, max length 512 (768 for Human), DeepSpeed ZeRO-3 on 4 A100 80GB, about 7 h (source: App. B, Table A.1, p. 13).
- FLAN-T5-XXL: 16000 steps, batch 16, lr 5e-5, no warm-up, input max 128, output 256 (384 for Human), about 25 h (source: App. B, Table A.1, p. 13).
- Greedy decoding for evaluation (source: App. B, p. 13). Code and checkpoints: github.com/jayelm/gisting (source: App. B, p. 13; not inspected).
- Gist tokens share one embedding; only the activations differ by position (source: fn. 2, p. 3).
- Applying this to a frame policy: a "prompt" would be a span of past frames/actions and the gist would be written by masking rather than recurrence; none of that is specified (source: assumption, user's lens).

## Limitations / open questions

- Admitted: compression loses nuance and needs evaluation before deployment (Sec. 8, p. 9).
- Admitted: wall-time gains are small at these short prompts; larger gains expected for very long prompts such as k-shot prompts exceeding a context window (Sec. 6, p. 8, Sec. 8, p. 9).
- Admitted future work: retrofitting a frozen LM, and "gist pretraining" with other objectives (Sec. 8, p. 9).
- Open question (user's lens): gisting compresses a static prompt once, with no recurrence or accumulation; whether gist masking can be applied to rolling history (gist of the last N frames fed to the next window) is not tested here, and credit assignment is only the single-step masked loss.
- Open question: eval is win rate vs an uncompressed model, so what the gist encodes (and what it drops, e.g. exact values) is not probed beyond failure examples.

## Related

- related [[chevalier2023adapting]]: concurrent work compressing arbitrary context into accumulated summary vectors, cited in that paper as related work (not recurrent here).
- related [[bulatov2022recurrent]]: memory tokens that carry state recurrently; gisting has no recurrence.
- related [[vaswani2017attention]]: the attention mask is the only modification.
- related [[magne2026nitrogen]]: the frames-to-gamepad policy whose context a gist-style mask might shorten.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
