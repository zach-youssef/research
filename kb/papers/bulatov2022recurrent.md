---
id: bulatov2022recurrent
title: Recurrent Memory Transformer
authors: [Aydar Bulatov, Yuri Kuratov, Mikhail S. Burtsev]
year: 2022
venue: NeurIPS 2022
url: https://arxiv.org/abs/2207.06881
arxiv: 2207.06881
doi:
pdf_sha256: 1d68ae152a85755944574e3a1b0e49407b0133beee558c3d0fa586196c5e91d6
tags: [memory-tokens, recurrence, long-context, transformer, bptt, language-modeling]
status: read
read: 2026-11-25
---

# Recurrent Memory Transformer

## TL;DR

Adds m learned memory tokens at the start (read) and end (write) of each segment of a decoder-only
transformer, the write outputs feeding the next segment, trained by BPTT through 0-4 segments. Beats
Transformer-XL on copy/reverse over many segments, and matches it on WikiText-103 with less memory.

## Problem

Self-attention mixes local and global information in the same element-wise representations and
its cost is quadratic in input length, which limits sequence length (Sec. 1, pp. 1-2).
Transformer-XL's cache carries m x N vectors per segment (all N layers) and stores information
inside token states; the paper wants a compact, dedicated memory that needs no change to the
backbone (Sec. 1, Sec. 3, pp. 2-4).

## Method

- Input of a segment is extended with m memory tokens (real-valued vectors) at the start and the
  same m at the end: H~0 = [H_mem o H0 o H_mem] (Sec. 3, p. 4).
- The start group is "read memory": sequence tokens attend to it. The end group is "write
  memory": it attends to all segment tokens and, after the N layers, its output H_write is the
  updated memory (Sec. 3, p. 4).
- Recurrence: H_mem(tau+1) := H_write(tau); the next segment's input is built from it. Segments
  are processed sequentially, tokens inside a segment in parallel (Sec. 3, p. 4).
- Memory tokens attend to all tokens in their read/write block; the causal mask applies only to
  the sequence tokens (Sec. 3, p. 4, Fig. 6d, p. 9). Write tokens at the end are needed because
  in a decoder-only model start tokens cannot see later tokens (Sec. 3, p. 4).
- Training is BPTT: gradients flow through the memory between segments, unlike Transformer-XL's
  stop-gradient. The number of previous segments to backpropagate through (`#BPTT_unrolls`) is
  varied from 0 to 4; deeper is costlier in GPU memory (Sec. 3, pp. 4-5).
- Memory sits only at the input and output, so an existing transformer can be augmented
  (the paper does this for BERT, RoBERTa, DeBERTa, T5) (Sec. 4, p. 5, Sec. 5, pp. 7-8).
- Tr-XL + RMT is also tried: the Tr-XL cache as short-term memory, RMT as long-term; for
  that, an auxiliary loss on the memory tokens with weight 0.01 and a fixed special token
  predicted from memory is used (Sec. 5, p. 7, App. A.5, p. 15).
- No loss is imposed on the memory content in the base model: it is trained only through the
  task loss of later segments (Sec. 3, p. 4; reader's reading).

## Assumptions

- A1: Text/symbol sequences are split into fixed-length segments processed in order; memory is
  carried segment to segment, and training uses the same segmentation (Sec. 3-4, pp. 4-5).
- A2: Training uses BPTT through memory, and the stated results use 0-4 unrolled previous
  segments (Sec. 3, pp. 4-5).
- A3: The tasks are supervised sequence prediction (next-token loss), on synthetic algorithmic
  data or language modelling, not reinforcement learning or perception (Sec. 4, p. 5).
- A4: Inputs are discrete tokens (character or word), with a decoder-only backbone
  (Sec. 4, p. 5; reader's inference for "discrete").
- A5: Model size is small (12-16 layers, hidden 410-512 for LM; 4-6 layers for algorithmic
  tasks) (Sec. 4, p. 5, App. A.1, p. 14).

## Claims

- C1: On Copy and Reverse, RMT (no BPTT) stays near perfect as segments grow to 9 and 6 while Transformer-XL with the same memory/cache size (= segment length) falls to the no-memory baseline at 9 segments (copy) [A1, A3, A4, A5] (Fig. 3, Sec. 5, p. 6).
- C2: On copy at 360 tokens split into up to 9 segments, RMT stays perfect while Tr-XL fails from 6 segments; with segment 120 and a fixed 60-token memory (Fig. 4b), RMT degrades only beyond about 720 tokens of source+target [A1, A3, A4, A5] (Fig. 4, p. 7).
- C3: On Associative Retrieval RMT solves the task up to 4 segments with Tr-XL close behind; at 5 segments RMT drops slightly and Tr-XL average rises higher [A1, A3, A4, A5] (Fig. 3c, Sec. 5, p. 6).
- C4: On quadratic equations (180 tokens, 6 segments, memory 30) RMT reaches 0.99 +- 0.002 accuracy, equal to the unsegmented baseline's 0.99, while Tr-XL gets 0.93 +- 0.02 [A1, A3, A4, A5] (Table 1, p. 6).
- C5: On WikiText-103 with 150-token segments, RMT with memory 10 or 25 (BPTT 3 / 2) reaches perplexity 25.04 / 24.85, comparable to Tr-XL with memory 75 (24.68) and better than MemTr (29.63) and the no-memory baseline (29.95); Tr-XL (ours) with 150 memory gets 24.12 [A1, A2, A3, A4, A5] (Table 2, p. 7).
- C6: With 50-token segments (more recurrence steps), RMT with memory 1 and BPTT 3 (28.40, Table 5) beats Tr-XL with memory 10 (28.98); RMT with 5 memory tokens is on par with Tr-XL with cache 50, and RMT's gain from more memory fades from 5 to 50 tokens [A1, A2, A3, A4, A5] (Table 2, p. 7, Fig. 5b, p. 8, Table 5, p. 17).
- C7: For a fixed memory size, more BPTT unrolls improve RMT perplexity markedly (e.g. memory 10, 50-token segments: 30.69 at 0 unrolls to 26.25 at 4), but the paper observes instabilities and out-of-memory problems for larger memory with deeper unrolls [A2, A3, A5] (Fig. 5a, p. 8, Table 5, p. 17, Sec. 5, p. 7).
- C8: Combining the Tr-XL cache with RMT memory gives the best WikiText-103 result, 23.99 (cache 150 + memory 10, 2x steps) vs 24.12 for Tr-XL alone [A1, A2, A3, A4, A5] (Table 2, p. 7).
- C9: On enwik8 (segment 128 chars), RMT with memory 5 (BPTT 2) gets 1.231 bpc, close to Tr-XL with memory 40 (1.230) [A1, A2, A3, A4, A5] (Table 4, p. 16, Sec. 5, p. 7).
- C10: Augmenting pretrained BERT/RoBERTa/DeBERTa/T5-base (512 input tokens) with 10 memory tokens and recurrence and fine-tuning on Hyperpartisan news (up to 4 segments, about 2000 tokens) gives F1 up to 98.11 (RoBERTa-base, 4 segments), above the listed long-input models (best 96.60) [A1, A2, A3] (Table 3, p. 8).
- C11: Attention maps show RMT learning separate write (sequence tokens into memory, in reverse order for Reverse), read, and rewrite (read memory to write memory) operations, whereas Tr-XL must mix token representations across segments; with memory 6 for a 12-token segment RMT compresses the whole segment into memory [A3, A4] (Sec. 5, pp. 8-9, Fig. 6, p. 9, Fig. 9, p. 18).

## Implementation notes

- The code is the LM-RMT repository, built on the Transformer-XL code (source: footnote 1, p. 2, github.com/booydar/LM-RMT; not inspected).
- LM: WikiText-103 16 layers, 10 heads, hidden 410, FF 2100; enwik8 12 layers, 8 heads, hidden 512, FF 2048; Adam, linear schedule from 0.00025 for 200k (WikiText) / 400k (enwik8) steps (source: Sec. 4, p. 5).
- Algorithmic tasks: lr 1e-4, reduce-on-plateau factor 0.5, 4 layers/4 heads (copy, reverse), 6 layers/4 heads (associative retrieval), 6 layers/6 heads (quadratic), no loss on the source part (source: App. A.1-A.3, pp. 14-15).
- Memory token initialisation, and whether read and write memories share parameters, are not stated (source: assumption, check the authors' code).
- Compute: 1080Ti/V100/A100, from 1 hour up to 3-4 days per run, LM runs 10-30 h on 2 A100 80GB (source: App. A, pp. 14-15).
- For a frame-based policy, the "segment" would be a window of frames and actions; memory slots would be latent vectors with no discrete vocabulary (source: assumption, user's lens).

## Limitations / open questions

- Admitted: training instabilities and GPU-RAM growth with deeper BPTT and larger memory (Sec. 5, p. 7; checklist, p. 13).
- Admitted: RMT must store information for more steps in the same memory when segments shrink, making LM harder (Sec. 4, p. 5).
- Admitted: memory is theoretically unbounded in horizon but practically limited by capacity and access efficiency (Sec. 1, p. 2).
- Open question (user's lens): memory is written by the model itself and passed on, the "self-written summary" idea; but only BPTT depth 0-4 segments was tried, so how far a policy would carry strategy (weapon, economy, rank over thousands of frames) is untested.
- Open question: BPTT 0 solved copy/reverse here, but LM needs BPTT; would a policy need BPTT through many segments, and at what memory cost per frame window?
- Open question: no memory-content loss is used; for a RAM-blind agent, can an offline probe from memory tokens to RAM variables (weapon level, score) show what is stored?
- Open question: all tasks are discrete tokens with exact-recall targets; behaviour on continuous perception and noisy rewards is not tested.

## Related

- extends [[vaswani2017attention]]: leaves the transformer unchanged and adds recurrent memory tokens at input and output.
- related [[lee2022multigame]]: a Decision Transformer whose context is a short window of frames; RMT would be one way to extend its horizon.
- related [[magne2026nitrogen]]: frames-in, gamepad-out policy whose context is a few frames; the target setting for memory tokens.
- related [[reed2022generalist]]: Gato's fixed context length is the limit memory tokens would relax.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
