---
id: chevalier2023adapting
title: Adapting Language Models to Compress Contexts
authors: [Alexis Chevalier, Alexander Wettig, Anirudh Ajith, Danqi Chen]
year: 2023
venue: EMNLP 2023
url: https://arxiv.org/abs/2305.14788
arxiv: 2305.14788
doi:
pdf_sha256: 641d7ec338ae41551faa52a00912e5efea198ff01c48ae527a58a755c5a1e0d9
tags: [memory-tokens, context-compression, summary-vectors, long-context, soft-prompts, language-modeling]
status: read
read: 2026-11-25
---

# Adapting Language Models to Compress Contexts

## TL;DR

Fine-tunes OPT and Llama-2 to compress each segment into 50 accumulated summary vectors used as a
soft prompt, trained unsupervised with random segment lengths and stop-gradients. Cuts perplexity
over up to 30,720 tokens, and compresses ICL demonstrations and retrieved passages.

## Problem

LM context windows are finite and long-context inference is expensive (Abstract, p. 1).
Recurrent Memory Transformer (RMT) extends context with memory tokens but passes only the
previous segment's summary, and is costly to fine-tune from pre-trained models; Wingate et al.'s
prompt compression needs per-context optimisation (Sec. 1-2, pp. 1-2).

## Method

- Extends the vocabulary with kappa summary tokens <Sum>_i; appending them to a segment makes the
  model output summary vectors (last-layer states at those positions), reused as a soft prompt of
  length kappa for the next segment (Sec. 3, p. 3, Fig. 1, p. 1). Their embeddings are initialised
  from the embedding of </s> (fn. 3, p. 3).
- Summary accumulation: the document is split into S_1..S_n; the summaries sigma_1..sigma_(i-1)
  of all earlier segments are concatenated and prepended to S_i, so the soft prompt grows as
  (i-1)*kappa; this gives every segment a direct path to all earlier ones, unlike RMT
  (Sec. 3, p. 3).
- Positional embeddings: with OPT's absolute positions, none is added to summary tokens or
  vectors (order is kept by their token embeddings), so any number of compression steps can be
  used; with RoPE, the usual position is applied (Sec. 3, p. 3).
- Objective: unsupervised cross-entropy next-token loss over the whole document, each segment
  conditioned on the accumulated summaries; no distillation (Sec. 3.1, Eq. 1, p. 3).
- Randomized segmenting: segment lengths m_i are randomly varied (each fitting the context
  window), so the model compresses contexts of variable length (Sec. 3.1, pp. 3-4).
- BPTT with stop-gradients: BPTT with gradient checkpointing per segment; summary vectors are
  computed and cached, and their gradients stopped after 2 compression steps; this assumes that
  learning to compress S_i only needs to predict the adjacent S_(i+1) (Sec. 3.1, p. 4).
- Use: ICL demonstrations compressed into summaries (Sec. 5, p. 6), fused summaries of retrieved
  passages (Sec. 6.1, pp. 7-8, App. F, p. 15), passage re-ranking (Sec. 6.2, p. 8).

## Assumptions

- A1: Inputs are text tokens; the model is a pre-trained decoder LM (OPT 1.3B/2.7B, Llama-2 7B) adapted by fine-tuning (Sec. 4, p. 4).
- A2: Training is next-token prediction on long documents, split into 2-20 segments of 1,024-2,048 tokens (Sec. 4.1-4.2, p. 4, App. A, p. 13).
- A3: Only the summary vectors, not raw earlier tokens, carry earlier context into later segments at evaluation (Sec. 4, pp. 4-5).
- A4: Stopping gradients after 2 compression steps loses nothing, i.e. compression only needs to serve the adjacent segment (Sec. 3.1, p. 4).
- A5: Fine-tuning compute is a single NVIDIA A100 80GB, with FlashAttention and gradient checkpointing; Llama-2 uses LoRA (Sec. 4.2-4.3, p. 5, App. A, p. 13).

## Claims

- C1: With OPT-2.7B and kappa=50, AutoCompressor lowers held-out perplexity on 2,048 tokens as more previous segments are compressed, to 5.93 in-domain and 8.10 out-of-domain at 6,144 context tokens (3 segments), against 6.28 / 8.53 for the fine-tuned OPT without context; RMT reaches 6.01 / 8.20 [A1, A2, A3, A4, A5] (Table 1, p. 5).
- C2: Extended full attention on 2,048 tokens of plain context is about as good (5.94 in-domain) but it needs 2,048 extra tokens and cannot go beyond 4,096, while AutoCompressor attends to at most 150 soft prompts for 6,144 tokens; for short contexts (128-512) AutoCompressor is better, but the gap to full attention is larger out of domain [A1, A2, A3, A5] (Table 1, p. 5, Sec. 4.1, p. 4).
- C3: Fine-tuned on 30,720-token Books3 documents, AutoCompressor-2.7B improves from 11.86 (no context) to 11.21 (14,336 tokens) and 11.18 (28,672 tokens), while RMT-1.3B stays at 12.50 for both and the RMT-2.7B run runs out of memory; AutoCompressor-1.3B uses 38 GB vs RMT's 54 GB, AutoCompressor-2.7B 75 GB [A1, A2, A3, A4, A5] (Table 2, p. 5, Sec. 4.2, pp. 4-5).
- C4: Llama-2-7B AutoCompressor (LoRA, trained on 6,144 tokens, 15B RedPajama tokens) gets 5.07 perplexity with 6,144 compressed tokens vs 5.52 for Llama-2 with no context; compressing 4,096 tokens into 100 vectors (5.08) matches extended full attention with 512 plain tokens (5.06), while extended full attention on 6,144 plain tokens reaches 4.76 [A1, A2, A3, A5] (Table 3, p. 5, Sec. 4.3, p. 5).
- C5: Removing summary accumulation, randomized segmenting or RMT-style design hurts: randomized segmenting helps short contexts, accumulation helps beyond one segment, and stopping gradients does not change perplexity while reducing GPU memory [A1, A2, A5] (Fig. 2, Sec. 4.4, p. 6, Table 2, p. 5).
- C6: kappa=50 summary tokens performs best overall among 20, 50, 70, 100 (6.93 at 6,144 tokens vs 7.00, 6.95, 7.00); more summary vectors do not help, which the authors guess needs a larger training budget [A1, A2] (Table 7, p. 14, Sec. 4.4, p. 6).
- C7: Summary vectors preserve short-context quality: with no context, the AutoCompressor's perplexity is within 1% of the fine-tuned OPT-2.7B (6.31 vs 6.28 in-domain) [A1, A2] (Table 6, p. 14).
- C8: Perplexity gains from summary vectors appear over the whole 2,048-token segment and are larger than extended full attention toward its end; tokens that benefit most are names, dates and places, sometimes inferred rather than copied [A1, A2, A3] (Fig. 5, App. D, p. 14, Sec. 4.4, p. 6, Table 9-10, pp. 16-17).
- C9: Compressing ICL demonstrations (1-3 segments of up to 750 tokens; about 30 SST-2 demonstrations per segment, so up to about 90) into 50/100/150 summary vectors with the Llama-2-7B AutoCompressor beats 150 plain demonstration tokens on 8 of 11 tasks and 750 tokens of plain demonstrations on 8 tasks (AG News, SST-2, BoolQ, WiC, WSC, CB, COPA, MultiRC), and the OPT AutoCompressor beats the RMT baseline on 8 of 11 tasks; accuracy is not monotone in the number of summary vectors [A1, A2, A3] (Table 4, Sec. 5, p. 7, Table 12, p. 18).
- C10: The Llama-2 AutoCompressor has clearly worse zero-shot accuracy on some tasks and slightly worse ICL than the original model, which the authors attribute to fine-tuning-data domain mismatch (a suspicion) [A1, A2] (Sec. 5, p. 7, Table 12, p. 18).
- C11: Fused Summaries of retrieved passages (no extra fine-tuning; OPT-2.7B, 10B-token index, top-k from Contriever) beat REPLUG and Fused Passages at 50-token passages (3.74% perplexity gain at top-10 vs 2.35% / 2.60%); Fused Summaries top-10 beat REPLUG top-2 with 512-token passages with 1.7x throughput, but REPLUG top-10 with 512-token passages (8.30%) beats it; summary storage is about 5 TB per domain [A1, A3] (Table 5, Sec. 6.1, p. 8).
- C12: Pre-computed summary vectors for 21M Wikipedia passages (2.1 TB at kappa=20, 5.4 TB at kappa=50, half precision) put re-ranking on the Pareto front of Recall@20 vs throughput, versus OPT models of 125M-2.7B with plain text [A1, A3] (Fig. 4, Sec. 6.2, pp. 8-9).
- C13: Admitted limits: summary vectors drop some information that full attention retains, more vectors do not always help, and accumulation is still quadratic in the number of segments (at a lower rate than full attention) [A1] (p. 9).

## Implementation notes

- OPT fine-tuning: lr 2e-5, batch 130K tokens, 1,000 warm-up steps, Adam, 2B Pile tokens (source: App. A.1, p. 13). The 8K setup is trained on up to 6,144 tokens in four segments of 1,024-2,048, first two summing to 3,072 (source: App. A.1, p. 13).
- 30K setup: 20 segments, segments 2i and 2i+1 sum to 3,072 tokens, each 1,024-2,048; Books3 only (Gutenberg held out) (source: App. A.2, p. 13, fn. 4, p. 4).
- Llama-2: LoRA rank 16 on QKV and output projections plus new summary-token embeddings, lr 4e-4, batch 200K tokens, 5,000 warm-up steps, 15B RedPajama tokens (source: App. A.3, p. 13, Sec. 4.3, p. 5).
- RMT baselines, kappa=50, no accumulation, randomized segmenting or stop-gradients: 8K setup RMT-2.7B on four segments of 2,048 (source: Sec. 4.1, p. 4, App. A.1, p. 13); 30K setup RMT-1.3B on 15 segments of 2,048, the 2.7B run runs out of memory (source: App. A.2, p. 13).
- ICL uses contextual calibration and balanced sampling chosen per task on validation, 7 seeds, GPT-3 prompt templates (source: Sec. 5, p. 6, Table 11, p. 18).
- Code and models: github.com/princeton-nlp/AutoCompressors (source: footnote 1, p. 1; not inspected).
- Sampling of a policy's "segment" is not defined for frames; one would replace the segment by a window of frames and actions (source: assumption, user's lens).

## Limitations / open questions

- Admitted: only OPT up to 2.7B and Llama-2 7B tested; summary vectors ignore some information full attention has; quadratic growth with segments (p. 9).
- Admitted: authors suspect the training signal for summaries is weak because pre-trained models predict well from plain tokens in the current segment (p. 9).
- Open question (user's lens): compression is unsupervised via next-token loss and accumulates one summary per segment; for a policy, would the signal be action-prediction loss, and how large does the accumulated prompt get over a whole run (n*kappa tokens)?
- Open question: stop-gradient after 2 compression steps means the training horizon for credit is about 2 segments, even though evaluation reaches 20 segments; does this carry over to long-term strategy that pays off later?
- Open question: the model is only evaluated on perplexity, ICL and retrieval; nothing measures fine factual retention (e.g. a counter) over many steps; for an RAM-blind agent, probing summaries against RAM offline is a possible check.

## Related

- extends [[bulatov2022recurrent]]: builds on RMT, adding summary accumulation, randomized segmenting and stop-gradients.
- related [[vaswani2017attention]]: the underlying transformer and its context-length cost.
- related [[magne2026nitrogen]]: the frames-to-gamepad policy whose context AutoCompressor-style summaries might extend.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
