---
id: zhang2024beacon
title: Long Context Compression with Activation Beacon
authors: [Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, Zhicheng Dou]
year: 2024
venue: arXiv preprint
url: https://arxiv.org/abs/2401.03462
arxiv: 2401.03462
doi:
pdf_sha256: 22b123483f23ff87f94bacd75e548e16125b108f8dbc6e46954ccffe967a8098
tags: [memory-tokens, context-compression, kv-cache, long-context, beacon-tokens, llm]
status: read
read: 2026-11-25
---

# Long Context Compression with Activation Beacon

Version read: arXiv v3 (11 Oct 2024, "Preprint. Under review", 17 pages). Earlier versions
(v1 Jan 2024, v2) were not read and may differ in title and numbers.

## TL;DR

A plug-in for a frozen LLM: beacon tokens are interleaved into each chunk and only their per-layer
keys/values are accumulated (raw tokens' activations dropped). Matches the fine-tuned uncompressed
model on LongBench at 32K (adaptive x2/x4/x8 on Llama-2, x4 on Qwen-2), 2x faster at 128K.

## Problem

LLMs pay quadratic attention cost and a KV cache that grows with context. Prior context
compression squeezes a context into a few soft tokens (ICAE, AutoCompressor) that must be
re-encoded, are fixed-ratio and compress all-at-once; token deletion (LongLLMLingua, SnapKV)
depends on the question or on attention that fails past the window (Sec. 1, p. 2).

## Method

- Chunks of size w (1024 for Llama-2, 2048 for Qwen-2) are split into units of alpha tokens; a
  beacon token is placed at the end of every unit, giving k = w/alpha beacons per chunk
  (Eq. 1-2, Sec. 3.1, p. 4). One shared beacon embedding is used for all beacons (Sec. 3.2, p. 6).
- The LLM encodes one chunk at a time, attending to the accumulated beacon activations of all
  earlier chunks plus the current chunk. The keys/values of beacon tokens at every layer are
  appended to the accumulated cache (K^ac, V^ac, Eq. 7-8); raw tokens' activations are dropped
  (Fig. 1, Fig. 2, pp. 3-4, Eq. 3-8, p. 5).
- Only beacon tokens use new Q/K/V projection matrices W^b; all other LLM parameters are frozen
  (Eq. 5, p. 5; Sec. 3.2, p. 6). Generation and compression happen in the same forward pass.
- Interleaving makes each beacon attend over a different span, which the paper calls
  fine-grained compression; the alternative of putting all beacons at the chunk end is ablated
  (Sec. 3.1, p. 5; Sec. 4.6, p. 10).
- Training: compression-based next-token loss over all tokens except the first chunk (Eq. 11,
  p. 6). Because chunk l depends only on layer l-1 outputs of earlier chunks, gradients flow
  through all chunks (no stop-gradients, unlike RMT / AutoCompressor). The ratio alpha_i is
  sampled per chunk from {2,4,8,16,32} (Sec. 3.2, p. 6).
- Cost: KV cache shrinks by the average ratio alpha; FLOPs from Eq. 9-10 and App. B (p. 6, p. 17).
- Data: 1B RedPajama tokens (docs between 2400 and 20480 tokens), then fine-tuning on LongAlpaca,
  BookSum and 16K synthetic GPT-3.5 QA (13K over books, 3K over papers) plus 5000 pre-training samples mixed in, all under 20K (Sec. 4.1, p. 7;
  App. A, p. 17).

## Assumptions

- A1: The backbone LLM is frozen; only beacon projections and the beacon embedding are trained (Sec. 3.2, p. 6)
- A2: Training sequences are at most 20K tokens (Sec. 4.1, p. 7)
- A3: Inputs are natural-language text with a standard causal LLM (Llama-2-7B chat with a 4K window, Qwen-2-7B instruct, Qwen-2-72B in the FLOPs plot) (Sec. 4.1, p. 7; Fig. 3, p. 5)
- A4: Compression is query-independent and done once; every chunk at inference uses one chosen ratio (Sec. 4.2, p. 8; Sec. 3.2, p. 6)
- A5: The kept beacon activations are stored for the whole history, so memory still grows linearly at 1/alpha of the raw length (Sec. 3.1, p. 4)
- A6: Evaluation tasks are QA, summarisation, few-shot learning, code completion and needle retrieval over documents; the needle score is graded by ChatGPT on 1-10 (Sec. 4.2, pp. 7-8)

## Claims

- C1: On LongBench at 32K, Llama-2-7B + Beacon (adaptive x2/x4/x8 by length) scores 34.9 / 27.5 / 25.0 / 61.4 / 57.8 on Single-Doc / Multi-Doc / Summ. / Few-shot / Code, near the fine-tuned uncompressed Full-FT (34.8 / 27.5 / 23.2 / 61.8 / 57.8) and above the 4K Llama-2 (24.7 / 22.4 / 24.6 / 63.2 / 57.7) on Single-Doc, Multi-Doc and Summ., level on Code (57.7) and below on few-shot (63.2) [A1, A2, A3, A4, A6] (Table 1, p. 7)
- C2: On Llama-2-7B the compression baselines are far lower at 32K: AutoCompressor 12.9 / 16.4 / 16.3 / 23.8 / 39.4 and ICAE 19.5 / 19.2 / 19.5 / 24.8 / 27.8 (same fine-tuning data); SnapKV, limited to the 4K window, gets 24.2 / 22.6 / 16.3 / 60.1 / 57.7 [A1, A2, A3, A4, A6] (Table 1, p. 7)
- C3: On Qwen-2-7B (uniform x4) Beacon gets 40.5 / 40.3 / 26.8 / 68.4 / 66.4, above uncompressed Full (38.8 / 37.5 / 26.7 / 70.1 / 60.3) except few-shot, and within 0.5 of Full-FT on Single-Doc (41.0), Multi-Doc (40.6) and few-shot (68.5), above it on Code (66.1) [A1, A2, A3, A4, A6] (Table 1, p. 7)
- C4: Needle-in-a-Haystack is mostly retrieved correctly up to 32K (Llama-2) and 128K (Qwen-2) with adaptive ratios, though training contexts were under 20K; a few cells fail (Llama-2 has one 5.0 cell and one 1.0 cell at 32K; Qwen-2 has cells at 5.0, 7.0 and 9.0) [A1, A2, A3, A4, A6] (Fig. 4, p. 7, Sec. 4.2, p. 8)
- C5: In multi-needle multi-turn retrieval at x8 (Llama-2 at 32K), Beacon keeps accuracy 9.75 / 9.40 / 9.05 for turns 1-3 matching Full-FT (9.75 / 9.45 / 9.10) while AutoCompressor, ICAE, LongLLMLingua and SnapKV stay at 1.0-2.2; for Qwen at 128K it is 9.70 / 9.35 / 9.10, with SnapKV at 8.85-9.45 [A1, A2, A3, A4, A6] (Table 2, p. 8)
- C6: Beacon's latency is mostly lower than other compressors (1.8x faster than AutoCompressor; on Qwen-2 at 128K, 9.3x and 3.6x faster than LongLLMLingua and SnapKV over three turns), with an exception: on Llama-2 at turn 1 SnapKV is faster (0.859 s vs 1.153 s). The paper attributes the gain to compression being query-independent and soft tokens not being re-encoded; at 128K it is 2x faster than Full-FT and with x8 it cuts KV cache 8x [A1, A3, A4, A5] (Table 2, p. 8, Sec. 4.3, p. 9)
- C7: The forward-FLOPs saving grows with length: x4 or more at 256K at the x8 ratio (about 2x at x2) for the models plotted [A3, A5] (Fig. 3, p. 5, Sec. 3.1, p. 6)
- C8: Compression quality on needle retrieval is best across ratios x2-x16 at 1K and 4K and at x8/x16 at 32K (the 32K panel has only those two ratios); SnapKV is on par at 1K and 4K (slightly above Beacon at 1K x16), but at 32K the baselines are near the floor while Beacon stays high [A1, A3, A4, A6] (Fig. 5, p. 9, Sec. 4.4, p. 9)
- C9: Short-context ability is preserved: Llama-2-7B MMLU 47.5 to 46.6, ARC-C 48.5 to 48.4, BoolQ 86.2 to 86.5, GSM8K 9.2 to 9.3; Qwen-2-7B MMLU 70.1 to 69.1, ARC-C 62.7 to 62.7, BoolQ 87.1 to 87.2, GSM8K 76.0 to 76.2. The authors conjecture the frozen backbone is the reason [A1, A3] (Table 3, p. 9, Sec. 4.5, pp. 9-10)
- C10: Ablation (Qwen-2-7B, LongBench Single-Doc, 32K, x4): default 40.5; without interleaved fine-grained compression 35.2; instance-wise instead of chunk-wise random ratio 37.7; without pre-training 34.9; without fine-tuning 35.5 [A1, A2, A3, A4, A6] (Table 4, p. 10, Sec. 4.6, p. 10)

## Implementation notes

- Chunk size 1024 (Llama-2) and 2048 (Qwen-2); ratios sampled from {2,4,8,16,32}; adaptive inference ratio x2 for 4K-8K, x4 for 8K-16K, x8 for 16K-32K contexts (Sec. 4.1-4.2, pp. 7-8)
- Pre-training lr 5e-5, fine-tuning lr 1e-5, linear decay, no warmup, batch 8, one 8xA800 machine, FlashAttention-2, HuggingFace (Sec. 4.1, p. 7)
- The last chunk may be shorter than w, and each chunk must be divisible by its alpha (footnote 3 and Eq. 2, p. 4)
- Beacon projection initialisation, beacon positional handling and the beacon KV memory layout beyond Eq. 3-8 are not described (assumption: read the authors' code at the repo they name, https://github.com/FlagOpen/FlagEmbedding, not consulted by this reader)
- Needle-in-a-Haystack uses the gkamradt harness and NeedleBench-style multi-needle for latency (Sec. 4.2-4.3, pp. 8-9)

## Limitations / open questions

- The paper has no explicit limitations section; the baselines for Llama-2 use a different, adaptive ratio than for Qwen-2 (uniform x4), so cross-model numbers aren't comparable (Sec. 4.2, p. 8)
- All results are language tasks with an LLM-graded needle score; nothing about non-text tokens (Sec. 4, pp. 7-10)
- Open question (reader's lens, shmup policy with NitroGen-style frames in / 18-frame action chunks out): Beacon's compression is query-independent and trained by next-token loss on text. A policy would need the beacons to carry what matters for later actions (weapon system, economy, rank); would a loss on future action chunks work instead of next-token prediction? The paper's unsupervised-compression recipe gives no evidence on that.
- Open question: beacons keep every layer's K/V, so a long game still grows memory at 1/alpha. Compared with RMT-style memory tokens that write a fixed-size summary ([[bulatov2022recurrent]], [[chevalier2023adapting]] accumulates summaries linearly too), is a bounded-size beacon cache (dropping old beacons) workable? The paper doesn't test dropping.
- Open question: does it need per-frame tokens (hundreds per frame) before compression to pay off? The paper tests 1K-2K token chunks only.

## Related

- extends [[chevalier2023adapting]]: same idea of accumulated per-segment compression, but keeps per-layer K/V of beacons instead of last-layer summary vectors, uses no stop-gradients, and samples the ratio per chunk
- related [[bulatov2022recurrent]]: contrasts with RMT's final-layer memory passing and gradient truncation (Sec. 3.2, p. 6)
- related [[kim2024compressed]]: CCM, another compressor that keeps compression-token KVs, cited as the paper's reference [28] (Sec. 2, p. 3)
- related [[mu2023learning]]: gist tokens compress an instruction all at once, the paper's closest all-at-once baseline family (Sec. 2, p. 3)
- related [[cherepanov2026recurrent]]: memory tokens in a decision transformer for long-horizon RL; the setting a beacon policy would be compared in
- related [[magne2026nitrogen]]: the frames-in, gamepad-out policy a beacon-style memory would be added to

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
