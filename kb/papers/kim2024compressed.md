---
id: kim2024compressed
title: Compressed Context Memory For Online Language Model Interaction
authors: [Jang-Hyun Kim, Junyoung Yeom, Sangdoo Yun, Hyun Oh Song]
year: 2024
venue: ICLR 2024 conference paper (read as arXiv v2)
url: https://arxiv.org/abs/2312.03414
arxiv: 2312.03414
doi:
pdf_sha256: e82b7a8469e45b632df424aa0d84f65a2d1b01071a76e90ee913b36e66c16aae
tags: [memory, context-compression, kv-cache, lora, language-model, online-inference, long-context, recurrent-memory, ccm]
status: read
read: 2026-11-25
---

# Compressed Context Memory For Online Language Model Interaction

## TL;DR

CCM compresses each context segment into the keys/values of a few ⟨COMP⟩ tokens via a conditional
LoRA, keeping a growing (concat) or running-average (merge) memory, trained in one masked pass.
On LLaMA-7B it nearly matches full context with ~1/5 to 1/8 of the KV memory.

## Problem

In online use (conversation, personalization, in-context learning) the context grows each step, so
memory and attention cost grow (Sec. 1, p. 1). Earlier compressors (Gisting, ICAE, Compressive
Transformer) target a fixed context and recompress everything; recurrent ones (RMT, AutoCompressor)
need sequential training (Sec. 1, p. 2). CCM wants a memory that is updated incrementally and trained
efficiently on a frozen-in-spirit pretrained LM.

## Method

- **Setting** (Sec. 2, p. 3, Table 2): at step t the model gets context c(t) (the previous input,
  output and feedback), input I(t), and predicts O(t); inference uses a compressed memory Mem(t)
  instead of the full C(t), f(· | Mem(t), I(t)) (Eq. 3, Sec. 3.1, p. 3).
- **Compression** (Sec. 3.1, pp. 3-4, Fig. 2): append ⟨COMP⟩ to c(t); at every layer it attends to
  c(t) and Mem(t-1); its keys/values are h(t) ∈ R^{2×L×d} (Eq. 1). Length 1 for notation; larger lengths allowed.
- **Update** (Sec. 3.1, p. 4, Eq. 2): CCM-concat appends h(t), so memory grows with t; CCM-merge
  keeps a fixed size with Mem(t) = (1−a_t) Mem(t−1) + a_t h(t), a_t = 1/t (arithmetic mean) in the main
  runs, EMA tried in App. C.
- **Parallelized training** (Sec. 3.1, p. 4-5, Fig. 3, Alg. 1): the sequence
  [c(1), ⟨COMP⟩, …, c(t), ⟨COMP⟩, I(t)] is run once with a mask so each c(j) sees only Mem(j−1) and
  I(t) sees only Mem(t); gradients flow through all steps in one pass instead of a recurrence.
- **Conditional LoRA** (Sec. 3.1, p. 5, Fig. 4, Eq. 4): LoRA weights apply only on ⟨COMP⟩ tokens
  (x'_h = W x_h + m ΔW x_h); plain LoRA overfit to I(t) alone (train loss without context 2.69 to
  1.84 while test stayed 2.59 on MetaICL/LLaMA-7B).
- **Complexity** (Sec. 3.2, p. 6, Table 3): CCM-merge compression costs O(l_c) per step, concat O(t + l_c)
  versus O(t l_c) for compressing the full context again.

## Assumptions

- A1: Context arrives as segments c(t) with a bounded length (average 50 tokens, max 256 or less)
  and the task has a limited number of steps (16, or 12 for dialogue) (App. A, p. 16, Table 12).
- A2: The base LM is first fine-tuned on the dataset, then a compression LoRA is trained per
  application (a single unified adapter is a separate experiment) (Sec. 4, p. 7, App. B, p. 16).
- A3: Compression runs with a fixed ⟨COMP⟩ length set to give a compression factor of about 8
  (Sec. 4, p. 7).
- A4: Information to keep is application-relevant text (demonstrations, user profiles, dialogue);
  merge works best when segments share information (Sec. 4.1, p. 8).
- A5: Total tokens per sample are capped at 1024 for training (Sec. 4, p. 7).
- A6: Models are decoder LMs from LLaMA (7B, 13B) plus Flan-T5-Large for one check (App. C.2, p. 19).
- A7: The streaming run uses the general-data adapter of Table 4 (Pretrain+MetaICL+SODA, 500k) with an
  attention sink and a KV cap of 160, on PG19 well beyond A1's segment and step bounds (Sec. 4.1, p. 9).

## Claims

- C1: On MetaICL at step 16 with LLaMA-7B, CCM-merge reaches 69.6% accuracy and CCM-concat 70.0% against
  full context 70.8%, with context KV length 8 and 128 against 800, and throughput of 69.9 and
  24.4 versus 5.3 samples/s on an A100 (maximum batch 950, 300 versus 60) [A1, A2, A3] (Table 1, p. 2).
- C2: Full-context-level accuracy needs about half the KV memory for CCM-concat and about 1/8 for
  CCM-merge; the system gains 18% accuracy over no context at step 16 at negligible memory
  increment [A1, A2, A3] (Sec. 4.1, p. 8, Fig. 6, p. 7).
- C3: At matched compression factor (about 8; CCM-merge's factor is higher, Sec. 4.1, p. 8), CCM-concat
  beats Gisting-online and Compressive Transformer at every step in Tables 23-25; CCM-merge trails a
  baseline at DailyDialog steps 1 (7.47 vs Gisting-online 7.42) and 8 (6.53 vs Compressive 6.52). At the
  last step, CCM concat / merge vs Gisting-online and Compressive: MetaICL 70.0 / 69.6 vs 57.7 and 67.8;
  LaMP 84.7 / 83.9 vs 78.7 and 82.8; DailyDialog perplexity 5.91 / 6.27 vs 7.44 and 6.38 (full context:
  70.8, 85.1, 5.56) [A1, A2, A3] (Sec. 4.1, p. 8, Fig. 7, p. 8, Tables 23-25, p. 20).
- C4: CCM-concat is better than merge when each segment carries distinct information (dialogue) and
  about equal on MetaICL and LaMP, where segments share information; the merge loss grows
  as diverse information is added [A4] (Sec. 4.1, p. 8).
- C5: The conditional LoRA beats default LoRA for every method on MetaICL: CCM-concat 70.0 vs 69.4,
  CCM-merge 69.6 vs 66.3, Gisting 66.9 vs 64.6 [A2] (Sec. 4.2, p. 9, Table 5, p. 10).
- C6: Against recurrent compressors on OPT-2.7B (MetaICL step 16), CCM trains about 7x faster per sample
  (195 ms vs 1330 ms) and gets 53.5 / 52.3 (concat / merge) vs AutoCompressor 48.1 (50.9 fine-tuned) and full context 54.2;
  RMT 44.4 (50.0 fine-tuned) vs 52.3 / 52.2 [A1, A2] (Sec. 4.2, pp. 10-11, Table 8, p. 11, Table 22, p. 20).
- C7: Against fixed-context Gisting (same inference cost as merge) on MetaICL, CCM-merge is 3 points
  better (69.6 vs 66.9) and peak KV memory during compression is 66 MB against 588 MB [A1, A2] (Sec. 4.2, p. 10, Table 6, p. 10).
- C8: In streaming with attention sink on PG19 (KV limit 160, CCM size 8, 64 tokens compressed to 2 per
  step), CCM-concat has lower perplexity than StreamingLLM at an identical KV size [A7] (Sec. 4.1, p. 9, Fig. 8, p. 9).
- C9: Versus MemoryBank-style text summarization on DailyDialog, CCM-concat has perplexity 5.98 at
  24 compressed tokens against 7.06 at 60 (full context 5.59 at 222; CCM-merge 6.34 at 2) [A1, A2] (Sec. 4.2, p. 11, Table 9, p. 11).
- C10: ⟨COMP⟩ length matters little: 1 token costs about 1% accuracy and cuts compression overhead to
  2% (LaMP, 4/50 = 8% at length 4); inference FLOPs favour compression beyond 504 processed tokens at length 1 [A1, A3] (Sec. 4.2, p. 10, Table 18, p. 19, Table 17, p. 18).
- C11: A single unified adapter (trained on MetaICL and SODA) loses a little: MetaICL concat 68.7 vs 70.0, LaMP 75.2 (vs 76.4 full context),
  DailyDialog 7.61 (vs 7.02), still best among baselines [A1, A2] (Sec. 4.1, p. 8, Table 15, p. 17).
- C12: Training data matters: adding source tasks improves matching evaluation sets, and more data helps
  (MetaICL gap to full context -4.9% with pretraining data only, -1.1% with 750k mixed) [A2] (Sec. 4.1, pp. 8-9, Table 4, p. 9).
- C13: EMA (a_t = 0.5) and arithmetic-average merging give similar DailyDialog perplexity (6.38 vs 6.34 at step 12) [A1] (App. C.1, p. 17, Table 16, p. 17).
- C14: Admitted limits: performance still declines against full context, and a broadly applicable
  compressor for arbitrary applications remains open [] (Sec. 6, p. 12).

## Implementation notes

- LoRA: rank 8, alpha 16, dropout 0.05, targets q,k,v,o projections; ⟨COMP⟩ embeddings are
  trained and shared across steps (source: Table 14, p. 17, App. B, p. 16).
- Training: MetaICL 2000 steps, batch 128, 256k samples, lr 3e-4 cosine, FP16, ⟨COMP⟩ length 8 (LaMP 300 steps / 4; DailyDialog 1000 steps / 2); 3-24 h on one A100 80GB (source: Table 13, p. 16).
- Evaluation: multi-choice accuracy by average log-likelihood of answer tokens (MetaICL, LaMP); perplexity for DailyDialog (source: App. B, p. 16-17; Sec. 4, p. 7).
- Codes: github.com/snu-mllab/context-memory (source: abstract, p. 1; not inspected).
- Applying to a game policy needs a port the paper doesn't describe: what c(t) is (a chunk of
  frames or tokens), where ⟨COMP⟩ is inserted, and the mask (source: assumption).

## Limitations / open questions

- Admitted: C14. Memory quality drops for diverse information under merge (C4).
- Only text LMs of 7B-13B and one encoder-decoder check; short contexts (50 tokens per step, 16 steps) (Table 12, p. 16).
- Open question (user's lens, vs recurrent memory tokens): CCM keeps per-layer K/V of ⟨COMP⟩ rather than
  carrying token embeddings forward as in RMT ([[bulatov2022recurrent]]) and AutoCompressor
  ([[chevalier2023adapting]]); RATE ([[cherepanov2026recurrent]]) puts RMT-style memory in an action
  policy. Differences stated in the paper: CCM's training is one parallel pass (7x faster than them,
  Table 8) and needs no extra forward for compressed token embeddings at inference (Sec. 1, p. 2). Whether
  that advantage holds for a transformer policy over frames is untested.
- Open question: the compression is trained to support the next answer. For a shmup policy,
  the loss is action prediction; strategy state (weapon level, economy, rank) is retained only if
  it changes actions. Offline RAM probes (never given to the agent) could test what ⟨COMP⟩ keys/values encode.
- Open question: CCM-merge's running mean weights all steps equally (a_t = 1/t); a run with an
  early upgrade shop and late boss would dilute it. Concat grows linearly, so a cap or a streaming eviction (as in Fig. 9, p. 9) would be needed.
- Open question: a shmup is real time; ⟨COMP⟩ every 18-frame action chunk would make
  c(t) a chunk, a design the paper doesn't test.

## Related

- related [[bulatov2022recurrent]]: RMT, a recurrent-memory baseline in CCM's Table 8 and Table 22.
- related [[chevalier2023adapting]]: AutoCompressor, the other recurrent compressor in Table 8 and Table 22.
- related [[mu2023learning]]: Gisting, CCM's main fixed-context baseline throughout (Gisting-online in Fig. 7).
- related [[cherepanov2026recurrent]]: RMT-style memory tokens inside an RL action policy, the M1 lens of this KB.
- related [[schmidt2024actions]]: both compress experience; LAPO compresses a transition into a latent action, CCM a context into KV memory (a reader's comparison).
- related [[magne2026nitrogen]]: the frames-in, gamepad-out policy whose long-term state the memory would serve.
- related [[lee2022multigame]]: its context is 4 frames (156 tokens), the kind of limit a compressed memory would extend.
- related [[reed2022generalist]]: Gato's context is 1024 tokens, so only a few image timesteps fit ([[reed2022generalist#A7]]).

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
