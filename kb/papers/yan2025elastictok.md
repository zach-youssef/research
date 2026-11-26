---
id: yan2025elastictok
title: "ElasticTok: Adaptive Tokenization for Image and Video"
authors: [Wilson Yan, Volodymyr Mnih, Aleksandra Faust, Matei Zaharia, Pieter Abbeel, Hao Liu]
year: 2025
venue: ICLR 2025
url: https://arxiv.org/abs/2410.08368
arxiv: 2410.08368
doi:
pdf_sha256: 9c66d870827b17b54815d1c70fa3ae853660a8906fdb06368ce143cca8a81287
tags: [video, tokenizer, adaptive-tokenization, variable-length, temporal-redundancy, autoencoder, fsq, vae, long-context, vision-language]
status: read
read: 2026-11-25
---

# ElasticTok: Adaptive Tokenization for Image and Video

<!-- Read from arXiv v2 (2025-02-02, printed "Published as a conference paper at ICLR 2025"), 21 PDF pages: pp. 1-10 main text, pp. 11-14 references, pp. 15-21 Apps. A-G. -->

## TL;DR

A ViT autoencoder trained with a random-length prefix mask per 4-frame block (block-causal over past
blocks) so a block can be encoded with 256 to 4096 tokens; at inference a search picks the fewest
tokens meeting an MSE target. Matches fixed-length baselines' reconstruction with 1.3× to 5× fewer tokens.

## Problem

Image and video tokenizers emit a fixed number of tokens whatever the content, so they must be sized
for the worst case and waste tokens on simple inputs, which makes long videos cost hundreds of millions
of tokens; too few tokens make encodings lossy (Sec. 1, pp. 1-2). Codecs already spend a variable number
of bytes per frame, but neural compressors are hard to use for downstream training (Sec. 5, p. 10).

## Method

- **Elastic training** (Sec. 3.1, Alg. 1, pp. 3-4): for each input, sample ℓ ~ U{M_min..M_max} and a
  mask m keeping the first ℓ of N encoder tokens; the encoder E(x, m) sees the mask, its output is
  masked (z ⊙ m) and decoded. Works with any autoencoder (VQ-VAE, FSQ, VAE); here ViTs. A lower
  bound M_min is needed because very few tokens destabilise training.
- **Multi-block video** (Sec. 3.1, p. 3, Fig. 2, p. 2): video is cut into blocks of 4 frames, a mask
  is sampled per block, and encoder and decoder use block-causal attention, so each block is encoded
  conditioned on the previous ones. No added parameters; training grows progressively from 1 block to
  256 blocks (up to 1024 frames, ~40 s) with RingAttention (Sec. 2.1, p. 2, Sec. 4.1, p. 5, App. B, p. 15).
- **Inference** (Sec. 3.2, Alg. 2, pp. 3-4): either fix the token count, or give a reconstruction
  threshold (e.g. pixel MSE) and search for the shortest length meeting it: full search, binned search
  (100 lengths), binary search (assumes loss is monotone in length) or a neural regressor that predicts
  the length in one pass. Multi-block search runs block by block with a KV cache. Not conditioning the
  encoder on the mask lets the encoder run once and only the decoder be searched, about 2× faster.
- **Models** (Sec. 4.1, p. 5, App. A, p. 15): 210M-parameter ElasticTok-FSQ (discrete, patch 1×8×8,
  4096 tokens per 256×256 4-frame block, M_min 256, FSQ levels 8,8,8,5,5,5 ≈ 64k codes) and
  ElasticTok-VAE (continuous, patch 2×8×8, 2048 tokens per block, M_min 128, latent dim 8); LLaMA-2
  style blocks, MSE + LPIPS loss, no GAN. Mask enters the encoder as learned 0/1 embeddings added to
  the patch embeddings.
- **Tokens per frame:** FSQ 64 to 1024 per 256×256 frame (256 to 4096 per 4-frame block); VAE 32 to
  512. **Temporal redundancy** is exploited by conditioning each block on past blocks, so predictable
  blocks meet the threshold with fewer tokens; token use spikes at scene changes and fast motion
  (Fig. 1, p. 1, Fig. 3, p. 5).

## Assumptions

- A1: Training data are 256 × 256 frames: COYO-700M images (~350M after filtering) and 6M scraped web videos of 4 s to 2 min at 24 FPS (Sec. 4.1, p. 5, App. C, pp. 15-16).
- A2: Video is split into 4-frame blocks, and each block keeps a prefix (the first ℓ) of its tokens, ℓ sampled uniformly in [M_min, M_max] during training (Sec. 3.1, p. 3, Sec. 4.1, p. 5).
- A3: Reconstruction quality is judged by per-block pixel MSE thresholds (0.001, 0.003, 0.015), and results report the share of blocks or images meeting the threshold (Sec. 4.2, p. 6).
- A4: Baselines are fixed-length tokenizers with the same architecture, training process and total FLOPs, one trained per token count (Sec. 4.1, p. 5).
- A5: Unless noted, inference uses binary search over token counts, which assumes the loss is monotone in token count, approximately true (Sec. 3.2, p. 4, Sec. 4.1, p. 5).
- A6: The encoder is conditioned on the mask, unless noted (Sec. 4.1, p. 5, App. F, p. 17).
- A7: Encoding blocks attend to all previous blocks in context, up to 1024 frames (512K VAE or 1M FSQ tokens) (App. A, p. 15).
- A8: The ImageNet comparison uses the image-only model trained on ImageNet alone (reader's inference, Sec. 4.1, p. 5, App. B, p. 15).

## Claims

- C1: On ImageNet (FSQ), ElasticTok meets the reconstruction threshold on as many images as the fixed baselines with 3.5× fewer tokens at MSE 0.015 and 1.3× fewer at 0.003 [A2, A3, A4, A5, A6, A8] (Fig. 4, p. 6).
- C2: On video (FSQ), it matches the baselines' satisfy rate with 5× fewer tokens at MSE 0.015, 2.4× at 0.003 and 1.3× at 0.001; strict thresholds (0.001) nearly always use all tokens [A1, A2, A3, A4, A5, A6, A7] (Fig. 4, Sec. 4.2, p. 6).
- C3: For the VAE variant on video at the same 50% average token use, the share of blocks meeting the threshold is 50% vs 40% (MSE 0.001), 88% vs 78% (0.003) and 99% vs 99% (0.015) against a fixed 50% baseline [A1, A2, A3, A4, A5, A6, A7] (App. F, Table 3, p. 16).
- C4: To match a fixed baseline's worst-case per-video MSE, ElasticTok-FSQ uses on average 75% vs 78% (0.001), 37% vs 57% (0.003) and 9.4% vs 15% (0.015) of tokens [A1, A2, A3, A4, A5, A6, A7] (App. F, Table 4, p. 17).
- C5: The satisfy rate rises with the number of frames in context, from 65.4% to a peak of 88.4%, then falls to 80.3% at the largest context (1M tokens), which the authors attribute to too little training at that length; the caption places the peak "around 100 frames" [A1, A2, A3, A5, A6, A7] (Fig. 6, p. 7, Sec. 4.2, p. 6).
- C6: A VLM (OpenLLaMA-3B) fine-tuned on ElasticTok-VAE tokens gains accuracy on GQA, POPE, MSVD-QA and MSRVTT-QA as more tokens are used, and matches a VLM fine-tuned on a fixed-token baseline tokenizer (GQA 54% vs 54%, POPE 82% vs 82%, MSVD 52% vs 53%, MSRVTT 37% vs 37%) [A1, A2, A4] (Fig. 7, Table 1, Sec. 4.3, pp. 7-8).
- C7: Search methods trade error for cost (FSQ): full search 0% error with 4096 function evaluations, binned 0.5% with 100, binary 7% with 12, neural regression 9% with 1 [A2, A3] (Table 2, p. 9, Sec. 4.4, p. 8).
- C8: Without encoder conditioning on the mask, reconstruction is slightly worse at equal average tokens (33.1% vs 36% at MSE 0.001, 75.3% vs 79.6% at 0.003, 98.4% vs 98.5% at 0.015), but inference is about 2× faster [A1, A2, A3, A5] (App. F, Table 5, p. 17, Sec. 3.2, p. 4).
- C9: Token count correlates with high-frequency content (Sobel gradient magnitude) on 2k 5-second videos: Pearson r = 0.771 for single blocks and 0.676 for multi-block video (0.77 and 0.67 in the text), the lower value attributed to conditioning on past frames [A1, A2, A3, A5] (Fig. 9, p. 10, Sec. 4.5, pp. 8-9).
- C10: Any scalar, even non-differentiable, objective can drive the search; a CLIP cosine objective at the same average token count spends more tokens on text and fewer on other fine detail than MSE (qualitative) [A2] (Fig. 8, p. 9, Sec. 4.4, p. 8).
- C11: Sampling too few tokens during training destabilises it, so a lower bound M_min (128 or 256) is used [A2] (Sec. 3.1, p. 3).

## Implementation notes

- Hidden size 1024, FFN 2048, 10 encoder and 10 decoder layers, RoPE θ 5e6 (VAE) or 5e7 (FSQ), KL weight 1e-8 for the VAE (source: App. A, p. 15).
- Training schedule: ImageNet model 200k iterations at batch 256; long-video runs (1) to (7) go from 1 to 256 blocks, batch 256 down to 4, image/video split 50/50 then 25/75, AdamW lr 2e-4, weight decay 1e-4; each run starts from the previous one (source: App. B, p. 15).
- Long-video models are trained on TPU v4-512, the image-only model on v4-256 (source: Sec. 4.1, p. 5).
- The neural regressor adds 1D strided-conv downsampling (×4) and 4-layer transformer stages to the encoder, pools, and a 2-hidden-layer MLP outputs a normalised length in [-1, 1] per block, trained by MSE on 100K sequences labelled at MSE target 0.003 (source: App. E, p. 16).
- Relative weights of the MSE and LPIPS losses are not stated (source: assumption: tune, start equal).
- Video scenes are cut with PySceneDetect and filtered by OCR, NSFW, motion and aesthetic scores (source: App. C, p. 16).
- No code release is named in the paper; only the project page largeworldmodel.github.io/elastictok (source: p. 1).
- For a game-frame policy (the KB's lens): at a 4-frame block and 60 FPS, an 18-frame action chunk is 4.5 blocks; aligning chunks to 16 or 20 frames would keep token budgets per chunk whole (source: assumption).

## Limitations / open questions

- It slightly underperforms fixed baselines at the extremes of encoding length, perhaps because one model must hold both coarse global and fine local codes; spreading kept tokens differently than a left-aligned prefix helped in early tests (Sec. 6, p. 10).
- Performance drops at the longest context (1M tokens), attributed to limited training compute there (Sec. 4.2, p. 6).
- Table 1's baseline is described as fixed at 50% tokens in the text (p. 7) but at 100% in its caption (p. 8); the note doesn't resolve which.
- Faster search methods have 5-10% error in the chosen length (Sec. 4.4, p. 8).
- The authors suggest extending it to other temporal data, including state-action-reward trajectories, and to task-conditioned encoders that keep only relevant information (Sec. 6, p. 10).
- Open question: bullet-hell frames are dense in high-frequency detail (small sprites, bullets), which C9 ties to high token use, so the savings measured on web video may shrink in shmups; scrolling backgrounds, by contrast, are well predicted from past blocks.
- Open question: MSE spends tokens on pixels, not on what the policy needs. C10 allows any scalar objective; a policy-relevant one (action-prediction loss, or offline-only probes against game RAM object positions) could drive the length search without the agent ever reading RAM.
- Open question: variable token counts per chunk break fixed-shape batching and make per-step latency content-dependent; a real-time policy would likely need the 1-NFE regressor or a fixed cap.
- Open question: ElasticTok shortens the visual stream but keeps every past block in context (up to 1M tokens); it doesn't by itself give a bounded memory for long-term strategy (compare ryoo2023turing).

## Related

- related [[hafner2025scalable]]: Dreamer 4's causal tokenizer also encodes frames conditioned on past frames, with a fixed token budget.
- related [[he2021masked]]: masking-based autoencoder training; ElasticTok masks a random-length suffix instead of random patches.
- related [[magne2026nitrogen]]: a game policy with a fixed 256 tokens per frame; elastic counts are an alternative input budget.
- related [[ryoo2023turing]]: bounded token memory, where ElasticTok keeps every past block in context.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
