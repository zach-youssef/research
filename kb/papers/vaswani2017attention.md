---
id: vaswani2017attention
title: Attention Is All You Need
authors: [Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin]
year: 2017
venue: NIPS 2017 conference paper
url: https://arxiv.org/abs/1706.03762
arxiv: 1706.03762
doi:
pdf_sha256: bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697
tags: [transformer, attention, sequence-modeling, positional-encoding, machine-translation, architecture]
status: read
read: 2026-11-16
---

# Attention Is All You Need

<!-- Read from arXiv v7 (2023-08-02), 15 PDF pages; pp. 10-12 are mostly references; pp. 13-15 are attention visualizations (Fig. 3-5). -->

## TL;DR

The Transformer replaces recurrence and convolution with stacked multi-head self-attention and
position-wise feed-forward layers plus sinusoidal positional encodings; the big model reaches 28.4
BLEU on WMT14 EN-DE (over 2 above prior ensembles) after 3.5 days on 8 P100 GPUs.

## Problem

Recurrent models compute hidden states one position after another, which precludes parallelism
within a training example and becomes critical at long sequence lengths; attention was used almost
only together with recurrence (Sec. 1, p. 2). Convolutional models (Extended Neural GPU, ByteNet,
ConvS2S) compute positions in parallel, but the operations needed to relate two positions grow with
their distance, linearly or logarithmically, which makes distant dependencies harder to learn
(Sec. 2, p. 2).

## Method

- **Encoder-decoder** (Sec. 3, Fig. 1, pp. 2-3): the encoder maps (x_1…x_n) to z = (z_1…z_n); the
  decoder generates (y_1…y_m) auto-regressively, consuming earlier outputs.
- **Stacks** (Sec. 3.1, p. 3): encoder of N = 6 layers, each multi-head self-attention then a
  position-wise feed-forward network; each sub-layer wrapped as LayerNorm(x + Sublayer(x)); all
  outputs d_model = 512. Decoder of N = 6 layers adds a third sub-layer attending over the encoder
  output, and masks its self-attention so position i only sees outputs before i (outputs are also
  shifted right by one).
- **Scaled dot-product attention** (Sec. 3.2.1, Eq. 1, p. 4): Attention(Q, K, V) =
  softmax(QKᵀ/√d_k)V. The 1/√d_k factor counters large dot products pushing softmax into regions
  with tiny gradients (footnote 4: for unit-variance components q·k has variance d_k).
- **Multi-head** (Sec. 3.2.2, pp. 4-5): h heads, each attending over its own learned linear
  projections of Q, K, V to d_k, d_k, d_v dims; outputs concatenated and projected by W^O. Base:
  h = 8, d_k = d_v = d_model/h = 64.
- **Three uses** (Sec. 3.2.3, p. 5): encoder-decoder attention (queries from the decoder, keys and
  values from the encoder output), encoder self-attention (every position sees all positions of the
  previous layer), masked decoder self-attention (illegal connections set to −∞ before softmax).
- **Feed-forward** (Sec. 3.3, Eq. 2, p. 5): FFN(x) = max(0, xW1 + b1)W2 + b2 applied to each
  position separately and identically, d_ff = 2048; parameters differ between layers.
- **Embeddings** (Sec. 3.4, p. 5): learned token embeddings; the two embedding layers and the
  pre-softmax linear layer share one weight matrix; embedding weights are multiplied by √d_model.
- **Positional encoding** (Sec. 3.5, p. 6): since there is no recurrence or convolution, order must
  be injected: PE(pos, 2i) = sin(pos/10000^(2i/d_model)), PE(pos, 2i+1) = cos(…), wavelengths a
  geometric progression from 2π to 10000·2π, added to the input embeddings at the bottom of both
  stacks. Learned positional embeddings are the tested alternative.
- **Why self-attention** (Sec. 4, Table 1, pp. 6-7): compares per-layer complexity, minimum
  sequential operations and maximum path length against recurrent, convolutional and restricted
  (neighbourhood r) self-attention layers.
- **Training** (Sec. 5, pp. 7-8): WMT14 EN-DE (~4.5M pairs, BPE ~37k shared vocab) and EN-FR (36M
  sentences, 32k word-piece); batches of ~25k source and ~25k target tokens grouped by length;
  Adam with warmup then inverse-sqrt decay (Eq. 3); residual dropout and label smoothing.
- **Evaluation** (Sec. 6, pp. 8-10): BLEU on newstest2014 (Table 2), variants on newstest2013 dev
  (Table 3), and English constituency parsing on WSJ Section 23 (Table 4).

## Claims

- C1: Transformer (big) reaches 28.4 BLEU on WMT14 EN-DE newstest2014, more than 2.0 BLEU above the best previously reported models including ensembles (Table 2, Sec. 6.1, p. 8)
- C2: On WMT14 EN-FR the big model outperforms all previously published single models at less than 1/4 of the previous state of the art's training cost; Table 2 and the abstract give 41.8 BLEU while the Sec. 6.1 text gives 41.0 (Table 2, Sec. 6.1, p. 8, p. 1)
- C3: The base model (27.3 EN-DE, 38.1 EN-FR) surpasses all previously published EN-DE models and ensembles at a fraction of their training cost: 3.3·10^18 FLOPs vs 2.3·10^19 for big and, e.g., 7.7·10^19 for the ConvS2S ensemble on EN-DE (Table 2, Sec. 6.1, p. 8)
- C4: On 8 P100 GPUs a base training step takes about 0.4 s (100k steps, 12 hours) and a big step 1.0 s (300k steps, 3.5 days) (Sec. 5.2, p. 7)
- C5: Per layer, self-attention costs O(n²·d) with O(1) sequential operations and O(1) maximum path length, vs O(n·d²), O(n), O(n) for recurrent, O(k·n·d²), O(1), O(log_k n) for (dilated) convolution, and O(r·n·d), O(1), O(n/r) for self-attention restricted to a neighbourhood r (Table 1, p. 6)
- C6: Self-attention layers are faster than recurrent ones when the sequence length n is smaller than the representation dimension d, which is most often the case for word-piece and byte-pair sentence representations (Sec. 4, pp. 6-7)
- C7: With computation held constant, single-head attention is 0.9 BLEU worse than the best setting (24.9 vs 25.8 dev BLEU), and quality also drops with too many heads (32 heads: 25.4) (Table 3, Sec. 6.2, p. 9)
- C8: Reducing the key size d_k hurts quality (d_k = 16: 25.1, 32: 25.4 vs 25.8 at 64), which the authors read as a sign that compatibility is not easy to determine and a richer function than dot product may help (Table 3, Sec. 6.2, p. 9)
- C9: Bigger models are better and dropout helps avoid overfitting: N = 2 gives 23.7 dev BLEU, d_model = 1024 gives 26.0, d_ff = 4096 gives 26.2, and P_drop = 0 gives 24.6 vs 25.8 at 0.1 (Table 3, Sec. 6.2, p. 9)
- C10: Label smoothing ε_ls = 0.1 hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU; without it dev PPL is 4.67 and BLEU 25.3, vs 4.92 and 25.8 with it (Sec. 5.4, p. 8, Table 3, p. 9)
- C11: Learned positional embeddings give nearly identical results to the sinusoids (dev PPL 4.92 for both, BLEU 25.7 vs 25.8) (Table 3, Sec. 6.2, p. 9, Sec. 3.5, p. 6)
- C12: Sinusoids were chosen on two untested hypotheses: PE_{pos+k} is a linear function of PE_pos for any fixed offset k, which may let the model attend by relative position, and they may extrapolate to sequence lengths longer than those seen in training (Sec. 3.5, p. 6)
- C13: Because each head has reduced dimension, the total cost of 8-head attention is similar to single-head attention with full dimensionality (Sec. 3.2.2, p. 5)
- C14: Dot-product attention is much faster and more space-efficient than additive attention in practice; additive outperforms unscaled dot product for larger d_k, which the authors suspect is due to large dot products giving small softmax gradients (Sec. 3.2.1, p. 4)
- C15: A 4-layer Transformer (d_model = 1024) without task-specific tuning reaches 91.3 F1 on WSJ Section 23 trained on WSJ only (~40k sentences) and 92.7 semi-supervised; the authors say this beats all previously reported models except the Recurrent Neural Network Grammar (93.3), and beats the BerkeleyParser (90.4) with WSJ-only data (Table 4, Sec. 6.3, p. 10)
- C16: Individual encoder heads learn to perform different tasks, and many show behaviour related to sentence structure, e.g. layer-5 heads following the long-distance dependency "making … more difficult" and sharp attention for anaphora ("its" → "Law") (Sec. 4, p. 7, Fig. 3, Fig. 4, Fig. 5, pp. 13-15)
- C17: The base model has 65M parameters and the big model 213M (N = 6, d_model = 1024, d_ff = 4096, h = 16, P_drop = 0.3 for EN-DE; the EN-FR big model used P_drop = 0.1), reaching dev PPL 4.33 and BLEU 26.4 (Table 3, Sec. 6.1, p. 8, p. 9)

## Implementation notes

- Base model: N = 6, d_model = 512, d_ff = 2048, h = 8, d_k = d_v = 64, P_drop = 0.1, ε_ls = 0.1, 100k steps (source: Table 3, p. 9).
- Adam β1 = 0.9, β2 = 0.98, ε = 1e-9; lrate = d_model^−0.5 · min(step^−0.5, step · warmup^−1.5) with warmup_steps = 4000, i.e. linear rise then inverse-sqrt decay (source: Sec. 5.3, Eq. 3, p. 7).
- Dropout on each sub-layer output before the residual add and norm, and on the sums of embeddings and positional encodings in both stacks (source: Sec. 5.4, p. 8); Sec. 6.3 also mentions tuning "attention" dropout, whose placement and value for translation are not given (source: Sec. 6.3, p. 9).
- Layer norm comes after the residual add (post-LN) (source: Sec. 3.1, p. 3).
- Decoding: base uses the average of the last 5 checkpoints (written every 10 min), big the last 20; beam 4, length penalty α = 0.6, max output length input + 50 (source: Sec. 6.1, p. 8). Parsing: beam 21, α = 0.3, input + 300 (source: Sec. 6.3, p. 10).
- Not stated: weight initialization, padding masks in encoder self-attention (padding tokens appear in Fig. 3-5), gradient clipping (source: assumption, the paper is silent; the authors' code is github.com/tensorflow/tensor2tensor, p. 10, not read).
- For the user's design (reader's inference): self-attention itself never sees positions (Sec. 3.5 gives this as why position must be injected), so an encoder without positional encodings treats its input as an unordered set; segment tokens with no natural order could skip PE, or carry spatial (centroid, box) and frame-index encodings instead (source: assumption from Sec. 3.5, p. 6).
- For the user's design (reader's inference): K tokens per frame over N frames give n = K·N, so per-layer attention cost grows as (K·N)²·d (C5); the paper's proposed (untested, future-work) remedy is restricted attention over a neighbourhood r, O(r·n·d) (source: assumption from Table 1, p. 6, Sec. 4, p. 7, Sec. 7).
- For the user's design (reader's inference): an action policy needs only an encoder stack plus output heads; the auto-regressive decoder of Sec. 3.1 is for generating sequences, and the paper does not describe a pooled or class-token readout (source: assumption from Sec. 3.1, p. 3).
- For the user's design (reader's inference): C12's extrapolation argument favours sinusoids when the number of frames or segments at test time exceeds training; learned embeddings were only shown equal at seen lengths (source: assumption from Sec. 3.5, p. 6, Table 3 row E, p. 9).

## Limitations / open questions

- All results are on text: translation and parsing; images, audio and video, and local restricted attention for large inputs, are named as future work (Sec. 7, p. 10).
- Extrapolation of sinusoidal encodings to longer sequences is a hypothesis, not tested (Sec. 3.5, p. 6).
- Self-attention averages attention-weighted positions, reducing effective resolution; multi-head attention is the stated counter (Sec. 2, p. 2).
- Table 3 variants were evaluated without checkpoint averaging on the dev set (Sec. 6.2, p. 9).
- The EN-FR BLEU of the big model is 41.8 in Table 2 and the abstract but 41.0 in the Sec. 6.1 text (p. 8).
- Table 4 also lists Luong et al. (2015) multi-task at 93.0 F1, above the semi-supervised Transformer's 92.7, although the text names only the RNN Grammar as better (Table 4, Sec. 6.3, p. 10).
- Open question: are segment tokens from a segmentation model, with no canonical order and a varying count, better served by no positional encoding plus spatial features, or by a canonical ordering (e.g. raster by centroid)?
- Open question: with post-LN and warmup tuned for supervised translation, how stable is a large Transformer under PPO's non-stationary targets?

## Related

- related [[he2021masked]]: MAE's ViT encoder and decoder are Transformer stacks with sine-cosine positional embeddings over image patches.
- related [[kirillov2023segment]]: SAM's mask decoder uses token self-attention and two-way cross-attention; its segments are candidate input tokens for a Transformer policy.
- related [[schulman2017proximal]]: PPO is the candidate trainer for a Transformer policy with shared policy and value heads.
- related [[serrano2024crossdomain]]: cross-domain RL transfer; a Transformer over a shared token space is one way to share parameters across games.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
