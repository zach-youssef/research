---
id: yu2024tokenizer
title: Language Model Beats Diffusion — Tokenizer is Key to Visual Generation
authors: [Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, Lu Jiang]
year: 2024
venue: ICLR 2024
url: https://arxiv.org/abs/2310.05737
arxiv: 2310.05737
doi:
pdf_sha256: b393503a10d53239ff4ebc67124d99fcc96777e78e4d525b1dd17ba43a474537
tags: [video, tokenizer, magvit, lookup-free-quantization, vq-vae, causal-3d-cnn, masked-language-model, video-generation, video-compression, action-recognition]
status: read
read: 2026-11-25
---

# Language Model Beats Diffusion — Tokenizer is Key to Visual Generation

<!-- Read from arXiv v3 (2024-03-29, printed "Published as a conference paper at ICLR 2024"), 19 PDF pages: pp. 1-9 main text, pp. 10-14 references, pp. 15-19 Apps. A-B. MAGVIT-v2. -->

## TL;DR

MAGVIT-v2: a temporally causal 3D-CNN video tokenizer with lookup-free quantization (binary latents,
2^18 vocabulary) that also tokenizes single images. With it a 307M masked LM beats diffusion models on
ImageNet 512 (FID 1.91) and Kinetics-600 (FVD 4.3), and raters prefer its compression over HEVC.

## Problem

Language models lag diffusion models on image and video generation (e.g. ImageNet 256: 3.41 vs 1.79
FID), which the paper attributes to weak visual tokenizers (Sec. 1, p. 1). Larger VQ vocabularies
improve reconstruction but hurt generation past a few thousand codes, so LM vocabularies stay at 1-8k
(Sec. 3.1, pp. 3-4). MAGVIT's 3D CNN can't tokenize images and flickers on long videos (Sec. 2, p. 3).

## Method

- **Lookup-free quantization (LFQ)** (Sec. 3.1, Eq. 2-5, p. 4): the codebook embedding dimension is
  reduced to zero; the latent is log2 K independent binary dimensions, q(z_i) = sign(z_i), and the
  token index is the binary number Σ 2^(i-1) 1{z_i > 0}. An entropy penalty
  E[H(q(z))] - H(E[q(z)]) encourages code use; the batch-entropy term is approximated with sub-groups
  of dimensions for K > 2^18. Trained with reconstruction, GAN, perceptual and commitment losses and
  LeCAM regularization.
- **Causal 3D CNN** (Sec. 3.2, Fig. 2c, p. 5): every 3D convolution pads k_t - 1 frames before and
  none after, so each output depends only on past frames and the first frame is encoded alone; this
  lets one model tokenize images and videos. Temporal stride-s downsampling maps 1 + s·t frames to
  1 + t latent frames.
- **Architecture changes** (Sec. 3.2, p. 5, Fig. 7, p. 16): strided convolutions instead of average
  pooling, depth-to-space upsampling, temporal downsampling moved to the last encoder blocks, 3D blur
  pooling in the discriminator, adaptive group normalization in the decoder conditioned on the
  quantized latents.
- **Token factorization** (Sec. 3.2, p. 5): a 2^18 vocabulary is split into two 2^9 sub-tokens,
  embedded separately and summed; output uses weight-tied per-subspace heads.
- **Generation** (Sec. 2, pp. 2-3, Sec. 4.2, p. 6): the MAGVIT masked LM (MLM), iterative
  non-autoregressive decoding with a decreasing masking schedule; an AR-LM is also tested (App. B).
- **Tokens per frame** (App. A.1-A.2, p. 15): a 17-frame 128 × 128 clip becomes 5 × 16 × 16 = 1,280
  tokens (the first frame 256 tokens, then 256 per 4 frames, so 64 per frame amortised); images get
  16 × 16 tokens at 256 or 512 resolution (16× or 32× downsampling). Each token is 18 bits.
  **Temporal redundancy** is handled by learned 4× temporal downsampling, with the first frame as a
  separately encoded keyframe; the compression ratio is fixed, not content-adaptive.

## Assumptions

- A1: The Kinetics-600 video tokenizer (128 × 128, 17 frames, stride 1, inflated from a 2D ImageNet model) is the one used for K600 generation, compression and recognition; image tokenizers are trained on ImageNet (App. A.1-A.2, p. 15).
- A2: Comparisons with diffusion models use the same training data (ImageNet or Kinetics), a comparable model size and a similar training budget (Sec. 1, p. 1, Sec. 4.2, p. 7).
- A3: Generation uses the MAGVIT masked LM transformer (~307M parameters) with token factorization into two 2^9 heads, unless stated (Sec. 4.2, p. 6).
- A4: The tokenizer is temporally causal with 4× temporal and 8× (video) spatial compression, the first frame encoded on its own (Sec. 3.2, p. 5, App. A.2, p. 15).
- A5: Video compression quality is judged by 16 human raters in pairwise forced choice on the 30 MCL-JCV videos at 640 × 360 (Sec. 4.3, p. 7, App. A.3, p. 17).
- A6: Evaluated videos and images are natural footage (Kinetics, UCF-101, SSv2, MCL-JCV, ImageNet) (reader's inference, Sec. 4.1, p. 6).
- A7: Tokenizers for the UCF-101 results follow MAGVIT's training setup; their training data are not stated (reader's inference, Sec. 4.1, p. 6).

## Claims

- C1: With VQ, larger vocabularies improve reconstruction FID but generation FID worsens past 2^14; with LFQ both improve up to 2^16 (ImageNet 128 × 128, MLM of 306-372M parameters) [A1, A3] (Fig. 1, Sec. 3.1, p. 4).
- C2: Video generation: Kinetics-600 frame prediction FVD 4.3 (24 steps) or 5.2 (12 steps) vs MAGVIT 9.9 and a non-causal LFQ baseline 11.6; UCF-101 class-conditional FVD 58 vs MAGVIT 76, with 307M parameters [A1, A2, A3, A4, A6, A7] (Table 1, p. 7).
- C3: ImageNet 512 × 512 class-conditional generation: FID 1.91 and IS 324.3 with guidance (64 steps) vs VDM++ 2.65 (512 steps, 2B parameters), and 3.07 FID without guidance; at 12 steps 4.61 [A1, A2, A3, A6] (Table 2, p. 7).
- C4: ImageNet 256 × 256: FID 1.78 and IS 319.4 with guidance (307M, 64 steps) vs MDT 1.79 (676M, 250 steps), 3.65 FID without guidance; the margin over diffusion is smaller than at 512 [A1, A2, A3, A6] (Table 7, p. 19, Sec. 4.2, p. 7).
- C5: In the rater study raters prefer MAGVIT-v2 to MAGVIT, HEVC and VVC at similar bit rates (Elo); at 0.0384 bpp it has LPIPS 0.104 vs MAGVIT 0.144, VVC 0.153 and HEVC 0.199, but lower PSNR (26.18 vs VVC 32.65) and MS-SSIM (0.894 vs 0.966) than the codecs; the abstract words it as "comparable to" VVC [A1, A4, A5, A6] (Fig. 6, p. 7, Table 3, p. 8).
- C6: As masked-modelling targets for SSv2 action recognition, its tokens give 67.38% vs MAGVIT 67.22% and raw pixels 64.83%; as detokenized inputs to a frozen pixel-trained ViViT they give 62.40% (SSv2), 75.34% (K400) and 77.93% (K600) vs 63.08%, 76.13% and 78.92% on raw pixels [A1, A4, A6] (Table 4, Sec. 4.4, p. 8).
- C7: Among causal tokenizers on UCF-101, the causal 3D CNN (58M) gets first-frame FID 7.06 and FVD 96.33, vs C-ViViT (90M) 28.02 and 437.54 and C-ViViT + MAGVIT (67M) 13.52 and 316.70; MAGVIT (39M, non-causal) has FVD 107.15 [A6, A7] (Table 5a, p. 9).
- C8: On ImageNet 128 image tokenization, each change lowers FID from MAGVIT's 2.65: LFQ 2.48, large vocabulary 1.34, up/downsamplers 1.21, deeper model 1.20, adaptive normalization 1.15 (LPIPS 0.1292 to 0.0685) [A1, A6] (Table 5b, p. 9).
- C9: On UCF-101 video tokenization, FVD falls from MAGVIT's 24.55 to 16.12 (LFQ and large vocabulary), 15.37 (up/downsamplers), 11.11 (late temporal downsampling), 8.90 (deeper) and 8.62 (3D blur pooling) [A6, A7] (Table 5c, p. 9).
- C10: With an AR-LM on UCF-101, MAGVIT-v2 tokens give FVD 109 vs MAGVIT's 265 (840M vs 306M parameters, without factorization or weight tying) [A4, A6, A7] (Table 8, p. 19).
- C11: Because the first frame depends on no other frame, the same tokenizer encodes single images and videos with one shared vocabulary [A4] (Sec. 3.2, p. 5).

## Implementation notes

- Tokenizer hyper-parameters: base channels 128, channel multipliers 1,2,2,4, 4 residual blocks, latent 5 × 16 × 16, vocabulary 2^18, entropy loss weight 0.1 annealed from 3× over 2,000 steps, reconstruction 5.0, adversarial 0.1, perceptual 0.1, commitment 0.25, LeCAM 0.001, r1 penalty 10, Adam β1 = 0, β2 = 0.99, peak lr 1e-4 with warmup and cosine decay, EMA 0.999, batch 256 (source: App. A.2, p. 15).
- Video tokenizer: 190 epochs on Kinetics-600 at batch 256; discriminator trained from scratch rather than inflated; inflation fills the last temporal slice to fit causal padding (source: App. A.1, p. 15).
- Image MLM: 1,080 epochs, batch 1024, class label dropped 10% for classifier-free guidance, temperature 30 (512) or 15 (256), guided scale 25 at temperature 15 (source: App. A.1, p. 15).
- K600 frame prediction: the first 2 × 16 × 16 tokens (5 frames) condition the MLM; 360 epochs, cosine schedule, temperature 32 (source: App. A.1, p. 15).
- No code release is named in the paper; samples at magvit.cs.cmu.edu/v2 (source: p. 6).
- For a 256 × 256 game frame at the paper's 8× video downsampling, 32 × 32 = 1,024 tokens per latent frame, or 256 per frame amortised over 4 (source: assumption, scaling App. A.2).
- The authors note running it on CPUs like standard codecs needs further work (source: Sec. 4.3, p. 8).

## Limitations / open questions

- Text-to-image and text-to-video results are not given (App. B, p. 18).
- At equal bit rate it trails HEVC and VVC on PSNR and MS-SSIM, which the authors attribute to codecs' block artifacts that those metrics miss (Sec. 4.3, p. 8).
- Action recognition from its tokens still trails the state of the art (Sec. 4.4, p. 9).
- Open question: tokens are trained for reconstruction and generation, with a fixed 4× temporal ratio; it's untested whether they keep small, policy-critical sprites (bullets, pickups) in shmup frames, and C6 only bounds the loss for natural-video recognition.
- Open question: the causal design fits an online policy (no future frames), and the first-frame keyframe plus 4-frame groups resembles an I/P split; a 4-frame latent step aligned to an 18-frame action chunk doesn't divide evenly (4 × 4 + 1 = 17 frames per 5 latents).
- Open question: binary LFQ codes are compact (18 bits per token); could a policy write its own memory in such codes, or would continuous memory tokens (ryoo2023turing) be easier to train?
- Open question: game RAM could be used offline to test whether frozen MAGVIT-v2 tokens linearly encode object positions or HUD state (weapon level, score) in game frames.

## Related

- related [[hafner2025scalable]]: Dreamer 4 uses a learned causal tokenizer for game video, continuous rather than LFQ codes.
- related [[cheng2025jointly]]: JOWA tokenises Atari frames with a VQ-VAE for a world-action model; LFQ is an alternative quantizer.
- related [[reed2022generalist]]: Gato tokenises images as patches; MAGVIT-v2 argues discrete learned tokens matter for LM-style models.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
