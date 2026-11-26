---
id: mete2024quest
title: "QueST: Self-Supervised Skill Abstractions for Learning Continuous Control"
authors: [Atharva Mete, Haotian Xue, Albert Wilcox, Yongxin Chen, Animesh Garg]
year: 2024
venue: NeurIPS 2024 (read the arXiv v3 version)
url: https://arxiv.org/abs/2407.15840
arxiv: 2407.15840
doi:
pdf_sha256: cda97b0ca9ea522d9f797d32a7c34a945283c2dccebed3ca57d60f062911387b
tags: [skill-tokens, action-chunking, finite-scalar-quantization, imitation-learning, few-shot, transformer, robotics, discrete-latents]
status: read
read: 2026-11-25
---

# QueST: Self-Supervised Skill Abstractions for Learning Continuous Control

## TL;DR

Stage 1 compresses a 32-step action chunk into 8 discrete FSQ skill tokens with a causal, state-independent
autoencoder; stage 2 is a transformer predicting the tokens from images and a task embedding. Beats VQ-BeT and
diffusion on LIBERO-90 (88.6%) and 5-shot LIBERO-LONG (66.7%; 68.8% in Table 1); robot manipulation only.

## Problem

Robot policies transfer poorly across tasks. Continuous Gaussian-latent skill models suffer
posterior collapse; earlier discrete ones (VQ-BeT, PRISE, TAP) use encoders that don't capture
transferable structure, or need state prediction and MPC (Sec. 1, p. 2; Sec. 2.3, p. 3). The paper wants
shareable, state-independent low-level skills in a discrete latent space, usable on new tasks from few demos.

## Method

- **Setting**: demonstrations with continuous actions, RGB (front and gripper camera) plus proprioception, and a
  task label (language or one-hot) (Sec. 3.1, p. 3).
- **Finite Scalar Quantization** (Eq. 1, Sec. 3.2, p. 3-4): project to few dims, bound with tanh and round with
  straight-through gradients. Implicit codebook of size prod(alpha_i); no codebook-collapse tricks needed.
- **Stage 1, skill codebook** (Sec. 4.1, Eq. 2-4, p. 4-5): an action sequence a_t..a_{t+T-1} goes through causal
  strided 1D convs and masked self-attention, downsampled by F, giving n = T/F tokens (T = 32, F = 4,
  n = 8). The decoder is a transformer that cross-attends from fixed sinusoidal positions to all tokens, trained with
  an L1 reconstruction loss. The encoder and decoder are causal and state-independent: no observation enters.
  Tokens are fixed-length in time (each covers F steps); variable-length motion is expressed by sequences of codes.
- **Stage 2, skill prior** (Sec. 4.2, Eq. 5-6, p. 5): decoder-only transformer predicts the n tokens
  autoregressively given observation history (h frames) and the task embedding, with negative log-likelihood.
  Image encoder ResNet18 (first 4 layers) + spatial softmax, proprioception MLP (App. B.2, p. 15).
- **Inference** (Sec. 4.3, p. 6): top-k (k = 5), temperature 1 sampling of the 8 tokens, decode to 32 actions, execute
  the first T_a, then replan (receding horizon).
- **Few-shot** (Sec. 4.2, Eq. 7, p. 6): with 5 demos of a held-out task, tokenise them with the frozen encoder, fine-tune
  the prior, and optionally the decoder (loss weight 10).

## Assumptions

- A1: Skills are temporal action abstractions that do not depend on the state: the autoencoder sees only actions (Sec. 4.1, p. 5).
- A2: Actions are continuous-valued and the setting is imitation from demonstrations with task labels (Sec. 3.1, p. 3).
- A3: Held-out evaluation tasks are structurally similar to the pretraining tasks (p. 10).
- A4: Pretraining data has sufficient breadth: LIBERO-LONG tasks are pairs of tasks from the LIBERO-90 pretraining set (App. C, Table 7, p. 17).
- A5: The reported runs fix the chunk length T = 32, the downsampling F = 4 and a codebook of about 1000; F and the codebook size matter, few-shot more than multitask (Fig. 4, p. 9).
- A6: The setting is robot manipulation with one or two camera images plus proprioception, evaluated in simulation (Sec. 5.1, p. 6).

## Claims

- C1: On LIBERO-90 multitask BC (4 seeds), QueST gets 88.6% success versus VQ-BeT 81.3, Diffusion Policy 75.4, ResNet-T 84.4, PRISE 54.4 and ACT 50.8 [A1, A2, A5, A6] (Fig. 2a, p. 7, Sec. 5.2, p. 7).
- C2: In 5-shot transfer to 8 unseen LIBERO-LONG tasks, QueST scores 66.7% (65.2% with the decoder frozen), above the best baseline PRISE at 52.7 and ResNet-T 51.9 [A1, A2, A3, A4, A5, A6] (Fig. 2b, p. 7, Sec. 5.3, pp. 7-8).
- C3: On 10 LIBERO-LONG tasks trained only on that data, QueST gets 68.8% versus VQ-BeT 59.3, Diffusion Policy 50.1, ResNet-T 44.1 [A1, A2, A5, A6] (Fig. 2c, p. 7, Sec. 5.4, p. 8).
- C4: On Meta-World ML45, QueST is about equal to the others in multitask (91.7 vs ACT 90.8, Diffusion 90.3) and slightly ahead in 5-shot on 5 held-out tasks (71.9 vs ACT 70.8) [A1, A2, A3, A5, A6] (Fig. 3, p. 8).
- C5: Replacing FSQ by VQ drops LIBERO-90 / few-shot to 81.2 / 62.5 from 88.6 / 68.8, so FSQ's codebook utilisation helps [A1, A2, A3, A4, A5, A6] (Table 1, p. 8).
- C6: Conditioning the decoder on observations hurts (81.9 / 61.3), as does a mirrored decoder (86.3 / 45.4) [A1, A2, A3, A4, A5, A6] (Table 1, p. 8, Sec. 5.6, p. 9).
- C7: Causality matters: non-causal encoder 82.0 / 58.8, non-causal decoder 85.1 / 61.6, fully non-causal 78.5 / 56.1, versus 88.6 / 68.8 [A1, A2, A3, A4, A5, A6] (Table 2, p. 8, App. C, p. 17).
- C8: Best settings are F = 4 and a codebook around 1024; F = 16 and small codebooks (16, 64) degrade, and few-shot is more sensitive than multitask [A1, A2, A3, A4, A5, A6] (Fig. 4, p. 9, Sec. 5.6, p. 9).
- C9: The pipeline runs at 33 Hz on the authors' setup, versus 100 Hz ResNet-T, 100 Hz VQ-BeT, 50 Hz ACT and 12 Hz Diffusion Policy [A5, A6] (Sec. 5.5, p. 8).
- C10: Skill-token embeddings of similar motions (approaching, picking, placing) from different tasks cluster together in t-SNE of the first 11 LIBERO-90 tasks, from stage 1 trained without task labels [A1, A5, A6] (Sec. 5.7, Fig. 5, pp. 9-10).
- C11: Fine-tuning the decoder on predicted tokens adds 2.8 points in few-shot (66.0 frozen vs 68.8 with loss scale 10; 66.0 with scale 100), and the text calls the drop without decoder fine-tuning marginal (1.5%) [A1, A2, A3, A4, A5, A6] (App. C, Table 7, p. 17, Sec. 5.3, p. 8).
- C12: VQ-BeT with a chunk of 32 loses about 14% versus smaller chunks because one latent can't hold several motion primitives, which the authors use to explain QueST's multiple tokens per chunk [A1, A2, A6] (App. B.5, p. 16).
- C13: Varying QueST's chunk size T over 16, 48 and 64 instead of 32 changed LIBERO success by less than 1%, which the authors read as robustness to T; the paper gives no per-setting numbers and no Meta-World result for this [A1, A2, A5, A6] (App. B.5, p. 16).

## Implementation notes

- Stage 1: encoder/decoder dim 256; 2 encoder layers, 4 decoder layers, 4 heads; conv kernels [5,3,3], strides [2,2,1]; dropout 0.1; FSQ levels [8,5,5,5] giving 1000 codes (source: Table 3, p. 15).
- Stage 2: vocab 1000 plus a start token (1000), block size 8, 6 layers, 6 heads, embedding 384, dropout 0.1, beam size 5, temperature 1.0, decoder loss scale 10, T_a = 8, observation history 1 (source: Table 4, p. 15). QueST's execution horizon is T_a = 8 (source: Table 4, p. 15); the T = 32, T_a = 16 (LIBERO) and T = 16, T_a = 8 (Meta-World) values in App. B.3 belong to the Diffusion Policy baseline, not QueST (source: App. B.3, pp. 15-16).
- Hardware: 8 RTX 1080 Ti 10 GB, one GPU per run (source: App. B.4, p. 16).
- Seeds: 4 for multitask LIBERO, 5 for Meta-World, 9 for LIBERO few-shot (source: Sec. 5.2-5.3, p. 7).
- Few-shot protocol: 100 epochs for QueST, ResNet-T and VQ-BeT, 200 for ACT and Diffusion (source: App. D.1, p. 18).
- Mapping to a gamepad policy, reader's guess: an action chunk of 16-18 steps (NitroGen-style) would need T = 16-18 and F = 2 or 3 (n = 8 or 6); T = 16 is reported as within 1% of T = 32 on LIBERO (C13), so F is the open choice; a digital gamepad is not continuous, so L1 reconstruction would be replaced by cross-entropy on buttons (source: assumption).
- Code: the project page is https://quest-model.github.io; the repository and commit are not named in the PDF (source: App. A, p. 15).

## Limitations / open questions

- Authors: held-out tasks are structurally similar to training tasks, so a more diverse set may defeat the learned skill space; they suggest stage 1 on Open X-Embodiment with a bigger codebook, and other inductive biases (geometric invariance, dynamics consistency) (p. 10).
- Authors: only causality is built in as an inductive bias (p. 10).
- Open question (reader, skill lens): a token here is a fixed-length slice (F steps) of low-level motion, with no explicit start or termination; the 'slow' decision is the prior's next-token choice, which is not slower than the control rate (8 tokens per 32-step chunk, replanning every T_a).
- Open question: no long-term strategy: observation history is 1 and 'long horizon' means two chained tasks (LIBERO-LONG), not economy or rank.
- Open question: would state-independent skills transfer across games that share genre but differ in sprites and physics, given the manipulation tasks share a robot and scene?
- Reader: the few-shot LIBERO-LONG figure is 66.7 in Fig. 2b but 68.8 in Table 1 and Table 7, and frozen-decoder is 65.2 in Fig. 2b but 66.0 in Table 7; the paper doesn't explain the gap, which may reflect the seed sets (Fig. 2 caption says nine seeds for few-shot, Table 1's four) (Fig. 2b, p. 7; Table 7, p. 17).
- The reference to "Appendix ??" in Sec. 5.6 (p. 9) is broken in the PDF; the ablation discussion is App. C.

## Related

- related [[magne2026nitrogen]]: action chunks as the output unit for a gamepad policy; QueST would replace the chunk with discrete tokens.
- related [[gao2025adaworld]]: latent actions learned from video; QueST's tokens are learned from actions only, state-independent.
- related [[schmidt2024actions]]: LAPO, another VQ latent-action bottleneck, learned from video without action labels.
- related [[ye2025latent]]: LAPA, VQ latent actions for pretraining a policy.
- related [[kujanpaa2023hierarchical]]: VQ-VAE codes as subgoals, versus QueST's codes as motion primitives.
- related [[reed2022generalist]]: discretised action tokens in a generalist transformer policy.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
