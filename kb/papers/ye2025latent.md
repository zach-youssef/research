---
id: ye2025latent
title: Latent Action Pretraining From Videos
authors: [Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, Minjoon Seo]
year: 2025
venue: ICLR 2025 conference paper (read as arXiv v2)
url: https://arxiv.org/abs/2410.11758
arxiv: 2410.11758
doi:
pdf_sha256: efab4f379e1a30ef7be36c87885c2c509d880ee9578f4cfdcc20d9321c93dce9
tags: [latent-actions, vision-language-action, robotics, vq-vae, pretraining, behavior-cloning, human-video, cross-embodiment, lapa]
status: read
read: 2026-11-25
---

# Latent Action Pretraining From Videos

## TL;DR

LAPA learns discrete latent actions between a frame and one a fixed window later (VQ-VAE, NSVQ),
labels robot and human video with it, pretrains a 7B VLM to predict them, then swaps the head and
fine-tunes on a small labelled set. It beats OpenVLA (50.1% vs 43.9%) on real tasks at ~1/30 the compute.

## Problem

VLA models need action labels from teleoperation, which limits scale; web video has no labels and
a different embodiment from robots (Sec. 1, p. 1). LAPA wants robot-foundation-model pretraining
from video without ground-truth actions (Sec. 1, pp. 1-2).

## Method

- **Stage 1, latent action quantization** (Sec. 3.1, p. 4, App. A, p. 15, Fig. 8): follows Genie's LAM with
  changes. The encoder takes frame x_t and x_{t+H} and outputs latent action z_t; the decoder takes z_t
  and x_t and reconstructs x_{t+H}; it attends to z_t with cross-attention instead of Genie's additive
  embedding. Encoder: patch embedding, spatial transformer per frame, causal transformer; decoder
  spatial only. The latent is d = e_2 − e_1 passed through a CNN, giving a sequence of s tokens from a
  codebook of size |C|, nearest-neighbour quantized (Eq. 1), then NSVQ noise replaces the quantization
  error (Eq. 2); stop-gradient on the patch embedding of x_t; L2 reconstruction (Eq. 3-4, p. 15).
  Only two frames are used, no past frames (footnote 1, p. 4).
- **Stage 2, latent pretraining** (Sec. 3.2, p. 4): the stage-1 encoder acts as an inverse dynamics
  model to label every frame; a pretrained VLM (7B LWM-Chat-1M, Sec. 4.2, p. 5) predicts z_t from the
  current image and the language instruction through a new single-layer latent-action head
  of vocabulary |C|; the vision encoder is frozen, the language model is unfrozen.
- **Stage 3, action fine-tuning** (Sec. 3.3, pp. 4-5): continuous actions (delta end-effector) are
  discretized per dimension into equal-count bins; the latent head is discarded and a new action
  head is trained, with the language model unfrozen (footnote 2, p. 5: reusing the latent
  head's logits plus an added head, as LAPO does, did worse for the 7B model).
- Decoder as world model (Sec. 3.1, p. 4; Sec. 5.2, p. 10): the stage-1 decoder renders rollouts from
  LAPA's predicted latent actions.

## Assumptions

- A1: Pretraining video comes with a language instruction (or task description) per clip (Sec. 3.2, p. 4).
- A2: Frame pairs are separated by a fixed window H chosen from the video's frame rate (0.6 s ahead
  for robot video, 2.4 s for human video) so that the delta is dynamic enough (App. F, p. 19).
- A3: A small action-labelled set from the target embodiment is available for stage 3 (100 to
  7k trajectories here) (Sec. 3.3, p. 4, Table 3, p. 16).
- A4: The latent generation space is 8^4 (vocabulary 8, sequence length 4) in all main experiments
  except Language Table (Sec. 5.1, p. 9).
- A5: The setting is single-camera tabletop-style manipulation with 7-DoF arms (Sec. 4.1, p. 5).
- A6: Language Table's latent size in Table 1 is not stated in the pages read; the paper varies it
  (vocabulary 8 to 1024, length 1 to 4, Fig. 16, p. 25) (guess: not 8^4, since Sec. 5.1 excepts it).
- A7: The Language Table latent-action analysis uses vocabulary 8 and sequence length 1 (App. E, p. 18).

## Claims

- C1: On Language Table, pretrained on 181k simulated trajectories and fine-tuned on 1k labelled
  (in-domain), LAPA gets 62.0 / 49.6% (seen / unseen) vs Scratch 15.6 / 15.2, VPT 44.0 / 32.8, UniPi
  22.0 / 13.2 and ActionVLA 77.0 / 58.8 [A1, A3, A6] (Sec. 4.3, p. 6, Table 1, p. 6).
- C2: Cross-task (7k labelled): LAPA 73.2 / 54.8 vs VPT 72.0 / 60.8 and ActionVLA 77.0 / 58.8; cross-environment
  (real-world pretraining 442k, 1k simulated labels): LAPA 33.6 / 29.6 vs VPT 18.0 / 18.4, Scratch 15.6 / 15.2,
  ActionVLA 64.8 / 54.0 [A1, A3, A6] (Sec. 4.3, p. 6, Table 1, p. 6).
- C3: On 54 real Franka rollouts per model, LAPA pretrained on Open-X averages 50.1% vs OpenVLA 43.9%
  (+6.22 points, strict criterion 35.19% vs 27.78%); pretrained on Bridge, LAPA 36.8% vs OpenVLA 30.8%,
  ActionVLA 32.6%, Scratch 21.2%. It is weaker on pick-and-place, mostly early grasp failures [A1, A3, A5] (Sec. 4.4, p. 7, Table 2, p. 7, Fig. 3, p. 7, App. G.3, p. 27).
- C4: Pretrained on Something-Something V2 human video only (220k videos, no robot actions), LAPA gets 52.1% on SIMPLER
  vs VPT 45.8, Scratch 34.4, UniPi 0.7, and 34.0% real-world average, above OpenVLA (Bridge) 30.8 [A1, A2, A3, A5] (Sec. 4.5, p. 8, Fig. 4, p. 8, Table 2, p. 7, Table 12, p. 22).
- C5: Pretraining Open-X LAPA took 8 H100 for 34 h (272 H100-hours) against OpenVLA's 21,500 A100-hours,
  about 30-40x less (the authors convert H100 to A100 with a 2-3x factor) [A5] (Sec. 4.6, p. 8).
- C6: Scaling model size (30M-300M quantizer), data share and latent vocabulary raises SIMPLER success;
  latent length helps less than vocabulary on Language Table, and a very large window H hurts. Data
  scaling on human video is mild: LAPA on 10% of Something-Something V2 gets 50.0% on SIMPLER vs 52.1%
  on all of it [A2, A4] (Sec. 5.1, p. 9, Fig. 5, p. 9, App. F, p. 19, Fig. 15, Fig. 16, p. 25, Table 12, p. 22).
- C7: Latent actions cluster by real motion: each of 8 Language Table codes matches a movement
  direction, the same code gives similar motion across different Open-X embodiments, and on human video codes
  also capture camera movement [A4, A7] (Sec. 5.2, pp. 9-10, Fig. 6, p. 10, App. E, pp. 18-19).
- C8: Decoding LAPA's pretraining-only latent predictions through the stage-1 decoder gives plausible
  closed-loop rollouts for a language instruction (qualitative) [A1, A5] (Sec. 5.2, p. 10, Fig. 7, p. 10).
- C9: Admitted limits: weaker at fine-grained motion such as grasping, latency in real-time use, and
  no test beyond manipulation video (driving, navigation, scenery) [] (Sec. 6, p. 10).
- C10: In-domain on SIMPLER with Bridgev2 pretraining (no action labels for LAPA, VPT, UniPi; ActionVLA
  uses the labels), average success is LAPA 57.3%, VPT 51.0, Scratch 34.4, UniPi 1.3, ActionVLA 63.5 and
  OpenVLA fine-tuned on SIMPLER 36.4 (the paper attributes OpenVLA's weakness to poor real-to-sim
  transfer); grasping rate 71.9 vs 66.7 (VPT) and 80.2 (ActionVLA) [A1, A3, A5] (Sec. 4.1, p. 5, Table 11, p. 21).
- C11: Head to head on the real robot, LAPA (Open-X) beats OpenVLA in 65.4% of trials excluding ties
  (31.5% wins, 16.7% losses, 51.9% ties) [A1, A3, A5] (App. D, p. 18, Fig. 11, p. 24).

## Implementation notes

- Quantizer: ~300M parameters, a C-ViViT variant with cross-attention decoder; window H = 3 for
  Bridge (5 Hz); codebook replacement early in training (source: Sec. 3.1, p. 4, App. F, p. 19). Vocabulary 8 and sequence length 4 hold for the main runs except Language Table (source: Sec. 5.1, p. 9), which covers the real-robot runs.
- Fine-tuning uses LoRA for OpenVLA baselines; LAPA and ActionVLA use batch 128 with image augmentation (source: App. C, p. 17).
- The scale of the VLM is 7B (LWM-Chat-1M); one epoch is sufficient for LAPA pretraining (source: Sec. 4.6, p. 9).
- Authors' code and checkpoints: latentactionpretraining.github.io (source: abstract, p. 1; "will open-source"; not inspected).

## Limitations / open questions

- Admitted: C9. LAPA fine-tuning relies on labelled data in the target action space (Sec. 3.3, p. 4).
- Evaluation is on 54 real rollouts per model with the authors' partial-success scoring; large
  standard errors in Table 1 (Sec. 4.3, p. 6).
- Open question (user's lens, grounding): LAPA's route is "label all frames with the IDM, train a
  policy to predict the latent, replace the head with a few hundred labelled real actions". For
  shmup superplay video the real-action fine-tune set could be a few runs with gamepad logs
  (a NitroGen-style set); untested here.
- Open question: H is the temporal granularity of one latent (0.6 s for robots), akin to an action
  chunk; NitroGen's 18-frame chunk could set H. Whether a single |C|^s code captures dodging plus
  firing over a window is unknown.
- Open question: latent actions are per-window deltas, not memory; on a long run they say what
  the ship did, not what weapon or rank state it is in.

## Related

- extends [[bruce2024genie]]: takes Genie's LAM, adds cross-attention, NSVQ, and a decoder that
  supports a VLA instead of a controllable world model (Sec. 3.1, p. 4).
- related [[schmidt2024actions]]: LAPO's latent policy with a decoder head; LAPA found a re-initialised head better at 7B (footnote 2, p. 5).
- related [[baker2022video]]: VPT is a baseline whose IDM needs labelled actions (Sec. 4.2, p. 5).
- related [[gao2025adaworld]]: continuous latent actions for world models, a different use of the same frame-pair idea.
- related [[jiang2026olaf]]: builds on latent actions across contexts.
- related [[magne2026nitrogen]]: frames-in, gamepad-out policy; LAPA's pipeline is a label-then-ground template.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
