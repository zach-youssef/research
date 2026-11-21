---
id: zhang2020correspondence
title: Learning Cross-Domain Correspondence for Control with Dynamics Cycle-Consistency
authors: [Qiang Zhang, Tete Xiao, Alexei A. Efros, Lerrel Pinto, Xiaolong Wang]
year: 2020
venue: arXiv preprint
url: https://arxiv.org/abs/2012.09811
arxiv: 2012.09811
doi:
pdf_sha256: d9a6222cf56bc47ed257057f343d64f5a39c5708846993e3b2173660e5d763e2
tags: [reinforcement-learning, transfer-learning, cross-domain, cycle-consistency, unpaired-alignment, state-estimation, sim-to-real]
status: read
read: 2026-11-21
---

# Learning Cross-Domain Correspondence for Control with Dynamics Cycle-Consistency

<!-- Read from arXiv v1 (2020-12-17, the only arXiv version), 15 PDF pages headed "Preprint. Under review."; pp. 10-12 are references, pp. 13-15 are App. A-C. The survey cites it as Zhang et al. 2021; that published version was not read. -->

## TL;DR

Learns an observation map G (e.g. images to simulator state) and action maps H, P between two
domains from unpaired, mostly random transitions, requiring T_Y(G(x_t), H(x_t, a_t)) ≈ G(x_{t+1}) via a
frozen forward model plus GANs; a Y policy then runs in X without fine-tuning (MuJoCo, real xArm).

## Problem

Cross-domain transfer (human to robot, sim to real, robot to robot) needs a correspondence
between domains (Sec. 1, p. 1). Invariant-feature methods need paired, aligned trajectories,
which are expensive, and invariance may not suit every downstream task (p. 1). Unpaired
translation such as CycleGAN only aligns visual observations and can't reason about physics or
morphology (p. 2); Kim et al. 2019 need RL policies in both domains, which ties the
correspondence to one task (Sec. 2, p. 3). The paper wants observation and action
correspondence from unpaired, randomly collected data; most of its experiments assume the
downstream task and rewards are unknown (pp. 2-3).

## Method

- Data: transitions τ_X = (x_t, a_t, x_{t+1}) and τ_Y = (y_t, u_t, y_{t+1}); maps G: X → Y,
  H: X×A → U and P: Y×U → A, its inverse (Sec. 3, p. 3). Observation correspondence x ⇔ y and
  action correspondence (x_t, a_t) ⇔ (y_t, u_t) are defined on p. 4.
- G is trained with a GAN loss against a discriminator D_Y on y (Eq. 1, p. 4); H and P with
  GAN losses (D_U, D_A) plus a CycleGAN-style action cycle ||P(y, H(x, a)) − a||_1 (Eq. 2, p. 4).
  The paper argues these alone leave G free to map x_t to y_{t+1} (p. 4).
- Dynamics cycle-consistency: ||G(x_{t+1}) − T_Y(G(x_t), H(x_t, a_t))||_1 (Eq. 3, p. 5). T_Y is
  not differentiable, so a forward model F is regressed on τ_Y with an L1 loss (Eq. 4, p. 5),
  pre-trained and then frozen. Full loss: λ0 L_dyn-cyc + λ1 (action GANs + action cycle) +
  λ2 L_adv(G, D_Y) (Eq. 5, p. 5).
- Alg. 1 (p. 5): pre-train F, then alternate e1 steps of H, P (G fixed, λ2 = 0) with e2 steps
  of G (H, P fixed, λ1 = 0). Action maps start from the identity, or for new joints from a copy
  of the nearest joint's action (Sec. 3, p. 5; App. A.4, pp. 13-14).
- Settings (Fig. 2, p. 4; pp. 5-6): cross-physics (G = identity, learn H, P), cross-modality
  (H, P = identity, learn G, e.g. images to state), both jointly, and cross-morphology (both
  maps between agents with different limb counts). Transfer Y → X at test time: y = G(x), run
  the Y policy, map its action back with P (p. 6).
- G is a ResNet-18 with an MLP head on the current and two previous frames stacked; D, F, H, P
  are MLPs (Sec. 3, p. 6; App. B.1, p. 14).
- Evaluation (Sec. 4, p. 6): MuJoCo HalfCheetah, FetchReach, Walker, Hopper (Gym) and
  FingerSpin (DM Control); 50k unpaired training samples per domain in most settings, 10k for
  evaluation; Y policies from DDPG+HER (FetchReach) or TD3, 5 seeds, not fine-tuned in X.

## Claims

- C1: Observation and action correspondences are learned from unpaired data, randomly collected and independent of the RL task in most experiments (cross-physics data come from a pre-trained Y policy, C3), and the domain-Y policy transfers to X without fine-tuning on X (p. 1, Sec. 2, p. 3, Sec. 4, p. 7, Sec. 6, p. 9).
- C2: Cross-physics (larger armature or different torso mass in X, App. A.1 Table 6), transferring Y's policy through P beats direct deployment and a domain-randomized policy on four of five tasks and ties at 100% on FetchReach, with widely overlapping stds over 5 seeds (e.g. Hopper ±794 vs DR ±869); the paper calls it superior on all tasks: HalfCheetah 3997±438 vs 3651 and 3763±752 (oracle Y 6270, oracle X 6769), FingerSpin 562 vs 483 and 492, Walker2d 667 vs 516 and 546, Hopper 1919±794 vs 1542 and 1683±869 (Table 1, Sec. 4, pp. 6-7).
- C3: In the cross-physics setting the unpaired training data come from a pre-trained policy in domain Y, not random actions, and a Cycle-GAN action-translation baseline learns a near-random mapping, so its numbers are not reported (Sec. 4, p. 7, App. C, p. 15).
- C4: Cross-modality (RGB images in X, simulator state in Y, random actions, no paired data), the state-estimation L1 error is 0.57 / 0.23 / 0.05 on HalfCheetah / FingerSpin / FetchReach, against 2.07 / 1.92 / 0.94 for an image-state Cycle-GAN and 2.18 / 1.61 / 0.87 for a random projection (Table 2, Sec. 4, p. 7).
- C5: Running the state policy on images through G gives 1504±256 on HalfCheetah, 341±39 on FingerSpin and 92% on FetchReach, against an image-trained oracle of 3689, 765 and 100% (state oracle 6270, 804, 100%); Cycle-GAN gets −119, 0 and 0% (Table 2, p. 7).
- C6: Walker and Hopper are left out of the cross-modality setting because even supervised state estimation with the same number of image-state pairs works poorly on them (Sec. 4, p. 7).
- C7: On HalfCheetah cross-modality, the state-estimation L1 error falls as training data grow, from about 1.0 at the smallest size plotted to 0.57 at 50k samples (Fig. 3, Sec. 4, pp. 7-8).
- C8: Removing the discriminator raises the HalfCheetah L1 error from 0.57 to 1.63 and lowers the transferred task reward from 1504 to 439 (Fig. 3, Sec. 4, pp. 7-8).
- C9: With paired image-state data, adding dynamics cycle-consistency to the supervised loss raises the training-state L1 error slightly (0.13 to 0.16) but raises the transferred task reward (3855 to 4127), which the authors read as regularization that helps on novel states (Fig. 3, Sec. 4, pp. 7-8).
- C10: Cross-modality-and-physics from random unpaired data, the joint model transfers with 1251±297 on HalfCheetah, 305±43 on FingerSpin and 92% on FetchReach, against 856, 243 and 92% when only the observation map is trained, and −226, 0 and 0% for Cycle-GAN (oracle Y 6270, 804, 100%) (Table 3, Sec. 4, p. 8).
- C11: Cross-morphology from random unpaired data, transferring a 2-leg HalfCheetah policy to a 3-leg one scores 2471±382 (oracle Y 6270) and a 3-limb Swimmer policy to a 4-limb one 204±56 (oracle Y 366), while Cycle-GAN (−43, 14) and the nearest-joint action-copy initialization alone (−37, −15) fail (Table 4, Sec. 4, p. 8, App. A.4, p. 14).
- C12: On a real xArm, from 11k unpaired real (image, action, next image) triplets from an uncalibrated camera and random simulator trajectories, end-effector estimation L1 error is 0.025 on 1,000 random and 0.033 on 100 smooth test samples, against 0.18 and 0.21 for Cycle-GAN and 0.30 and 0.18 for a random G; using 7 joint positions as the simulator state gives 0.031 and 0.044 (Table 5, Sec. 5, pp. 8-9).
- C13: Optimizing the full objective end to end collapses (G and H find shortcuts that fool the forward model), and training F jointly lets G and F map everything to zero; hence F is pre-trained and frozen and training alternates (Sec. 3, p. 5).
- C14: GAN losses with a domain cycle loss alone don't fix the correspondence: G can map x_t to y_{t+1} and back without violating domain cycle-consistency, which the dynamics cycle is meant to constrain (Sec. 3, p. 4).

## Implementation notes

- F: MLP (hidden 64, 128, 32) on (state, action), Adam, lr 0.001 halved every 3 epochs, 20 epochs, all settings (source: App. B.1-B.2, p. 14). Sec. 3 p. 6 says lr 0.0001 for F and "the same learning rate" for the other networks (50 epochs), while App. B.2 gives 0.001 for F and for G, H, P in cross-physics and cross-modality; the appendix is taken as the more specific (source: assumption).
- D: five-layer MLP (hidden 32, 64, 128, 32) on states; H, P: MLPs (hidden 32, 64, 128, 32) on (state, action); G: ResNet-18 plus MLP head (256, 64, 32), identity in cross-physics (source: App. B.1, p. 14).
- Images 256×256, current plus two previous frames concatenated as G's input (source: App. A.2, p. 13).
- Cross-physics λ0 = 200, λ1 = 1, λ2 = 0; cross-modality λ0 = 200, λ1 = 0, λ2 = 3; Adam, batch 32, 50 epochs, lr 0.001 cut by 1/3 every 10 epochs (source: App. B.2, p. 14).
- Joint (cross-morphology) training: 10 epochs, batch 32, lr 0.0001, alternating every 5000 steps (e1 = e2 = 5000); the λ values for it aren't given, presumably the two settings' values (source: App. B.2, p. 14; the λ values are an assumption).
- Sec. 3 p. 5 says "set γ = 0 in Eq. 5" for cross-modality, but Eq. 5 has no γ; it matches λ1 = 0 in App. B.2 (source: assumption).
- Sec. 4 says 50k unpaired "trajectories" per domain; read here as transitions (the training tuples), not checked (source: assumption).
- Reference policies: TD3 400k steps, 25k initial exploration steps, discount 0.99, target update 0.005, exploration noise 0.1; DDPG+HER 50 epochs of 400 episodes, ε 0.3, discount 0.98; evaluation over 50 episodes and 5 seeds (source: App. B.3, pp. 14-15).
- Domain-randomization ranges and the physics values per task (source: App. A.1 Table 6, p. 13).
- Code: the project page https://sjtuzq.github.io/cycle_dynamics.html is linked (p. 1); no code was read for this note.

## Limitations / open questions

- Transfer stays below the target-domain oracle in Tables 1-2 except FetchReach cross-physics
  (100% = 100%), e.g. 1504 vs 3689 on image HalfCheetah (Table 2, p. 7); Tables 3-4 give only
  the source-domain oracle (pp. 6-8).
- Cross-modality was only shown where supervised state estimation works; Walker and Hopper
  were excluded (Sec. 4, p. 7). Cross-physics used data from a pre-trained Y policy (p. 7).
- Needs a forward model trained on domain Y's transitions, so Y must have a learnable state
  (Sec. 3, p. 5); X is never modelled. That it should be low-dimensional is an inference, loosely
  supported by end-effector states beating 7 joint positions (Sec. 5, p. 9).
- Joint-position observations are harder than end-effector ones on the real robot (Table 5,
  p. 9); cross-morphology needs a hand-made nearest-joint initialization (App. A.4, p. 14).
- Open question: correspondence is pairwise, one Y per X; for many games, a shared canonical Y
  (or a learned latent Y) would be needed, and the paper doesn't test one.
- Open question: H and P are continuous maps with an L1 cycle; game action spaces are mostly
  discrete and differ in size, which the method doesn't address.
- Open question: the forward model is deterministic L1 regression; stochastic game dynamics
  might blur the dynamics cycle.
- Open question: G stacks 3 frames, the same "N frames" idea the survey's embedding uses, but
  its target is a known simulator state, not an unsupervised embedding.

## Related

- related [[serrano2024crossdomain]]: the survey cites this paper as [66], a single-source policy-transfer method learning state-action maps that align transition dynamics for zero-shot transfer (its pp. 5, 11).

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
