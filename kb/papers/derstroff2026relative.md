---
id: derstroff2026relative
title: "Reframing Object-Centric Reinforcement Learning: From Absolute to Relative Representations"
authors: [Cedric Derstroff, Jannis Blüml, Janosch Moos, Michael Girstl, Zihan Ye, Debora Clever, Kristian Kersting]
year: 2026
venue: arXiv preprint
url: https://arxiv.org/abs/2610.28197
arxiv: 2610.28197
doi:
pdf_sha256: 63fe8dbf930a7b473a1140a3fd27eccdd11169b6df78e6f0013d3879e9ddbdfd
tags: [reinforcement-learning, object-centric, relative-representation, egocentric, invariance, transformer, ppo, atari, hackatari, robustness, state-representation]
status: read
read: 2026-11-22
---

# Reframing Object-Centric Reinforcement Learning: From Absolute to Relative Representations

## TL;DR

ORCA re-expresses extracted objects' positions and velocities relative to an anchor object (player
or ball); ORBiT, a PPO set transformer on them, gets 1.10 HNS IQM on 13 Atari games vs 0.54 on
absolute coordinates and loses 4% on HackAtari variants vs 30%; it needs a decision-relevant anchor.

## Problem

Object-centric RL (OCRL) feeds the policy object features in a fixed global frame, so the same
relative configuration (a cauldron 5 px right of the player in Asterix) is a different input at
every screen position and must be relearned; agents become fragile under translation and
platform shifts even when the relative scene is unchanged (Sec. 1, p. 2, Fig. 1, p. 2). Anchors
and relative frames are common in multi-agent RL and robotics but, the authors say, largely
absent from OCRL (p. 2, p. 10).

## Method

- **Input.** At step t an upstream detector (SAM3, OCAtari or similar) gives a set of N_t objects,
  each with intrinsic attributes c (in the experiments a one-hot type) and extrinsic ones
  (position p, velocity v = p_t − p_{t−1} in Atari; Foosball adds rotation) (Sec. 2, p. 3).
- **ORCA** (Object-Relative Coordinates fixed to Anchor): pick an anchor ψ_t with position,
  velocity and orientation Q_t; each object becomes [c, R_t(p − p^ψ), R_t(v − v^ψ)] with
  R_t = Q_tᵀ (Eq. 1-3, pp. 3-4). The anchor encodes itself as zero position and velocity; its
  absolute pose is discarded (p. 4). Atari and Foosball use R = I (translation only); Tag uses
  the full co-moving, co-rotating kinematic transform (App. G, p. 23). No learnable parameters;
  it wraps any object-vector environment (p. 4).
- **Anchor choice.** The player where the game has one; the ball in Foosball (11 controlled
  figurines); the anchor is treated as a task-dependent design parameter chosen by hand (p. 5).
- **Relations.** Pairwise displacements are differences of two anchor-relative tokens (Eq. 4,
  p. 4), so attention can recover them without an explicit scene graph; graphs are not evaluated
  (pp. 4-5).
- **ORBiT.** Shared linear embedding per object token, ℓ transformer encoder layers without
  positional encodings, mean pooling, linear policy and value heads, trained end-to-end with PPO
  and no auxiliary losses (Sec. 2, p. 5, Fig. 3, p. 5, App. C, p. 20). OCT is the identical
  model on absolute coordinates; PPObj is an MLP over concatenated object vectors, PPObj-ORCA the
  same with ORCA features; PPO^CNN uses pixels (p. 6).
- **Evaluation.** 13 Atari games (App. E, p. 22), 40M frames / 10M steps, 5 seeds, CleanRL PPO;
  zero-shot tests on unseen perturbations: Translate (shift game content inside the frame, same
  offset added to object coordinates), Shuffle (permute object order) and one HackAtari variant
  per game (Table 6, p. 22); retained performance ϱ = (r_var − r_min)/(r_org − r_min) (Eq. 5,
  p. 6); IQM with 95% CI via rliable (App. A, p. 18). Foosball: full game, 8-DoF continuous
  control, self-play, ~6.6B steps on 4,096 environments (p. 6, App. F, p. 22). Tag: a 2D arena
  where the player catches a target among distractors and landmarks, all with headings (App. G,
  p. 23).
- **Theory.** Theorem 1: the ORCA observation is unchanged when one SE(d) transform is applied
  to all objects and the anchor; Corollary 1: so is any policy or value function of it (App. B,
  pp. 18-19).

## Assumptions

- A1: An object extractor supplies, at every step, the set of objects with type, position and velocity; the experiments read them from OCAtari's RAM extraction (or simulator state) and assume the extractor is perfect (Sec. 2, p. 3, Sec. 3, p. 6, Sec. 4, p. 11).
- A2: An anchor object exists at every step and can be localized (Sec. 2, p. 4).
- A3: The anchor is chosen by hand per environment, as the decision-relevant object (the player, or the ball in Foosball) (p. 5, Sec. 4, p. 11).
- A4: The environments are 2D (Sec. 4, p. 11, App. B, p. 18).
- A5: In Atari and Foosball the anchor has no meaningful orientation, so R = I and only translation invariance is exercised (Sec. 2, p. 4, App. B, p. 20).
- A6: Objects are described by a one-hot type from a vocabulary fixed at initialization plus positions and velocities (no size, shape or appearance features) (Sec. 2, p. 3, p. 5, Table 2, p. 21).
- A7: The scenes contain static objects that can serve as landmarks, so the anchor's discarded absolute pose is recoverable from them; the authors say this holds in most environments (Sec. 2, p. 4, App. H, p. 24).
- A8: One agent is trained per game and tested on perturbed versions of that same game; no test is on a game unseen in training (reader's inference, Sec. 3, p. 6, App. E, p. 22).
- A9: The perturbations are never seen in training (Sec. 3, p. 6).
- A10: Translate is lossless: content is shifted inside the empty border without cutting any, which excludes Freeway, Frostbite and Kangaroo (Sec. 3, p. 7, Table 8, p. 25).
- A11: All methods share the PPO hyperparameters of Huang et al. and Moos et al., without per-method tuning (Table 4, p. 21).
- A12: The encoder is permutation-invariant: a mean-pooled transformer without positional encodings (App. B, p. 19, App. C, p. 20).
- A13: Foosball is simulated with full state of the ball, goals and every figurine observed, and training uses mixed self-play with randomized rod configurations, evaluation the full game (App. F, p. 22).
- A14: In Tag, the target and distractors differ only in type, and the distractor and landmark counts change only at test time (App. G, p. 23, Table 10, p. 27).
- A15: The perturbed scenes keep the training game's object types (reader's inference, Table 6, p. 22).

## Claims

- C1: On 13 Atari games, ORBiT reaches 1.10 HNS IQM (95% CI [1.04, 1.16]) vs 0.54 ([0.51, 0.58]) for OCT, a gain of 0.56 (CI [0.49, 0.63]), about human level [A1, A2, A3, A5, A6, A7, A8, A11] (Sec. 3, pp. 6-7, Fig. 4, p. 7).
- C2: OCT falls below the pixel baseline PPO^CNN (0.69, CI [0.66, 0.72]) while ORBiT clears it; the authors read this as the transformer failing to capture position-dependent correlations without an anchor [A1, A2, A3, A5, A6, A7, A8, A11] (p. 7, Fig. 4, p. 7).
- C3: ORCA also helps the MLP: PPObj rises from 0.62 to 0.77 HNS IQM (gain 0.15, CI [0.09, 0.22]); neither ORCA alone (PPObj-ORCA) nor the transformer alone (OCT) clears human level, only the combination; the gap between the two gains is mainly driven by a few games such as Asterix [A1, A2, A3, A5, A6, A7, A8, A11] (p. 7, Fig. 4, Table 8, p. 25).
- C4: Under zero-shot Translate, OCT loses 50% and PPObj 44% of their performance with absolute inputs, vs about 1% (ORBiT) and 5% (PPObj-ORCA) with ORCA; PPO^CNN collapses by 88% (IQM of ϱ over the 10 games with a Translate variant) [A1, A2, A3, A5, A6, A7, A8, A9, A10, A11, A15] (Sec. 3, pp. 7-8, Fig. 5, p. 8).
- C5: Under Shuffle, OCT and ORBiT are unaffected while PPObj drops 85-93% with or without ORCA: permutation robustness comes from the encoder, not from ORCA [A1, A6, A7, A8, A11, A12] (p. 8, Fig. 5, p. 8, App. B, p. 19).
- C6: On the HackAtari variants (one per game, 10 games), absolute-coordinate OCT and PPObj drop 30% and 62%; with ORCA, PPObj drops 7% and ORBiT 4%, comparable to PPO^CNN (7%) [A1, A2, A3, A5, A6, A7, A8, A9, A11, A15] (p. 8, Fig. 5, p. 8, Table 6, p. 22).
- C7: Per game, some HackAtari variants still break ORBiT: on MsPacman set_level_2 it falls from 2449 ± 1067 to 214 ± 183 (PPO^CNN from 3907 to 466), while on Pong hidden_enemy it holds (13 ± 7 to 14 ± 6) where OCT falls from 18 to −20 [A1, A2, A3, A5, A6, A7, A8, A9, A11, A15] (Table 8, pp. 25-26).
- C8: ORBiT improves on OCT in every Atari game except Kangaroo and Pong; Kangaroo's higher OCT mean comes from one seed (12680), and the per-seed IQM favours ORBiT (3727 vs 3435) [A1, A2, A3, A5, A6, A7, A8, A11] (App. H, p. 24, Table 8, pp. 25-26, Table 9, p. 26).
- C9: Theorem 1: the ORCA observation set is invariant to any rigid transform in SE(d) applied jointly to all objects and the anchor, and by Corollary 1 so are any policy and value function on it, with no augmentation or auxiliary loss [A2] (Sec. 2, p. 4, App. B, pp. 18-19).
- C10: Every pairwise displacement between objects is recoverable as the difference of two anchor-relative tokens, so ORCA keeps the relational content a displacement-edge graph would give [A2] (Eq. 4, Sec. 2, p. 4).
- C11: In Foosball, across 50,000 goals per algorithm pair (5×5 seeds, 1,000 goals per side), PPO^MLP scores 330 and OCT 73 against ORBiT, and OCT scores about 3.9k against PPO^MLP's 46.1k [A1, A2, A3, A4, A5, A7, A11, A13] (Sec. 3, pp. 8-9, Fig. 6b, p. 9, Table 7, p. 24).
- C12: To the authors' knowledge, ORBiT is the first agent trained from scratch via self-play to play the full game of Foosball [A13] (p. 9).
- C13: In Tag (rotations included), ORBiT reaches 22.4 catches per 500 steps (CI [21.9, 22.9]) vs 1.8 ([1.2, 2.9]) for OCT; PPObj-ORCA 21.6, PPObj 5.6 [A1, A2, A3, A4, A14] (p. 9, Table 10, p. 27, Fig. 7b, p. 10).
- C14: Trained with 5 distractors, ORBiT keeps 21.9-22.5 catches with 0, 1 or 10 at test time, and 21.9-22.2 with 0 or 4 landmarks; OCT stays at 1.1-2.4, falling slightly as distractors grow; PPObj can't be evaluated past its fixed slot count [A1, A2, A3, A4, A14] (p. 9, Table 10, p. 27).
- C15: The gain needs a decision-relevant anchor: in Tag, ORBiT gets 22.4 (player), 20.7 (target), 0.7 (a distractor) and 1.4 (a static landmark, close to OCT) catches; PPObj-ORCA gets 10.3, 1.0 and 5.4 with target, distractor and landmark anchors [A1, A2, A4, A14] (p. 9, Table 10, p. 27, Fig. 7b, p. 10).
- C16: Each Atari run takes about 4 h 15 min on average on one V100 [A11] (Sec. 3, p. 6).
- C17: In early experiments, mean pooling of the encoder's output tokens worked best among max, mean, first-token and sum aggregation [] (App. C, p. 20).

## Implementation notes

- ORBiT: 3 transformer blocks, embedding dimension 128, 8 heads, width and height not included in the features (source: Table 2, p. 21).
- Atari PPO: lr 2.5e-4, 1e7 steps, 10 envs, batch 1,280, minibatch 320, 4 epochs, GAE λ 0.95, γ 0.99, value coef 0.5, entropy 0.01, clip 0.1, value clipping, max grad norm 0.5, seeds 0-4 (source: Table 4, p. 21).
- Foosball PPO: lr 3e-4, 6e7 max epochs, 4,096 envs, batch and minibatch 4,096, 1 epoch, value coef 4.0, entropy 1e-4, clip 0.2, max grad norm 1.0; rewards: distance-to-ball shaping, win 150, loss, stagnation and termination penalties 150 (source: Table 3, Table 4, p. 21).
- Atari environment: ALE 0.8.1, Gymnasium 0.29.1, v5, frameskip 4, sticky actions 0.25, minimal action set, 10 evaluation episodes; "Buffer Window Size 4", which the paper doesn't explain (likely the number of past object states kept, as in OCAtari) (source: Table 1, p. 20; the reading of the buffer is an assumption).
- Velocity is the frame difference of positions (source: Sec. 2, p. 3).
- Foosball object token: 11-d, one-hot type (5), x, y, vx, vy, rotation and rotational velocity about Y (source: App. F, p. 22).
- Tag dynamics: unicycles with throttle and steering in [−1, 1], speed clipped to [0, 0.5], target and distractors driven by smoothed random noise that steers away from the player; arena side 10, catch radius 1, 500-step episodes, 3M training steps, 10 envs (source: App. G, p. 23).
- How objects that leave the screen or disappear are tokenized (masked, dropped, or zero-filled) is not stated; for OCAtari-style fixed slots, dropping inactive objects from the set is a guess (source: assumption).
- Code: anonymous.4open.science/r/ORBiT_TMLR, to be released after acceptance (source: App. D, p. 20).

## Limitations / open questions

- 2D only; one hand-picked anchor per environment; a perfect object extractor is assumed, and the effect of detection quality is untested; features are only positions and velocities (Sec. 4, p. 11).
- Every Atari or Foosball result tests translation invariance only; rotation is exercised only in Tag (App. B, p. 20).
- No explicit scene-graph or GNN baseline is evaluated (p. 5).
- Validation on physical hardware with imperfect detection is future work (p. 9, p. 11).
- The authors expect the advantage to grow with how often a task reuses relative configurations and to shrink where the world frame is itself the task structure (Sec. 4, p. 10); untested.
- Open question (reader): the HackAtari tests are perturbations of the training game, not unseen games; does a relative object frame help a single policy transfer across games of one genre, where the type vocabulary (A6) and the anchor's identity change?
- Open question (reader): for a frame-based pipeline, ORCA needs a detector plus a per-game anchor rule; can the anchor be found from pixels without game knowledge (e.g. the object the actions move)?
- Open question (reader): MsPacman set_level_2 (C7) breaks ORBiT, both PPObj variants and PPO^CNN, so ORCA doesn't cover layout changes that alter which relative configurations matter.

## Related

- uses [[vaswani2017attention]]: ORBiT is a Transformer encoder without positional encodings over object tokens.
- uses [[schulman2017proximal]]: all agents train with PPO (CleanRL implementation).
- related [[kurin2021cage]]: also a transformer over an unordered token set (limbs), without structural priors.
- related [[gupta2022metamorph]]: one PPO transformer over per-entity tokens, transferring across morphologies.
- related [[kirillov2023segment]]: the paper names SAM3, a successor of SAM, as a possible upstream object detector.
- related [[lee2022multigame]]: multi-game Atari transformer from pixels, the cross-game setting ORCA doesn't test.
- related [[ravi2024sam]]: video segmentation that tracks object identities across frames, a candidate source of the per-step object set A1 assumes (velocities would need differencing); the paper itself names only SAM3.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
