---
id: liu2018context
title: "Imitation from Observation: Learning to Imitate Behaviors from Raw Video via Context Translation"
authors: [YuXuan Liu, Abhishek Gupta, Pieter Abbeel, Sergey Levine]
year: 2018
venue: ICRA 2018 conference paper
url: https://arxiv.org/abs/1707.03374
arxiv: 1707.03374
doi:
pdf_sha256: 09698e7ff3a0d36c878065d2f14697d3688c526cd1c48ba267c71200a61a8b79
tags: [imitation-learning, imitation-from-observation, reward-learning, video, context-translation, viewpoint-invariance, reinforcement-learning, robotics]
status: read
read: 2026-11-21
---

# Imitation from Observation: Learning to Imitate Behaviors from Raw Video via Context Translation

<!-- Read from arXiv v2 (2018-06-18), 8 PDF pages, all read: main text pp. 1-8, references p. 8. Figure values were read off zoomed renders of Fig. 4, 6 and 8 and are marked as such. -->

## TL;DR

Translates a demo video from one context (viewpoint, background, object layout) into another,
given only that context's first frame, and rewards RL for tracking the translated features; it does
well on 4 simulated tasks, where TPIL and GAIL get 0%, and on 4 real Sawyer tasks (66-100%).

## Problem

"Imitation from observation": learning a policy from videos of a behaviour only, with no actions
and with each demonstration recorded in a different context, e.g. another viewpoint, background,
or object positions and types (Sec. I, pp. 1-2). Behavioural cloning needs actions; IRL can in
principle use observations but has been hard to run on images (Sec. II, p. 2). Third-person
imitation learning (TPIL, Stadie et al.) needs expert and non-expert policies and handles only two
contexts; pretrained visual features (Sermanet et al.) give invariance but no explicit translation
(Sec. II, p. 2). A raw pixel distance to a demonstration is a poorly shaped reward (Sec. III, p. 3).

## Method

- Setting (Sec. III, p. 3): demos D_k = [o_0^k … o_T^k] come from a POMDP with observation,
  dynamics and expert policy all conditioned on a context ω drawn i.i.d. from p(ω); the learner's
  context ω_l is from the same distribution. Systematic domain shift (e.g. a different embodiment,
  so ω_l is not from p(ω)) is explicitly out of scope.
- Context is never labelled: the only assumption is that a demo's first frame o_0^k implicitly
  carries its context (Sec. IV, p. 3).
- Translation model M(o_t^i, o_0^j) (Fig. 2, Sec. IV, pp. 3-4): Enc1 encodes the source frame (z1),
  Enc2 the target context's first frame (z2), a translator z3 = T(z1, z2), and Dec(z3) predicts
  o_t^j; skip connections from Enc2 to Dec. F(o_t^i, o_0^j) = z3 is the feature extractor.
  Trained on random pairs of demos (D_i, D_j) that are assumed time-aligned, so frame t of one is
  the target for frame t of the other.
- Loss (Sec. IV, p. 4): L = Σ_(i,j) (L_trans + λ1 L_rec + λ2 L_align), with L_trans = ||Dec(z3) - o_t^j||²,
  an autoencoder loss L_rec = ||Dec(Enc1(o_t^j)) - o_t^j||², and L_align = ||z3 - Enc1(o_t^j)||², which
  puts translated features and the encoder's features of real target frames in one space.
- Reward (Sec. V-A, p. 4): R_feat(o_t^l) = -||Enc1(o_t^l) - (1/n) Σ_i F(o_t^i, o_0^l)||², tracking
  the mean of all n demos translated into the learner's context; plus a weak pixel term
  R_img = -||o_t^l - (1/n) Σ_i M(o_t^i, o_0^l)||², combined as R = R_feat + w_rec R_img, w_rec a small
  tuned constant. R_img guards against Enc1 seeing out-of-distribution frames from poor policies.
- RL (Sec. V-B, pp. 4-5): TRPO in simulation; on the real robot, GPS's trajectory-centric local
  policy optimisation with z3 in the state, a squared-distance cost, and no image term. Simulated
  striking and real pushing weight later time steps more (quadratic ramp).
- Networks (Sec. VI-A, p. 5): simulation encoders are four 5×5 stride-2 convs (64-512 filters)
  plus two FC 1024 layers, LeakyReLU 0.2; T has one hidden layer of 1024 on [z1, z2]. Real-world
  models are smaller (feature size 100), with dropout 0.5 and Enc1, Enc2 tied. Adam, lr 1e-4.
  Videos: 3000 (reach), 4500 (sim push), 894 (sweep), 3500 (strike), 135 (real push), 85 (real
  paper sweep), 100 (almond sweep), 60 (ladling); 36×64 pixels for simulated sweep, 48×48 otherwise.
- Simulated tasks (MuJoCo, Fig. 3, Sec. VI, p. 5): reach (colour/appearance variation), push
  (positions, distractors), sweep (viewpoint), strike (target position); demos are rollouts of a
  TRPO expert trained on a hand-written reward in random contexts. Metric: final object-to-goal
  distance divided by the initial one, thresholded to success; 10 random environment conditions,
  100 RL iterations of 12,500 samples each (Sec. VI-B, p. 5).
- Real tasks (Sec. VI-D, pp. 6-7): a human demonstrates with the same tool the 7-DoF Sawyer later
  uses; videos are cropped around task-relevant areas to avoid embodiment shift.

## Claims

- C1: Learning from demos in random contexts, the method succeeds on all four simulated tasks, close to a TRPO oracle on the true reward (figure reading: ours about 0.85 reach, 0.85 push, 1.0 sweep, 0.95 strike vs oracle about 0.9, 0.9, 1.0, 1.0, with error bars overlapping) (Fig. 4, Sec. VI-B, pp. 5-6).
- C2: TPIL and GAIL reach 0% success on all four simulated tasks; pretrained Inception-v3 features do reasonably only on sweep (about 0.7 read from the figure, wide error bar) and are about 0.25 on reach, 0 on push and about 0.05 on strike (Fig. 4, Sec. VI-B, p. 6).
- C3: In ablations on simulated reach and push, removing any of L_trans, L_rec, L_rec with L_align, R_feat or R_img lowers success; the full method is best on both tasks (figure reading: reach 0.85 vs 0.45-0.75, push 0.85 vs 0.6 without R_feat and 0.4 without R_img; the error bars overlap and the loss ablations were not run on push) (Fig. 6, Sec. VI-C, p. 6).
- C4: Without L_align (and L_rec) tying the translated features to the encoder's own features, the feature-tracking reward is ineffective, since learner and expert features need not lie in the same space (Sec. IV, p. 4, Fig. 6).
- C5: On real Sawyer pushing with GPS, trained only from human videos from different viewpoints, the method succeeds about 0.96 of the time (figure reading) and beats pretrained-feature (about 0.25) and kinesthetic joint-angle (about 0) rewards (Fig. 8, Sec. VI-D1, p. 7).
- C6: On real crumpled-paper sweeping, the method and the joint-angle baseline both work well (about 1.0 in the figure) while pretrained features fail (about 0); on almond sweeping the method scores highest (about 0.75 vs 0.62 and 0.67, read from the figure, with overlapping error bars) (Fig. 8, Sec. VI-D2, p. 7).
- C7: On real almond ladling the method reaches 66% success, while the pretrained-feature and joint-angle baselines stay at about 0.17 and 0.11 (figure reading); the authors explain the joint-angle failure by its interpolating between the start pose and the turned-spoon end pose (Fig. 8, Sec. VI-D3, p. 7).
- C8: The translation model generalises to the learner's context not seen in training, producing plausible translated videos for holdout demos (qualitative) (Fig. 5, p. 6, Sec. VII, pp. 7-8).

## Implementation notes

- Values of λ1, λ2 and w_rec, the RL iterations on the real robot, and the success thresholds are not given (source: Sec. IV, Sec. V-A, Sec. VI-B; the authors point to their project site, not read).
- Error bars in Fig. 4, 6 and 8 are not defined (std, stderr or CI) (source: Fig. 4, Fig. 6, Fig. 8 captions).
- Pretrained-feature baseline: the best-performing Inception-v3 layer, distance to features averaged over training demos; for GPS its 221952-d features are PCA'd to 100 (source: Sec. VI-B, p. 6; Sec. VI-D1, p. 7).
- GAIL baseline uses a convolutional discriminator on images following Li et al. [39] (source: Sec. VI-B, p. 6).
- Pairs (D_i, D_j) are drawn uniformly from the training set each step (source: assumption; the paper says "chosen randomly", Sec. IV, p. 4).
- Survey carry-over (reader's note): context is inferred from one target frame, which is like conditioning an encoder on a per-game "context frame"; the feature space is shared only because L_align ties it to Enc1 (source: assumption, Sec. IV, p. 4).

## Limitations / open questions

- Demos must be time-aligned; the authors suggest iterative time alignment as future work (Sec. IV, p. 3).
- Contexts of learner and demos must come from one distribution; embodiment shift (human vs robot arm) is left to future work, e.g. domain adaptation, and avoided by same-tool demos and cropping (Sec. III, p. 3; Sec. VI-D, pp. 6-7).
- Needs many demos (60 to 4500 videos per task) and demos from multiple contexts, trained from scratch per task; the authors suggest multi-task models and higher-level pretrained representations (Sec. VII, p. 8).
- The feature encoder only sees expert data and is biased toward expert-like features on poor policies' frames, hence R_img (Sec. V-A, p. 4).
- Open question: training is per task and supervised by aligned demo pairs, not unpaired; would the translation-plus-alignment idea scale to many games with no shared timeline, e.g. with TCC-style soft alignment instead of index alignment?
- Open question: results come from 10 environment conditions per simulated task with undefined error bars; how much of the gap to the ablations survives more seeds?

## Related

- related [[zakka2021xirl]]: also rewards RL by embedding distance to action-free demo videos, but learns the embedding with TCC across embodiments instead of translating contexts.
- related [[dwibedi2019cycle]]: TCC aligns videos without the time-aligned pairs this method assumes.
- related [[zhang2020correspondence]]: learns cross-domain maps from unpaired data, where this method needs aligned demo pairs in a shared context distribution.
- related [[serrano2024crossdomain]]: survey of cross-domain RL transfer, the broader frame for imitation across contexts.
- related [[gupta2017invariant]]: also rewards tracking a shared feature space learned from time-aligned pairs, across two robots instead of contexts.
- related [[baker2022video]]: another way to learn behaviour from action-free video, via an inverse dynamics model.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
