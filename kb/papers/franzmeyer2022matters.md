---
id: franzmeyer2022matters
title: "Learn what matters: cross-domain imitation learning with task-relevant embeddings"
authors: [Tim Franzmeyer, Philip H. S. Torr, João F. Henriques]
year: 2022
venue: NeurIPS 2022
url: https://arxiv.org/abs/2209.12093
arxiv: 2209.12093
doi:
pdf_sha256: 2c76d4cca800e721b3969fb70781beb89410eb9a3fced1e52146eeef735c7190
tags: [imitation-learning, cross-domain, cross-embodiment, adversarial-imitation, mutual-information, state-representation, imitation-from-observation, unpaired-alignment, continuous-control]
status: read
read: 2026-11-22
---

# Learn what matters: cross-domain imitation learning with task-relevant embeddings

## TL;DR

UDIL picks the expert-state dimensions that best tell expert transitions from shuffled ones (mutual
information plus an elbow), then trains a learner encoder, policy and discriminator adversarially to
match that embedding. It beats XIRL on X-MAGICAL and GWIL on MuJoCo embodiment pairs, without proxy tasks.

## Problem

Cross-domain imitation learns a policy in the learner's domain from state-only demonstrations in
a different domain (another embodiment or environment). Earlier methods assume nearly identical
domains, need demonstrations from several experts, or need proxy tasks demonstrated in both
domains to learn the mapping; GWIL avoids proxies but is limited to short demonstrations and
Euclidean spaces (Sec. 1, pp. 1-2, Sec. 2, p. 3). The authors also note the problem is
under-defined: it isn't specified whether a humanoid imitating a cheetah should crawl or walk
(p. 2).

## Method

- **Setting.** Domains are (S, A, P, ζ); the expert MDP has a reward, the learner's has none;
  demonstrations are state trajectories only (Sec. 3, p. 3).
- **Cross-domain GAIfO.** Map learner states into the expert space with g and minimize
  −H(π_L) + D_JS between the learner's mapped state-transition distribution and the expert's
  (Eq. 3, Sec. 4.1, p. 4), building on GAIfO (Eq. 1, p. 3) and Kim et al.'s result that an
  injective reduction of the learner MDP to the expert's preserves optimality (p. 4).
- **Task-relevant embedding.** An encoder f: S_E → Z (|Z| ≪ |S_E|) keeps only the dimensions the
  reward depends on, so the IRL step can be written on z (Eq. 4, p. 5, App. 8.1.1, p. 14), and g
  only has to map into Z (Eq. 5, p. 5).
- **Finding f.** Pair each expert transition (s, s′) with a pseudo-random transition made of
  two independently sampled expert states, and choose f to maximize I((f(S), f(S′)); Y), Y the
  expert/random label (Eq. 6, Sec. 4.3, p. 5). In practice the MI of each state dimension with
  the label is estimated separately (Ross's estimator), dimensions are ranked, the cumulative MI
  curve's elbow is found (Satopaa's kneedle), and f keeps the dimensions up to the elbow (Sec. 5.1,
  p. 6, App. 8.3, pp. 15-16, Fig. 7, p. 15).
- **Time invariance.** f and g act on each state separately while the discriminator sees
  transitions; this asymmetry rules out degenerate g (Sec. 4.4, p. 5, App. 8.1.2, pp. 14-15).
- **UDIL objective.** min over g, π_L, max over discriminator D_θ of the GAN loss on (g(s_L), g(s′_L))
  vs (z, z′) (Eq. 7, Sec. 4.5, p. 5). g is trained by backpropagating the negative discriminator
  loss; π_L by RL with DAC (Sec. 5.2, pp. 7-8). In X-MAGICAL the authors instead use XIRL's
  simplified framework: reward = distance to the mean terminal state, here in the embedding
  (Sec. 5.1, pp. 6-7).
- **Experiments.** X-MAGICAL (Gripper and Longstick learners, demos of one other agent) against a
  single-demonstrator XIRL; MuJoCo Hopper, Walker, HalfCheetah pairs (fixed-length episodes)
  against GWIL; evaluation by the expert's reward component that transfers (distance travelled
  forward in MuJoCo) (Sec. 5, pp. 5-8).

## Assumptions

- A1: The expert data are state-only trajectories from one expert agent in one domain; there are no expert actions, no proxy tasks and no paired or multi-expert data (Sec. 3, p. 3, Sec. 5, p. 6).
- A2: States are low-dimensional vectors in which task-relevant quantities (positions, angles, velocities) occupy separate dimensions; pixels are not used (Sec. 5.1, p. 6, App. 8.3.1, p. 16, Sec. 6, p. 9).
- A3: The expert's reward depends on a sparse subset of the state dimensions, so truncating to them loses nothing the reward uses (App. 8.1.1, p. 14, Sec. 4.2, p. 4).
- A4: Task-relevant information shows in individual state dimensions: MI is estimated per dimension and summed, ignoring dependencies between dimensions (Sec. 5.1, p. 6, App. 8.3, p. 15).
- A5: Expert transitions are distinguishable from random pairs of expert states: the expert's policy doesn't induce a uniform distribution over transitions, and the environment is fully observable (Sec. 6, p. 9).
- A6: The learner's states can be mapped onto the expert's task-relevant embedding by an affine map (Sec. 5.2, p. 7, App. 8.3.2, p. 17).
- A7: Learner and expert perform the same task, so a mapping reducing the learner MDP to the expert's exists (reader's inference, Sec. 4.1, p. 4).
- A8: The learner domain has no reward; success is measured by a reward component shared by both domains (blocks' distance to the zone, distance travelled forward) (Sec. 5, p. 6, Sec. 5.2, p. 8).
- A9: The demonstrations come from experts trained on the ground-truth reward (Sec. 5.1, p. 6, App. 8.3.2, p. 17).
- A10: MuJoCo episodes have a fixed length (seals), so the learner isn't implicitly rewarded for longer episodes (Sec. 5.2, p. 7).
- A11: In X-MAGICAL, all agents act in the same environment with the same objects and goal, and only the embodiment changes (reader's inference, Sec. 5.1, p. 6).
- A12: The embedding f is fixed before adversarial training; only g, π_L and D are trained jointly (Sec. 4.2-4.5, pp. 4-5, Fig. 1, p. 2).
- A13: In the X-MAGICAL runs the learner's reward is the negative distance to the mean expert terminal state in the embedding, so the task is defined by a goal state (Sec. 5.1, pp. 6-7).

## Claims

- C1: On X-MAGICAL, UDIL consistently outperforms the single-demonstrator XIRL for Gripper from Longstick and Longstick from Gripper, close to the maximum score in both cases [A1, A2, A3, A4, A5, A8, A9, A11, A13] (Sec. 5.1, pp. 6-7, Fig. 2, p. 6).
- C2: UDIL also outperforms XIRL on every other tested pairing, Gripper or Longstick learners from Shortstick, Mediumstick or each other [A1, A2, A3, A4, A5, A8, A9, A11, A13] (App. 8.3.1, p. 16, Fig. 8, p. 16).
- C3: From demonstrations alone, without rewards, the MI ranking picks out task-relevant dimensions: in X-MAGICAL the 9 dimensions left of the elbow are the blocks' x and y positions and their distances to the target zone; in MuJoCo the torso's x position ranks highest, then its height, then the main joint angles, with sub-component velocities lowest [A1, A2, A4, A5] (Sec. 5.1, p. 6, Sec. 5.2, p. 7, App. 8.3, p. 16, Fig. 7, p. 15).
- C4: On six MuJoCo learner-expert pairs over six seeds, UDIL learns meaningful policies; Hopper and Walker cover about 50% of the distance they reach with ground-truth rewards, HalfCheetah about 13% of the expert's [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12] (Sec. 5.2, p. 8, Fig. 4, p. 8).
- C5: UDIL consistently outperforms GWIL on all six pairs; GWIL's results were highly stochastic across hyperparameter settings, which the authors attribute to the Gromov-Wasserstein distance [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12] (Sec. 5.2, p. 8, Fig. 4, p. 8, App. 8.3.2, p. 17).
- C6: The learners find locomotion strategies different from the experts' (qualitative: a hopper that swims or propels itself with its foot), which the authors hypothesize the task-relevant embedding enables [A1, A2, A3, A4, A5, A6, A7, A9, A10, A12] (Sec. 5.2, p. 8, Fig. 3, p. 7).
- C7: Growing the embedding (d ∈ {3, 6, all}, all meaning no reduction) generally lowers mean performance and robustness for Hopper and HalfCheetah; d = 3 yields lying-down, propelling strategies, larger d strategies closer to the demonstrations, so d trades expert-mimicry against finding new strategies [A1, A2, A5, A6, A7, A8, A9, A10] (Sec. 5.3, pp. 8-9, Fig. 3, p. 7, Fig. 5, p. 9).
- C8: The time-invariance constraint is essential: with the discriminator input reduced from (s, s′) to s, performance drops in both Hopper and HalfCheetah ablations [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12] (Sec. 5.3, p. 9, Fig. 5, p. 9).
- C9: In a 2D example, an affine g without time invariance has a trivial solution that ignores its input and still matches the expert's transition ordering, while a time-invariant affine g can't express it and must negate the input [] (App. 8.1.2, pp. 14-15, Eq. 9-10, p. 14).
- C10: Maximizing I(Z; Y) on single states would give non-informative representations, because random transitions are built from expert states; only transitions separate the two sets [A5] (Sec. 4.3, p. 5).
- C11: With a single expert trajectory, UDIL still outperforms GWIL by a large margin; HalfCheetah from Hopper scored higher with one trajectory than with 20, which the authors consider an outlier since no other pair showed it [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12] (Sec. 5.3, p. 9, App. 8.3.3, p. 18, Fig. 10, p. 18).
- C12: An oracle baseline that omits g and feeds the learner's matching state dimensions (order unknown) directly stays near zero reward on Hopper from HalfCheetah and HalfCheetah from Hopper, far below UDIL (curves read from the figure; the text gives no verdict) [A1, A2, A3, A4, A5, A7, A8, A9, A10] (App. 8.3.3, p. 18, Fig. 11, p. 18).
- C13: On X-MAGICAL, the adversarial (Eq. 7) version of UDIL also outperforms XIRL in both pairings, but is inconsistent against the simplified version, better once and worse once [A1, A2, A3, A4, A5, A8, A9, A11, A12, A13] (App. 8.3.1, pp. 16-17, Fig. 9, p. 17).

## Implementation notes

- Demonstrations: 20 expert trajectories per MuJoCo agent; the X-MAGICAL demonstrations are XIRL's dataset (source: Sec. 5.1, p. 6, App. 8.3.2, p. 17).
- Expert transitions use a frameskip of 15; as many pseudo-random transitions are drawn as there are expert ones (source: App. 8.3, p. 15).
- Per-dimension MI: Ross (2014) estimator for discrete-continuous MI; elbow: Satopaa et al.'s kneedle implementation, dimensions up to and including the elbow kept (source: Sec. 5.1, p. 6, App. 8.3, pp. 15-16).
- Learner encoder g: one linear layer, sigmoid-scaled weights with magnitude at most 5, optional bias; lr 1e-3 (Hopper, HalfCheetah) or 1e-4 (Walker); bias only for HalfCheetah; "train every n-enc steps" printed as 0.01, 0.01, 0.1, a value the paper doesn't explain (source: Table 1, p. 17, App. 8.3.2, p. 17).
- RL: reimplemented DAC with the original hyperparameters, Stable-Baselines3 agents, seals fixed-length Gym environments; 2.5M environment steps in the plots (source: App. 8.3.2, p. 17, Fig. 4, p. 8).
- X-MAGICAL: authors' environment and XIRL code on state vectors, training time doubled vs XIRL's defaults (source: App. 8.3.1, p. 16).
- Seeds: the main text says 0-6 and the appendix 0-5 (six seeds); the six-seed reading is the appendix's (source: Sec. 5, p. 6, App. 8.3.2, p. 17).
- Whether g is learned in the simplified X-MAGICAL setting is not stated; since all agents share one state layout, applying f to both agents' states is the likely reading (source: assumption).
- No code released with the paper; the checklist says it will be published (source: p. 13).

## Limitations / open questions

- Cross-domain imitation is under-defined, so incorrect policies can be learned; the MI objective can degenerate when the expert's policy makes transitions uniform or under partial observability; choosing d may be hard in complex domains; high-dimensional (pixel) observations need further work (Sec. 6, p. 9).
- GWIL is the only baseline with matching assumptions, and the authors found it highly stochastic, with no positive result in some runs (Sec. 5.2, p. 8, App. 8.3.2, p. 17).
- Open question (reader): with so unstable a baseline, how much of the MuJoCo margin reflects UDIL's strength rather than GWIL's failure?
- Open question (reader): the dimension-selection f needs semantically separated state dimensions (A2, A4); from frames, what would play this role, e.g. an object-centric or learned disentangled state?
- Open question (reader): g is affine (A6); the MuJoCo pairs share torso position semantics, so how far this goes for domains whose state spaces don't align linearly is untested.
- Open question (reader): the embedding is chosen per expert task; it isn't a shared embedding across many games, and the learner still trains per target domain.

## Related

- uses [[zakka2021xirl]]: XIRL is the X-MAGICAL baseline and supplies the demonstrations and goal-distance reward framework.
- related [[gupta2017invariant]]: learns a shared space from proxy-task pairs, the supervision UDIL removes.
- related [[ganin2016domain]]: the same adversarial idea of making mapped features indistinguishable across domains.
- related [[zhang2020correspondence]]: also aligns two domains from unpaired data, via dynamics cycle-consistency.
- related [[liu2018context]]: imitation from observation across contexts (viewpoint), by translation instead of an embedding.
- related [[serrano2024crossdomain]]: survey of cross-domain transfer methods; this paper fits its imitation-learning branch.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
