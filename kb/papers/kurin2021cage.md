---
id: kurin2021cage
title: "My Body is a Cage: the Role of Morphology in Graph-Based Incompatible Control"
authors: [Vitaly Kurin, Maximilian Igl, Tim Rocktäschel, Wendelin Böhmer, Shimon Whiteson]
year: 2021
venue: ICLR 2021
url: https://arxiv.org/abs/2010.01856
arxiv: 2010.01856
doi:
pdf_sha256: c5c9017d87758d69a83045c66195e4dad5109d0409f50407b0549d4d60034a13
tags: [reinforcement-learning, multi-task, transformer, graph-neural-network, morphology, incompatible-spaces, continuous-control, td3]
status: read
read: 2026-11-21
---

# My Body is a Cage: the Role of Morphology in Graph-Based Incompatible Control

<!-- Read from arXiv v2 (2021-04-14, ICLR 2021 camera-ready), 20 PDF pages, all read: main text
pp. 1-9, references pp. 10-14 (skimmed), App. A (hyperparameters) p. 15, Table 2 p. 16, App. B
p. 17, App. C pp. 18-19, App. D-E pp. 19-20. Code: https://github.com/yobibyte/amorpheus (not checked).
The method is AMORPHEUS; the id uses the title's most distinctive word. -->

## TL;DR

Ablations show GNN policies for incompatible multi-task control (SMP, NerveNet) gain nothing from the
robot's morphology graph; AMORPHEUS, a transformer over per-limb tokens with no graph, no positional
encoding and no mask, beats SMP on all its benchmarks and shows gait-coupled attention.

## Problem

Multi-task RL (MTRL) mostly assumes compatible tasks, with the same state and action dimensions;
incompatible ones (robots with different limbs) cannot use an MLP (Sec. 1, p. 1; Sec. 2.1, pp. 2-3).
GNNs handle graphs of any size and let practitioners inject structure; NerveNet and Shared Modular
Policies (SMP) build the graph from the physical morphology (limb nodes, edges between connected
limbs), but the benefit of restricting computation this way had not been validated (Sec. 1, p. 2).
The paper tests that assumption and proposes a method without it.

## Method

- **Setting** (Sec. 2.1-2.2, pp. 2-3): each limb is a node whose label holds its features (limb
  type, coordinates, angular velocity); each joint's torque is a node output. MTRL return is the
  average return over the N tasks. Only the dimension mismatch is addressed, not semantics (p. 3).
- **Baselines** (Sec. 2.2, pp. 3-4): NerveNet (MLP encoder per node, edge MLP and GRU node update
  over message passing, MLP decoder giving Gaussian means with a state-independent std) and SMP
  (tree-only, bottom-up then top-down passes, concatenated rather than permutation-invariant
  messages, per-node actor outputs and per-node critic losses; Huang et al. report the root choice
  can change performance by 15%).
- **Morphology ablations** (Sec. 3, pp. 4-5; App. B, p. 17, Fig. 9): replace the morphology graph
  by a star (all limbs to the torso), a line (structure unrelated to the body), or for NerveNet
  also a fully connected graph. Three seeds, average undiscounted MTRL return and standard error.
- **AMORPHEUS** (Sec. 4, pp. 5-6; Fig. 2): encode-process-decode; a linear encoder per node, a
  transformer (a GNN on the fully connected graph with attention aggregation, Eq. 1) and an MLP
  decoder per node; actor and critic share the architecture but no weights; one action per non-torso node,
  and per-node critic outputs with per-node value losses as in SMP. Node observation as in SMP: limb
  type (e.g. hip, shoulder), x position relative to the torso, positional and rotational
  velocities, rotations, joint angle and its range normalised to [0, 1]. The input features are
  concatenated into the decoder (called a residual connection) so nodes keep their own features.
  The PyTorch tutorial transformer was used with masking and positional encoding removed (App. A,
  p. 15), so tokens are distinguished only by their features. Trained with TD3 in the SMP codebase
  (Sec. 4.1, p. 6).
- **Benchmarks** (Sec. 4.1, p. 6; Table 2, p. 16): SMP's Walker++ (4 train, 2 zero-shot test
  morphologies), Humanoid++ (6/2), Cheetah++ (12/3), Hopper++ (3/0) and their unions
  (Walker-Humanoid, Walker-Humanoid-Hopper, Cheetah-Walker-Humanoid, Cheetah-Walker-Humanoid-Hopper;
  Fig. 3, p. 7); flipped environments are omitted. Also
  NerveNet's Walkers (Ostrich, HalfCheetah, FullCheetah, Hopper, HalfHumanoid), with its 3800 return
  limit removed (App. A, p. 15). 10^7 total environment steps per run (Fig. 3 and Fig. 4 axes).

## Claims

- C1: In SMP on Walker++ and Humanoid++, a star graph performs on par with the true morphology graph and a line graph performs worse, so SMP's performance is not contingent on morphology information (Fig. 1, Sec. 3, pp. 4-5).
- C2: In NerveNet on Walkers, morphology, line, star and fully connected graphs perform similarly, star marginally best; the fully connected variant learns more slowly at first, and unlike SMP the line graph is on par with morphology (Fig. 1, Sec. 3, p. 5).
- C3: AMORPHEUS reaches higher average return than SMP on all eight SMP benchmarks over 10^7 steps, by a wide margin on Walker++, Hopper++ and the two Cheetah unions, moderately on Walker-Humanoid-Hopper++, and narrowly, with overlapping standard-error bands, on Cheetah++, Humanoid++ and Walker-Humanoid++ (where SMP's mean briefly crosses above around 0.9×10^7 steps); the authors call this consistently outperforming, supporting the hypothesis that explicit graph structure is not needed (Fig. 3, Sec. 4.1, pp. 6-7).
- C4: On the union benchmarks, where SMP is unstable, AMORPHEUS learns without trouble: about 2600 vs about 850 on Cheetah-Walker-Humanoid++ and about 2400 vs about 500 with Hopper added, read from the curves at 10^7 steps (Fig. 3, Sec. 4.1, pp. 6-7).
- C5: On NerveNet's Walkers benchmark of five different morphologies, AMORPHEUS (about 5300) beats SMP (about 3500) and NerveNet (about 2000) at 10^7 steps; NerveNet is not directly comparable (different features and learning algorithm, MLP critic, a separate encoder per task) (Fig. 4, Sec. 4.1, p. 6, p. 8).
- C6: Zero-shot on 7 held-out morphologies (3 seeds, 100 rollouts), AMORPHEUS has the higher mean on 5, e.g. walker-3-main 666 vs 176 and humanoid-2d-8-right-knee 2717 vs 328, while SMP wins cheetah-5-back (3821 vs 3418) and cheetah-6-front (6019 vs 5082); variance is high for both, which the authors call initial results (Table 3, App. D, p. 19).
- C7: AMORPHEUS's attention masks are state dependent: a Walker++ model shows different masks within one Walker-7 rollout, unlike the fixed message passing of NerveNet and SMP (Fig. 5, Sec. 4.2, pp. 7-8).
- C8: First-layer attention cycles with the gait: in a Walker-7 rollout, nodes attend to an upper leg (column-wise mask sum about 3) when it is closer to the ground (normalised angle about 0) (Fig. 6, Sec. 4.2, pp. 7-8).
- C9: Earlier attention layers change more slowly within a rollout than later ones, for three Walker++ models on Walker-7, which the authors read as AMORPHEUS being able to learn a rigid structure if needed (Fig. 13, Sec. 4.2, App. C.2, pp. 7-8, p. 19).
- C10: Early in training the attention masks are spread over the whole graph; later their weights become less uniform (Fig. 10, Fig. 11, Fig. 12, Sec. 4.2, App. C.1, p. 8, p. 18).
- C11: Without the input-to-decoder residual connection, AMORPHEUS still roughly matches the full model and beats SMP on most benchmarks, so the authors cannot attribute the gain to the residual alone; but on Humanoid++ the no-residual variant falls below SMP (about 3700 vs 4500 at 10^7 steps, read from the curves; one seed started improving much later) and on Cheetah++ and Walker-Humanoid++ it is about level with SMP (Fig. 14, App. E, pp. 19-20).
- C12: Unlike SMP, which is limited to the maximum number of children per node seen at initialisation, AMORPHEUS applies to any morphology without connectivity constraints (Sec. 4, p. 6).
- C13: SMP also does better with the smaller learning rate 0.0001 on Walker++, and that rate is used for SMP throughout (Fig. 7, App. A, p. 15).
- C14: Removing NerveNet's 3800 return limit on Walkers slightly lowers NerveNet's performance; the better option is plotted for it (Fig. 8, App. A, p. 15).

## Implementation notes

- AMORPHEUS: learning rate 0.0001, gradient clipping 0.1, LayerNorm (passed to TransformerEncoder), 3 attention layers, 2 heads, attention hidden size 256, encoder output size 128; 3 runs per benchmark (source: Table 1, App. A, p. 15).
- All other hyperparameters (TD3, replay, exploration noise, decoder MLP sizes) follow SMP with two-directional message passing (source: App. A, p. 15, and Huang et al. 2020, not in the KB).
- No positional encoding and no attention mask (source: App. A, p. 15); without a mask, each batch must hold one morphology, so no padding is needed (source: assumption, consistent with the per-morphology collection MetaMorph describes, Gupta et al. 2022 App. B.1).
- Experience is collected sequentially per morphology into a separate replay buffer per morphology, and updates are sequential per task (source: MetaMorph, Gupta et al. 2022, App. B.1; Sec. 6, p. 9 of this paper mentions "sequential task updates").
- Left and right limbs are told apart by the relative x position in the node features, not by structure (source: Sec. 4.1, p. 6).

## Limitations / open questions

- Transformer cost is quadratic in the number of nodes; the authors consider this minor for
  continuous control, where node counts are small, and note transformers parallelise better than
  SMP's level-by-level tree passes (Sec. 6, p. 9).
- The learning algorithm was not studied: averaging gradients instead of sequential task updates,
  or balancing tasks with bandits or PopArt, are named as future directions (Sec. 6, p. 9).
- Zero-shot generalisation is preliminary and high-variance (Table 3, App. D, p. 19).
- All benchmarks are planar MuJoCo agents built from a few archetypes (Sec. 4.1, p. 6); MetaMorph
  (Gupta et al. 2022, Sec. 2) later calls this low-diversity, at most 7 DoF control.
- Open question (reader): the node features (limb type, relative position, joint range) are a
  morphology description in themselves, so "no morphology" means no graph edges, not no body
  information; MetaMorph (Gupta et al. 2022, App. B.1) finds these features weaker than richer descriptors.
- Open question (survey lens): with no positional encoding, the policy is a set function over
  entity tokens with one output per token, the shape a game policy over a variable set of
  entities and per-entity actions would need; whether node features learned from frames could
  replace hand-given limb types is untested here.

## Related

- uses [[vaswani2017attention]]: the transformer encoder at the core, without positional encoding.
- related [[reed2022generalist]]: Gato's Modular RL domain is from the same SMP benchmark family, handled by flat tokenisation instead of per-limb tokens.
- related [[serrano2024crossdomain]]: survey of cross-domain transfer in RL, the frame for this survey batch.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
