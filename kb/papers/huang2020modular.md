---
id: huang2020modular
title: "One Policy to Control Them All: Shared Modular Policies for Agent-Agnostic Control"
authors: [Wenlong Huang, Igor Mordatch, Deepak Pathak]
year: 2020
venue: ICML 2020
url: https://arxiv.org/abs/2007.04976
arxiv: 2007.04976
doi:
pdf_sha256: df1d5ce3e9171ef14921e139620f41437c8d5d1add4637fcc6404b80daba5954
tags: [reinforcement-learning, cross-embodiment, morphology-transfer, modular-policy, message-passing, zero-shot, multi-task, continuous-control]
status: read
read: 2026-11-21
---

# One Policy to Control Them All: Shared Modular Policies for Agent-Agnostic Control

## TL;DR

SMP runs one shared module per limb of a planar MuJoCo agent, passing 32-d messages leaves-to-root
then root-to-leaves; trained with TD3 on up to 18 variants at once, it beats a monolithic multi-task
baseline and the decentralized schemes, and controls held-out variants zero-shot.

## Problem

RL controllers are trained per morphology with tuned hyperparameters; a shared pre-trained
controller is hard because agents differ in limb count, sensors and action dimensions (Sec. 1,
p. 1). Centralized monolithic policies can't exploit modularity or reuse across bodies (p. 2).
Unlike self-assembling agents (Pathak et al. 2019), the module here must work for fixed, given
morphologies and learn one unified policy (p. 2, Sec. 3.4, p. 5).

## Method

- **Morphology as a graph** (Sec. 2.1, p. 3): K limbs are nodes, joints are edges; assumed a
  tree rooted at the torso (any limb can be root, Sec. 5.4). Each limb has an actuator moving it
  relative to its parent.
- **Shared module** (Sec. 2.2-2.3, p. 3, Eq. 1): the same π_θ is instantiated at every limb of
  every agent, takes that limb's local state s_t^k and outputs its torque; θ maximizes the sum
  over the N agents of their discounted whole-body returns. Trained with an off-the-shelf TD3.
- **Message schemes** (Sec. 3.2-3.3, p. 4): bottom-up (Eq. 2: action and parent message from
  local state plus aggregated child messages), top-down (Eq. 3: from local state plus the
  parent's message), and **both-way** (Eq. 4): a bottom-up net π_θ1(s^k, f(child messages))
  emits only the message to the parent, leaf to root; then a top-down net
  π_θ2(m^{k→p(k)}, m^{p(k)→k}) emits the action and the child messages, root to leaf. The root
  can thus act as an emergent centralized controller. Messages are 32-d (Sec. 3.1, p. 4).
- **Variable child count** (Sec. 3.4, p. 5): the top-down net always emits max_n K_n messages
  (children use the first ones, the rest are ignored); the bottom-up net takes max_n K_n child
  messages, zero-padded. A recurrent aggregator is mentioned as an alternative, not used.
- **Cycles** (pp. 4-5): proposed only, repeating both-way passes until messages converge as in
  loopy belief propagation; not evaluated.
- **Benchmarks** (Sec. 4, pp. 5-6): Walker2D, Humanoid (constrained to 2D), Hopper and
  HalfCheetah from Gym; variants are connected limb subsets containing the torso, minus
  infeasible ones. Collections: walker++ (12), humanoid++ (8), cheetah++ (15), walker-hopper++
  (12+3), walker-hopper-humanoid++ (12+3+8); 20% held out. Cheetah isn't mixed with others (its
  integrator makes joint training unstable).
- **Baseline** (Sec. 5.1, pp. 6-7): a monolithic 4-layer TD3 policy over the whole agent's
  joint-coordinate state plus a task descriptor (limb count, one-hot env id), states and actions
  zero-padded to the largest env, following Chen et al. 2018.
- Pseudo-code: Alg. 1-2 (App. A.5, p. 12): one episode per environment per round, then a TD3
  update per environment's buffer in turn; nodes in topological order.

## Assumptions

- A1: The agents are planar MuJoCo locomotors: Walker2D, Hopper, HalfCheetah and a Humanoid constrained to 2D, and their limb-subset variants (Sec. 2.1, p. 3, Sec. 4, p. 5).
- A2: Each morphology is given to the policy as a connected tree whose nodes are limbs and whose edges are joints, rooted at one limb (the torso by default) (Sec. 2.1, p. 3, Sec. 3.1, p. 4, Sec. 5.4, p. 7).
- A3: Each limb other than the torso has one actuator moving it relative to its parent, and the action is one torque per limb (Sec. 2.1, p. 3, Sec. 4, p. 6).
- A4: Every limb's local state has the same features with the same meaning in all agents: global position, positional and rotational velocity, exponential-map rotation, and joint position and range normalised to [0, 1] (Sec. 4, p. 6).
- A5: All agents share one goal, walking forward, with Gym's default reward per category and walker++'s reward (x displacement, alive bonus, squared-action penalty) for mixed collections (Sec. 2, p. 3, App. A.2, p. 11).
- A6: Morphologies are static; the agent can't change its shape (p. 5).
- A7: The number of messages per node is fixed at the largest child count across the training agents (Sec. 3.4, p. 5).
- A8: Held-out morphologies are connected limb subsets containing the torso, from the same agent families as the training ones, with locomotion-infeasible variants removed, and 20% of each collection held out (Sec. 4, pp. 5-6).
- A9: One policy is trained with off-the-shelf TD3 on all training morphologies in parallel, one environment and replay buffer each, for about 1.2-2e7 steps (Sec. 2.3, p. 3, App. A.2, p. 11, Fig. 3, p. 5).
- A10: Cheetah variants are never trained with the other categories, because their different integrator makes joint training unstable (Sec. 4, pp. 5-6).
- A11: Results are episode returns over 1,000 steps, averaged over four seeds (Sec. 4, p. 5).
- A12: The multi-task baseline is a 4-layer TD3 MLP over the whole agent's joint-coordinate state plus a task descriptor (limb count, one-hot environment id), with states and actions zero-padded to the largest environment (Sec. 5.1, pp. 6-7).
- A13: Every limb's full local state is observed at every step; there is no partial observability or perception from pixels (reader's inference, Sec. 4, p. 6).

## Claims

- C1: Trained jointly on 80% of the variants of each of the five collections, SMP with both-way messages reaches higher mean reward across agents than the monolithic multi-task baseline, which performs poorly on the collections with walker and humanoid agents and is unstable even on the cheetahs [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13] (Sec. 5.1, p. 7, Fig. 3, p. 5).
- C2: Both-way message passing outperforms top-down-only, bottom-up-only and no messages in all five collections; e.g. on walker++ both-way ends near 4500 mean reward against roughly 1800 for bottom-up only [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A13] (Sec. 5.2, p. 7, Fig. 3, p. 5).
- C3: Within walker++ (one policy for ten walker variants), the decentralized schemes learn locomotion to some extent for the four- to seven-limb walkers but nothing meaningful for the three- and two-limb ones, while both-way learns all six shown, including gaits such as hopping for the two-limb walker [A1, A2, A3, A4, A5, A6, A7, A8, A9, A11, A13] (Sec. 5.2, p. 7, Fig. 4, p. 6).
- C4: In Fig. 4 the jointly trained both-way curves end below the single-morphology monolithic reference for Walker-1 to Walker-3 (7, 6, 5 limbs) and above it for Walker-4 to Walker-6 (4, 3, 2 limbs), so "matching" per-agent oracles holds per variant only approximately [A1, A2, A3, A4, A5, A6, A7, A8, A9, A11, A13] (Fig. 4, p. 6, Sec. 1, p. 2).
- C5: Zero-shot on the held-out 20% of variants (2, 2, 3, 3 and 5 agents), both-way reaches the highest reward of the message schemes in every collection, with no fine-tuning; on humanoid++ bottom-up only nearly catches up late in training, and the authors point to failure videos as well as successes [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A13] (Sec. 5.3, p. 7, Fig. 5, p. 7).
- C6: The root can be any limb: on one walker morphology, training reward at 1M steps over four seeds (mean ± standard deviation) is 3709.87 ± 580.87 with the left foot as root, 3215.04 ± 447.82 with the torso, and 3592.70 ± 111.13 for the monolithic baseline; the ranges overlap [A1, A2, A3, A4, A5, A6, A9, A11, A13] (Sec. 5.4, p. 7).
- C7: A one-dimensional t-SNE of the root's message over an episode is flat at the start of training and cyclic after 1M steps for both walker and hopper, matching the alternating gait, and the agent's pose tracks the message; the authors read this as a centralized controller emerging from decentralized modules [A1, A2, A3, A4, A5, A6, A9, A13] (Sec. 6, p. 8, Fig. 6, p. 8).
- C8: In walker, after PCA to one dimension and averaging over an episode, the feet's actions and states correlate more with top-down messages from closer limbs: action correlation about 0.9 at distance 0 falls to about 0.55 (left) and 0.27 (right) at distance 2, and state correlation from about 0.42 and 0.23 to near 0-0.08 [A1, A2, A3, A4, A5, A6, A9, A13] (App. A.3, p. 11, Fig. 7, p. 11).
- C9: Trained on a single standard morphology, both-way SMP performs comparably to a monolithic policy on Walker, Humanoid, Cheetah and Hopper, although the modular problem is harder; in the Walker panel bottom-up only ends above both-way [A1, A2, A3, A4, A5, A6, A9, A11, A13] (App. A.4, p. 11, Fig. 8, p. 12).
- C10: Without messages the limbs act as independent agents and learn nothing meaningful, even on a single morphology [A1, A2, A3, A4, A5, A6, A9, A13] (App. A.4, p. 11, Fig. 8, p. 12).
- C11: Sending each child its own message (the caching trick) works better in practice than sending all children the same message; no numbers are given [A2, A3, A7] (Sec. 3.4, p. 5).

## Implementation notes

- TD3 with Adam, learning rate 4e-4, tau 0.046, exploration noise 0.13, uniformly random actions for the first 10,000 steps; initial positions and velocities randomized each episode (source: App. A.2, p. 11).
- Both modules are 4-layer fully connected nets with ReLU and tanh non-linearities; layer widths, batch size, discount and TD3 policy delay are not given (source: App. A.2, p. 11; widths etc. would come from the authors' code https://github.com/huangwl18/modular-rl, not checked).
- Messages are 32-d and normalized before being passed; the normalization isn't specified (source: App. A.2, p. 11; L2 normalization is an assumption).
- One environment and replay buffer per morphology, 1e6 each, total capped at 1e7 (1e7/n each when n > 10); vectorized envs from OpenAI Baselines and dynamic batching across limbs that are neither ancestor nor descendant (source: App. A.2, p. 11).
- Limb state: global position, positional and rotational velocity, 3D rotation as an exponential map, and joint range as (position, low, high) normalized to [0, 1] (source: Sec. 4, p. 6).
- The torso has no actuator: it keeps a module for message passing and its torque is ignored (source: Sec. 4, p. 6).
- Rewards: Gym's default per single-category collection; multi-category collections use walker++'s (x displacement, alive bonus, squared-action penalty) (source: App. A.2, p. 11).
- Four seeds per experiment, mean and standard error (Sec. 4, p. 5; Sec. 5.4 says standard deviation for its table); episodes of 1,000 steps; training runs about 1.2-2e7 steps (source: Fig. 3, p. 5).

## Limitations / open questions

- All agents are planar MuJoCo locomotors with one goal (walk forward), static morphologies and
  tree-shaped graphs; cycles are only proposed (Sec. 2.1, p. 3, Sec. 3.3, pp. 4-5, Sec. 3.4, p. 5).
- The held-out variants are limb subsets of the same agent families, and cheetah couldn't be
  trained with the others (Sec. 4, pp. 5-6); hopper alone has only three variants.
- Seed spread is wide (C6; shaded bands in Fig. 3, Fig. 5) and the zero-shot study has no
  monolithic baseline (Fig. 5, p. 7).
- Max child count is fixed by the training agents (Sec. 3.4, p. 5), so a test agent with a
  wider joint wouldn't fit (reader's reading).
- Open question (survey lens): SMP needs a known entity graph and per-entity local features with
  the same meaning across agents. A game frame gives neither: would object slots or patch tokens
  from an unsupervised encoder supply the "limbs", with attention instead of a fixed tree?
- Open question: the top-down net sees the local state only through the bottom-up message
  (Eq. 4, p. 4); does this bottleneck matter at 32 dimensions for larger agents?

## Related

- related [[devin2016modular]]: also splits control into modules for transfer, but per robot and task with a learned interface; SMP shares one module across all limbs of all agents.
- related [[hejna2020morphological]]: morphology transfer by a shared high level and retrained low level, against SMP's zero-shot single policy.
- related [[serrano2024crossdomain]]: surveys cross-domain transfer RL; SMP sidesteps an explicit inter-domain mapping by making the policy morphology-agnostic.
- related [[vaswani2017attention]]: later work (Kurin et al. 2021, Gupta et al. 2022 MetaMorph; not in the KB on main) replaces SMP's tree message passing with transformer attention over limbs.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
