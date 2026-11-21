---
id: chen2021embeddings
title: Transfer with Action Embeddings for Deep Reinforcement Learning
authors: [Yu Chen, Yingfeng Chen, Zhipeng Hu, Tianpei Yang, Changjie Fan, Yang Yu, Jianye Hao]
year: 2021
venue: arXiv preprint
url: https://arxiv.org/abs/1909.02291
arxiv: 1909.02291
doi:
pdf_sha256: 059c7c24a1b6d37ce263116c8651acfa3f60f2ea4d9a568f8ac2e6178b5d6516
tags: [reinforcement-learning, transfer-learning, cross-domain, action-embedding, discrete-actions, sac, games]
status: read
read: 2026-11-21
---

# Transfer with Action Embeddings for Deep Reinforcement Learning

<!-- Read from arXiv v3 (2021-05-10; first version 2019-09-05), 11 PDF pages. arXiv's metadata
title is "Learning Action-Transferable Policy with Action Embedding" (the title cited by
serrano2024crossdomain); the printed v3 title is used here. pp. 7-8 are references; App. A-D are
pp. 8-11 and were read. No journal ref on arXiv; no code link in the paper. -->

## TL;DR

TRACE learns a d-dim embedding per discrete action through a forward transition model, has a SAC
policy output a point in that space mapped to the nearest action, and transfers policy and model to
tasks with other action (and state) spaces, learning faster there than SAC, layer-swap BT or MIKT.

## Problem

Cross-domain transfer work maps differing *state* spaces into a common feature space but ignores
that similar actions in different action spaces share semantics, e.g. the "damage", "control" and
"evasion" skills of different MMORPG roles (Sec. 1, p. 1). Prior action embeddings were given a
priori, learned from expert demonstrations (Act2Vec), via inverse dynamics, or used for
generalisation or transfer within the same state-action space (Sec. 1, p. 1; Sec. 2.2, p. 2). The paper asks
how to learn embeddings that capture action semantics and how to transfer a policy between tasks
whose state and discrete action spaces differ but whose rewards and dynamics are similar
(Sec. 3, p. 2).

## Method

- **Action embeddings** (Sec. 4.1, p. 3; Fig. 1, Eq. 1): a matrix W^ae ∈ R^{|A|×d}, row i =
  e(a_i). Intended properties: actions with similar effects have nearby embeddings, and
  p(s_{t+1}|s_t, e(a_t)) approximates p(s_{t+1}|s_t, a_t). A transition model f_θD learns them: a
  VAE-like latent z_t ~ N(μ_t, σ_t) with [μ_t, σ_t] = f_θD1(s_t, e(a_t)) (for stochastic
  environments), then s̃_{t+1} = f_θD2(s_t, e(a_t), z_t). Loss: E[‖s̃_{t+1} − s_{t+1}‖²] +
  β·KL(N(μ_t, σ_t) ‖ N(0, I)), trained jointly over θD and W^ae. A recurrent model (LSTM) is
  suggested for non-Markov transitions.
- **Policy with embeddings** (Sec. 4.2, pp. 3-4; Alg. 1, p. 3): the SAC policy outputs a
  proto-action â ∈ R^d; the executed action is g(â) = argmin_a ‖â − e(a)‖₂ (nearest neighbour, as
  in Dulac-Arnold et al. 2015); (s, â, r, s') is stored, SAC updates the policy (and state
  embedding), and a second batch updates θD and W^ae. The method is said to extend to other RL
  algorithms "with appropriate adaptions", but only SAC is used.
- **Same-domain transfer** (same states, different actions; Sec. 4.2, p. 4): copy the policy, copy
  the transition model and freeze it, and learn the target's W^ae_T ∈ R^{|A_T|×d} with the
  transition loss; the frozen model pulls target embeddings into alignment with the source's, so
  the nearest-neighbour step finds the most relevant target action.
- **Cross-domain transfer** (different states and actions; Sec. 4.2, p. 4; Fig. 2, Alg. 2,
  p. 3): a state embedding f_θse maps each task's states to a common-size space, trained with the
  policy; the transition model takes (f_se(s_t), a_t, f_se(s_{t+1})). On the target, policy and
  transition model are initialised from the source, f_se and W^ae are re-initialised at random,
  and everything is trained as on the source; the transition model is fine-tuned, not frozen.
- **Experiments** (Sec. 5, pp. 4-6; App. A, pp. 8-9): TRACE-PT (transfer policy and transition
  model) vs SAC from scratch, BT (swap input and output layers of the source network and fine-tune
  everything; an ablation without action embeddings), MIKT (Wan et al. 2020) and TRACE (no
  transfer); all SAC-based, 10 runs each. Tasks: 11×11 n-step gridworld (each action is n
  successive moves, 4^n actions for n = 1, 2, 3; same states); InvertedPendulum and
  InvertedDoublePendulum in MuJoCo and Roboschool (mP, mDP, rP, rDP; states of 4, 11, 5, 9 dims,
  continuous actions discretised to 101, 51, 91, 71 values); one-vs-one combat in a commercial
  game with two classes, She Shou (SS) and Fang Shi (FS), hand-made states of 48 and 60 dims,
  10 actions each.

## Claims

- C1: In the 3-step gridworld (64 actions, d = 4, 10,000 transitions from a random policy), embeddings of actions with the same net effect sit close together in 16 separate groups with near-perfect symmetry across the four directions, and show analogies such as e(↑↑←) + e(↑←→) − e(←→←) ≈ e(↑↑↑) (Sec. 5.1, p. 5, Fig. 3a, p. 4)
- C2: After same-domain transfer from the 2-step to the 1-step gridworld, source and target embeddings are aligned, e.g. e(↑) ≈ 0.5·e(↑↑) + 0.5·e(↑↓) (Sec. 5.1, p. 5, Fig. 3b, p. 4)
- C3: In gridworld same-domain transfer (target n = 1, 2, 3 from sources n = 3, 1, 2), TRACE-PT learns faster than all other methods on all three tasks; MIKT learns faster than SAC but slower than BT (Sec. 5.1, p. 5, Fig. 4, p. 5)
- C4: SAC from scratch outperforms TRACE without transfer on all gridworld tasks, which the authors attribute to mapping discrete actions into a continuous space, hard to learn especially with few actions; TRACE-PT curves show no jump-start because target embeddings start random, but they adapt quickly (Sec. 5.1, p. 5, Fig. 4, p. 5)
- C5: Across MuJoCo and Roboschool pendulum tasks with different state and action spaces (mDP→mP, rDP→rP in the same engine; mP→rDP, rP→mDP across engines), TRACE-PT learns faster than SAC and BT in all tasks, and MIKT is slightly better than SAC (Sec. 5.2, p. 5, Fig. 5, p. 6, Fig. 10, p. 10)
- C6: BT does well only in the simplest transfers and in most cases transfers negatively, e.g. failing on rP→mDP and mP→rDP, while TRACE-PT handles all transfer tasks (Sec. 5.2, pp. 5-6, Fig. 5, p. 6, App. B.2, p. 10)
- C7: For Roboschool/MuJoCo pendulums, the 1-D PCA projection of learned embeddings of the discretised actions follows a roughly linear trend in the action index, with local oscillations, as expected for a discretised continuous action (App. B.1, p. 10, Fig. 11, p. 10)
- C8: In the commercial game (5 sampled skills, 50,000 transitions, d = 6), skills with special effects (Silence, Stun) are clearly separated, damage skills lie close together, and DoT and instant damage are distinguished (Sec. 5.3, p. 6, Fig. 6, p. 6)
- C9: For SS→FS and FS→SS policy transfer, measured by win rate over the last 100 episodes, TRACE-PT is more sample-efficient than MIKT, while BT may transfer negatively (Sec. 5.3, p. 6, Fig. 7, p. 6, Fig. 12, p. 10)
- C10: Transferring only the policy (TRACE-P) also speeds learning in same-domain transfer but trails TRACE-PT, because the target's embeddings, learned anew with a new transition model, may not align with the source's; in cross-domain transfer TRACE-P is competitive with TRACE-PT, especially on rDP→rP, which the authors explain by the transition model and action embeddings being learned from state embeddings that are retrained on the target, which limits transfer (Sec. 5.4, p. 6, Fig. 4, p. 5, Fig. 5b, p. 6)
- C11: Transferring only the transition model (TRACE-T) performs about like TRACE without transfer in both settings, which the authors read as the training cost lying mainly in policy training (Sec. 5.4, p. 6)
- C12: With d from 1 to 6 on the 2-step gridworld, d = 1 fails to learn action semantics; without transfer, training slows as d grows; transfer performance is only slightly affected by d, and the best d depends on the environment's dynamics and action semantics (App. C, p. 10, Fig. 13, p. 11, Fig. 14, p. 11)
- C13: Freezing the transition model in cross-domain transfer (rP→mDP, rP→rDP) gives unstable training and negative transfer compared with fine-tuning it (App. D, pp. 10-11, Fig. 15, p. 11)

## Implementation notes

- Gridworld: SAC actor and critic hidden [200, 100], actor lr 1e-5, critic lr 1e-3, τ = 0.999, α = 0.2, γ = 0.99; transition model hidden [64, 32], lr 1e-3, action_embed_dim 2 (Sec. 5.1 uses d = 4 for the embedding visualisation); no state embedding; reward −0.05 per move, +10 at the goal; episodes end at the goal or after 20 actions (source: Table 1, App. A.2, p. 8).
- MuJoCo/Roboschool: state_embed_dim 5 with hidden [200, 100], SAC as above, action_embed_dim 3, transition hidden [64, 32], lr 1e-3; episodes end after 100 actions or when poles fall (source: Table 3, App. A.3, p. 8).
- Combat game: state_embed_dim 25 with hidden [200, 100], SAC hidden [300, 200], action_embed_dim 6, transition hidden [128, 64], z_dim 8, z hidden [32], β = 1e-2, lr 1e-3; rewards from HP changes, a penalty for unready skills, ±10 for win/loss; 100-action limit (source: Table 5, App. A.4, p. 9).
- The latent z_t is used only in the combat game; gridworld and MuJoCo are deterministic, so the loss reduces to MSE (source: App. A.1, p. 8).
- τ = 0.999 is read as a target-network momentum (1 − 0.001 per step), not a soft-update rate of 0.999 (source: assumption from Tables 1, 3, 5).
- Batch sizes, replay size, update ratios, which policy layers are "middle" vs task-specific, and how SAC's entropy term is defined over the continuous proto-action are not given (source: assumption; Sec. 4.2, p. 3 only says SAC is used).
- Sec. 5.3 says 5 of 15 skills were sampled, but Table 4 lists 14 She Shou skills, while App. A.4 describes the action space as six unique class skills plus common operations such as move, attack and dummy; Fig. 11's caption swaps the colours of its legend; App. B calls the method AE-SAC (source: Sec. 5.3, p. 6, App. A.4, p. 8, Table 4, p. 9, Fig. 11, p. 10).
- For the user's design (reader's inference): with per-game action tables W^ae and one shared policy head emitting a proto-action, a transformer policy could share its trunk across games with different controls; at transfer time TRACE needs target-environment interaction to learn the new W^ae (no zero-shot), and its nearest-neighbour step makes the policy output deterministic given â (source: assumption from Sec. 4.2, pp. 3-4).
- For the user's design (reader's inference): TRACE's policy is a continuous policy over the proto-action â, and the transitions store â, not g(â) (Alg. 1, p. 3, Sec. 4.2, p. 4), so PPO could use it directly, with â's log-probability and g treated as part of the environment; only SAC is tested, and the open points are a continuous-action (e.g. Gaussian) PPO head and the extra variance and clipping effects of a many-to-one mapping, consistent with Serrano et al. grouping the method with parameter transfer usable by on-policy learners (source: [[serrano2024crossdomain#C3]]; the PPO reading, the Gaussian head and the variance point are assumptions).

## Limitations / open questions

- Discrete action spaces only; continuous control is discretised, and extending to continuous actions and constraining state-embedding alignment are left to future work (Sec. 3, p. 2, Sec. 6, p. 7).
- Without transfer, TRACE learns slower than plain SAC in gridworld (Sec. 5.1, p. 5).
- The tasks assume related rewards and dynamics between source and target (Sec. 3, p. 2); the combat-game states are hand-engineered vectors, not pixels (Sec. 5.3, p. 6).
- Reader's observation: results are learning curves only (10 runs, bootstrapped 95% CIs), with no tabulated numbers or significance tests, and with one source per target; source-target pairs differ between the main text and App. B (Fig. 4, p. 5, Fig. 9, p. 9).
- Open question: would embeddings learned from forward dynamics on pixel embeddings (rather than low-dimensional states) still cluster by effect across games, e.g. "jump" in two platformers with different buttons?
- Open question: can an embedding table for a new game be learned from a few minutes of random play before the policy acts, giving near zero-shot transfer?

## Related

- related [[serrano2024crossdomain]]: the survey's [79]; it describes transferring the transition model and middle policy layers and groups the method with parameter transfer ([[serrano2024crossdomain#C3]]).
- related [[liu2025prototypes]]: complementary half of cross-domain transfer; it shares a state encoder across domains, this note shares the action side.
- related [[schulman2017proximal]]: the actor-critic the user's design would use in place of SAC; Implementation notes say how its proto-action policy would fit PPO.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
