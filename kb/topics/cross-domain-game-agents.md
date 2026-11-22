---
id: cross-domain-game-agents
title: "Cross-domain game agents: a visual state embedding, one transformer policy, unseen games"
tags: [cross-domain, transfer, video-games, state-representation, segmentation, object-centric, world-model, domain-adaptation, transformer, ppo, generalist-agent, zero-shot]
updated: 2026-11-22
---

# Cross-domain game agents

The lens is the research direction in the survey plan, in three stages:

1. **Embedding.** Learn, without supervision, a state embedding of a video game from its visual
   output over N frames, meant to work across games.
2. **Policy.** Train one large transformer policy, run locally, on that embedding with PPO or
   another actor-critic method.
3. **Transfer.** Play unseen games that share genre conventions and skills, zero-shot or after a
   little fine-tuning.

A side question: could a frame's SAM segmentation map be the state encoding, or its base, given
that a transformer can take a variable-length set of segments? The section "Segmentation maps as
the state" answers it.

Each statement about a paper's results cites the claim it rests on (`[[id#Cn]]`), and its
conditions (`[[id#An]]`) where they matter. The four combined statements that carry the idea
each have a `### Compatibility` block ("Combining claims" in `docs/knowledge-base.md`). A
statement with an open `conflicts` row is labelled a **conjecture**.

## Summary

- **Stage 1, embedding.** No paper in the KB learns a state embedding from frames of visually
  different games and shows that it supports control in a game it hasn't seen. The nearest
  results:
  - A frozen encoder pretrained on random-action frames transfers to unseen DMControl domains
    [[liu2025prototypes#C4]], but all those domains share one rendering style
    ([[liu2025prototypes#A14]]).
  - Latent actions inferred from frame pairs cluster by action across Habitat, Minecraft and
    DMLab [[gao2025adaworld#C3]].
  - Reconstruction carries most of DreamerV3's performance [[hafner2024mastering#C11]], and
    world-model features beat a VLA backbone for behaviour cloning in Minecraft
    [[hafner2025scalable#C5]]. Both are single-game or per-task.
  - Against a representation-only route: CPC and BERT pretraining fine-tune worse than a
    multi-game decision transformer on held-out Atari games [[lee2022multigame#C7]].
- **Stage 2, policy.** Every generalist game policy in the KB is trained by imitation or
  offline:
  - Gato is pure BC ([[reed2022generalist#A5]]); NitroGen, Game-TARS and SIMA use no RL
    ([[magne2026nitrogen#A6]], [[wang2025gametars#A7]], [[simateam2024instructable#A7]]);
    MGDT learns offline from DQN replay ([[lee2022multigame#A4]]).
  - PPO trains one transformer over 100 robot morphologies [[gupta2022metamorph#C1]], but on
    simulator state ([[gupta2022metamorph#A14]]).
  - PPO trains an object-set transformer to about human level on Atari
    [[derstroff2026relative#C1]], but with one agent per game ([[derstroff2026relative#A8]]).
  - The one RL fine-tune of a pretrained game policy, VPT's, is in one game; RL from scratch
    fails there [[baker2022video#C8]], and the run needs a KL term to the pretrained policy
    [[baker2022video#C11]].
- **Stage 3, transfer.** Zero-shot play of held-out games is reported only for imitation-trained
  generalists, and only where the games share conventions:
  - SIMA, in 3D games ([[simateam2024instructable#C8]], [[simateam2024instructable#A3]],
    [[simateam2024instructable#A18]]);
  - Game-TARS, given a per-game prompt with the controls ([[wang2025gametars#C21]]);
  - NitroGen, whose benchmark may overlap its training games ([[magne2026nitrogen#C11]],
    [[magne2026nitrogen#A15]]).
  - Game-specific skills don't transfer ([[simateam2024instructable#C9]],
    [[magne2026nitrogen#C16]]). Gato transfers negatively to visually distinct Atari Boxing
    [[reed2022generalist#C11]].
- **SAM as the state.** It is a plausible base, not an established one. A permutation-invariant
  transformer over an object set is robust to object order and count [[derstroff2026relative#C5]],
  [[derstroff2026relative#C14]]. But the result assumes perfect, typed objects read from RAM
  ([[derstroff2026relative#A1]], [[derstroff2026relative#A6]]). SAM's masks are class-agnostic
  and it never saw game frames ([[kirillov2023segment#A6]], [[kirillov2023segment#A14]]).
- **Main open problems:**
  1. extracting a typed, tracked object or state set from game pixels without per-game RAM
     access;
  2. alignment across many games rather than two domains;
  3. PPO for one policy over many real games on local hardware, with no simulator resets or
     massive parallelism;
  4. a shared action interface;
  5. reward and progress signals that mean the same across games.

## Stage 1: a state embedding from frames

### Pixel encoders trained with or before the agent

- **CURL.** A contrastive loss on frame stacks, added to off-policy RL:
  - It reaches 17.5% median human-normalised score on Atari100k [[srinivas2020curl#C6]].
  - An encoder trained only by the contrastive loss gives near-optimal policies on most of 6
    DMControl tasks [[srinivas2020curl#C11]].
  - Contrasting whole frame stacks beats single frames on most tasks [[srinivas2020curl#C10]].
  - Conditions: one environment, no transfer ([[srinivas2020curl#A9]]), no visual shift
    ([[srinivas2020curl#A15]]), off-policy learners only; on-policy ones were not tested
    ([[srinivas2020curl#A8]]).
- **MAE.** Masked patch reconstruction:
  - A 75% masking ratio works well [[he2021masked#C2]], and leaving mask tokens out of the
    encoder cuts pretraining compute [[he2021masked#C3]].
  - Linear probing and fine-tuning disagree: fine-tuning one block adds 7.5 points
    [[he2021masked#C12]].
  - Its features are evaluated after supervised training, not zero-shot
    ([[he2021masked#A12]]), on static natural photos ([[he2021masked#A21]],
    [[he2021masked#A1]]). It is also the pretraining of SAM's and SAM 2's encoders
    ([[kirillov2023segment#A3]], [[ravi2024sam#A3]]).
- **CRPTpro.** Prototypical pretraining on random-action frames:
  - The frozen encoder transfers to unseen domains without fine-tuning
    [[liu2025prototypes#C4]], and more diverse pretraining domains generalise better
    [[liu2025prototypes#C7]].
  - Its loss uses frames only, so domains with different action spaces can share pretraining
    ([[liu2025prototypes#A7]]). That fits the lens, but "unseen" means other DMControl domains
    in the same rendering ([[liu2025prototypes#A13]], [[liu2025prototypes#A14]]).
- **Bisimulation.** A learned behavioural distance on Space Invaders tracks game events, not
  pixel differences [[castro2020scalable#C12]], within one deterministic MDP
  ([[castro2020scalable#A11]], [[castro2020scalable#A4]]).
- **How many frames.** The evidence is mixed:
  - Context length matters for DT on Atari even with frame stacking [[chen2021decision#C7]].
  - Frame stacks help CURL [[srinivas2020curl#C10]].
  - NitroGen found no gain from more than one past frame in its action-game tasks
    [[magne2026nitrogen#C8]].

### Segmentation and object-centric encoders

- **SAM.** Class-agnostic promptable masks:
  - SA-1B averages about 100 masks per image [[kirillov2023segment#C3]].
  - Zero-shot single-point masks beat RITM on 16 of 23 datasets [[kirillov2023segment#C8]].
  - The decoder runs in ~50 ms on CPU once the image embedding exists
    [[kirillov2023segment#C5]], the encoder's cost being paid once per image
    ([[kirillov2023segment#A11]]).
  - Mask embeddings show some, imperfect, semantic similarity, which the authors call
    preliminary [[kirillov2023segment#C19]].
  - It sees one image at a time ([[kirillov2023segment#A1]]).
- **SAM 2.** SAM for video, with a memory:
  - 130 FPS in image mode on an A100, against SAM's 21.7 [[ravi2024sam#C6]]; 43.8 FPS for
    video with Hiera-B+ [[ravi2024sam#C8]].
  - On Virtual KITTI 2, driving rendered by a game engine, it beats SAM+Cutie
    [[ravi2024sam#C18]]. That is the KB's only segmentation result on engine-rendered frames,
    and those frames are near-photoreal ([[ravi2024sam#A30]]).
  - It tracks only the objects it is prompted with ([[ravi2024sam#A5]]), and runs memory and
    decoder per object ([[ravi2024sam#A6]]).
- **ORCA (ORBiT).** Objects expressed relative to an anchor object, fed to a mean-pooled
  transformer with no positional encodings ([[derstroff2026relative#A12]]):
  - It reaches 1.10 HNS IQM on 13 Atari games with PPO [[derstroff2026relative#C1]].
  - It is robust to translation and HackAtari variants ([[derstroff2026relative#C4]],
    [[derstroff2026relative#C6]]), and to shuffled or extra objects
    ([[derstroff2026relative#C5]], [[derstroff2026relative#C14]]).
  - It needs a decision-relevant anchor [[derstroff2026relative#C15]], and some variants still
    break it [[derstroff2026relative#C7]].
  - A run takes about 4 h on one V100 [[derstroff2026relative#C16]].
- **Entities from patches.** MGDT's attention heads attend to patches holding entities (the
  player, objects, free space) with no object supervision [[lee2022multigame#C20]].

### Segmentation maps as the state

What the KB supports:

- **The set-transformer half works.** A permutation-invariant transformer over an object set is
  unaffected by object order [[derstroff2026relative#C5]] and keeps its score when distractor
  and landmark counts change at test time [[derstroff2026relative#C14]]. Relative coordinates
  make the input invariant to rigid moves of the whole scene [[derstroff2026relative#C9]].
  In robots, one transformer over variable-length limb sets transfers zero-shot to unseen
  morphologies ([[gupta2022metamorph#C9]], [[kurin2021cage#C6]]).
- **The extractor half is unproven for games.**
  - ORCA reads objects from RAM and assumes a perfect extractor ([[derstroff2026relative#A1]]).
    Its objects carry a one-hot type from a fixed vocabulary plus position and velocity, with
    no appearance features ([[derstroff2026relative#A6]]).
  - SAM gives class-agnostic masks with no types ([[kirillov2023segment#A6]]), from one image
    at a time with no temporal context ([[kirillov2023segment#A1]]), and was trained and
    evaluated without game frames, sprites or HUDs ([[kirillov2023segment#A14]]). Velocities
    and identities over N frames would need SAM 2's tracking, which follows only prompted
    objects ([[ravi2024sam#A5]]) at per-object cost ([[ravi2024sam#A6]]).
  - The descriptors matter: MetaMorph fails to learn without morphology inputs
    [[gupta2022metamorph#C5]], the limb-set analogue of objects without types. SAM's mask
    embeddings are the obvious type substitute, but their semantics are preliminary
    [[kirillov2023segment#C19]].
- **Cost.** SAM 2 runs at 43.8 FPS for one prompted object at batch size 1 with a compiled
  encoder on an A100 ([[ravi2024sam#C8]], [[ravi2024sam#A22]]). Full-frame automatic
  segmentation, many tracked objects, and a local consumer GPU are all unmeasured.

So a SAM segmentation map is a plausible base for the state: segments, each with a mask
embedding, box and motion, as tokens of a set transformer. Whether it is a suitable encoding is
a conjecture until the five conflicts in Compatibility (a) are tested (types, a perfect
extractor, a hand-picked anchor, one agent per game, no unseen game), along with its main
unverified row, mask quality on game imagery.

### World models and latent actions

- **DreamerV3.** One configuration with fixed hyperparameters beats PPO in all 8 domains
  [[hafner2024mastering#C1]]. Minecraft needed one A100 for about 9 days
  [[hafner2024mastering#C4]]. Reconstruction carries most of the performance, which the authors
  suggest could enable pretraining on unsupervised data [[hafner2024mastering#C11]]. But every
  task is learned from scratch, and one world model across domains is future work
  ([[hafner2024mastering#A1]]).
- **Dreamer 4.** In Minecraft only ([[hafner2025scalable#A1]]):
  - Behaviour cloning on world-model features beats a Gemma 3 VLA and training from scratch
    [[hafner2025scalable#C5]].
  - It runs at 21 FPS on one H100 [[hafner2025scalable#C11]].
  - Action conditioning learned in the Overworld carries to Nether and End video that had no
    actions [[hafner2025scalable#C13]], but not without video of those regions
    [[hafner2025scalable#C14]].
- **AdaWorld.** Latent actions pretrained on about 2000M frames from 1016 games plus video
  ([[gao2025adaworld#A2]]):
  - Latent actions cluster by action across environments [[gao2025adaworld#C3]].
  - With 100 samples per action or trajectory it adapts to four unseen environments, in
    action-conditioned video prediction (PSNR, LPIPS), not control [[gao2025adaworld#C7]].
  - With no weight updates it averages 44.83% in planning [[gao2025adaworld#C13]].
  - Adding robot video helps on 2D games [[gao2025adaworld#C16]].
  - Conditions: the agent's action must drive the dominant change between frames
    ([[gao2025adaworld#A10]]), and the test scenes come from seen Procgen games
    ([[gao2025adaworld#A6]]).

### Cross-domain alignment

- **DANN.** A gradient-reversed domain classifier:
  - It closes 52.9% of the MNIST→MNIST-M gap [[ganin2016domain#C9]] but fails MNIST→SVHN
    [[ganin2016domain#C11]].
  - Conditions: exactly two domains ([[ganin2016domain#A3]]), a shared label space
    ([[ganin2016domain#A2]]), supervised i.i.d. data, not RL ([[ganin2016domain#A22]]). It
    relies on marginal alignment also aligning the classes ([[ganin2016domain#A8]]).
- **UDIL.** Unsupervised cross-domain imitation beats XIRL on X-MAGICAL
  [[franzmeyer2022matters#C1]], and its time-invariance constraint is essential
  [[franzmeyer2022matters#C8]]. It works on low-dimensional vectors, not pixels
  ([[franzmeyer2022matters#A2]]).
- **State correspondence (Zhang et al.).** It learns an unpaired map from images to simulator
  state [[zhang2020correspondence#C4]], but end-to-end training collapses
  [[zhang2020correspondence#C13]].
- **XIRL and TCC.**
  - XIRL's temporal-cycle-consistency reward generalises to a held-out embodiment
    [[zakka2021xirl#C2]], where SimCLR does poorly [[zakka2021xirl#C6]].
  - In few-shot phase classification, TCC with one labelled video matches supervised learning
    with about 50 [[dwibedi2019cycle#C6]].
  - Imitation from human video in other contexts works on a real robot
    [[liu2018context#C5]].
- **Action alignment (TRACE).** Action embeddings learned from transitions separate skills by
  effect in a commercial combat game [[chen2021embeddings#C8]], and transfer policies between
  character classes [[chen2021embeddings#C9]]. They need low-dimensional states
  ([[chen2021embeddings#A4]]), and target embeddings are learned online, not zero-shot
  ([[chen2021embeddings#A10]]).
- **The survey's view.** Most cross-domain methods assume some shared structure whose presence
  in a new setting is hard to check [[serrano2024crossdomain#C6]]. No method detects that no
  source will help [[serrano2024crossdomain#C5]]. None of the reviewed domains is a video game
  ([[serrano2024crossdomain#A13]]).

## Stage 2: one transformer policy

### Generalist game policies (all imitation or offline)

- **DT and MGDT.**
  - DT is comparable to CQL on 3 of 4 Atari games from 1% of the DQN replay
    [[chen2021decision#C1]], one model per environment ([[chen2021decision#A23]]).
  - One DT-200M trained offline on 41 Atari games reaches 126% human-normalised IQM
    [[lee2022multigame#C1]], improving with size [[lee2022multigame#C3]].
  - On 5 held-out games it adapts only by fine-tuning [[lee2022multigame#C4]]
    ([[lee2022multigame#A16]]), and all games share Atari's 18 actions, a property the authors
    say their conclusions rest on ([[lee2022multigame#A1]]).
- **Gato.**
  - Single-domain specialists of the same architecture are stronger
    [[reed2022generalist#C17]].
  - Prompting with demonstrations couldn't adapt it to new tasks
    [[reed2022generalist#C8]].
  - The 1.18B model ran on an RTX 3090 at a context of 1 and still overran 20 Hz
    [[reed2022generalist#C22]].
- **VPT.**
  - An inverse dynamics model is about two orders of magnitude more data-efficient than a
    causal BC model [[baker2022video#C2]].
  - RL from the pretrained model works where RL from scratch fails [[baker2022video#C8]], with
    a KL to the pretrained policy against forgetting [[baker2022video#C11]].
- **SIMA.** One agent beats per-environment specialists by 67% on average
  [[simateam2024instructable#C4]], at an unreported scale
  ([[simateam2024instructable#A17]]).
- **NitroGen.**
  - BC on 40,000 h of gamepad video completes 38-62% of tasks without fine-tuning
    [[magne2026nitrogen#C11]].
  - Pretraining helps fine-tuning on held-out games, by up to 52% relative on combat
    ([[magne2026nitrogen#C13]], [[magne2026nitrogen#C14]]).
  - The authors call it a reactive "system-1" model and expect RL post-training to add more
    [[magne2026nitrogen#C17]].
- **Game-TARS.** Adding other games' data raises its Minecraft score to 65%, from about 45%
  with Minecraft data only [[wang2025gametars#C15]].
- **Jev.** A Doom bot driven through a text state rather than images, a vendor claim
  [[almeida2026jev#C11]].

### Morphology and modular transfer: what a token-set policy can share

These are robot results, but they are the KB's evidence on one policy over inputs whose
structure varies, the role games' object sets would play.

- **MetaMorph.**
  - One PPO-trained transformer over 100 morphologies matches per-robot MLPs on two of three
    terrains [[gupta2022metamorph#C1]], about 5x more sample-efficiently
    [[gupta2022metamorph#C2]].
  - It is strong zero-shot on 100 unseen robots [[gupta2022metamorph#C9]].
  - Without morphology inputs it fails [[gupta2022metamorph#C5]]. Without replay balancing
    across robots it fails for 10-15% of them [[gupta2022metamorph#C6]].
  - Its token order matters zero-shot unless randomised in training
    [[gupta2022metamorph#C16]].
- **AMORPHEUS** beats SMP on all eight SMP benchmarks [[kurin2021cage#C3]] and on 5 of 7
  held-out morphologies [[kurin2021cage#C6]]. **SMP** beats a monolithic multi-task policy
  [[huang2020modular#C1]] and generalises zero-shot to held-out variants
  [[huang2020modular#C5]].
- **DGN** keeps 62-113% of its standing-up reward with half or double the limbs
  [[pathak2019assembling#C2]].
  **HCP-E** reaches 88-92.5% zero-shot reaching on held-out robot types
  [[chen2018hardware#C2]].
- **Modular and hierarchical splits.**
  - Devin's composed modules transfer some tasks zero-shot [[devin2016modular#C1]] and
    initialise others [[devin2016modular#C3]].
  - A high-level policy transfers across bodies [[hejna2020morphological#C1]] but overfits to
    the morphology it trained with [[hejna2020morphological#C3]].
  - The survey proposes hierarchical decoupling as a route to reuse
    [[serrano2024crossdomain#C13]].

### PPO and data sampling

- **PPO.**
  - On 49 Atari games it wins 30 on average reward over training [[schulman2017proximal#C13]],
    beats A2C on sample complexity and matches ACER while being much simpler
    [[schulman2017proximal#C15]], a few lines of change to vanilla policy gradient
    [[schulman2017proximal#C17]].
  - It allows a joint policy-value architecture [[schulman2017proximal#C17]].
  - Every policy is trained from scratch on one environment ([[schulman2017proximal#A15]]).
- **SGS.** Success-guided sampling over a finite set of task configurations:
  - It reaches 73% on 13-terrain locomotion where PLR gets 54% [[zhang2026balanced#C1]].
  - It keeps improving as parallel environments grow to 1M [[zhang2026balanced#C3]].
  - More configurations help up to a point [[zhang2026balanced#C9]].
  - Conditions: reset to any configuration ([[zhang2026balanced#A3]]), GPU simulation at up
    to 1M environments ([[zhang2026balanced#A8]]), and state, not pixels
    ([[zhang2026balanced#A7]]).
  - Across games, "configuration" would become a (game, start state) pair; the game-level
    analogue of MetaMorph's replay balancing.
- **On-policy transfer.** About 43% of reviewed cross-domain methods transfer knowledge only an
  off-policy learner can use [[serrano2024crossdomain#C2]]. For on-policy learners, reward
  shaping or parameter transfer can still help [[serrano2024crossdomain#C3]].

## Stage 3: zero-shot and few-shot transfer

- **SIMA.**
  - Held-out environments get non-trivial zero-shot performance, sometimes matching the
    specialist [[simateam2024instructable#C8]].
  - Generic skills transfer and environment-specific ones don't
    [[simateam2024instructable#C9]].
  - Goat Simulator 3, whose data never touched the encoders, transfers comparably, so
    transfer isn't driven by the visual components alone [[simateam2024instructable#C10]].
- **Game-TARS.** On unseen games it is above GPT-5 and o4-mini on every MiniWorld map
  [[wang2025gametars#C12]]; on VizDoom it is strictly ahead on 2 of 4 maps and tied with
  Gemini-2.5-Pro on the other two [[wang2025gametars#C10]]; on four Poki web games against
  first-time humans it is the slowest of all agents on Race [[wang2025gametars#C8]]. Every target
  game supplies a prompt naming its keys and rules ([[wang2025gametars#C21]],
  [[wang2025gametars#A11]]).
- **NitroGen.** It completes tasks without fine-tuning [[magne2026nitrogen#C11]]. The authors
  conclude that common patterns (combat, navigation) transfer and game-specific mechanics need
  training [[magne2026nitrogen#C16]].
- **Negative and partial results.**
  - Gato on Atari Boxing: a random initialisation beats every pretrained variant
    [[reed2022generalist#C11]].
  - MGDT measures held-out games only after fine-tuning [[lee2022multigame#C4]].
  - AdaWorld with no updates equals random on Maze [[gao2025adaworld#C13]].
- **Within one design space.** Zero-shot to unseen robots works when they come from the
  training design space ([[gupta2022metamorph#C9]], [[gupta2022metamorph#A12]]): the robot
  analogue of "same genre".

## Compatibility blocks

### Compatibility: SAM object set into a cross-game transformer policy

Statement (**conjecture**): SAM or SAM 2 segments of game frames, encoded as an object set, give
a transformer policy a state that works across games, with ORCA's order and count robustness,
at interactive rates.

Assume:
- S1: inputs are rendered video-game frames (2D or 3D, stylised, sprites, HUD)
- S2: one policy is trained across many games
- S3: the target game is unseen in training
- S4: training and play run on local hardware (one consumer GPU)
- S5: objects come from SAM or SAM 2 masks of the frames, not from RAM or simulator state

| Claim | Assumptions |
|---|---|
| [[kirillov2023segment#C3]] | [[kirillov2023segment#A4]], [[kirillov2023segment#A7]], [[kirillov2023segment#A20]] |
| [[kirillov2023segment#C5]] | [[kirillov2023segment#A1]], [[kirillov2023segment#A2]], [[kirillov2023segment#A11]] |
| [[kirillov2023segment#C19]] | [[kirillov2023segment#A3]], [[kirillov2023segment#A4]], [[kirillov2023segment#A6]], [[kirillov2023segment#A14]], [[kirillov2023segment#A23]], [[kirillov2023segment#A24]], [[kirillov2023segment#A25]] |
| [[ravi2024sam#C6]] | [[ravi2024sam#A1]], [[ravi2024sam#A2]], [[ravi2024sam#A3]], [[ravi2024sam#A4]], [[ravi2024sam#A5]], [[ravi2024sam#A14]], [[ravi2024sam#A16]], [[ravi2024sam#A20]], [[ravi2024sam#A22]], [[ravi2024sam#A30]] |
| [[ravi2024sam#C8]] | [[ravi2024sam#A1]], [[ravi2024sam#A2]], [[ravi2024sam#A3]], [[ravi2024sam#A6]], [[ravi2024sam#A7]], [[ravi2024sam#A8]], [[ravi2024sam#A9]], [[ravi2024sam#A22]] |
| [[derstroff2026relative#C1]] | [[derstroff2026relative#A1]], [[derstroff2026relative#A2]], [[derstroff2026relative#A3]], [[derstroff2026relative#A5]], [[derstroff2026relative#A6]], [[derstroff2026relative#A7]], [[derstroff2026relative#A8]], [[derstroff2026relative#A11]] |
| [[derstroff2026relative#C5]] | [[derstroff2026relative#A1]], [[derstroff2026relative#A6]], [[derstroff2026relative#A7]], [[derstroff2026relative#A8]], [[derstroff2026relative#A11]], [[derstroff2026relative#A12]] |
| [[derstroff2026relative#C14]] | [[derstroff2026relative#A1]], [[derstroff2026relative#A2]], [[derstroff2026relative#A3]], [[derstroff2026relative#A4]], [[derstroff2026relative#A14]] |
| [[derstroff2026relative#C15]] | [[derstroff2026relative#A1]], [[derstroff2026relative#A2]], [[derstroff2026relative#A4]], [[derstroff2026relative#A14]] |

Union: S1, S2, S3, S4, S5, [[kirillov2023segment#A4]], [[kirillov2023segment#A7]], [[kirillov2023segment#A20]], [[kirillov2023segment#A1]], [[kirillov2023segment#A2]], [[kirillov2023segment#A11]], [[kirillov2023segment#A3]], [[kirillov2023segment#A6]], [[kirillov2023segment#A14]], [[kirillov2023segment#A23]], [[kirillov2023segment#A24]], [[kirillov2023segment#A25]], [[ravi2024sam#A1]], [[ravi2024sam#A2]], [[ravi2024sam#A3]], [[ravi2024sam#A4]], [[ravi2024sam#A5]], [[ravi2024sam#A14]], [[ravi2024sam#A16]], [[ravi2024sam#A20]], [[ravi2024sam#A22]], [[ravi2024sam#A30]], [[ravi2024sam#A6]], [[ravi2024sam#A7]], [[ravi2024sam#A8]], [[ravi2024sam#A9]], [[derstroff2026relative#A1]], [[derstroff2026relative#A2]], [[derstroff2026relative#A3]], [[derstroff2026relative#A5]], [[derstroff2026relative#A6]], [[derstroff2026relative#A7]], [[derstroff2026relative#A8]], [[derstroff2026relative#A11]], [[derstroff2026relative#A12]], [[derstroff2026relative#A4]], [[derstroff2026relative#A14]]

Gaps:

| Pair | Status | Gap | What would close it |
|---|---|---|---|
| [[kirillov2023segment#A14]] / S1 | unverified | SAM saw no game frames, sprites or HUDs; mask quality on them is unknown | experiment: score SAM automatic masks against OCAtari's RAM objects on Atari frames |
| [[kirillov2023segment#A4]] / S1 | unverified | the ~100 masks per image are a statistic of high-resolution photos; game frames may give far fewer or more | experiment: count masks per frame on a range of 2D and 3D games |
| [[kirillov2023segment#A6]] / [[derstroff2026relative#A6]] | conflicts | ORCA needs a one-hot type per object from a fixed vocabulary; SAM's masks carry no class | experiment: replace ORCA's types by clustered SAM mask embeddings and rerun its Atari benchmark |
| [[derstroff2026relative#A1]] / S5 | conflicts | ORCA's results assume a perfect RAM extractor; SAM masks are noisy, split or merge objects, and give no velocity | experiment: ORBiT on SAM-derived objects vs RAM objects on the same games |
| [[kirillov2023segment#A1]] / [[derstroff2026relative#A1]] | unverified | ORCA objects have velocities; SAM is per-image, so velocity needs tracking across frames | experiment: SAM 2 tracking or frame differencing for velocities, ablated in ORBiT |
| [[ravi2024sam#A30]] / S1 | unverified | SAM 2 saw no non-photoreal game imagery; its only engine-rendered test is near-photoreal driving | experiment: SAM 2 tracking quality on pixel-art and cartoon-rendered games |
| [[ravi2024sam#A6]] / S4 | unverified | per-object memory and decoding make cost grow with object count; the A100 figure doesn't carry to a consumer GPU | experiment: SAM 2 FPS vs object count on the local GPU |
| [[ravi2024sam#A22]] / S4 | unverified | 43.8 FPS is measured on an A100 with torch.compile | experiment: SAM 2 FPS at batch size 1 on the local GPU |
| [[ravi2024sam#A5]] / S5 | unverified | SAM 2 segments and tracks only prompted objects; game objects must be prompted automatically and again as new ones enter | experiment: automatic-mask FPS plus per-frame re-prompting cost on the local GPU |
| [[kirillov2023segment#A7]] / S4 | unverified | automatic masks prompt a 32x32 grid plus 20 crops per image; that cost per frame is unmeasured | experiment: automatic-mask FPS per game frame on the local GPU |
| [[kirillov2023segment#A11]] / S4 | unverified | the ~50 ms CPU prompt time ([[kirillov2023segment#C5]]) excludes the image encoder; SAM runs at 21.7 FPS on an A100 ([[ravi2024sam#C6]]) | experiment: end-to-end per-frame latency, encoder included, on the local GPU |
| [[derstroff2026relative#A3]] / S2 | conflicts | the anchor is hand-picked per game; across games, and in unseen ones, it must be found automatically | experiment: learned anchor selection (e.g. the segment whose motion best predicts the action) vs the hand-picked anchor |
| [[derstroff2026relative#A8]] / S2 | conflicts | ORCA trains one agent per game | experiment: one ORBiT agent across several Atari games with a shared type vocabulary |
| [[derstroff2026relative#A8]] / S3 | conflicts | ORCA never tests a game unseen in training | experiment: leave-one-game-out ORBiT on HackAtari or OCAtari games |
| [[derstroff2026relative#A4]] / S1 | unverified | ORCA's environments are 2D; many target games are 3D | paper: an object-centric RL result from pixels in 3D games |
| [[derstroff2026relative#A2]] / S5 | unverified | an anchor must exist and be localised every step; SAM may miss or merge the player | experiment: rate of anchor loss in SAM 2 tracks over long episodes |

### Compatibility: an unsupervised visual embedding aligned across games

Statement (**conjecture**): an encoder pretrained without labels on frames from many games
(contrastive, prototype or world-model objective), with its features aligned across games by a
DANN-style game classifier, gives a PPO policy an input that transfers across games.

Assume:
- S1: inputs are rendered video-game frames (2D or 3D, stylised, sprites, HUD)
- S2: one policy is trained across many games
- S3: the target game is unseen in training
- S4: training and play run on local hardware (one consumer GPU)
- S5: the encoder is pretrained on frames (and frame pairs) from many games, without labels or rewards
- S6: alignment is adversarial on the encoder features, with each game a domain
- S7: the policy on top is trained online with PPO from game reward

| Claim | Assumptions |
|---|---|
| [[srinivas2020curl#C11]] | [[srinivas2020curl#A1]], [[srinivas2020curl#A2]], [[srinivas2020curl#A3]], [[srinivas2020curl#A4]], [[srinivas2020curl#A5]], [[srinivas2020curl#A6]], [[srinivas2020curl#A8]], [[srinivas2020curl#A9]], [[srinivas2020curl#A10]], [[srinivas2020curl#A11]], [[srinivas2020curl#A12]], [[srinivas2020curl#A15]] |
| [[liu2025prototypes#C4]] | [[liu2025prototypes#A1]], [[liu2025prototypes#A2]], [[liu2025prototypes#A3]], [[liu2025prototypes#A4]], [[liu2025prototypes#A5]], [[liu2025prototypes#A6]], [[liu2025prototypes#A7]], [[liu2025prototypes#A8]], [[liu2025prototypes#A9]], [[liu2025prototypes#A11]], [[liu2025prototypes#A12]], [[liu2025prototypes#A13]], [[liu2025prototypes#A14]], [[liu2025prototypes#A18]] |
| [[hafner2024mastering#C11]] | [[hafner2024mastering#A1]], [[hafner2024mastering#A2]], [[hafner2024mastering#A3]], [[hafner2024mastering#A4]], [[hafner2024mastering#A8]], [[hafner2024mastering#A9]], [[hafner2024mastering#A11]], [[hafner2024mastering#A14]], [[hafner2024mastering#A15]], [[hafner2024mastering#A16]], [[hafner2024mastering#A17]], [[hafner2024mastering#A18]] |
| [[gao2025adaworld#C3]] | [[gao2025adaworld#A2]], [[gao2025adaworld#A9]], [[gao2025adaworld#A10]] |
| [[gao2025adaworld#C13]] | [[gao2025adaworld#A1]], [[gao2025adaworld#A2]], [[gao2025adaworld#A3]], [[gao2025adaworld#A4]], [[gao2025adaworld#A5]], [[gao2025adaworld#A6]], [[gao2025adaworld#A7]], [[gao2025adaworld#A10]], [[gao2025adaworld#A12]] |
| [[ganin2016domain#C9]] | [[ganin2016domain#A1]], [[ganin2016domain#A2]], [[ganin2016domain#A3]], [[ganin2016domain#A4]], [[ganin2016domain#A8]], [[ganin2016domain#A9]], [[ganin2016domain#A10]], [[ganin2016domain#A11]], [[ganin2016domain#A13]], [[ganin2016domain#A15]], [[ganin2016domain#A19]], [[ganin2016domain#A22]] |
| [[ganin2016domain#C11]] | [[ganin2016domain#A1]], [[ganin2016domain#A2]], [[ganin2016domain#A3]], [[ganin2016domain#A4]], [[ganin2016domain#A8]], [[ganin2016domain#A9]], [[ganin2016domain#A10]], [[ganin2016domain#A11]], [[ganin2016domain#A13]], [[ganin2016domain#A14]], [[ganin2016domain#A15]], [[ganin2016domain#A19]], [[ganin2016domain#A22]] |
| [[lee2022multigame#C7]] | [[lee2022multigame#A1]], [[lee2022multigame#A2]], [[lee2022multigame#A3]], [[lee2022multigame#A4]], [[lee2022multigame#A5]], [[lee2022multigame#A6]], [[lee2022multigame#A7]], [[lee2022multigame#A8]], [[lee2022multigame#A9]], [[lee2022multigame#A10]], [[lee2022multigame#A11]], [[lee2022multigame#A12]], [[lee2022multigame#A13]], [[lee2022multigame#A14]], [[lee2022multigame#A16]], [[lee2022multigame#A17]] |

Union: S1, S2, S3, S4, S5, S6, S7, [[srinivas2020curl#A1]], [[srinivas2020curl#A2]], [[srinivas2020curl#A3]], [[srinivas2020curl#A4]], [[srinivas2020curl#A5]], [[srinivas2020curl#A6]], [[srinivas2020curl#A8]], [[srinivas2020curl#A9]], [[srinivas2020curl#A10]], [[srinivas2020curl#A11]], [[srinivas2020curl#A12]], [[srinivas2020curl#A15]], [[liu2025prototypes#A1]], [[liu2025prototypes#A2]], [[liu2025prototypes#A3]], [[liu2025prototypes#A4]], [[liu2025prototypes#A5]], [[liu2025prototypes#A6]], [[liu2025prototypes#A7]], [[liu2025prototypes#A8]], [[liu2025prototypes#A9]], [[liu2025prototypes#A11]], [[liu2025prototypes#A12]], [[liu2025prototypes#A13]], [[liu2025prototypes#A14]], [[liu2025prototypes#A18]], [[hafner2024mastering#A1]], [[hafner2024mastering#A2]], [[hafner2024mastering#A3]], [[hafner2024mastering#A4]], [[hafner2024mastering#A8]], [[hafner2024mastering#A9]], [[hafner2024mastering#A11]], [[hafner2024mastering#A14]], [[hafner2024mastering#A15]], [[hafner2024mastering#A16]], [[hafner2024mastering#A17]], [[hafner2024mastering#A18]], [[gao2025adaworld#A2]], [[gao2025adaworld#A9]], [[gao2025adaworld#A10]], [[gao2025adaworld#A1]], [[gao2025adaworld#A3]], [[gao2025adaworld#A4]], [[gao2025adaworld#A5]], [[gao2025adaworld#A6]], [[gao2025adaworld#A7]], [[gao2025adaworld#A12]], [[ganin2016domain#A1]], [[ganin2016domain#A2]], [[ganin2016domain#A3]], [[ganin2016domain#A4]], [[ganin2016domain#A8]], [[ganin2016domain#A9]], [[ganin2016domain#A10]], [[ganin2016domain#A11]], [[ganin2016domain#A13]], [[ganin2016domain#A15]], [[ganin2016domain#A19]], [[ganin2016domain#A22]], [[ganin2016domain#A14]], [[lee2022multigame#A1]], [[lee2022multigame#A2]], [[lee2022multigame#A3]], [[lee2022multigame#A4]], [[lee2022multigame#A5]], [[lee2022multigame#A6]], [[lee2022multigame#A7]], [[lee2022multigame#A8]], [[lee2022multigame#A9]], [[lee2022multigame#A10]], [[lee2022multigame#A11]], [[lee2022multigame#A12]], [[lee2022multigame#A13]], [[lee2022multigame#A14]], [[lee2022multigame#A16]], [[lee2022multigame#A17]]

Gaps:

| Pair | Status | Gap | What would close it |
|---|---|---|---|
| [[ganin2016domain#A3]] / S6 | conflicts | DANN treats exactly two domains with a binary classifier; here every game is a domain | paper: Kline 2025 (arXiv 2505.17328), a DANN-style game classifier over 10 games |
| [[ganin2016domain#A22]] / S7 | conflicts | DANN's results are supervised i.i.d. prediction; an RL policy's data shift as it learns | experiment: DANN loss on a PPO agent's encoder across two Atari games, against no alignment |
| [[ganin2016domain#A2]] / S2 | unverified | DANN needs one label space; games differ in actions, rewards and what a state means | experiment: game-classifier alignment across games with mismatched action semantics and reward scales, against no alignment |
| [[ganin2016domain#A8]] / S6 | unverified | games differ in how often each kind of state occurs (label shift), where aligning marginals can mismatch classes | paper: Zhao et al. 2019 (arXiv 1901.09453), invariant representations under label shift |
| [[ganin2016domain#A14]] / S5 | unverified | adaptation fails from a narrow source to a more diverse target; a game set skewed to one style may be narrow | experiment: vary the pretraining game mix and measure held-out-game RL return |
| [[liu2025prototypes#A14]] / S1 | conflicts | CRPTpro's unseen domains share its rendering style; games differ visually | experiment: CRPTpro pretrained on some games, frozen, PPO on a visually different held-out game |
| [[liu2025prototypes#A3]] / S1 | unverified | CRPTpro is DMControl only | experiment: CRPTpro pretraining on game frames, frozen encoder, PPO on a training game |
| [[srinivas2020curl#A9]] / S2 | conflicts | CURL trains on one environment at a time | experiment: CURL's loss on a multi-game encoder |
| [[srinivas2020curl#A15]] / S3 | conflicts | CURL assumes no visual shift between training and test | experiment: CURL encoder evaluated on a held-out game |
| [[srinivas2020curl#A8]] / S7 | unverified | CURL was tested with off-policy SAC and Rainbow only | experiment: CURL auxiliary loss with PPO on Atari |
| [[hafner2024mastering#A1]] / S2 | conflicts | DreamerV3 learns each task from scratch; one world model across domains is future work | paper: a multi-game world model used as an RL encoder (e.g. OneLA, arXiv 2610.20160) |
| [[hafner2024mastering#A3]] / S1 | unverified | 64x64 images may lose small sprites, text and HUD detail | experiment: resolution ablation on text-heavy games |
| [[gao2025adaworld#A10]] / S1 | unverified | latent actions assume the agent drives the dominant frame change; camera motion and many NPCs break it | experiment: latent-action clustering on games with moving cameras |
| [[gao2025adaworld#A6]] / S3 | unverified | AdaWorld's Procgen test scenes come from seen games | experiment: AdaWorld latent actions on an unseen game |
| [[lee2022multigame#A16]] / S3 | unverified | MGDT measures representation transfer only by fine-tuning; its negative result on representations is not zero-shot either | experiment: frozen-encoder zero-shot comparison of CPC, MAE and DT features on held-out Atari games |

### Compatibility: a generalist transformer policy trained with PPO

Statement (**conjecture**): a NitroGen-, Game-TARS- or MGDT-style transformer policy trained
across many games with PPO from game reward, rather than by imitation, improves on its
imitation-only start and stays stable, within local compute.

Assume:
- S1: inputs are rendered video-game frames (2D or 3D, stylised, sprites, HUD)
- S2: one policy is trained across many games
- S3: the target game is unseen in training
- S4: training and play run on local hardware (one consumer GPU)
- S5: the policy is a large transformer trained online with PPO from game reward, optionally starting from an imitation-pretrained generalist
- S6: all games are driven through one fixed action interface (gamepad as in NitroGen, or keyboard and mouse) with a factored categorical or Gaussian head (added to close a gap)

| Claim | Assumptions |
|---|---|
| [[schulman2017proximal#C13]] | [[schulman2017proximal#A1]], [[schulman2017proximal#A2]], [[schulman2017proximal#A3]], [[schulman2017proximal#A4]], [[schulman2017proximal#A5]], [[schulman2017proximal#A6]], [[schulman2017proximal#A11]], [[schulman2017proximal#A13]], [[schulman2017proximal#A14]], [[schulman2017proximal#A15]] |
| [[schulman2017proximal#C17]] | [[schulman2017proximal#A1]], [[schulman2017proximal#A2]], [[schulman2017proximal#A3]], [[schulman2017proximal#A4]], [[schulman2017proximal#A5]] |
| [[gupta2022metamorph#C1]] | [[gupta2022metamorph#A1]], [[gupta2022metamorph#A2]], [[gupta2022metamorph#A3]], [[gupta2022metamorph#A4]], [[gupta2022metamorph#A5]], [[gupta2022metamorph#A6]], [[gupta2022metamorph#A7]], [[gupta2022metamorph#A8]], [[gupta2022metamorph#A10]], [[gupta2022metamorph#A13]], [[gupta2022metamorph#A14]] |
| [[gupta2022metamorph#C6]] | [[gupta2022metamorph#A1]], [[gupta2022metamorph#A2]], [[gupta2022metamorph#A3]], [[gupta2022metamorph#A4]], [[gupta2022metamorph#A5]], [[gupta2022metamorph#A6]], [[gupta2022metamorph#A7]], [[gupta2022metamorph#A8]], [[gupta2022metamorph#A10]], [[gupta2022metamorph#A13]], [[gupta2022metamorph#A14]] |
| [[zhang2026balanced#C3]] | [[zhang2026balanced#A1]], [[zhang2026balanced#A2]], [[zhang2026balanced#A3]], [[zhang2026balanced#A4]], [[zhang2026balanced#A5]], [[zhang2026balanced#A6]], [[zhang2026balanced#A7]], [[zhang2026balanced#A8]], [[zhang2026balanced#A9]], [[zhang2026balanced#A10]], [[zhang2026balanced#A11]], [[zhang2026balanced#A12]], [[zhang2026balanced#A13]], [[zhang2026balanced#A14]], [[zhang2026balanced#A15]], [[zhang2026balanced#A19]], [[zhang2026balanced#A21]] |
| [[baker2022video#C8]] | [[baker2022video#A1]], [[baker2022video#A2]], [[baker2022video#A3]], [[baker2022video#A4]], [[baker2022video#A5]], [[baker2022video#A6]], [[baker2022video#A7]], [[baker2022video#A9]], [[baker2022video#A10]], [[baker2022video#A11]], [[baker2022video#A12]], [[baker2022video#A13]], [[baker2022video#A14]], [[baker2022video#A15]], [[baker2022video#A16]], [[baker2022video#A20]], [[baker2022video#A21]], [[baker2022video#A22]], [[baker2022video#A23]], [[baker2022video#A25]] |
| [[magne2026nitrogen#C17]] | [[magne2026nitrogen#A5]], [[magne2026nitrogen#A6]], [[magne2026nitrogen#A11]] |
| [[lee2022multigame#C1]] | [[lee2022multigame#A1]], [[lee2022multigame#A2]], [[lee2022multigame#A3]], [[lee2022multigame#A4]], [[lee2022multigame#A5]], [[lee2022multigame#A6]], [[lee2022multigame#A7]], [[lee2022multigame#A8]], [[lee2022multigame#A9]], [[lee2022multigame#A10]], [[lee2022multigame#A11]], [[lee2022multigame#A13]] |

Union: S1, S2, S3, S4, S5, S6, [[schulman2017proximal#A1]], [[schulman2017proximal#A2]], [[schulman2017proximal#A3]], [[schulman2017proximal#A4]], [[schulman2017proximal#A5]], [[schulman2017proximal#A6]], [[schulman2017proximal#A11]], [[schulman2017proximal#A13]], [[schulman2017proximal#A14]], [[schulman2017proximal#A15]], [[gupta2022metamorph#A1]], [[gupta2022metamorph#A2]], [[gupta2022metamorph#A3]], [[gupta2022metamorph#A4]], [[gupta2022metamorph#A5]], [[gupta2022metamorph#A6]], [[gupta2022metamorph#A7]], [[gupta2022metamorph#A8]], [[gupta2022metamorph#A10]], [[gupta2022metamorph#A13]], [[gupta2022metamorph#A14]], [[zhang2026balanced#A1]], [[zhang2026balanced#A2]], [[zhang2026balanced#A3]], [[zhang2026balanced#A4]], [[zhang2026balanced#A5]], [[zhang2026balanced#A6]], [[zhang2026balanced#A7]], [[zhang2026balanced#A8]], [[zhang2026balanced#A9]], [[zhang2026balanced#A10]], [[zhang2026balanced#A11]], [[zhang2026balanced#A12]], [[zhang2026balanced#A13]], [[zhang2026balanced#A14]], [[zhang2026balanced#A15]], [[zhang2026balanced#A19]], [[zhang2026balanced#A21]], [[baker2022video#A1]], [[baker2022video#A2]], [[baker2022video#A3]], [[baker2022video#A4]], [[baker2022video#A5]], [[baker2022video#A6]], [[baker2022video#A7]], [[baker2022video#A9]], [[baker2022video#A10]], [[baker2022video#A11]], [[baker2022video#A12]], [[baker2022video#A13]], [[baker2022video#A14]], [[baker2022video#A15]], [[baker2022video#A16]], [[baker2022video#A20]], [[baker2022video#A21]], [[baker2022video#A22]], [[baker2022video#A23]], [[baker2022video#A25]], [[magne2026nitrogen#A5]], [[magne2026nitrogen#A6]], [[magne2026nitrogen#A11]], [[lee2022multigame#A1]], [[lee2022multigame#A2]], [[lee2022multigame#A3]], [[lee2022multigame#A4]], [[lee2022multigame#A5]], [[lee2022multigame#A6]], [[lee2022multigame#A7]], [[lee2022multigame#A8]], [[lee2022multigame#A9]], [[lee2022multigame#A10]], [[lee2022multigame#A11]], [[lee2022multigame#A13]]

Gaps:

| Pair | Status | Gap | What would close it |
|---|---|---|---|
| [[schulman2017proximal#A15]] / S2 | conflicts | PPO's results train one policy per environment, from scratch | experiment: multi-game PPO on a small Atari set vs per-game PPO at equal frames |
| [[gupta2022metamorph#A14]] / S1 | conflicts | MetaMorph's multi-robot PPO observes full simulator state, not pixels | experiment: MetaMorph-style PPO with object tokens from frames |
| [[gupta2022metamorph#A7]] / S2 | unverified | MetaMorph shares one dense reward across robots; game scores differ in scale and density | experiment: per-game reward normalisation vs raw scores in multi-game PPO |
| [[gupta2022metamorph#A8]] / S4 | unverified | 10^8 iterations; about 1.5 GPU days on an RTX 2080 for its own state-based setup ([[gupta2022metamorph#C12]]), unknown for pixels | experiment: measure frames per second of a pixel transformer PPO on the local GPU |
| [[zhang2026balanced#A7]] / S1 | conflicts | SGS's PPO observes compact state, not pixels | experiment: SGS over (game, start state) configurations with a pixel policy |
| [[zhang2026balanced#A8]] / S4 | conflicts | SGS's gains rise with 4K to 1M parallel GPU environments; real games run a few instances locally | experiment: SGS vs uniform sampling at 8-64 environments |
| [[zhang2026balanced#A3]] / S2 | unverified | SGS resets any environment to any configuration; commercial games rarely expose that | experiment: SGS over checkpointed episode starts reachable without arbitrary resets, vs uniform sampling |
| [[baker2022video#A1]] / S2 | unverified | VPT's RL fine-tune is in one game | experiment: KL-regularised PPO fine-tune of NitroGen on two or more games |
| [[baker2022video#A21]] / S4 | conflicts | VPT's RL used 80 GPUs and 56,719 CPUs | experiment: KL-regularised PPO fine-tune at local scale |
| [[magne2026nitrogen#A6]] / S5 | unverified | NitroGen expects RL post-training to help but never runs it | experiment: PPO post-training of NitroGen on held-out games |
| [[magne2026nitrogen#A11]] / S5 | unverified | NitroGen's flow-matching action head gives no tractable likelihood for PPO's ratio (reader's inference) | experiment: PPO with a categorical or Gaussian head distilled from the flow head |
| [[magne2026nitrogen#A5]] / S5 | unverified | NitroGen sees one frame with no reward input; a critic needs reward and may need more context | experiment: value head on NitroGen features, with 1 vs 4 frames |
| [[lee2022multigame#A4]] / S5 | unverified | MGDT learns offline from DQN replay; online PPO on it is untested | experiment: online PPO fine-tune of a multi-game DT |
| [[lee2022multigame#A1]] / S2 | unverified | MGDT relies on Atari's shared 18 actions | assume: S6 |
| [[schulman2017proximal#A2]] / S5 | unverified | PPO needs a stochastic policy with a tractable likelihood; a gamepad or keyboard space mixes buttons and sticks | assume: S6 |

### Compatibility: zero-shot play of an unseen game of the same genre

Statement (conditional, with open conflicts so a **conjecture**): a policy trained on many games
plays an unseen game of a shared genre zero-shot well enough to beat random, with game-specific
skills left to few-shot fine-tuning.

Assume:
- S1: inputs are rendered video-game frames (2D or 3D, stylised, sprites, HUD)
- S2: one policy is trained across many games
- S3: the target game is unseen in training
- S4: training and play run on local hardware (one consumer GPU)
- S5: the target shares genre conventions (controls, objectives, visual idioms) with training games
- S6: at test time the agent gets only frames, no prompt, reward or demonstrations (zero-shot), or a small fine-tuning budget (few-shot)
- S7: the policy is trained with PPO (stage 2 of the lens), not only by imitation

| Claim | Assumptions |
|---|---|
| [[simateam2024instructable#C8]] | [[simateam2024instructable#A1]], [[simateam2024instructable#A2]], [[simateam2024instructable#A3]], [[simateam2024instructable#A4]], [[simateam2024instructable#A5]], [[simateam2024instructable#A6]], [[simateam2024instructable#A7]], [[simateam2024instructable#A8]], [[simateam2024instructable#A9]], [[simateam2024instructable#A10]], [[simateam2024instructable#A11]], [[simateam2024instructable#A12]], [[simateam2024instructable#A13]], [[simateam2024instructable#A15]], [[simateam2024instructable#A16]], [[simateam2024instructable#A17]], [[simateam2024instructable#A18]], [[simateam2024instructable#A20]], [[simateam2024instructable#A25]] |
| [[simateam2024instructable#C9]] | [[simateam2024instructable#A1]], [[simateam2024instructable#A2]], [[simateam2024instructable#A3]], [[simateam2024instructable#A4]], [[simateam2024instructable#A5]], [[simateam2024instructable#A6]], [[simateam2024instructable#A7]], [[simateam2024instructable#A8]], [[simateam2024instructable#A9]], [[simateam2024instructable#A10]], [[simateam2024instructable#A11]], [[simateam2024instructable#A12]], [[simateam2024instructable#A13]], [[simateam2024instructable#A15]], [[simateam2024instructable#A16]], [[simateam2024instructable#A17]], [[simateam2024instructable#A18]], [[simateam2024instructable#A25]] |
| [[wang2025gametars#C10]] | [[wang2025gametars#A1]], [[wang2025gametars#A2]], [[wang2025gametars#A3]], [[wang2025gametars#A4]], [[wang2025gametars#A5]], [[wang2025gametars#A6]], [[wang2025gametars#A7]], [[wang2025gametars#A8]], [[wang2025gametars#A9]], [[wang2025gametars#A10]], [[wang2025gametars#A11]], [[wang2025gametars#A12]], [[wang2025gametars#A13]], [[wang2025gametars#A14]], [[wang2025gametars#A19]], [[wang2025gametars#A21]], [[wang2025gametars#A26]] |
| [[wang2025gametars#C21]] | [[wang2025gametars#A11]], [[wang2025gametars#A17]], [[wang2025gametars#A18]] |
| [[magne2026nitrogen#C11]] | [[magne2026nitrogen#A1]], [[magne2026nitrogen#A3]], [[magne2026nitrogen#A4]], [[magne2026nitrogen#A5]], [[magne2026nitrogen#A6]], [[magne2026nitrogen#A7]], [[magne2026nitrogen#A8]], [[magne2026nitrogen#A9]], [[magne2026nitrogen#A10]], [[magne2026nitrogen#A11]], [[magne2026nitrogen#A12]], [[magne2026nitrogen#A13]], [[magne2026nitrogen#A14]], [[magne2026nitrogen#A15]] |
| [[magne2026nitrogen#C16]] | [[magne2026nitrogen#A3]], [[magne2026nitrogen#A8]], [[magne2026nitrogen#A14]], [[magne2026nitrogen#A16]], [[magne2026nitrogen#A17]], [[magne2026nitrogen#A18]], [[magne2026nitrogen#A19]] |
| [[reed2022generalist#C11]] | [[reed2022generalist#A1]], [[reed2022generalist#A3]], [[reed2022generalist#A4]], [[reed2022generalist#A5]], [[reed2022generalist#A9]], [[reed2022generalist#A10]], [[reed2022generalist#A12]], [[reed2022generalist#A15]], [[reed2022generalist#A19]], [[reed2022generalist#A20]], [[reed2022generalist#A21]], [[reed2022generalist#A22]], [[reed2022generalist#A27]] |
| [[gupta2022metamorph#C9]] | [[gupta2022metamorph#A1]], [[gupta2022metamorph#A2]], [[gupta2022metamorph#A3]], [[gupta2022metamorph#A4]], [[gupta2022metamorph#A5]], [[gupta2022metamorph#A6]], [[gupta2022metamorph#A7]], [[gupta2022metamorph#A8]], [[gupta2022metamorph#A10]], [[gupta2022metamorph#A12]], [[gupta2022metamorph#A13]], [[gupta2022metamorph#A14]] |

Union: S1, S2, S3, S4, S5, S6, S7, [[simateam2024instructable#A1]], [[simateam2024instructable#A2]], [[simateam2024instructable#A3]], [[simateam2024instructable#A4]], [[simateam2024instructable#A5]], [[simateam2024instructable#A6]], [[simateam2024instructable#A7]], [[simateam2024instructable#A8]], [[simateam2024instructable#A9]], [[simateam2024instructable#A10]], [[simateam2024instructable#A11]], [[simateam2024instructable#A12]], [[simateam2024instructable#A13]], [[simateam2024instructable#A15]], [[simateam2024instructable#A16]], [[simateam2024instructable#A17]], [[simateam2024instructable#A18]], [[simateam2024instructable#A20]], [[simateam2024instructable#A25]], [[wang2025gametars#A1]], [[wang2025gametars#A2]], [[wang2025gametars#A3]], [[wang2025gametars#A4]], [[wang2025gametars#A5]], [[wang2025gametars#A6]], [[wang2025gametars#A7]], [[wang2025gametars#A8]], [[wang2025gametars#A9]], [[wang2025gametars#A10]], [[wang2025gametars#A11]], [[wang2025gametars#A12]], [[wang2025gametars#A13]], [[wang2025gametars#A14]], [[wang2025gametars#A19]], [[wang2025gametars#A21]], [[wang2025gametars#A26]], [[wang2025gametars#A17]], [[wang2025gametars#A18]], [[magne2026nitrogen#A1]], [[magne2026nitrogen#A3]], [[magne2026nitrogen#A4]], [[magne2026nitrogen#A5]], [[magne2026nitrogen#A6]], [[magne2026nitrogen#A7]], [[magne2026nitrogen#A8]], [[magne2026nitrogen#A9]], [[magne2026nitrogen#A10]], [[magne2026nitrogen#A11]], [[magne2026nitrogen#A12]], [[magne2026nitrogen#A13]], [[magne2026nitrogen#A14]], [[magne2026nitrogen#A15]], [[magne2026nitrogen#A16]], [[magne2026nitrogen#A17]], [[magne2026nitrogen#A18]], [[magne2026nitrogen#A19]], [[reed2022generalist#A1]], [[reed2022generalist#A3]], [[reed2022generalist#A4]], [[reed2022generalist#A5]], [[reed2022generalist#A9]], [[reed2022generalist#A10]], [[reed2022generalist#A12]], [[reed2022generalist#A15]], [[reed2022generalist#A19]], [[reed2022generalist#A20]], [[reed2022generalist#A21]], [[reed2022generalist#A22]], [[reed2022generalist#A27]], [[gupta2022metamorph#A1]], [[gupta2022metamorph#A2]], [[gupta2022metamorph#A3]], [[gupta2022metamorph#A4]], [[gupta2022metamorph#A5]], [[gupta2022metamorph#A6]], [[gupta2022metamorph#A7]], [[gupta2022metamorph#A8]], [[gupta2022metamorph#A10]], [[gupta2022metamorph#A12]], [[gupta2022metamorph#A13]], [[gupta2022metamorph#A14]]

Gaps:

| Pair | Status | Gap | What would close it |
|---|---|---|---|
| [[simateam2024instructable#A7]] / S7 | unverified | SIMA's zero-shot result is for a BC agent; whether a PPO-trained policy keeps it is untested | experiment: zero-shot held-out-game score before and after PPO fine-tuning |
| [[wang2025gametars#A7]] / S7 | unverified | Game-TARS is imitation-only | experiment: Game-TARS-style agent's unseen-game score before and after PPO fine-tuning |
| [[magne2026nitrogen#A6]] / S7 | unverified | NitroGen is BC-only | experiment: NitroGen's no-fine-tune benchmark before and after PPO post-training |
| [[simateam2024instructable#A3]] / S1 | unverified | SIMA's environments are 3D first- or third-person; 2D games are untested | paper: SIMA 2 (arXiv 2512.04797) or a 2D zero-shot benchmark such as VideoGameBench (arXiv 2505.18134) |
| [[simateam2024instructable#A8]] / S6 | conflicts | SIMA is instructed in language per task; the lens has no language input | experiment: SIMA-style agent with the instruction removed, zero-shot on a held-out game |
| [[simateam2024instructable#A11]] / S3 | unverified | SIMA's pretrained encoders may have been fine-tuned on the held-out game's data | experiment: hold the target game out of every training stage |
| [[simateam2024instructable#A17]] / S4 | unverified | SIMA's model and compute are unreported | paper: a zero-shot game result at a stated, local scale |
| [[wang2025gametars#A11]] / S6 | conflicts | Game-TARS needs a per-game prompt with the keys, rules and sometimes a tutorial | experiment: Game-TARS on its unseen games with the prompt removed |
| [[wang2025gametars#A8]] / S4 | unverified | Game-TARS is a VLM with 2.5B active parameters (plus a 632M encoder) | experiment: distil to a local-size policy and remeasure |
| [[wang2025gametars#A19]] / S3 | unverified | the VizDoom runs, prompts and pausing are not stated | paper: a reproduction with stated protocol |
| [[magne2026nitrogen#A15]] / S3 | conflicts | NitroGen's no-fine-tune benchmark games may appear in its pretraining data | experiment: rerun the benchmark on games verified absent from the data |
| [[magne2026nitrogen#A12]] / S6 | unverified | NitroGen's games are paused during inference; real-time play is untested | experiment: same tasks in real time on the local GPU |
| [[magne2026nitrogen#A18]] / S5 | unverified | the transfer gain depends on how well the target's mechanics and visuals are covered (the authors' untested explanation) | experiment: transfer gain vs similarity of the held-out game to the training mix |
| [[reed2022generalist#A22]] / S3 | unverified | Gato's transfer depends on the target looking like pretraining data; unseen games may not | experiment: zero-shot score vs visual distance (e.g. encoder-feature distance) from the training games |
| [[reed2022generalist#A5]] / S7 | unverified | Gato is BC-only; its Boxing failure may or may not carry to RL-trained policies | experiment: held-out Atari Boxing with an RL-trained multi-game policy |
| [[gupta2022metamorph#A14]] / S1 | conflicts | MetaMorph's zero-shot robots are observed by state, not pixels | paper: an analogous zero-shot result from pixels |
| [[gupta2022metamorph#A12]] / S5 | unverified | unseen robots come from the training design space; whether a genre is as tight a family is open | experiment: zero-shot within-genre vs cross-genre held-out games |

## Open questions and candidate experiments

The gaps above reduce to a few experiments, in rough order of cost. Each is small enough for one
local GPU on Atari or emulated games.

1. **SAM on game frames.** Run SAM or SAM 2 automatic masks on Atari frames and score them
   against OCAtari's RAM objects: recall, splits and merges, and end-to-end FPS on the local GPU
   (encoder, automatic prompting, re-prompting per frame). It closes the SAM-imagery and cost
   rows of (a) and decides whether stage 1 can start from segments.
2. **ORBiT from SAM objects.** Train ORBiT on SAM-derived objects, with clustered mask
   embeddings as types and SAM 2 tracks for velocity, against RAM objects on the same games.
   It tests the two conflicts of (a) on ORCA's own benchmark.
3. **Multi-game ORBiT and leave-one-out.** One agent over several games with an automatic anchor
   rule, then on a held-out game and on HackAtari variants as a genre-shift template. It tests
   stage 3 in the object-set setting.
4. **Multi-game PPO baseline.** Per-game vs one shared pixel transformer with PPO on a small
   Atari set at equal frames, with and without per-game reward normalisation and game-level
   success sampling (SGS-style). It closes the PPO rows of (c).
5. **Frozen encoder, held-out game.** Pretrain CURL or CRPTpro or MAE encoders on frames from
   some games, freeze them, and run PPO on a visually different held-out game; add a DANN game
   classifier as an ablation. It closes the rendering-style and alignment rows of (b).
6. **RL post-training of an imitation generalist.** KL-regularised PPO on NitroGen (needs a
   tractable action head) on two games, measuring held-out zero-shot before and after. It
   tests whether RL keeps or erodes imitation transfer, the S7 rows of (d).

Open questions without an experiment yet:
- What plays the role of ORCA's object type, or MetaMorph's morphology inputs, when objects come
  from an unsupervised segmenter? The KB shows the descriptors matter
  ([[gupta2022metamorph#C5]]) but not what game-agnostic ones would be.
- How many frames does the state need? The KB's evidence points both ways
  ([[chen2021decision#C7]], [[magne2026nitrogen#C8]]).
- Can an agent tell that no training game helps for a target? No method in the reviewed
  literature does [[serrano2024crossdomain#C5]].

## Papers that would close gaps

None fetched. Entries are in `~/code/.supervisor/plans/res/optional-papers.md` unless marked.

- **Zhao et al. 2019** (arXiv 1901.09453): invariant representations under label shift; the
  ([[ganin2016domain#A8]]) row of (b).
- **Kline 2025** (arXiv 2505.17328): a DANN-style game classifier over 10 games; the
  two-domain conflict of (b). Its downstream transfer isn't measured, per the list.
- **OCAtari** (arXiv 2306.08649) and **HackAtari** (arXiv 2505.21731): the object ground truth
  for experiment 1 and the variant benchmark for experiment 3.
- **Yoon et al. 2023**, object-centric pretraining for RL (id unverified): slots from frames,
  the unsupervised alternative to RAM extraction in (a). **Haramati et al. 2024** (id
  unverified), an object-centric transformer from pixels, for the same row.
- **OneLA** (arXiv 2610.20160): latent actions across games, robots and driving with no
  alignment loss; the single-world-model row of (b).
- **Genie** (arXiv 2402.15391) and **LAPA** (arXiv 2410.11758): latent actions from unlabelled
  video at scale, extending AdaWorld's evidence.
- **GATN** (arXiv 2507.03026): a domain-agnostic representation and policy adapter evaluated on
  Atari and MuJoCo; a candidate cross-domain RL baseline.
- **SIMA 2** (arXiv 2512.04797) and **VideoGameBench** (arXiv 2505.18134): zero-shot play beyond
  SIMA's 3D setting, the latter with secret games and real-time play; rows of (d).
- **The 2026 survey of generalist game agents** (arXiv 2605.09965): a map of the field to check
  this topic against.
