---
id: simateam2024instructable
title: Scaling Instructable Agents Across Many Simulated Worlds
authors: [SIMA Team, Maria Abi Raad, Arun Ahuja, Catarina Barros, Frederic Besse, Andrew Bolt, Adrian Bolton, Bethanie Brownfield, Gavin Buttimore, Max Cant, Sarah Chakera, Stephanie C. Y. Chan, Jeff Clune, Adrian Collister, Vikki Copeman, Alex Cullum, Ishita Dasgupta, Dario de Cesare, Julia Di Trapani, Yani Donchev, Emma Dunleavy, Martin Engelcke, Ryan Faulkner, Frankie Garcia, Charles Gbadamosi, Zhitao Gong, Lucy Gonzales, Kshitij Gupta, Karol Gregor, Arne Olav Hallingstad, Tim Harley, Sam Haves, Felix Hill, Ed Hirst, Drew A. Hudson, Jony Hudson, Steph Hughes-Fitt, Danilo J. Rezende, Mimi Jasarevic, Laura Kampis, Rosemary Ke, Thomas Keck, Junkyung Kim, Oscar Knagg, Kavya Kopparapu, Rory Lawton, Andrew Lampinen, Shane Legg, Alexander Lerchner, Marjorie Limont, Yulan Liu, Maria Loks-Thompson, Joseph Marino, Kathryn Martin Cussons, Loic Matthey, Siobhan Mcloughlin, Piermaria Mendolicchio, Hamza Merzic, Anna Mitenkova, Alexandre Moufarek, Valeria Oliveira, Yanko Oliveira, Hannah Openshaw, Renke Pan, Aneesh Pappu, Alex Platonov, Ollie Purkiss, David Reichert, John Reid, Pierre Harvey Richemond, Tyson Roberts, Giles Ruscoe, Jaume Sanchez Elias, Tasha Sandars, Daniel P. Sawyer, Tim Scholtes, Guy Simmons, Daniel Slater, Hubert Soyer, Heiko Strathmann, Peter Stys, Allison C. Tam, Denis Teplyashin, Tayfun Terzi, Davide Vercelli, Bojan Vujatovic, Marcus Wainwright, Jane X. Wang, Zhengdong Wang, Daan Wierstra, Duncan Williams, Nathaniel Wong, Sarah York, Nick Young]
year: 2024
venue: arXiv preprint (Google DeepMind tech report)
url: https://arxiv.org/abs/2404.10179
arxiv: 2404.10179
doi:
pdf_sha256: 86faa74036e338cb7e96589789f77cc1a4d5157f2014009eb922ef80084fb316
tags: [embodied-agents, instruction-following, imitation-learning, multi-game, video-games, keyboard-mouse, language-grounding, transfer-learning]
status: read
read: 2026-11-21
---

# Scaling Instructable Agents Across Many Simulated Worlds

<!-- Read from arXiv v3 (2024-10-11), 31 PDF pages: text pp. 1-21, references pp. 21-28 (skimmed), author contributions pp. 29-31. No appendix. Authors listed alphabetically after "SIMA Team" (p. 1). -->

## TL;DR

SIMA trains one language-conditioned agent by BC on human play in 3D research environments and
games, screen pixels and text in, keyboard and mouse out; it beats per-environment specialists by
67% on average, sometimes matches them zero-shot, and gets 34% on No Man's Sky tasks (human experts 60%).

## Problem

Grounding language in perception and action remains far below human level, and prior agents
worked in one environment or a few, often with privileged state, handcrafted action spaces or
simplified command grammars (Sec. 1, pp. 1-3). SIMA aims at an agent that follows arbitrary
instructions in any simulated 3D world through the interface humans use, on the bet from large
language models that broad training data is what drives generality (p. 2).

## Method

- **Interface** (Sec. 1, p. 3): agents see only the screen a human would (no internal state,
  rewards or privileged information) and act through the same keyboard and mouse; games run in
  real time, asynchronously, and each instance needs a GPU, so massive parallel actors are not
  possible. The same interface everywhere means no per-game observation or action design.
- **Environments** (Sec. 3.1, pp. 6-8, Fig. 2): first-person or over-the-shoulder 3D worlds.
  Commercial: Goat Simulator 3, Hydroneer, No Man's Sky, Satisfactory, Teardown, Valheim, Wobbly
  Life; research: Construction Lab (new), Playhouse, ProcTHOR (primarily for data collection), WorldLab.
  Games run on Google Cloud with rendering to a virtual display, and scripted loading of
  save-games from expert play makes scenarios repeatable (p. 7). Results focus on single-player play.
- **Data** (Sec. 3.2, pp. 9-10, Fig. 3): human gameplay with video, instructions and dialogue,
  actions and annotations: free single-player play annotated with instructions post-hoc, and
  two-player setter-solver sessions where one player instructs another on a shared view. Offline
  resizing, heuristic filtering of low-quality data, and remixing and weighting across
  environments and collections (no ratios given). Instructions cluster into a hierarchy (navigate,
  object management, build, combat, menu use, …) shared across environments.
- **Agent** (Sec. 3.3, pp. 10-11, Fig. 4): pretrained SPARC image-text encoder and Phenaki video
  model, fine-tuned on SIMA data by behavioral cloning and video prediction respectively, plus a
  text encoder; trained-from-scratch transformers cross-attend to these and the instruction, a
  Transformer-XL attends to past memory states, and the resulting state feeds a policy that emits
  keyboard-and-mouse actions for sequences of 8 actions. Losses: behavioral cloning plus an
  auxiliary goal-completion prediction. Tasks are short, about 10 s or less.
- **Classifier-free guidance** (Sec. 3.3, p. 11): π_CFG = π(image, language) + λ(π(image,
  language) − π(image, ·)), shifting logits away from the language-free policy.
- **Latency** (Sec. 3.4.1, p. 13): BC targets are actions offset in time from the visual input,
  and evaluation buffers observations and actions to match; TPU scheduling and on-device state
  caching reduce latency.
- **Evaluation** (Sec. 3.4, pp. 11-13): ground truth in the research environments (with
  distractor objects that fail the episode, and goal switching mid-episode); OCR of on-screen
  completion text in No Man's Sky and Valheim; human judges (game experts with at least 16 h of
  play, typically 5 per video, failing any episode with irrelevant actions first). 1,485 unique
  tasks in 9 skill categories across 7 environments, from a verb list.
- **Comparisons** (Sec. 4.2, pp. 16-17): main SIMA agent (all environments except Hydroneer and
  Wobbly Life, which are kept for qualitative zero-shot tests); zero-shot agents trained on N − 1
  environments; no-pretraining (a from-scratch ResNet replaces SPARC/Phenaki); no-language; and
  environment-specialized agents (one environment's data, same pretrained encoders), which
  normalize scores to 100%. Ablations: one seed each, evaluated at 1.2M training steps.

## Assumptions

- A1: The agent uses a human-compatible interface: it sees only the screen a human player would, with no internal game state, rewards or other privileged information, and acts through the same keyboard-and-mouse controls (Sec. 1, p. 3, Fig. 4, p. 10)
- A2: All games share one keyboard-and-mouse action encoding, control rate and observation resolution; the paper gives none of them, so a new game whose controls or timing don't fit that encoding falls outside the setup (reader's inference, Sec. 3.2, p. 10, Sec. 3.3, pp. 10-11)
- A3: Environments are 3D worlds seen first-person or over the shoulder, so basic embodied interactions such as navigation are shared, and results cover single-player play only (Sec. 2, p. 4, Sec. 3.1, pp. 6-8, Fig. 2, p. 8)
- A4: Environments run in real time, asynchronously to the agent; latency is handled by training BC on actions offset in time from the visual input and mirroring the offset with buffering at evaluation, plus TPU scheduling and on-device state caching (Sec. 1, p. 3, Sec. 3.4.1, p. 13)
- A5: Training data are contracted human gameplay with recorded keyboard-and-mouse actions: free single-player play annotated with instructions post hoc, and two-player setter-solver sessions on a shared view (Sec. 3.2, pp. 9-10)
- A6: Data are resized offline, filtered by heuristics for low quality, and remixed and weighted across environments and collections, with no ratios given (Sec. 3.2, p. 10)
- A7: Supervision is behavioural cloning plus an auxiliary goal-completion prediction loss; no RL or reward signal is used (Sec. 3.2, p. 9, Sec. 3.3, p. 11)
- A8: Instructions are open-ended natural language for short tasks of about 10 s or less; longer tasks must be broken into a sequence of instructions by the user (Sec. 1, p. 3, Sec. 3.3, p. 10)
- A9: The agent combines pretrained SPARC image-text and Phenaki video models, fine-tuned on SIMA data by BC and video prediction, with a text encoder, from-scratch transformers that cross-attend to them, a Transformer-XL memory, and a policy that emits sequences of 8 keyboard-and-mouse actions (Sec. 3.3, pp. 10-11, Fig. 4)
- A10: Classifier-free guidance is applied at inference, shifting logits away from the language-free policy with a weight lambda whose value isn't given (Sec. 3.3, p. 11)
- A11: The pretrained encoders were fine-tuned on data from a subset of the environments, which can include a zero-shot agent's held-out environment, but not Goat Simulator 3 (Sec. 4.2, p. 16)
- A12: Success is judged by ground truth in the research environments (distractor interactions fail the episode, goals can switch mid-episode), by OCR of on-screen completion text in No Man's Sky and Valheim, and otherwise by game-expert human judges (at least 16 h of play, typically 5 per video) who fail an episode with irrelevant actions first (Sec. 3.4, pp. 12-13)
- A13: The evaluation set is 1,485 tasks in 9 skill categories over 7 of the more than 10 environments (limited data and evaluations for the rest), built from a list of common English verbs plus verbs from gameplay, and started from save-games of expert play (Sec. 3.4, p. 13, Sec. 3.1, p. 7, p. 9)
- A14: Main-agent results average 3 seeds, multiple episodes per task in research environments and one per task in video games, with 95% CIs over tasks and seeds (Sec. 4.1, p. 15)
- A15: Comparison agents run one seed each and are evaluated at 1.2M training steps, except the Goat Simulator 3 specialist, picked as the best 200k-step checkpoint (400k steps) (Sec. 4.2, p. 17)
- A16: Scores in the comparisons are normalised per environment to an environment-specialised agent trained only on that environment's data with the same pretrained encoders (Sec. 4.2, pp. 16-17)
- A17: Dataset size, model size and training compute are not reported, so the results are tied to an unstated scale (reader's inference, Sec. 3.2-3.3, pp. 9-11)
- A18: Zero-shot transfer relies on held-out games sharing generic skills and conventions with the training games, such as consistent object colours, left mouse to grab or interact, and common navigation (Sec. 4.2, p. 19, Fig. 2, p. 8)
- A19: WorldLab shares an engine and some implementation details with the other internal research environments, and part of its data comes from earlier versions with different dynamics and task distributions (Sec. 4.2, p. 19)
- A20: Core premise: a language-first agent with a common human interface needs no per-game observation or action design, so new environments are easy to add and learned skills can transfer zero-shot, with language and grounded learning mutually reinforcing (Sec. 1, p. 3, Sec. 2, pp. 5-6)
- A21: Evaluation tasks are designed so the instruction is needed, ideally testing several instructions from one initial state rather than letting the scene's affordances reveal the goal (Sec. 3.4, p. 11)
- A22: The no-pretraining ablation replaces SPARC and Phenaki with a from-scratch ResNet, because training those encoders through agent training performed poorly in preliminary experiments (Sec. 4.2, p. 16)
- A23: The human baseline is players from the data collection with game experience, on a separate No Man's Sky task subset, scored by the same judges who weren't told they were rating humans (Sec. 4.3, p. 20)
- A24: The classifier-free guidance ablation is evaluated only on Construction Lab, Playhouse and WorldLab (Fig. 10, Sec. 4.2, p. 19)
- A25: Each game runs on Google Cloud with accelerated rendering to a virtual display, one GPU per commercial-game instance, and scripted save-game loading for repeatable scenarios, with developer partnerships for the commercial games (Sec. 1, p. 3, Sec. 3.1, pp. 6-7)

## Claims

- C1: SIMA completes a range of instructed tasks in all 7 evaluated environments but far from perfectly; success is higher in the simpler Playhouse and WorldLab, lower in commercial games, and also lower in the new Construction Lab [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A13, A14, A17, A25] (Fig. 6, Sec. 4.1, pp. 15-16)
- C2: Averages are over multiple episodes per task (one per task in video games), multiple tasks per environment and 3 seeds, with 95% CIs across tasks and seeds [] (Sec. 4.1, p. 15)
- C3: Success varies across skill categories and within clusters; skills needing precise actions or spatial understanding ("combat", "use tools", "build") tend to be harder [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A13, A14, A17, A25] (Fig. 7, Sec. 4.1, pp. 15-17)
- C4: SIMA outperforms environment-specialized agents overall by 67% on average relative to specialist performance, which the authors take as positive transfer across environments [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A13, A14, A15, A16, A17, A20, A25] (Fig. 8, Sec. 4.2, p. 17)
- C5: Per environment, a permutation test on mean per-task differences finds SIMA significantly better than the specialist in every one of the 7 environments (p = 0.001, 0.002, 0.036, 0.0002, 0.008, 0.004, 0.0002) [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A13, A14, A15, A16, A17, A25] (Sec. 4.2, p. 17, Fig. 9, p. 18)
- C6: SIMA substantially outperforms the no-pretraining ablation overall (p < 0.001), but per environment the benefit varies in size and significance (p = 0.0002, 0.14, 0.041, 0.0002, 0.244, 0.052, 0.032) [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A13, A14, A15, A16, A17, A22, A25] (Sec. 4.2, p. 17)
- C7: The no-language ablation performs very poorly (all permutation tests p < 0.001), showing that the agent uses language and that the tasks are not solvable by plausible language-free behavior [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A13, A14, A15, A16, A17, A21, A25] (Sec. 4.2, p. 17, Fig. 8, p. 18)
- C8: Agents trained on N − 1 environments and tested zero-shot on the held-out one reach non-trivial performance, almost always beat the no-language ablation, and in some cases match or exceed the specialist; the aggregate zero-shot bar sits a little under the specialist's 100% (read off the chart) [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A15, A16, A17, A18, A20, A25] (Fig. 8, Fig. 9, p. 18, Sec. 4.2, p. 19)
- C9: Zero-shot agents do well on generic skills shared across games (navigation such as "go down the hill", grabbing objects by color, left mouse to interact) but fall short on environment-specific skills; at similar scores, zero-shot and specialist agents succeed on different skills [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A15, A16, A17, A18, A25] (Sec. 4.2, p. 19)
- C10: On Goat Simulator 3, whose data never fine-tuned the pretrained encoders, the zero-shot agent performs comparably to the specialist, so transfer is not driven by the visual components alone [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A15, A16, A17, A18, A25] (Sec. 4.2, p. 16, p. 19)
- C11: Zero-shot is especially strong on WorldLab, which the authors attribute to its tasks' larger share of domain-general skills, its engine and implementation shared with the other internal environments, and a train-test distribution shift that weakens the specialist [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A13, A15, A16, A17, A18, A19, A25] (Sec. 4.2, p. 19)
- C12: On Construction Lab, Playhouse and WorldLab, removing classifier-free guidance (λ = 0) makes the agent noticeably worse, but without CFG it still clearly beats the no-language ablation [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A13, A15, A16, A17, A24, A25] (Fig. 10, Sec. 4.2, p. 19)
- C13: On a separate No Man's Sky task set, expert human players succeed on only 60% of tasks under the same judges and criteria, SIMA 34% and the no-language baseline 11%; full success may be unattainable because judges disagree on ambiguous tasks [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A12, A17, A23, A25] (Fig. 11, Sec. 4.3, p. 20)
- C14: Offline action log-probabilities on held-out data correlate at most weakly with agent performance beyond the most basic skills, and static-image evaluations don't reliably predict success on prolonged tasks, so online evaluation is needed [A5, A7, A12] (Sec. 3.4, pp. 11-12)
- C15: In preliminary experiments the pretrained SPARC and Phenaki models offered complementary benefits (Sec. 3.3, p. 11), and training the SPARC/Phenaki encoders through agent training performed poorly [A9, A22] (Sec. 4.2, p. 16)
- C16: The agent can reach instructed targets that are not in view at the start, e.g. "go to the spaceship" in No Man's Sky and "go to the HUB" in Satisfactory [A1, A3, A8, A9] (Fig. 5, p. 14, Sec. 4, p. 15)

## Implementation notes

- Model sizes, encoder dimensions, number of layers, image resolution, frame rate, action
  discretization and the exact keyboard-and-mouse action format are not given (source: the paper
  is silent; Sec. 3.3 is high level).
- "Sequences of 8 actions" per policy step (source: Sec. 3.3, p. 11); whether all 8 are executed
  open-loop before re-planning is not said (source: assumption, unresolved).
- The CFG weight λ used in the main results is not given; only λ = 0 for the ablation (source:
  Sec. 4.2, p. 19).
- Data mixture weights across environments and collection types, filtering heuristics and
  dataset size in hours are not given (source: Sec. 3.2, p. 10, silent).
- The p-values in C5 and C6 are given "respectively" for the environments without naming the
  order; Fig. 9's order (Construction Lab, Goat Simulator 3, No Man's Sky, Playhouse,
  Satisfactory, Valheim, WorldLab) is the likely one (source: assumption from Fig. 9, p. 18).
- Specialist baseline for Goat Simulator 3: evaluated every 200k steps and the best (400k)
  kept, which can only understate SIMA's advantage (source: footnote 4, p. 17).
- Per-bar values in Fig. 6-11 other than those quoted in the text are not printed; reading them
  off the charts gives only rough numbers (source: Fig. 6-11, pp. 16-20).

## Limitations / open questions

- A work in progress: many skills and tasks are out of reach; tasks are short-horizon
  (about 10 s); evaluation results cover only 7 of the 10+ environments because of limited data
  and evaluations (Sec. 5, p. 20, Sec. 3.3, p. 10, footnote 1, p. 9).
- OCR evaluation is limited to tasks that games signal with on-screen text, and human
  evaluation is slow and costly (Sec. 3.4, p. 12).
- Ablations ran with one seed each, unlike the main agent's 3 (Sec. 4.2, p. 17).
- Zero-shot results may be confounded: the pretrained encoders were fine-tuned on data that can
  include the held-out environment, except for Goat Simulator 3 (Sec. 4.2, p. 16).
- Not released; deployment kept in a controlled, closed environment (Sec. 3.5, p. 15).
- Open question: the architecture and data details needed to reproduce or scale the result
  (model size, data volume per environment, mixture weights) are not in the report, so its
  scaling claims are about the number of environments, not about model or data size.
- Open question: the keyboard-and-mouse action space is shared across games, so it avoids
  per-game action heads; how the action tokens are structured is not described.

## Related

- related [[lee2022multigame]]: also one model across many games, but Atari-only with a shared 18-action space, offline RL-style return conditioning instead of language instructions.
- related [[reed2022generalist]]: also one set of weights across many environments, but Gato mixes modalities and control tasks and takes no free-form language instructions; SIMA fixes one pixels-and-text in, keyboard-and-mouse out interface.
- related [[baker2022video]]: also behaviour-clones a transformer on human keyboard-and-mouse play from pixels, but in Minecraft only and from IDM-labelled video; SIMA uses recorded human play with language labels across many games.
- related [[serrano2024crossdomain]]: survey of cross-domain RL transfer; SIMA sidesteps state/action mapping by giving every environment a human-like pixels-in, keyboard-and-mouse-out interface.
- related [[vaswani2017attention]]: SIMA's state and policy are built from transformers (cross-attention plus Transformer-XL memory).

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
