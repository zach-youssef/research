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

## Claims

- C1: SIMA completes a range of instructed tasks in all 7 evaluated environments but far from perfectly; success is higher in the simpler Playhouse and WorldLab, lower in commercial games, and also lower in the new Construction Lab (Fig. 6, Sec. 4.1, pp. 15-16)
- C2: Averages are over multiple episodes per task (one per task in video games), multiple tasks per environment and 3 seeds, with 95% CIs across tasks and seeds (Sec. 4.1, p. 15)
- C3: Success varies across skill categories and within clusters; skills needing precise actions or spatial understanding ("combat", "use tools", "build") tend to be harder (Fig. 7, Sec. 4.1, pp. 15-17)
- C4: SIMA outperforms environment-specialized agents overall by 67% on average relative to specialist performance, which the authors take as positive transfer across environments (Fig. 8, Sec. 4.2, p. 17)
- C5: Per environment, a permutation test on mean per-task differences finds SIMA significantly better than the specialist in every one of the 7 environments (p = 0.001, 0.002, 0.036, 0.0002, 0.008, 0.004, 0.0002) (Sec. 4.2, p. 17, Fig. 9, p. 18)
- C6: SIMA substantially outperforms the no-pretraining ablation overall (p < 0.001), but per environment the benefit varies in size and significance (p = 0.0002, 0.14, 0.041, 0.0002, 0.244, 0.052, 0.032) (Sec. 4.2, p. 17)
- C7: The no-language ablation performs very poorly (all permutation tests p < 0.001), showing that the agent uses language and that the tasks are not solvable by plausible language-free behavior (Sec. 4.2, p. 17, Fig. 8, p. 18)
- C8: Agents trained on N − 1 environments and tested zero-shot on the held-out one reach non-trivial performance, almost always beat the no-language ablation, and in some cases match or exceed the specialist; the aggregate zero-shot bar sits a little under the specialist's 100% (read off the chart) (Fig. 8, Fig. 9, p. 18, Sec. 4.2, p. 19)
- C9: Zero-shot agents do well on generic skills shared across games (navigation such as "go down the hill", grabbing objects by color, left mouse to interact) but fall short on environment-specific skills; at similar scores, zero-shot and specialist agents succeed on different skills (Sec. 4.2, p. 19)
- C10: On Goat Simulator 3, whose data never fine-tuned the pretrained encoders, the zero-shot agent performs comparably to the specialist, so transfer is not driven by the visual components alone (Sec. 4.2, p. 16, p. 19)
- C11: Zero-shot is especially strong on WorldLab, which the authors attribute to its tasks' larger share of domain-general skills, its engine and implementation shared with the other internal environments, and a train-test distribution shift that weakens the specialist (Sec. 4.2, p. 19)
- C12: On Construction Lab, Playhouse and WorldLab, removing classifier-free guidance (λ = 0) makes the agent noticeably worse, but without CFG it still clearly beats the no-language ablation (Fig. 10, Sec. 4.2, p. 19)
- C13: On a separate No Man's Sky task set, expert human players succeed on only 60% of tasks under the same judges and criteria, SIMA 34% and the no-language baseline 11%; full success may be unattainable because judges disagree on ambiguous tasks (Fig. 11, Sec. 4.3, p. 20)
- C14: Offline action log-probabilities on held-out data correlate at most weakly with agent performance beyond the most basic skills, and static-image evaluations don't reliably predict success on prolonged tasks, so online evaluation is needed (Sec. 3.4, pp. 11-12)
- C15: In preliminary experiments the pretrained SPARC and Phenaki models offered complementary benefits (Sec. 3.3, p. 11), and training the SPARC/Phenaki encoders through agent training performed poorly (Sec. 4.2, p. 16)
- C16: The agent can reach instructed targets that are not in view at the start, e.g. "go to the spaceship" in No Man's Sky and "go to the HUB" in Satisfactory (Fig. 5, p. 14, Sec. 4, p. 15)

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
