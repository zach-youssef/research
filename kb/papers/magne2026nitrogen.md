---
id: magne2026nitrogen
title: "NitroGen: An Open Foundation Model for Generalist Gaming Agents"
authors: [Loïc Magne, Anas Awadalla, Guanzhi Wang, Yinzhen Xu, Joshua Belofsky, Fengyuan Hu, Joohwan Kim, Ludwig Schmidt, Georgia Gkioxari, Jan Kautz, Yisong Yue, Yejin Choi, Yuke Zhu, Linxi "Jim" Fan]
year: 2026
venue: CVPR 2026 conference paper (read as arXiv v1)
url: https://arxiv.org/abs/2601.02427
arxiv: 2601.02427
doi:
pdf_sha256: c652f6da2880821f3780c7153f89aa416f4fa66ee337466cc0bd87226ea2dea0
tags: [generalist-agent, behavioral-cloning, imitation-learning, multi-game, video-games, gamepad, action-space, action-labeling, flow-matching, transfer-learning, pretraining]
status: read
read: 2026-11-21
---

# NitroGen: An Open Foundation Model for Generalist Gaming Agents

## TL;DR

Labels 40k h of web gameplay video (1,000+ games) by parsing the on-screen gamepad overlays some
creators show, then behaviour-clones a 500M SigLIP 2 + flow-matching DiT policy on one frame to a
shared 16-button + 2-stick gamepad space; fine-tuning on a held-out game gains 5-52% relative.

## Problem

Embodied agents lack large, diverse, action-labelled datasets: LLM agents need hand-built game
APIs or perception modules, RL agents are narrow and need dedicated simulators, and pixel BC has
relied on costly demonstrations for a few titles, e.g. VPT's Minecraft-only labels (Sec. 1, p. 1;
Sec. 5, p. 10). There was also no open framework to train and evaluate one agent across many
commercial games (p. 1). The concurrent Game-TARS (arXiv 2510.23691) is cited as using ~20k h
of contractor and reasoning data instead (Sec. 5, p. 10).

## Method

- **Label source.** Some streamers run "input overlay" software that draws the gamepad with the
  pressed buttons in a screen corner; 71,000 h of such video were collected by keyword search plus
  curation for diversity across games, genres and skill levels (Sec. 2.1, p. 4; Fig. 2a, p. 3).
- **Action extraction, three stages** (Sec. 2.1, pp. 4-5; Fig. 2b, p. 3):
  1. template matching: 25 frames per video, SIFT and XFeat keypoints against ~300 controller
     templates, an affine fit needing at least 20 inliers; the best match gives the crop;
  2. parsing: a fine-tuned SegFormer takes two consecutive crops (concatenated spatially) and
     outputs a segmentation mask placing each stick on an 11 × 11 grid plus binary button states;
     trained on 8M synthetic frames made by drawing three overlay tools' templates (random states,
     opacity, size, compression) over dataset frames. At inference, stick contours are tracked over
     the video, the centre is the mean of frames classified as centred, and positions are scaled
     to [-1, 1] by the 99th percentile of |x| and |y| over the video;
  3. filtering: keep only chunks where at least 50% of timesteps have a non-zero action, and mask
     the overlay in every frame so the policy can't read it as a shortcut (p. 5).
- **Shared interface.** Observation: one RGB frame. Action: a 16-dim binary vector (4 d-pad, 4
  face, 2 shoulders, 2 triggers, 2 stick clicks, start, back) plus a 4-dim continuous vector for
  the two sticks, the same for every game (Sec. 2.2, p. 6).
- **Universal simulator.** A library that intercepts the game's system clock so any title that
  uses it for physics can be stepped frame by frame behind a Gymnasium API, without game code
  changes; real-time/asynchronous play is left for later (Sec. 2.2, pp. 5-6; App. B.1, p. 16).
- **Benchmark.** 10 commercial games (2D: three side-scrollers, two procedurally generated
  top-down roguelikes; 3D: two open-world, two combat action-RPGs, one sports), 30 tasks (11
  combat, 10 navigation, 9 game-specific), each with fixed start and goal states chosen so the
  first frame is enough context; success is judged by humans (Sec. 2.2, pp. 6-7).
- **Model.** GR00T N1 with the language and state encoders removed and one action head: SigLIP 2
  encodes a 256 × 256 frame into 256 tokens; a DiT (alternating self-attention over action tokens
  and cross-attention to frame tokens) denoises a 16-step action chunk, one token per timestep,
  decoded per timestep by an MLP (Sec. 2.3, p. 7). Trained with conditional flow matching,
  `a_t = (1-t)ε + t·a`, loss `‖π(a_t, ψ(o), t) - (a - ε)‖²`, t from a shifted beta favouring small
  t; inference is 16 Euler steps from Gaussian noise (Eq. 1, Eq. 2, App. A, p. 16).
- **Transfer protocol.** Pre-train on everything but one game, fine-tune on that game with limited
  data, compare against the same architecture trained from scratch with the same data and compute
  (Sec. 3, p. 9; Fig. 7, p. 9).

## Assumptions

- A1: Action labels come only from web gameplay videos that show a gamepad input overlay, matched against about 300 controller templates; videos without an overlay, or with keyboard and mouse input, are not used (Sec. 2.1, pp. 4-5).
- A2: The overlay parser is trained only on synthetic frames drawn with three overlay tools' templates (Open Joystick Display, Input Overlay, GamePad Viewer) over dataset frames, with random states, opacity, size and compression (Sec. 2.1, p. 5).
- A3: Every game is controlled through one shared gamepad layout: a 16-dim binary button vector plus a 4-dim continuous vector for two analog sticks, with the same physical button or stick per slot in every game (Sec. 2.2, p. 6).
- A4: Stick positions are normalised per video to [-1, 1] by the 99th percentile of |x| and |y|, around a centre estimated from frames classified as centred, so stick magnitude is relative to each video rather than absolute (Sec. 2.1, p. 5).
- A5: The observation is one 256 × 256 RGB frame, with the overlay masked out; there is no game-state, reward or language input (Sec. 2.1, p. 5, Sec. 2.2, p. 6, Sec. 2.3, p. 7).
- A6: Supervision is behaviour cloning of human play of mixed skill (casual and competitive) only; there is no reward and no RL (Sec. 2.1, p. 4, Sec. 2.3, p. 7, Sec. 4, p. 10).
- A7: Training keeps only chunks in which at least 50% of timesteps have a non-zero button or stick action, 55% of the data (Sec. 2.1, p. 5).
- A8: Training data is 40,000 h over more than 1,000 games, skewed toward gamepad-played action games (Action-RPG 34.9%, Platformer 18.4%, Action-Adventure 9.2% of hours), with keyboard-only, strategy and simulation games under-represented (Sec. 2.1, Fig. 3, p. 4, Sec. 4, p. 10).
- A9: Labels are noisy: overlay software delays, parsing errors, creator artifacts in frames, and per-player sensitivity settings and button remappings that change what an input means (Sec. 3, p. 9).
- A10: One action label per source-video frame, so the policy's timestep and action timing follow the source videos' frame rate, which the paper doesn't state (reader's inference, Sec. 2.1, p. 5).
- A11: The policy is GR00T N1 without language and state encoders: a SigLIP 2 encoder (256 tokens per frame) and a flow-matching DiT that outputs 16-step action chunks, sampled with 16 Euler steps (Sec. 2.3, p. 7, App. A, p. 16).
- A12: Games are run synchronously by a library that intercepts the system clock and pauses the game while the model infers; this works only for titles that use the system clock for physics and interactions, and real-time play is not tested (Sec. 2.2, p. 5, App. B.1, p. 16).
- A13: Benchmark tasks have fixed start and goal states and were chosen so that the first frame gives enough context to elicit the behaviour, with no language or goal specification (Sec. 2.2, p. 7).
- A14: Task success is judged by humans, on 3 tasks per game and 5 rollouts per task, with no spread reported (Sec. 2.2, p. 7, Fig. 6, p. 8).
- A15: Some benchmark games, or their fixed layouts, may appear in the pre-training data (reader's inference, Sec. 3, p. 8).
- A16: The transfer study pre-trains on all data except one held-out game and compares fine-tuning with the same architecture trained from scratch on the same data and compute (Sec. 3, Fig. 7, p. 9).
- A17: The held-out game's fine-tuning data is that game's share of the same overlay-labelled web video (reader's inference, Sec. 3, p. 9).
- A18: The transfer gain depends on how well the target game's mechanics and visual style are represented in the pre-training data (the authors' explanation, untested) (Sec. 3, p. 9).
- A19: Cross-game transfer of a button or stick relies on shared control conventions across games (e.g. left stick moves the character), since nothing in the action space encodes per-game semantics (reader's inference, Sec. 2.2, p. 6, Sec. 3, p. 9).
- A20: The extraction benchmark is six games recorded with OBS and an input recorder, with randomised overlay opacity, size and controller type over Xbox and PlayStation families (Sec. 3, p. 8, Fig. 5, p. 7).
- A21: The pause test replays 5-min human recordings from several games, in segments expected to be deterministic (no enemy randomness) (App. B.1, p. 16).

## Claims

- C1: The collected overlay video totals 71,000 h; after filtering, the dataset is 40,000 h over more than 1,000 games, 38,739 videos from 818 creators, with a mean video length of 1 h 50 min, which the authors call the largest labelled video-action dataset for games to date [A1, A7, A8] (Sec. 2.1, Fig. 3, p. 4).
- C2: Coverage is uneven: 846 games have over 1 h, 91 over 100 h and 15 over 1,000 h (233 games have under 1 h); by hours, Action-RPG is 34.9%, Platformer 18.4% and Action-Adventure 9.2%, the rest spread over seven genres [A1, A7, A8] (Fig. 3, p. 4).
- C3: On a ground-truth benchmark recorded with OBS from six games, with randomised overlay opacity, size and controller type, the extraction pipeline reaches an average joystick R² of 0.84 (per family 0.77 PS5 to 0.92 Xbox One) and an average per-frame button accuracy of 0.96 (0.91 PS4 to 0.98 Xbox One and Series X), "across the most popular controller families" [A1, A2, A20] (Fig. 5, p. 7, Sec. 3, p. 8).
- C4: Estimating stick positions as segmentation masks on an 11 × 11 grid works significantly better than regressing stick coordinates directly; no numbers are given [A1, A2] (Sec. 2.1, p. 5).
- C5: Training on the raw 71,000 h makes the model over-predict the null action; keeping only chunks where at least 50% of timesteps have a non-zero button or stick action fixes this and keeps 55% of the data [A1, A3, A5, A6, A7, A11] (Sec. 2.1, p. 5).
- C6: The labels are noisy: overlay software adds small delays and parsing adds errors, frames carry creator artifacts (chats, subscribe prompts, progress trackers), and player sensitivity settings and custom button mappings change what an input means [A1, A4, A9] (Sec. 3, p. 9).
- C7: Pausing and resuming games through the clock-intercepting simulator doesn't change their behaviour, tested on 5-min human recordings from several games in parts expected to be deterministic (e.g. no enemy randomness): replays of recorded human actions from the same start diverge visually after about 1 min for games with continuous actions and about 3 min for discrete-only games, the same with and without random high-frequency pauses, which the authors attribute to error accumulation [A12, A21] (App. B.1, p. 16).
- C8: Conditioning on more than one past frame gave no benefit, even with larger temporal gaps, and generating 16-action chunks improved temporal consistency over single-action generation; the authors suggest the first frame of these action-game tasks already gives enough context [A3, A5, A6, A8, A10, A11, A13] (Sec. 2.3, p. 7).
- C9: More than 16 denoising steps give no measurable improvement [A11] (App. A.2, p. 16).
- C10: EMA weights (decay 0.9999) consistently outperform non-EMA weights; all results use EMA weights [A11] (Sec. 2.3, p. 8).
- C11: The 500M model, without fine-tuning, reaches average task completion (3 tasks per game, 5 rollouts per task) of 61.2% combat, 55.0% navigation and 56.3% game-specific on 3D games, 46.0%, 52.0% and 61.5% on 2D top-down games, and 44.8%, 37.9% and 54.0% on 2D side-scrollers; no spread is reported [A1, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15] (Fig. 6, p. 8).
- C12: The authors find no significant difference between games with fixed layouts the model may have partly seen in training and procedurally generated ones, and read this as the model both memorising and adapting; no per-category numbers are given [A1, A3, A5, A6, A7, A8, A11, A12, A13, A14, A15] (Sec. 3, p. 8).
- C13: Fine-tuned on a held-out isometric roguelike, the pre-trained model beats from-scratch training at every data size: 53.0% vs 48.1% (60 h), 65.6% vs 57.8% (120 h) and 81.0% vs 76.0% (240 h), about 10% relative on average, and both scale with data [A1, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A16, A17, A18, A19] (Fig. 7a, p. 9).
- C14: On a held-out 3D action-RPG with 30 h of fine-tuning data, the gain depends on the task type: combat 73.3% vs 48.3% (52% relative, the abstract's "up to 52%"), navigation 60.0% vs 48.0% (25%), game-specific 66.6% vs 63.3% (5%) [A1, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A16, A17, A18, A19] (Fig. 7b, p. 9, Sec. 3, p. 9).
- C15: Across data quantities the authors report a 25% average relative gain on the 3D action-RPG vs 10% on the roguelike, and attribute it to action-RPGs being better represented in the pre-training data than the roguelike's mechanics and visual style; this is their explanation, not tested [A8, A14, A16, A17, A18] (Sec. 3, p. 9).
- C16: The authors conclude that pre-training transfers skills for common gameplay patterns (combat, navigation), while game-specific mechanics still need training on the new game [A3, A8, A14, A16, A17, A18, A19] (Sec. 3, p. 9).
- C17: NitroGen is a reactive "system-1" model: it can't plan over long horizons or follow language and only reacts to its short context; the authors expect language and RL post-training to add these [A5, A6, A11] (Sec. 4, p. 10).
- C18: The dataset is biased toward action games played with a gamepad; keyboard-only games and complex manipulation are under-represented, which may limit transfer to strategy or simulation genres [A1, A3, A8] (Sec. 4, Fig. 3, p. 10).

## Implementation notes

- Policy optimiser: AdamW, weight decay 0.001, warmup-stable-decay schedule with a constant 1e-4
  phase, EMA 0.9999 (source: Sec. 2.3, p. 8).
- Augmentations: random brightness, contrast, saturation and hue, rotation in [-5°, 5°], random
  crops (source: Sec. 2.3, p. 8).
- Annotation SegFormer: AdamW, lr 1e-4 with linear decay, weight decay 0.1, batch 256, 8M
  synthetic frames (source: Sec. 2.1, p. 5).
- The action chunk is written `a ∈ R^{16×24}` (16 timesteps, 24 dims), while Sec. 2.2 lists
  16 + 4 = 20 dims; the extra 4 aren't explained (source: App. A.1, p. 16 vs Sec. 2.2, p. 6).
- Not stated: the action/frame rate (timestep length), how many of the 16 chunk actions run
  before replanning, DiT depth and width, the split of the 500M parameters, batch size, training
  steps and compute, the shifted-beta parameters (source: the paper is silent; GR00T N1 and π0
  hold the beta schedule, per App. A.1, p. 16).
- Not stated: the fine-tuning recipe (steps, learning rate, frozen parts) and the held-out games'
  titles (source: Sec. 3, p. 9 is silent).
- The 10 benchmark titles aren't listed in the text; Fig. 1 and Fig. 4 show frames from titles
  such as The Witcher 3, Cyberpunk 2077, Rocket League, Cuphead and The Binding of Isaac (source:
  Fig. 1, p. 2, Fig. 4, p. 6; the mapping to the benchmark is the reader's guess, an assumption).
- Whether the 10 benchmark games are in the pre-training data isn't stated; C12's wording implies
  at least some are (source: Sec. 3, p. 8; assumption).
- The released checkpoint differs from the paper's stated shape: its config has `action_horizon` 18
  and `action_dim` 25 (21 buttons from `BUTTON_ACTION_TOKENS`, plus 4 stick dims), 16 Euler steps
  (`num_inference_timesteps`), vision encoder `google/siglip2-large-patch16-256`, 493.6M parameters
  (source: released ng.pt `ckpt_config` @ nvidia/NitroGen 584c8dd; NitroGen 3260844
  `nitrogen/shared.py`; experiment 2026-11-24-nitrogen-mac-smoke). The paper says 16-step chunks
  over 16 buttons + 2 sticks (Sec. 2.2-2.3), and App. A.1 writes `a ∈ R^{16×24}`.
- The authors release the dataset, the evaluation suite and the weights at
  https://nitrogen.minedojo.org (source: p. 1; not fetched).

## Limitations / open questions

- Reactive only, no planning or language (C17); data skewed to gamepad action games (C18).
- Success is judged by humans from 5 rollouts per task, with no variance or confidence intervals,
  so small gaps (e.g. 66.6% vs 63.3% in C14) may be within noise (Fig. 6, p. 8; Fig. 7, p. 9).
- The transfer study covers two held-out games; the 3D action-RPG's data-quantity sweep behind
  the 25% average (C15) isn't shown (Fig. 7, p. 9).
- No comparison with other generalist agents or with VPT-style IDM labelling on the same data
  (Sec. 3, pp. 8-9).
- Synchronous play only: the game is paused while the model infers (Sec. 2.2, p. 5; App. B.1).
- Open question: does the released model (18-step chunks, 25 dims) differ from the paper's (16 steps,
  16 buttons + 2 sticks, or 16 x 24 in App. A.1), or is the paper's description simplified?
  `play.py` executes all 18 actions of a chunk (source: MineDojo/NitroGen 3260844,
  scripts/play.py; action_downsample_ratio 1 in nitrogen/inference_session.py).
- Open question: one frame of context means no memory; how does this fail on tasks where the
  current frame is ambiguous (partial observability), which the benchmark selected against (p. 7)?
- Open question: the same button means different things in different games and player mappings
  (C6); how much of the cross-game transfer comes from shared conventions (stick = move) that a
  learned, game-agnostic action embedding would have to rediscover?
- Open question (survey): the shared gamepad space plus a frozen general image encoder (SigLIP 2)
  is a baseline for the survey's cross-domain embedding; would a state embedding learned without
  labels from N frames beat it, and does PPO fine-tuning from this BC policy help where BC is weak
  (game-specific tasks, C14)?

## Related

- related [[baker2022video]]: VPT labels Minecraft video with an IDM trained on contractor data; NitroGen reads labels off overlays instead and cites VPT's null-action problem for its filter (p. 5).
- related [[simateam2024instructable]]: SIMA does BC across games with keyboard and mouse and language; NitroGen drops language and uses a gamepad space (Sec. 5, p. 10).
- related [[reed2022generalist]]: Gato is another multi-task BC generalist, cited as relying on demonstration or RL data (Sec. 5, p. 10).
- related [[lee2022multigame]]: multi-game transformer pre-training with fine-tuning on held-out games, on Atari with offline RL data; not cited by NitroGen.
- related [[hafner2024mastering]]: cited as a vision-based RL method that still needs a dedicated simulator per environment (Sec. 5, p. 10).

## Experiments

<!-- kb:begin experiments -->
- [2026-11-24-nitrogen-mac-smoke](../../experiments/2026-11-24-nitrogen-mac-smoke/results.md): C8 inconclusive
<!-- kb:end experiments -->
