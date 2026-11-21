---
id: wang2025gametars
title: "Game-TARS: Pretrained Foundation Models for Scalable Generalist Multimodal Game Agents"
authors: [Zihao Wang, Xujing Li, Yining Ye, Junjie Fang, Haoming Wang, Longxiang Liu, Shihao Liang, Junting Lu, Zhiyong Wu, Jiazhan Feng, Wanjun Zhong, Zili Li, Yu Wang, Yu Miao, Bo Zhou, Yuanfan Li, Hao Wang, Zhongkai Zhao, Faming Wu, Zhengxuan Jiang, Weihao Tan, Heyuan Yao, Shi Yan, Xiangyang Li, Yitao Liang, Yujia Qin, Guang Shi]
year: 2025
venue: arXiv preprint
url: https://arxiv.org/abs/2510.23691
arxiv: 2510.23691
doi:
pdf_sha256: 7405e678f498f7bec67967ab3e3174b8602339fc966374cb7bf3ad1678c0ea9c
tags: [generalist-agent, multi-game, video-games, keyboard-mouse, action-space, vision-language-model, react, imitation-learning, causal-confusion, transfer-learning, scaling, memory, vendor-claims]
status: read
read: 2026-11-21
---

# Game-TARS: Pretrained Foundation Models for Scalable Generalist Multimodal Game Agents

## TL;DR

Continually pre-trains a VLM (Seed-VL-1.5 MoE, 2.5B active) on a 500B+-token blend of keyboard/mouse
game trajectories (~233B, 500+ games), agent and multimodal data, one mouseMove/keyPress action space;
reports ~2x prior SOTA on Minecraft MCU and beats GPT-5 on most unseen web/FPS tests, own protocol.

## Problem

Game agents either act through game- or GUI-specific high-level APIs, which don't transfer across
interfaces, or are trained per environment; the authors want one agent that acts through the
human-native keyboard and mouse, scales with data and generalises to unseen games and other
computer-use domains (Sec. 1, pp. 3-4).

## Method

- **Action space.** Three primitives shared by all games and desktop apps: `mouseMove(dx, dy)`
  (relative), `mouseClick(button)`, `keyPress(keys)` (several keys at once), plus `no_op`; the
  model also emits free-text thought (Sec. 2.1, pp. 5-6; Table 1, p. 5). Actions are time-aligned
  to frames at 50-100 ms intervals (p. 6). Policy: `(r_t, a_t) ~ π(·|o_t, h_t)` with a sparse
  ReAct format where thoughts appear only at some steps (Eq. 1-3, pp. 5-6).
- **Label source: think-aloud play.** Annotators play while speaking their reasoning (at least 3
  key moments a minute); screen, keyboard/mouse and audio are recorded in parallel, speech is
  transcribed and refined by an LLM (Sec. 2.2, p. 7; Fig. 3).
- **Causal realignment.** Logged timestamps lag the frames, so an action can follow the frame
  showing its effect and the model would degenerate into an inverse-dynamics model; a visual
  grounding model tracks the cursor and matches its motion to the logged dx, dy to realign
  (p. 7). Inputs between two frames are merged into one action, frames without input become
  `no_op`, and thoughts are moved to the step they explain (p. 8).
- **Decaying loss.** Each step's cross-entropy is weighted `ω_t = γ^(k_t-1)`, γ = 1/2, where k_t
  counts consecutive repeats of the same action, instead of VPT-style removal of no-ops (Sec. 2.3,
  p. 8; Eq. 4-5).
- **Post-training** (Sec. 3, pp. 8-12):
  - instruction following: random key rebinding written into the system prompt (Automatic Action
    Space Augmentation) plus an inverse-dynamics auxiliary task (Eq. 6, p. 9);
  - in-context learning: action-effect video clips as multimodal prompts, with text descriptions
    randomly masked, trajectories of at least 80 frames, early action loss masked and random
    dx, dy scaling (Sec. 3.2, pp. 10-11);
  - sparse thinking by rejection fine-tuning: thoughts kept only at steps where the action-only
    model fails and the thought-conditioned prediction matches the ground truth (Eq. 7-8, p. 11);
  - memory: 32k-token context of recent frames plus a long-term summary that keeps only thought
    text, trained on sequences up to 128k tokens (Sec. 3.4, p. 12; Fig. 6);
  - cross-domain code, GUI and MCP agent trajectories mixed in (Sec. 3.5, pp. 12-13).
- **Training** (Sec. 4.1, pp. 13-14; Table 2): single-stage continual pre-training on a blend of
  500B+ tokens (61B pure-action from 50 open-world games, 45B offline-synthesised ReAct from 50
  games, 127B think-aloud from 500 games, 20B+ agent data, 200B general multimodal), then
  post-training with the ViT frozen.
- **Models** (p. 13): Game-TARS-MoE-mini from Seed-VL-1.5 (632M vision encoder, MoE LLM with
  2.5B active parameters); a Dense model from Qwen2.5-VL-7B-Instruct trained on game data only;
  an MoE-Large is named without details.
- **Evaluation** (Sec. 4.2.1, p. 14): Minecraft (a training game, new tasks), four Poki web games,
  four VizDoom maps and five MiniWorld maps (unseen). The four Poki prompts shown give the game's
  name, the subset of keys it uses and, for three of them, a tutorial; the VizDoom and MiniWorld
  prompts aren't shown (App. B.2, pp. 32-34).

## Claims

- C1: Over 90% of actions in the action sequences repeat the previous action (the rate varies by game), so plain cross-entropy is dominated by these easy samples, which the authors link to causal confusion in imitation learning (Sec. 2.3, p. 8).
- C2: The authors argue that VPT-style removal of no-op actions distorts the true action distribution, and that unaligned recording timestamps would turn the policy into an inverse-dynamics model that reads its action off the following frame (p. 7, p. 8).
- C3: An early multi-stage curriculum (pure action, then offline ReAct, then think-aloud, then multimodal) "had a detrimental effect", so the final model uses one blended stage; no numbers are given (Sec. 4.1, p. 13).
- C4: On MCU Minecraft tasks (800+ tasks over unseen seeded starts, each completable within 600 steps, at least 3 runs each), Game-TARS-MoE-mini reaches 72.0±21.5% embodied, 55.4±17.6% GUI and 66.1±25.2% combat success vs the best baseline per category of 42.1±20.4% (UI-TARS-1.5), 36.7±17.2% (UI-TARS-1.5) and 31.9±13.7% (OpenHA); the ± ranges overlap in every category (Table 3, p. 15).
- C5: The game-data-only Dense model (Qwen2.5-VL-7B base) reaches 50.4%, 39.1% and 38.1% on the same categories, above UI-TARS-1.5 but well below MoE-mini, with overlapping ± ranges (Table 3, p. 15).
- C6: MoE-mini also needs fewer steps per Minecraft task than UI-TARS-1.5: 188 vs 290 embodied, 288 vs 320 GUI, 271 vs 346 combat (Table 3, p. 15).
- C7: The authors report "approximately a 2x performance improvement in Minecraft compared to previous state-of-the-art expert models", citing OpenHA [74]; Fig. 1's open-world panel prints 64.3 for Game-TARS, 36.6 for UI-TARS-1.5 and 31.5 for "Expert", without defining the metric (p. 3, Fig. 1, p. 1).
- C8: On four unseen Poki web games (paused sandbox, identical prompts with rules and action space, best-of-5 over at least 5 runs for models, best of at least 4 first-time humans), printed bar labels give Game-TARS vs humans: Jump-Only 5 vs 3 levels in 20 s (read off, no labels), Temple Run 228 vs 223, Airplane Flying 660 vs 640, Race checkpoint time 10.75 s vs 9.41 s (lower is better), where Game-TARS is the slowest of all five agents (Fig. 7, p. 16, p. 17).
- C9: On the same web games GPT-5 scores Temple Run 187, Airplane Flying 60 and Race 7.46 s (the fastest), so Game-TARS beats GPT-5 on three of four games under this protocol; Fig. 7 also plots Claude-4-Sonnet and a pink series with no legend entry (the legend lists o4-mini, with no matching teal bar), while the text names only GPT-5 and humans as baselines (Fig. 7, p. 16, p. 17).
- C10: On four VizDoom maps (average episode reward, number of runs not stated), Game-TARS-mini scores 18.87, 10.04, 15.24 and 14 vs GPT-5 12.59, 8.99, 10.24 and 13.46; Gemini-2.5-Pro ties it on Battle-2 (10.04) and Defend the Line (15.24), so it is strictly ahead on two maps (Battle-1, and Defend the Center by 14 vs 13.46), against the caption's "significantly outperforms all baselines"; the text says "almost all maps" (Fig. 8, p. 16, p. 17).
- C11: The authors report qualitative FPS behaviours in VizDoom (corner-peeking, dodging projectiles, adaptive targeting) and read them as transfer from diverse game data (p. 17).
- C12: On five unseen MiniWorld maps, Game-TARS-mini is above GPT-5 and o4-mini on every map, at roughly 40-80% vs 0-40% (read off, no labels, metric and runs not stated); Fig. 1 summarises it as 0.54 vs GPT-5 0.3 and o4-mini 0.1 (Fig. 9, p. 16, Fig. 1, p. 1).
- C13: Inference-time scaling on Minecraft: with a small step budget (below about 100 max steps, read off) Game-TARS is below UI-TARS-1.5, but it keeps improving with more steps to about 69% vs about 40% at 2,500 steps (read off) (Fig. 10, p. 17).
- C14: With under 10B training tokens the unified keyboard/mouse space did not beat GUI-action agents; the authors describe the GUI space as strong with little data but hard to scale, and the unified one as needing more data with an upper bound that "approaches" it; Fig. 10 shows it passing UI-TARS-1.5's last point, read off (p. 3, Sec. 4.3.1, p. 18).
- C15: Training-time scaling on Minecraft: Minecraft-only data takes Game-TARS from about 29% to 45% (read off, ~5B to ~50B tokens); adding about 200B tokens of other games raises it to 65% (the text says from 42% to 65%), and cross-domain agent and multimodal data to 72% (Fig. 10, p. 17, Sec. 4.3.1, p. 18).
- C16: Sparse thinking: the RFT-trained "efficient" thinking model scores Minecraft 63%, Doom 11.5, Race 9.3 s and 2048 score 924 at 37 tokens per step, vs no-thinking 55%, 13.2, 12.4 s, 445 (22 tokens) and greedy always-think 45%, 7.2, 11.3 s, 986 (68 tokens); no-thinking is best on Doom and greedy is best on 2048 (Table 4, p. 18).
- C17: The authors attribute greedy thinking's losses to hallucinated reasoning loops that end actions early, and report that the efficient model thinks at decision points and skips thought on routine steps (pp. 18-19).
- C18: The decaying loss raises online success on Minecraft: embodied 44.90% → 46.15%, combat 14.54% → 22.22%, GUI 37.75% → 42.26%, with higher behaviour diversity (60→90%, 50→60%, 80→100%) (Table 5, p. 19).
- C19: With the decaying loss, steps fall for combat (305 → 247) and GUI (244 → 164) but rise slightly for embodied (168 → 172), against the text's "substantially shorter trajectories" (Table 5, p. 19).
- C20: On 1,000 offline trajectories (100k+ steps), the decaying loss lowers overall action accuracy from 59% to 47% but raises non-repetitive accuracy (N-Acc, accuracy on steps where the predicted action differs from the previous one) from 12% to 39%; the authors read the overall drop as the baseline overfitting repetitive actions (Table 5, p. 19, p. 20).
- C21: The reported unseen-game transfer is zero-shot, with no game-specific training, but the web-game prompts (identical for GPT-5 and Game-TARS) give basic rules and the action space; the four Poki prompts shown define the keys, and three add a tutorial (App. B.2, pp. 32-34, p. 17).

## Implementation notes

- Training recipe (source: Table 2, p. 13), pre-training vs post-training:
  - images 640×360 to 1920×1080 vs 640×360 to 1280×720;
  - ~100 games and ~20k game hours vs ~500 games and ~2k hours;
  - learning rate 2e-5 → 2e-6 vs 1.4e-5 → 2e-6; batch 2048 vs 512; warmup 1000 vs 750;
  - steps 16,000 vs 10,000; max sequence 32k vs 128k tokens, 80 vs 300 images; ViT frozen
    only in post-training.
- Token counts don't add up: Table 2 lists 208B image + 326B text = 526B total (the parts sum to
  534B), and 40B post-training tokens vs "approximately 20 billion" in the text; the text gives
  500+ games for the data while Table 2 gives ~100 for pre-training; the five listed data sources
  sum to about 453B+ (61+45+127+20+200), short of the 500B+ total (source: Table 2, p. 13,
  Sec. 4.1, pp. 13-14).
- Reading of C7 (source: assumption): the 31.5 "Expert" and 36.6 UI-TARS-1.5 values in Fig. 1
  equal those baselines' Table 3 category means, so Fig. 1 likely shows category averages; but
  Game-TARS's Table 3 mean is 64.5, and 64.3 comes only from the text's rounded 55.0 and 66.0.
  Against UI-TARS-1.5 the ratio is about 1.76x, not 2x (Fig. 1, p. 1, Table 3, p. 15).
- Training mix by genre (printed labels): RPG 33.4%, shooting 32.3%, adventure 15.2%, other
  10.5%, simulation 6.0%, racing 2.5%; top titles include Don't Starve, Minecraft, PUBG
  (TslGame), Overwatch, VALORANT, GTA V, Genshin Impact and Stardew Valley (source: Fig. 4, p. 9;
  the PUBG reading of TslGame is the reader's).
- ICL post-training: dx, dy randomly rescaled with |value| ≤ 180; an ultra-long subset of more
  than 240 steps; summary memory spans more than 2,000 steps (source: pp. 11-12).
- The Fig. 6 caption calls the context and the summary "long-term and short-term respectively",
  the reverse of the text; the text's reading is used here (source: Fig. 6 vs Sec. 3.4, p. 12).
- The Minecraft system prompt asks for thoughts in Chinese (source: App. B.1, p. 31).
- Not stated: the vision tokenisation and frame rate fed to the model, the number of frames per
  context at inference, inference latency (games are paused, p. 17 footnote 3), number of runs
  for VizDoom and MiniWorld, the MiniWorld metric, the VizDoom and MiniWorld prompts, and model or data release (source: the paper
  is silent).

## Limitations / open questions

- The paper has no limitations section; the evaluations are the authors' own protocols, with
  best-of-5 selection on web games, no variance outside Table 3, and paused games (C8-C12).
- Reporting slips: Table 3's 55.4/66.1 appear as 55.0%/66.0% in the text (p. 15); the text's "986
  average tokens consumed per step" for 2048 is Table 4's 2048 score (Table 4, p. 18, p. 19);
  Fig. 1 gives GPT-5 0.3 on MiniWorld, while its Fig. 9 bars average about 0.2, read off (Fig. 1,
  p. 1, Fig. 9, p. 16).
- Minecraft, the main quantitative result, is a training game (new tasks and seeds only), and
  its ± ranges overlap the baselines (C4, C5).
- Failures visible in the results: Race (C8), the Doom and 2048 thinking ablation (C16), small
  data budgets (C14), and causal confusion from repeated actions (C1), which the decaying loss
  only partly addresses (C20).
- Memory is text-only summaries of thoughts; frames older than the 32k window are dropped
  (Sec. 3.4, p. 12).
- Open question (survey): the unified keyboard/mouse space is the counterpart of NitroGen's
  gamepad; both rely on per-game semantics given at run time (prompt here, fine-tuning there).
  Would a learned, game-agnostic state embedding from N frames remove the need for the
  prompt-level action description (C21)?
- Open question (survey): the decaying loss and inverse-dynamics auxiliary task target the same
  repeated-action and causal-confusion problems a PPO policy initialised from BC would inherit;
  does RL fine-tuning on unseen games fix them more directly?

## Related

- related [[magne2026nitrogen]]: concurrent multi-game foundation agent with a shared gamepad space, overlay-parsed labels and no language or reasoning; Game-TARS uses keyboard/mouse with think-aloud data.
- related [[baker2022video]]: VPT, whose no-op removal Game-TARS argues against (C2) and a Minecraft baseline in Table 3 (p. 15).
- related [[simateam2024instructable]]: SIMA, also language-conditioned keyboard/mouse play across many games but evaluated on short-horizon, in-domain tasks per the authors (Sec. 5.2, p. 21).
- related [[reed2022generalist]]: Gato, cited as a single-weights generalist whose discretised action space still needs domain-specific tokenisation (Sec. 5.2, pp. 20-21).
- related [[hafner2024mastering]]: DreamerV3, a Minecraft baseline in Table 3 (p. 15).
- related [[lee2022multigame]]: multi-game transformer pre-training on Atari with a shared discrete action space; not cited by Game-TARS.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
