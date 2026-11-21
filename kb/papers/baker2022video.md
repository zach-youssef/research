---
id: baker2022video
title: "Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos"
authors: [Bowen Baker, Ilge Akkaya, Peter Zhokhov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, Jeff Clune]
year: 2022
venue: arXiv preprint
url: https://arxiv.org/abs/2206.11795
arxiv: 2206.11795
doi:
pdf_sha256: 87c5af9a2d444412e42f950cb5341c36608e4bed54c2fe0fbd1cadc406ecab85
tags: [imitation-learning, inverse-dynamics, behavioral-cloning, pretraining, reinforcement-learning, minecraft, video, transformer, action-space]
status: read
read: 2026-11-20
---

# Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos

<!-- Read from arXiv v1 (2022-06-23), 34 PDF pages; pp. 10-15 are references, pp. 16-34 the supplementary appendices A-I, all read. Author order on p. 1 is as printed (randomized). -->

## TL;DR

An IDM trained on ~2k h of labelled play pseudo-labels ~70k h of web Minecraft video; a 0.5B causal
transformer is behaviour-cloned on it from 128x128 pixels with native 20 Hz keyboard+mouse, and BC then
RL fine-tuning (PPG + KL to the prior) crafts a diamond pickaxe in 2.5% of 10-min episodes.

## Problem

Internet-scale pretraining works for text and images, but web data for sequential decision
domains (games, robotics, computer use) is mostly video without action labels, so a behavioural
prior can't be cloned directly; RL is sample-inefficient on hard exploration, and earlier
semi-supervised imitation methods rely on the policy exploring to label data and were tested in
low-data regimes (Sec. 1, pp. 1-2; Sec. 2, p. 3). Prior Minecraft work used lower frame rates and
crafting/attack macros; none used the full unmodified human action space with drag-and-drop
inventory and crafting (Sec. 1, p. 2; Sec. 2, p. 3).

## Method

- **Pipeline** (Fig. 2, Sec. 3, p. 4): (1) contractors play while keypresses and mouse are
  recorded; (2) train a non-causal IDM p_IDM(a_t | o_1..o_T) by action NLL; (3) filter web video to
  "clean" segments and pseudo-label them with the IDM; (4) train the causal policy by BC,
  min_θ Σ_t −log π_θ(a_t | o_1..o_t) with a_t from the IDM (Eq. 1, p. 4). The IDM is fixed during BC,
  unlike BCO-style joint training, because contractors cover most of the state space (Sec. 2, p. 3).
- **Rationale**: inverting dynamics is easier than modelling human intent from the past only,
  since the IDM can look at future frames (Sec. 1, p. 2; Sec. 3, p. 4).
- **Data** (App. A, pp. 16-19): ~270k h found by keyword search (Table 1) and metadata blacklist;
  a frame classifier (CLIP RN50x64 embeddings + RBF SVM, trained on 8800 mTurk-labelled frames,
  Table 2) is run at 3 frames/s; videos with ≥80% clean frames are kept, labels median-filtered
  (kernel 7) and split into clean segments ≥5 s, giving web_clean (~70k h). earlygame_keyword
  (~3000 h): first 5 min of web_clean videos whose metadata matches regexes like "episode 1",
  "new world" (App. A.3, pp. 18-19). Contractor data: 1962 h for the IDM (general play plus
  treechop); contractor_house ~420 h of 10-min house building (App. B.3-B.4, p. 20).
- **Observations** (App. C.1, p. 21): raw pixels as a human sees them, HUD and hand kept, rendered
  at 640x360 and downsampled to 128x128 (the smallest size where GUI elements stay discernible);
  a cursor image is drawn when a GUI is open; FOV 70, GUI scale 2, brightness 2. No other state
  is input (inventory etc. are diagnostics only) (App. C, p. 21). Episodes are not reset on death
  and policy memory is not masked (p. 21).
- **Actions** (App. C.2, pp. 21-23, Table 3; App. E.3, pp. 25-26): 20 Hz; 20 binary keys/buttons
  (WASD, jump, inventory, sneak, sprint, attack, use, drop, hotbar 1-9; arbitrary typing
  disallowed) plus relative mouse X/Y, each discretized into 11 foveated bins (fine near zero,
  coarse for big moves, Fig. 13, p. 23); mouse moves the camera, or the cursor when a GUI is open.
  The BC policy uses a "joint hierarchical" space: one 8461-way categorical over mutually
  exclusive key combinations times a binary "move mouse" flag (3^3·10·2^4·2 + 1), plus a 121-way
  head over joint mouse bins used only when the flag is on.
- **IDM architecture** (App. D.1, pp. 22-24): ~0.5B weights; input 128 frames of 128x128x3 scaled
  to [0,1]; a non-causal 3-D conv (128 filters, temporal kernel 5, spatial 1); a per-frame ResNet
  (stacks of widths 64, 128, 128, each conv3x3 + maxpool stride 2 + two residual blocks), flattened
  to 2^17 = 131072, dense to 256 then 4096; 4 unmasked residual transformer blocks (32 heads of
  dim 128, MLP 16384); per-key 2-way softmax heads and 11-way heads per mouse axis. Layernorm
  before and ReLU after every dense/conv layer.
- **BC policy** (App. E.1, p. 24): same as the IDM without the 3-D conv, with causally masked
  attention, Transformer-XL-style recurrence over past segments of the same video and relative
  position embeddings. Null actions: frames with no action are removed (App. E.2, p. 24).
- **RL fine-tuning** (Sec. 4.4, p. 7; App. G.1, pp. 26-29): phasic policy gradient (PPO-based,
  with GAE) on a 10-min "obtain diamond pickaxe" task; value and auxiliary value heads are single
  linear layers on the last transformer block (value head zero-initialized), with value targets
  normalized by an EMA mean/std. An auxiliary loss ρ·KL(π_pt, π_θ) to the frozen pretrained policy
  (Eq. 2, p. 27) replaces the entropy bonus, with ρ = 0.2 decayed ×0.9995 per iteration (Table 6,
  p. 28). Reward: tiered item rewards (base 1/2/4/8 for wood-stone, coal, iron, diamond tiers)
  divided by the rewarded quantity (Table 7, p. 29).
- **Compute** (Sec. 4.2, p. 5; App. D.2, F, G.1, pp. 24-29): IDM 4 days on 32 A100s; foundation
  model 30 epochs, ~9 days on 720 V100s; BC fine-tuning 16 A100s for ~6 h (contractor_house) or
  ~2 days (early_game); RL ~6 days on 80 GPUs plus 56,719 CPUs. All results are one training run
  each (App. D.2, p. 24).

## Claims

- C1: The best IDM, trained on 1962 h of contractor data, reaches 90.6% keypress accuracy and 0.97 R² for mouse movement on held-out contractor data (Sec. 4.1, Fig. 3, p. 5)
- C2: On the same contractor data the non-causal IDM is about two orders of magnitude more data efficient than a causal BC model and improves faster with more data (Sec. 4.1, Fig. 3, p. 5)
- C3: The 0.5B VPT foundation model, BC-trained on IDM-labelled web_clean, zero-shot chops logs, crafts planks and crafting tables (a median 50 s or ~970 actions for proficient humans), plus sticks, dyes, hunting, swimming and pillar jumping, in 60-min episodes (Sec. 4.2, Fig. 4, pp. 5-6)
- C4: Zero-shot crafting is far below humans: contractors craft 5.44 crafting tables per 60 min vs 0.19 for the foundation model (Sec. 4.2, p. 6)
- C5: Loss on the (out-of-distribution) contractor dataset starts rising after 7 epochs of foundation training while web_clean losses keep falling, yet rollout statistics do not decrease and BC fine-tuning results keep improving with longer foundation training (Sec. 4.2, Fig. 4, p. 6, Sec. 4.3, Fig. 5, p. 7)
- C6: BC fine-tuning to earlygame_keyword gives 2.5x crafting tables, 6.1x planks, 4.3x logs and 5.5x total crafting over zero-shot but no new behaviours; fine-tuning to contractor_house gives 213x crafting tables, 59x planks, 7x logs, 59x total crafting and new wooden and stone tool crafting (Sec. 4.3, Fig. 5, pp. 6-7)
- C7: BC fine-tuning to contractor_house from the VPT foundation model is starkly better than training on it from scratch (Sec. 4.3, Fig. 5, p. 7)
- C8: RL from a random initialization in the native action space gets almost no reward and never reliably collects logs, whereas RL from the VPT foundation model reaches reward near 13, learning up to iron ore and furnaces but not smelting iron ingots (Sec. 4.4, Fig. 7, pp. 7-8)
- C9: Pretraining, BC fine-tuning to earlygame_keyword, then RL reaches reward ~25: over 80% iron pickaxes, almost 20% diamonds and 2.5% diamond pickaxes within 10-min episodes; humans told to get a diamond pickaxe achieve 57%, 15% and 12%, so the model is human-level on iron pickaxes and diamonds (Sec. 4.4, Fig. 7, p. 8)
- C10: The authors know of no prior non-zero diamond-pickaxe success; prior diamond results (~0.1% in 15 min) used simplified action spaces (Sec. 4.4, p. 8)
- C11: Without the KL loss to the pretrained policy, RL from the early-game model only learns logs, planks, sticks and crafting tables and stalls after ~100k episodes, attributed to catastrophic forgetting of later skills (Sec. 4.4, Fig. 7, p. 8, App. G.2, Fig. 16, p. 30)
- C12: In a preliminary comparison, RL from the contractor_house model gained reward faster at first, but RL from the early-game model did better after ~800k episodes and smelted more iron ingots (App. G.2, Fig. 17, pp. 29-30)
- C13: Scaling foundation data (1 h to ~70k h; 71M models except the final 0.5B point) increases zero-shot log collection, mining and crafting; crafting tables appear zero-shot only above 5000 h, and after contractor_house fine-tuning stone tools emerge only at the full data scale (Sec. 4.5, Fig. 8, pp. 8-9)
- C14: BC models trained on earlygame_keyword labelled by IDMs of varying size need IDMs trained on ≥10 h for any crafting and ≥50 h for crafting tables, with gains plateauing after ~100 h, suggesting 100 h (about $2000 of contractor data) would suffice (Sec. 4.6, Fig. 9, p. 9, Sec. 5, p. 10)
- C15: The IDM's initial non-causal temporal 3-D convolution is "extremely important": loss, keypress accuracy and mouse R² are clearly better with it on the 1962 h dataset (App. D.1, Fig. 14, pp. 22-23)
- C16: Removing null-action frames helps: unfiltered BC models emit far more than humans' 35% null actions (often over 95%); filtering only runs of ≥3 nulls was slightly better than filtering all or runs of ≥21, found after the large models were trained (App. E.2, Fig. 15, pp. 24-25)
- C17: A factored action space (independent keys) samples many null actions though the filtered data has none, while the joint hierarchical action space does not; other rollout statistics are comparable (App. E.3, Fig. 15, pp. 25-26)
- C18: Zero-shot, the 71M model has the highest web_clean loss but the lowest contractor-data loss and best rollout performance (even some wooden tools), and 248M appears better at crafting than 0.5B; after contractor_house fine-tuning the order reverses and 0.5B is best in loss and rollouts (App. H, Fig. 18, Fig. 19, pp. 30-31)
- C19: Preliminary experiments suggested foundation models trained for 30 epochs needed 0.5B parameters to stay in the efficient learning regime (Sec. 4.2, p. 5, App. H, p. 30)
- C20: Conditioning a VPT model on closed-caption text embeddings (fine-tuned on ~17k h with captions) makes it weakly steerable: "explore"/"water" travel farther and "seed", "wood", "dirt" collect more of those items (1,000 episodes per variant, 95% CIs), but steering failed for flowers, hunting, crafting a named item and house building (App. I, Fig. 20, pp. 32-34)
- C21: Loss was not consistently correlated with downstream evaluation metrics, which made progress slow (Sec. 5, p. 10)

## Implementation notes

- IDM training: Adam, linear LR decay from 0.003, batch 128 sequences of 128 frames, weight decay 0.01, 20 epochs; temporally consistent per-segment augmentation (hue ±0.2, saturation/brightness/contrast 0.8-1.2, rotate ±2°, scale 0.98-1.02, shear ±2°, translate ±2 px) (source: App. D.2, p. 24).
- Pseudo-labelling: slide the 128-frame IDM with stride 64 and keep only predictions for frames 32-96, so window edges (effectively causal) are never used except at the video's start and end (source: App. D.3, p. 24).
- Foundation BC: LR 0.002147, weight decay 0.0625, 30 epochs, batch 880; BC fine-tuning: LR 0.000181, weight decay 0.039428, 2 epochs, batch 16 (source: Tables 4-5, p. 26). 71M and 248M models use 1/3 and 1/2 width, 15 epochs, with their own LR/batch/decay (source: App. H, p. 30).
- RL (Table 6, p. 28): LR 2e-5, weight decay 0.04, batch 40, 48 batches/iteration, context length 128, γ 0.999, GAE λ 0.95, PPO clip 0.2, max grad norm 5, max staleness 2, 2 PPG sleep cycles with value/aux-value/KL coefficients 0.5/0.5/1.0 and max sample reuse 6; no-KL run used LR 3e-6, random-init run an entropy bonus 0.01 and no KL (source: Table 6, p. 28).
- The BC policy's context length (frames per Transformer-XL segment and memory length) is not stated; 128 frames, matching the IDM input and the RL context length, is a guess (source: assumption from App. D.1, p. 22 and Table 6, p. 28).
- RL model size is given as ~248M (Sec. 4.4, p. 7) and as "220 million" for the same model (App. I, p. 32); RL scale as ~1.3M episodes / 1.4e10 frames (p. 7) and ~1.4M episodes of 12,000 frames = 16.8e9 frames (App. G.1, p. 29): the paper is inconsistent (source: pp. 7, 29, 32).
- Foveated mouse-bin edges are only shown as a plot (11 bins over about ±10° camera or ±80 px, Fig. 13, p. 23); exact bin boundaries are in the released code, not the paper (source: Fig. 13, p. 23; the code location is an assumption, p. 3 says code/weights/data are open-sourced).
- For the user's design (reader's inference): VPT's embedding is learned end to end from action prediction, not an unsupervised visual objective, and every frame is one token (a 4096-d vector from a per-frame ResNet), so a cross-game encoder trained by masked or contrastive objectives would replace its ResNet stem (source: assumption from App. D.1, p. 23 and Sec. 5, p. 9).
- For the user's design (reader's inference): the keyboard+mouse action space is game-agnostic for PC games, which is the paper's main argument for generality, but the hierarchical joint head's mutually-exclusive key sets are Minecraft-specific (source: assumption from App. E.3, p. 25; Sec. 5, p. 10).

## Limitations / open questions

- Only Minecraft is tested; generality to other domains is argued, not shown (Sec. 1, p. 2; Sec. 5, p. 10).
- Policies condition on past observations only and can't be told a task; text conditioning is preliminary and too weak to be practically useful (Sec. 5, pp. 9-10; App. I, p. 34).
- One training run per IDM, BC and RL result; no seeds or error bars on training variance (App. D.2, p. 24).
- Null filtering and some experiments were not redone with the later-best settings for compute reasons (App. E.2, p. 25).
- Zero-shot performance of larger models is worse, plausibly from visual domain shift between web video and the authors' game engine; the authors call their scaling investigation not conclusive (App. H, p. 31).
- RL needed a hand-designed tiered reward and very large compute (56,719 CPUs for rollouts), and the reward is sparse and sometimes deceptive (App. G.1, pp. 28-29).
- Open question: how much of the IDM's accuracy depends on Minecraft's simple dynamics and fixed camera conventions, i.e. would a single IDM label videos from several games with different control mappings?
- Open question: is a per-frame token at 128x128 enough for games whose state lives in small HUD/GUI text, given the authors chose 128x128 as the smallest size where Minecraft's GUI is readable?

## Related

- uses [[schulman2017proximal]]: RL fine-tuning uses PPG, a PPO variant, with PPO clipping 0.2 and GAE.
- uses [[vaswani2017attention]]: IDM and policy are residual transformer stacks (unmasked in the IDM, causal Transformer-XL style in the policy).
- related [[serrano2024crossdomain]]: imitation from unlabeled observation via an IDM is one of the transfer forms that survey covers; VPT stays within one domain.
- related [[he2021masked]]: a candidate unsupervised visual pretraining for the per-frame encoder, where VPT learns its encoder only through action prediction.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
