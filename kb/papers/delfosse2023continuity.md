---
id: delfosse2023continuity
title: "Boosting Object Representation Learning via Motion and Object Continuity"
authors: [Quentin Delfosse, Wolfgang Stammer, Thomas Rothenbächer, Dwarak Vittal, Kristian Kersting]
year: 2023
venue: ECML PKDD 2023 conference paper (read as arXiv v3)
url: https://arxiv.org/abs/2211.09771
arxiv: 2211.09771
doi:
pdf_sha256: 2b3268f8fd89444aca6593acc91489713b0fd55f81abf86ce641bd0f6b522ea3
tags: [object-centric, unsupervised-learning, object-detection, optical-flow, contrastive-learning, atari, representation-learning, reinforcement-learning]
status: read
read: 2026-11-22
---

# Boosting Object Representation Learning via Motion and Object Continuity

## TL;DR

MOC adds motion supervision (masks from background subtraction) and a contrastive object-continuity
loss to SPACE ("SPOC") or Slot Attention. On 8 Atari games SPACE's relevant-object F1 rises from 38% to
85% on average and encodings become linearly classifiable; swapped into RAM-trained Boxing/Pong agents it helps.

## Problem

Unsupervised object-discovery models are trained only for reconstruction. On Atari, SPACE fails to
detect objects in some games (Pong, MsPacman, Tennis; "Type I" error), and where it detects them its
object encodings separate classes poorly ("Type II" error), so they are hard to use in RL (Sec. 1,
pp. 1-2, Fig. 2, p. 2).

## Method

- **Motion supervision (M).** Binary motion masks α̂ from a cheap optical-flow stand-in: per pixel,
  the mode over a batch of frames is the background; |x_t − background| > η gives the mask, contours
  give boxes and so target z_pres and z_where per grid cell (largest box if several share a cell)
  (Sec. 2.1, pp. 3-4, App. C.1, p. 19). For SPACE, L_M is a weighted sum of squared errors on α,
  z_pres and z_where, and a decaying "guidance" mixes the motion targets into the decoder early on
  (Eq. 14-16, App. C.2, p. 20). For Slot Attention, a bipartite match between slot masks and motion
  masks with a BCE loss (Eq. 19-21, App. C.3, p. 21).
- **Object continuity (OC).** For objects detected in consecutive frames, if the most similar encoding
  in frame t+1 (cosine) is also the nearest by location, their similarity is rewarded with weight −β
  (β = 5), all other pairs are penalised with weight 1 (Eq. 2-3, Sec. 2.2, p. 4, Eq. 17-18, p. 20).
  For SPACE the search is limited to a 3×3 cell neighbourhood (App. C.6, p. 22).
- **Total loss.** L = L_B + (1 − λ_align)·L_M + λ_align·λ_OC·L_OC; for SPACE λ_align = 2^(−δ_align),
  where δ_align grows when the model's boxes and counts disagree with the motion estimate beyond
  tolerances, recomputed each epoch; for Slot Attention a linear ramp (Eq. 4, Sec. 2.3, p. 5,
  App. C.4, pp. 21-22). Inference stays single-image (Sec. 2.3, p. 5).
- **Atari-OCTA dataset.** 8 games (AirRaid, Boxing, Carnival, MsPacman, Pong, Riverraid,
  SpaceInvaders, Tennis); 8192 stacks of 4 consecutive frames per game for training and 1024 for
  validation and test, from DQN, "rational DQN" or random agents; labels from AtariARI RAM and
  per-game colour/position/size heuristics, with per-class precision and recall on 16 hand-labelled
  images; HUD objects are "non-relevant" (Sec. 3, p. 6, App. A, pp. 16-17, App. H, pp. 37-39).
- **Metrics.** Precision, recall and F1 with a match when the centre distance divided by the box
  diagonal ("center divergence") is under 0.5, not IoU; AMI of k-means clusters of encodings against
  labels; ridge-regression few-shot accuracy with 1, 4, 16, 64 encodings per class (App. F, pp. 29-31).
- **RL test.** Concept-bottleneck Dueling-DQN agents trained on RAM object positions ("Info*") on
  Boxing and Pong; at test, positions come from SPACE or SPOC and classes from a 16-shot classifier
  on the encodings; MLP of 3 layers on (x, y) (Sec. 3, pp. 8-9, App. G.3, p. 37).

## Assumptions

- A1: Relevant objects move, and the background is static enough for a per-pixel mode image to separate them (local mode only for Tennis); MOC is biased against stationary objects (App. C.1, p. 19, App. F.4, p. 31, Sec. 5, p. 10).
- A2: Training data are 4-frame sequences of Atari 2600 games from fixed DQN, rational-DQN or random agents, 128×128 inputs (App. A, p. 16, Table 3, p. 16, App. C.3, p. 20).
- A3: Detection is scored on "relevant" objects only (HUD removed with per-game box filters), matching by center divergence < 0.5 rather than IoU (Sec. 3, p. 6, Table 4, p. 17, App. F.1, pp. 29-30).
- A4: Ground truth comes from semi-automatic heuristic labels, not exact game state; some classes have low label precision, e.g. Riverraid fuel 0.432 and player 0.778, Carnival owl 0.769 (App. A.1, p. 16, App. H, pp. 37-39).
- A5: Hyperparameters are set per game (z_scale prior, tolerances, background components, slot counts), visually checked, with no extensive search (App. E, pp. 26-29, App. F.4, p. 31).
- A6: Every model trains for 5000 gradient steps at batch 16, results over 5 seeds (3 for Slot Attention) (App. F.4, p. 31, App. E.3, p. 27).
- A7: The SPACE baseline uses a foreground learning rate of 3e-5 instead of the original 1e-5 (App. E.2, p. 26).
- A8: The RL agents are trained on RAM positions and only evaluated with detector outputs, on two games with a fixed number of objects (Sec. 3, pp. 8-9, App. G.3, p. 37).

## Claims

- C1: Averaged over the 8 games, relevant-object F1 rises from 38% (SPACE) to 85% (SPACE+MOC), and Slot Attention's foreground ARI from 4% to 95% [A1, A2, A3, A4, A5, A6, A7] (Fig. 4, Sec. 3, p. 6).
- C2: Per-game relevant F1, SPACE vs SPACE+MOC: AirRaid 51.3 vs 91.5, Boxing 58.8 vs 92.1, Carnival 51.6 vs 93.0, MsPacman 0.5 vs 91.5, Pong 0.2 vs 92.8, Riverraid 65.0 vs 69.1, SpaceInvaders 86.5 vs 87.2, Tennis 0.0 vs 62.6 [A1, A2, A3, A4, A5, A6, A7] (Table 11, p. 32).
- C3: On all objects including the HUD, F1 for SPACE vs SPACE+MOC is AirRaid 49.1 vs 87.9, Boxing 24.5 vs 70.5, Carnival 48.6 vs 90.6, MsPacman 0.4 vs 90.5, Pong 10.7 vs 87.4, Riverraid 45.0 vs 76.6, SpaceInvaders 87.5 vs 85.2, Tennis 3.6 vs 40.2 [A1, A2, A4, A5, A6, A7] (Table 14, p. 34, Fig. 13, p. 25).
- C4: SPACE reaches F1 > 0.5 (on the test set per Table 1's caption; the p. 6 text says validation) on only 5 of 8 games, after 1800 steps on average (1820 in the text), while SPACE+MOC reaches it on all 8 after 280 on average, which the authors call more than 7× faster (6.5× from the stated averages) [A1, A2, A3, A5, A6, A7] (Table 1, p. 7, Sec. 3, pp. 6-7).
- C5: Motion supervision alone (SPACE+MOC without OC) gives about the same detection as full MOC, e.g. relevant F1 97.6 vs 92.8 on Pong and 89.7 vs 91.5 on AirRaid [A1, A2, A3, A5, A6] (Sec. 3, p. 7, Table 11, p. 32).
- C6: The simple optical-flow masks alone score F1 between 63.9 (Riverraid) and 98.8 (Pong) on the human-annotated evaluation sets; the authors note SPACE+MOC exceeds them on AirRaid (91.5 vs 85.4) and Riverraid (69.1 vs 63.9) and read this as the network compensating for imperfect guidance, though the SPACE+MOC numbers (Table 11) are on the test set against heuristic labels, a different label set [A1, A3, A4] (Table 5, p. 19, Table 11, p. 32).
- C7: Averaged over games, AMI of object encodings rises from 26% to 69% (SPACE to SPACE+MOC) and from 16% to 37% (Slot Attention), with lower variance across seeds for SPACE+MOC [A2, A4, A5, A6] (Fig. 5, Sec. 3, p. 7).
- C8: With one labelled encoding per class, a ridge classifier on SPACE+MOC encodings averages 65% accuracy, 3.5× SPACE's and 1.3× SPACE+MOC without OC's, so the OC loss mainly improves encodings, not detection; SPACE's average includes 0.0 ± 0.0 entries that likely mean no usable detections (see Limitations) [A2, A3, A4, A5, A6] (Fig. 6, Sec. 3, p. 8, Table 15, p. 34).
- C9: The authors report that SPACE's object encodings are sub-optimal even for games with high detection scores, shown by ridge-regression few-shot accuracies and a cluttered t-SNE of SpaceInvaders encodings, and separate detection failures (Type I) from sub-optimal representations (Type II) [A2, A3, A4, A6, A7] (Fig. 2, p. 2, Sec. 1, p. 2).
- C10: Swapping detectors into RAM-trained agents, mean scores over 5 seeds are Boxing −3.5 (SPACE) vs 8.4 (SPACE+MOC) vs 36 (RAM) vs 4.3 (human), Pong −21 vs −11 vs 20 vs 9.3; best seeds reach 22 on Boxing and 4.8 on Pong with SPACE+MOC [A1, A2, A6, A8] (Table 2, p. 9, Table 25, p. 37).
- C11: Getting the object count exactly right per frame stays hard: share of frames with the correct relevant-object count for SPACE+MOC ranges from 11.1 (SpaceInvaders) and 32.1 (Riverraid) to 77.1 (Boxing) [A1, A2, A3, A4, A6] (Table 9, p. 32).
- C12: With 5k steps at the higher learning rate, baseline SPACE gets F1 65.0 on Riverraid vs 27.1 with the original rate and 10k steps, and 86.5 vs 76.5 on SpaceInvaders, so the change does not handicap the baseline [A6, A7] (Table 8, p. 28, App. E.2, p. 26).
- C13: SPACE is highly sensitive to hyperparameters because either VAE can learn the whole scene; per the authors, SPACE's own authors could train MsPacman only jointly with other games [A5] (App. B.1, p. 18).
- C14: Training uses about 18 GB of GPU memory for batch 16 of 4-frame stacks over 4 GPUs (V100 or A100), and the 3×3 neighbourhood cuts the OC loss from O(c_t²) to O(9 c_t) for c_t detected objects [A6] (App. F.4, p. 31, App. C.6, p. 22).

## Implementation notes

- SPACE+MOC defaults: fg/bg lr 3e-5 / 1e-3, batch 16, 5000 steps, z_pres prior 0.1 → 1e-10 over 5000 steps, z_scale prior mean −2 → −2.5, grid 16×16, K = 3, Gumbel τ 2.5, fixed α and boundary loss removed, η = 0.5, λ_α/λ_pres/λ_where = 100/1000/10000, λ_guid 1 → 0 over 3000 steps, β_mismatch 0.1, β_underestimation 1.25, λ_OC 10 (source: Table 6, p. 27, App. E.1, p. 26).
- Per-game deviations (Boxing z_scale −1, Riverraid 4 background components, Tennis z_scale −2.5 → −3.0, …) (source: Table 7, p. 28).
- Slot Attention: ResNet18 encoder, hidden 64, batch 18, 50 epochs, 5000 max steps, λ_OC 1.0 for Pong, Carnival and SpaceInvaders and 0.1 elsewhere, slots 10 to 50 per game (source: App. E.3, pp. 26-29).
- Code and the Atari-OCTA generator: https://github.com/k4ntz/MOC (source: p. 6, footnote 5); PyTorch 1.9.0, sklearn 0.24.1 (source: App. F.4, p. 31).
- The main text says Atari-OCTA was labelled with OCAtari, while App. A describes AtariARI plus OpenCV colour heuristics; the per-game tables in App. H use the heuristics (source: Sec. 3, p. 6 vs App. A.1, p. 16, reader's reading).
- [[delfosse2024ocatari#C4]]'s SPACE and SPOC F1 values equal this paper's all-object F1 (Table 14, p. 34) for its 7 games (all but AirRaid), so they are F1 on all objects, HUD included, against this paper's heuristic labels (source: reader's comparison of the two tables).

## Limitations / open questions

- The flow stand-in assumes static backgrounds; Riverraid and Tennis, with moving backgrounds, stay lowest (Fig. 9, p. 23, Sec. 5, p. 10).
- Bias towards moving objects; stationary but important objects may be missed (Sec. 5, p. 10).
- Slot Attention + MOC has larger AMI variance; the RL agents are basic, without tracking such as Kalman filters (Sec. 5, p. 10).
- SPACE's few-shot accuracy is 0.0 ± 0.0 at every shot count on SpaceInvaders (relevant F1 86.5) and Carnival (51.6) in Tables 15-18 and "n/a" for SpaceInvaders in Table 8, while Fig. 2 shows a SPACE bar of roughly 40% on SpaceInvaders; given Fig. 2's "0% accuracy corresponds to no object detected", these zeros likely mean no usable detections or a value not computed, not poor encodings (p. 2, p. 28, pp. 32-35).
- Open question (reader): RL is tested on two games with frozen RAM-trained policies; training a policy on SPOC outputs, or across HackAtari variants ([[delfosse2025intelligence]]), isn't tested.
- Open question (reader): no compute comparison with RAM extraction at inference is given; F1 of 40-90 on all objects is still well below RAM extraction.

## Related

- extends [[lin2020space]]: adds motion and object-continuity losses to SPACE's training.
- related [[delfosse2024ocatari]]: OCAtari's RAM extraction, whose Table 2 quotes this paper's SPACE and SPOC F1.
- related [[yoon2023investigation]]: object-centric pretraining for RL with slot encoders, the Slot Attention side.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
