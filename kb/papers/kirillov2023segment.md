---
id: kirillov2023segment
title: Segment Anything
authors: [Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick]
year: 2023
venue: arXiv preprint
url: https://arxiv.org/abs/2304.02643
arxiv: 2304.02643
doi:
pdf_sha256: c6ca524e47200c3e542587ca76ac1dbc89cc67340f948c819250d16fdbd90cfb
tags: [segmentation, foundation-model, vision-transformer, promptable, dataset, zero-shot, state-representation]
status: read
read: 2026-11-14
---

# Segment Anything

<!-- Read from arXiv v1 (2023-04-05), 30 PDF pages; pp. 13-15 are references. -->

## TL;DR

SAM: an MAE-pretrained ViT-H image encoder run once per image plus a light prompt encoder and
mask decoder (~50 ms on CPU) that return 3 masks with predicted-IoU scores per prompt, trained on
SA-1B (11M images, 1.1B automatic masks); zero-shot it beats RITM on 16 of 23 datasets from 1 point.

## Problem

The authors want a "foundation model" for segmentation: one model that, through prompting,
solves segmentation tasks and image distributions it wasn't trained on, as LLMs do for text
(Sec. 1, pp. 1-2). Segmentation masks aren't abundant on the web, so there was no web-scale
training source; the largest prior dataset (Open Images) has 2.7M masks (Fig. 6, p. 7). Earlier
multi-task segmentation systems run a fixed set of tasks whose training and test tasks coincide
(Sec. 2, p. 4).

## Method

- **Task** (Sec. 2, p. 4): promptable segmentation. A prompt (foreground/background points, a
  box, a mask, free text) yields a "valid" mask: for an ambiguous prompt, a reasonable mask for at
  least one of the objects it could mean (shirt vs. person, Fig. 3). Used both as the pretraining
  objective and, through prompt engineering, as the interface to downstream tasks.
- **Image encoder** (Sec. 3, p. 5; App. A, p. 16): an MAE-pretrained ViT-H/16 ([[he2021masked]])
  with 14×14 windowed attention and four equally spaced global-attention blocks (after ViTDet).
  Input is 1024×1024 (rescale, pad the shorter side); output a 16× downscaled 64×64 embedding,
  reduced to 256 channels by a 1×1 then a 3×3 convolution, each with layer norm. Runs once per
  image; its FLOPs are amortised over all prompts.
- **Prompt encoder** (App. A, p. 16): a point is a positional encoding plus a learned
  foreground/background embedding; a box is two such tokens (top-left, bottom-right); text goes
  through CLIP's text encoder. Mask prompts are taken at 4× lower resolution than the image,
  downscaled 4× more by two stride-2 convolutions, mapped to 256 channels and added to the image
  embedding ("no mask" is a learned embedding).
- **Mask decoder** (App. A, pp. 16-17, Fig. 14): learned output tokens (like ViT's [class]) are
  concatenated with the prompt tokens. Two modified Transformer decoder layers, each: token
  self-attention, token-to-image cross-attention, a token MLP, image-to-token cross-attention.
  Embedding dim 256, MLP 2048 (tokens only), cross-attention reduced to 128, 8 heads. The image
  embedding is upsampled 4× by two transposed convolutions; a 3-layer MLP maps each output token to
  a dynamic linear classifier dotted with every location, giving mask logits at 1/4 input resolution.
- **Ambiguity** (Sec. 3, p. 5; App. A, p. 17): 3 mask outputs per prompt (whole, part, subpart);
  training backpropagates only the lowest-loss mask. A head on an extra output token predicts each
  mask's IoU, used to rank them. With more than one prompt, a fourth token gives a single mask.
- **Training** (Sec. 3, p. 5; App. A, p. 17): focal + dice loss at 20:1, IoU head by MSE. Simulated
  interactive training: 11 rounds per mask (an initial point or noisy box, 8 points sampled from the
  error region, 2 rounds with no new point), previous mask logits fed back as a prompt.
- **Data engine** (Sec. 4, pp. 5-6): (1) assisted-manual: annotators click points in a
  browser tool backed by SAM, freely labelling "stuff" and "things" without names; (2)
  semi-automatic: a box detector trained on stage-1 masks pre-fills confident masks and annotators
  add the rest; (3) fully automatic: a 32×32 point grid, 3 masks per point, filtered by predicted
  IoU and stability, then NMS, plus zoomed-in crops. Stage 3 was run on all 11M images.
- **Stability** (Sec. 4, p. 6; App. B, p. 18): a mask is stable if thresholding its soft mask at
  0.5−δ and 0.5+δ gives similar masks; concretely, kept when the IoU between the masks
  thresholded at logits −1 and +1 is ≥ 95.0.
- **Zero-shot protocols** (Sec. 7, pp. 8-11): single-point valid mask on a new suite of 23
  datasets (Table 7, p. 20), edge detection (16×16 grid, Sobel on mask probabilities), object
  proposals (64×64 grid), instance segmentation (ViTDet boxes as prompts), text-to-mask (trained
  on CLIP image embeddings of mask crops, prompted with CLIP text embeddings at test time, App. D.5).

## Assumptions

- A1: Each image is processed on its own: the image encoder runs once per image and masks are predicted per image, with no temporal context (Sec. 3, p. 5, App. A, p. 16).
- A2: The input is rescaled to 1024×1024 with the shorter side padded, giving a 64×64×256 image embedding and mask logits at 1/4 input resolution (App. A, pp. 16-17).
- A3: The image encoder is an MAE-pretrained ViT with 14×14 windowed attention and four global-attention blocks (Sec. 3, p. 5, App. A, p. 16).
- A4: The training images are SA-1B's licensed, high-resolution photographs (about 3300×4950 on average) from a provider working with photographers, with faces and licence plates blurred (Sec. 5, p. 6).
- A5: By default SAM is trained only on the data engine's automatically generated stage-3 masks (Sec. 7, p. 8, Sec. 7.6, p. 11).
- A6: Masks are class-agnostic: annotators labelled "stuff" and "things" without semantic constraints, objects they could name or describe, in order of prominence, and no names were collected, so SAM gets no semantic supervision (Sec. 4, p. 5, App. D.6, p. 22).
- A7: The automatic masks come from 32×32 grid prompting on the full image plus 20 zoomed-in crops of the original high-resolution image, kept when predicted IoU ≥ 88.0, stability ≥ 95.0 and coverage < 95% of the image, then box NMS at 0.7 and removal of components and holes under 100 px (Sec. 4, p. 6, App. B, p. 18).
- A8: The data engine relies on professional annotators with SAM in the loop: stage 1 started from a SAM trained on common public segmentation datasets and was retrained on new masks, stage 2 pre-filled masks from a box detector trained on stage-1 masks (Sec. 4, pp. 5-6).
- A9: Prompts are geometric (foreground/background points, boxes, mask logits) and training simulates 11 interactive rounds per mask, starting from a uniformly sampled point or a box with noise of 10% of its side (at most 20 px) (Sec. 3, p. 5, App. A, p. 17).
- A10: A single prompt yields three output masks (meant as whole, part, subpart) ranked by the predicted-IoU head; more than one prompt yields a single fourth mask (Sec. 3, p. 5, App. A, p. 17).
- A11: Prompt-time cost is measured with the image embedding precomputed: the encoder's cost is paid once per image and amortised over prompts (Sec. 3, p. 5, App. A, p. 16).
- A12: Point evaluation: the first point is the mask's "center" (maximum of the interior distance transform) unless random sampling is stated, later points come from the error region, only the top-ranked mask is scored unless "oracle" is stated, and ground truth holds one mask per point (Sec. 7.1, p. 8, App. D.1, p. 19).
- A13: The zero-shot suite is 23 existing datasets (egocentric, microscopy, X-ray, underwater, aerial, simulation, driving, paintings and others), subsampled to about 10k masks each when larger than 15k, with faces blurred (App. D.1, p. 19, Table 7, p. 20).
- A14: Neither SA-1B nor the evaluation datasets contain rendered game frames, sprites, pixel art or HUD overlays (reader's inference, Sec. 5, p. 6, Table 7, p. 20).
- A15: Human ratings come from professional annotators of the data-engine vendor on a 1-10 guideline, 1000 masks per dataset, with masks smaller than 24² px excluded (App. E, p. 23).
- A16: The baselines in the edge, proposal and instance experiments (HED, EDETR, ViTDet-H) were trained on the target dataset, while SAM saw neither BSDS500 nor LVIS images or annotations (Sec. 7.2-7.4, pp. 10-11, App. E, p. 23).
- A17: Downstream tasks are solved by prompt engineering inside a larger system: point-grid prompting plus NMS for edges and proposals (16×16 and 64×64 grids, no IoU or stability filtering, no crops for proposals) and an external detector's boxes for instances, followed by one refinement pass that feeds the most confident predicted mask back with the box (Sec. 7.2-7.4, p. 10, App. D.2-D.4, pp. 21-22).
- A18: Automatic-mask quality is judged against professional corrections made with SAM-assisted brush and eraser tools (Sec. 5, p. 6).
- A19: Training runs at a scale of 256 A100 GPUs for ViT-H (128 GPUs for ViT-B/L) on 1024×1024 inputs (App. A, pp. 17-18, Table 9, p. 28).
- A20: Country and income are inferred from image captions with a named-entity model and World Bank income levels, not from geo-tags (App. C, p. 18).
- A21: Fairness is measured on MIAP perceived gender and age labels (Open Images person masks matched to MIAP boxes) and on a proprietary perceived-Fitzpatrick dataset, with randomly sampled 1 and 3 points (Sec. 6, p. 8, App. C, p. 19).
- A22: Text prompting needs a separately trained variant: trained on CLIP ViT-L/14@336px image embeddings of stage-1 and stage-2 masks larger than 100² px and prompted at test time with CLIP text embeddings, relying on CLIP's image-text alignment (Sec. 7.5, p. 11, App. D.5, p. 22).
- A23: Mask embeddings are SAM's image embedding of a crop around the mask (and its horizontal flip), multiplied by the mask and averaged, compared only with masks in the same image (App. D.6, pp. 22-23).
- A24: The image encoder is ViT-H/16 unless another size is stated (Sec. 7, p. 8, App. A, p. 16).
- A25: Training uses all 11M SA-1B images unless a subsample is stated (Sec. 7.6, p. 11).

## Claims

- C1: SA-1B holds 11M licensed, privacy-protected images and 1.1B masks, 99.1% generated fully automatically; it has 11× more images and 400× more masks than Open Images, the largest prior segmentation dataset, and 36× more masks per image [A4, A7] (Sec. 5, p. 6, Fig. 6, p. 7)
- C2: On 500 images (~50k masks) re-annotated by professionals, 94% of automatic masks have IoU > 90% with the corrected mask and 97% have IoU > 75%, against prior estimates of 85-91% inter-annotator IoU [A4, A7, A18] (Sec. 5, p. 6)
- C3: SA-1B averages ~100 masks per image, and the per-image average is fairly consistent across regions and income levels (94-108) [A4, A7, A20] (Sec. 1, p. 2, Fig. 2, p. 3, Sec. 6, p. 7)
- C4: Data engine stage 1 collected 4.3M masks on 120k images, with annotation time per mask falling from 34 s to 14 s and masks per image rising from 20 to 44; stage 2 added 5.9M masks on 180k images, raising masks per image from 44 to 72 [A6, A8] (Sec. 4, p. 6)
- C5: Given a precomputed image embedding, the prompt encoder and mask decoder run in a web browser, on CPU, in ~50 ms [A1, A2, A11] (Sec. 3, p. 5)
- C6: The mask decoder needs less than 1% of the image encoder's compute [A2, A3, A11, A24] (App. A, p. 17)
- C7: Three output masks suffice for most common ambiguous cases, since nested masks are often at most three deep (whole, part, subpart) [A6, A10] (Sec. 3, p. 5)
- C8: From a single center point, SAM's top-ranked mask has higher mIoU than RITM on 16 of the 23 datasets, by up to ~47 IoU; choosing the best of its 3 masks against ground truth (oracle), it beats RITM on all 23 [A1, A2, A3, A4, A5, A9, A10, A12, A13, A14, A24, A25] (Sec. 7.1, Fig. 9a, p. 9)
- C9: In a human study on 7 datasets, SAM's single-point masks are rated higher than RITM's, and higher than an ambiguity-unaware single-output SAM, which is still above RITM; SAM's means fall between 7 and 9 on a 1-10 scale, and all differences are statistically significant [A1, A2, A3, A4, A5, A9, A10, A12, A13, A14, A15, A24, A25] (Sec. 7.1, Fig. 9b, p. 9, Table 8, p. 24)
- C10: SAM outperforms prior interactive segmenters at 1 point and is on par as points are added (up to 9), the gap shrinking with more points; with random rather than center first points the gap grows [A1, A2, A3, A4, A5, A9, A10, A12, A13, A14, A24, A25] (Fig. 9c, Fig. 9d, Sec. 7.1, p. 9)
- C11: Zero-shot edge detection on BSDS500 gives ODS .768, OIS .786, AP .794, R50 .928, above the older zero-shot methods (Canny ODS .600) and below EDETR (.840), with more edges than the annotations and so high recall at lower precision [A1, A2, A3, A4, A5, A9, A10, A14, A16, A17, A24, A25] (Table 3, Sec. 7.2, p. 10)
- C12: Zero-shot object proposals on LVIS v1 reach mask AR@1000 59.3 vs. 63.0 for ViTDet-H trained on LVIS; SAM is behind on small objects (45.5 vs. 51.7) and frequent categories, ahead on medium objects (81.6 vs. 80.8) and rare/common categories (65.8/63.9 vs. 58.3/63.3); the single-output ablation gets 54.9 [A1, A2, A3, A4, A5, A9, A10, A14, A16, A17, A24, A25] (Table 4, Sec. 7.3, p. 10)
- C13: Prompted with ViTDet-H boxes, SAM reaches mask AP 46.5 on COCO and 44.7 on LVIS v1 vs. ViTDet-H's 51.0 and 46.6; in a human study where both models take LVIS ground-truth boxes as input, raters score SAM's masks higher (8.1 vs. 7.9) [A1, A2, A3, A4, A5, A9, A10, A14, A15, A16, A17, A24, A25] (Table 5, Fig. 11, Sec. 7.4, pp. 10-11, App. E, p. 23)
- C14: Each data-engine stage raises mIoU on the 23-dataset suite; training on the automatic masks alone is only ~0.5 mIoU below training on all stages' data [A1, A2, A3, A4, A8, A9, A10, A12, A13, A14, A24] (Sec. 7.6, p. 11, Fig. 13, p. 12)
- C15: Training on 1M images (~10% of SA-1B, ~100M masks) gives results comparable to the full 11M, while 0.1M images causes a large mIoU decline [A1, A2, A3, A4, A5, A9, A10, A12, A13, A14, A24] (Sec. 7.6, p. 11, Fig. 13, p. 12)
- C16: Scaling the image encoder from ViT-B to ViT-H improves results substantially, but ViT-H gains only marginally over ViT-L, and further encoder scaling does not appear fruitful [A1, A2, A3, A4, A5, A9, A10, A12, A13, A14, A19, A25] (Sec. 7.6, p. 12, Fig. 13)
- C17: SAM performs similarly across perceived gender presentation and skin tone when segmenting people; by perceived age it does best on people perceived as older (62.9 vs. 54.5 mIoU for middle at 1 point, the only pair whose 95% confidence intervals don't overlap, though the older interval is wide); it segments clothing better on people perceived as masculine at 1 point, a gap that closes at 3 points [A1, A2, A3, A4, A5, A9, A21, A24, A25] (Table 2, Sec. 6, p. 8, Table 6, App. C, p. 19)
- C18: SAM can segment from simple and nuanced text prompts ("a wheel", "beaver tooth grille"), shown qualitatively as a proof of concept; when text alone fails, an added point often fixes it [A1, A2, A3, A14, A22, A24] (Sec. 7.5, Fig. 12, p. 11)
- C19: Mask embeddings (SAM's image embedding of a crop around a mask, multiplied by the mask and averaged) have nearest neighbours in the same image with some, albeit imperfect, shape and semantic similarity, although SAM gets no semantic supervision; the authors call this preliminary [A3, A4, A6, A14, A23, A24, A25] (App. D.6, Fig. 17, pp. 22-23)
- C20: About 4% of automatic masks have small spurious components and about 4% small spurious holes, both fixed by postprocessing [A7] (App. B, p. 18)
- C21: The released SAM was trained on 256 A100 GPUs for 68 hours (about 6963 kWh) [A19] (Table 9, p. 28)

## Implementation notes

- Image encoder input is 1024×1024 with the shorter side padded; output 64×64×256 (source: App. A, p. 16). How low-resolution game frames should be scaled into it is not discussed (source: assumption).
- Per-frame image-encoder latency is not given; the paper says only that overall performance is not real-time with the heavy encoder (source: Sec. 8, p. 12). Budget for measuring it on our hardware for ViT-B/L/H (source: assumption).
- Encoder sizes are plotted as 91M (ViT-B), 308M (ViT-L) and 636M (ViT-H) parameters (source: Fig. 13, p. 12).
- Mask logits come out at 256×256 (1/4 of the 1024 input); upsampling to the frame and the 0 logit threshold for binarising are used in App. B (source: App. A, p. 17, App. B, p. 18); the exact upsampling is in the authors' code, not the paper (source: assumption).
- What a mask carries in SAM: binary mask (from logits), predicted IoU, a stability score you compute yourself, and the decoder's 256-d output token; there is no class label, as SA-1B masks have no categories or text (source: App. A, pp. 16-17, App. B, p. 18, App. F.1, p. 25).
- A per-segment feature vector, if one is wanted, can follow App. D.6: mask-multiply SAM's image embedding and average spatially (the paper crops around each mask first) (source: App. D.6, p. 22). Using the decoder output token instead is untested here (source: assumption).
- "Segment everything" settings for SA-1B: 32×32 point grid on the full image plus 20 overlapping crops (2×2 with 16×16 grids, 4×4 with 8×8 grids); keep masks with predicted IoU ≥ 88.0 and stability ≥ 95.0; drop masks covering ≥ 95% of the image and masks touching inner crop borders; box NMS at 0.7 within then across crops; remove components and fill holes under 100 px (source: App. B, p. 18).
- That pipeline gives ~100 masks per photo on SA-1B (Fig. 6 spreads from <10 to >200); with a 64×64 grid, NMS 0.9, no IoU/stability filtering and no crops it gives ~900 per image; a 16×16 grid gives 768 masks before NMS (source: Sec. 1, p. 2, Fig. 6, p. 7, App. D.2-D.3, p. 21).
- No latency is reported for "segment everything"; its cost is one encoder pass per crop (21 with SA-1B's crops, if each crop is encoded separately) plus 3 masks per grid point from the decoder (source: assumption from App. B, p. 18 and Sec. 3, p. 5).
- SA-1B generation used a special SAM: trained on stages 1-2 data only, 177,656 iterations with large-scale jitter, point and mask prompts only, 4 sampled points, a 3-layer decoder (source: App. B, p. 18).
- Training recipe: AdamW (0.9, 0.999), lr 8e-4 after 250-iteration warmup, 90k iterations (~2 SA-1B epochs) with ×0.1 steps at 60k and 86,666, batch 256 images, weight decay 0.1, drop path 0.4, layer-wise lr decay 0.8, no augmentation, up to 64 masks per GPU, masks covering > 90% of the image dropped (source: App. A, pp. 17-18).

## Limitations / open questions

- Misses fine structures, sometimes hallucinates small disconnected components, and its boundaries
  are less crisp than "zoom-in" methods (Sec. 8, p. 12); weaker on small objects than an in-domain
  detector (Table 4, p. 10).
- Prompt handling is real-time but the whole model isn't with the heavy encoder (Sec. 8, p. 12).
- Text-to-mask is exploratory and not robust; how to prompt for semantic or panoptic segmentation
  is unclear; domain-specific tools should beat it in their domains (Sec. 8, p. 12).
- Most capabilities come from large-scale supervised training on the data engine's masks, not
  from the MAE self-supervised initialisation (Sec. 8, p. 12).
- Masks are single-image: the paper says nothing about video or temporal consistency; the suite
  includes video-derived datasets (OVIS, VISOR, Cityscapes, GTEA, NDISPark), which the reader takes
  to be used as single frames; Table 7 (p. 20) doesn't say so.
- Single-point prompts are ambiguous, so 1-point mIoU understates mask quality (Fig. 9, p. 9).
- Open question: a SAM segment set is a hierarchy of overlapping whole/part/subpart masks thinned by
  NMS, not a partition; which granularity should a policy's tokens use, and how stable is that
  choice from frame to frame?
- Open question: masks carry no identity across frames; a token-set state over N frames needs a
  tracker or matching step (SAM 2, arXiv 2408.00714, is worth reading next).
- Open question: the 23 test datasets include simulation (Hypersim) and paintings (DRAM) but no game
  frames; how SAM handles sprites, pixel art, HUD and text is untested here.
- Open question: is SAM's segment set as a policy state better than the raw ViT embedding it is
  computed from, given the extra cost of grid prompting and the loss of non-object information?

## Related

- uses [[he2021masked]]: SAM's image encoder is an MAE-pretrained ViT-H/16 (Sec. 3, p. 5).
- related [[serrano2024crossdomain]]: cross-domain RL transfer; segment sets are one candidate shared state space across games.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
