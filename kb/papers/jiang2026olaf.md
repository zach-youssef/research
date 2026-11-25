---
id: jiang2026olaf
title: "Olaf-World: Orienting Latent Actions for Video World Modeling"
authors: [Yuxin Jiang, Yuchao Gu, Ivor W. Tsang, Mike Zheng Shou]
year: 2026
venue: ICML 2026 conference paper (read as arXiv v2)
url: https://arxiv.org/abs/2602.10104
arxiv: 2602.10104
doi:
pdf_sha256: f108f25fdbcf882621692aef27c19b0b06a75a4405abffef226cbe130019e983
tags: [world-model, latent-actions, video-pretraining, representation-alignment, cross-context, action-transfer, diffusion, adaworld, v-jepa]
status: read
read: 2026-11-25
---

# Olaf-World: Orienting Latent Actions for Video World Modeling

## TL;DR

Adds SeqΔ-REPA to AdaWorld-style latent action models: a clip's mean latent is aligned (cosine) to the
feature change of a frozen V-JEPA 2. On MIND first- vs third-person, cross-domain probe F1 beats
AdaWorld (0.625 vs 0.482 and 0.590 vs 0.500), as does pose error after LoRA adaptation.

## Problem

- Latent action models infer a latent from a frame pair and predict the next frame. The paper names two failure modes: shortcut learning (the latent encodes context cues that correlate with the next frame instead of the control) and cross-context non-identifiability (the loss never compares latents across clips, so the same action can map to different latent directions in different contexts) (Sec. 1, p. 2, Sec. 3.1, pp. 3–4).
- Local objectives are blind to this: Proposition A.1 shows that any per-context bijection of the latent leaves the prediction loss unchanged, so a latent from context A need not mean the same action in context B (App. A.2–A.3, p. 14). With a diagonal-Gaussian β-VAE posterior only signed permutations of coordinates remain as exact symmetries, which can still vary by context (App. A, pp. 14–15).

## Method

- **LAM** (Sec. 3.1, p. 3, Eq. 1): causal-encoder inverse model q(z_i | x_{0:i+1}) and forward decoder p(x_{i+1} | x_i, z_i), trained with a step-wise β-VAE loss, prior N(0, I), z in R^32.
- **SeqΔ-REPA** (Sec. 3.1, p. 4, Eqs. 2–5, Fig. 3a): a frozen video encoder f (V-JEPA 2) gives pooled per-frame descriptors s_i. Effect direction τ = mean of (s_{i+1} − s_i). The clip's mean latent z̄ goes through a trainable MLP head h to the encoder's dimension, u = h(z̄), and the loss is 1 − cos(u, τ). Total loss = β-VAE + λ·SeqΔ-REPA. Temporal differences are meant to drop static appearance, so the reference is stable under context change.
- **Olaf-World pretraining** (Sec. 3.2, p. 4, Fig. 3b): the frozen LAM labels unlabelled video with per-frame latents; a latent image-to-video DiT (SkyReels-V2 1.3B) is trained with flow matching, the latent projected and added to the timestep embedding and mapped to AdaLN-Zero parameters; because the 3D VAE compresses time 4×, every 4 per-step latents are grouped into one conditioning vector.
- **Adaptation to a target environment** (Sec. 3.2, pp. 4–5): a small action adapter maps real actions to latents; for discrete actions an embedding table initialised with class prototypes (the mean inferred latent over segments labelled with that action), then the adapter and a LoRA on the backbone are fine-tuned.
- **Setup** (Sec. 4.1, p. 5): pretrain on MiraData 3D-rendering and city-walking categories; evaluate on MIND (Unreal Engine 5, frame-aligned labels, disjoint first-person and third-person subsets sharing an 8-action space: W/S/A/D, look up/down/left/right). Baselines: AdaWorld re-trained with the same backbone, data and budgets but its own LAM objective; DirectAct (ground-truth actions as conditioning).

## Assumptions

- A1: Both the LAM and the world model are pretrained on MiraData 3D-rendering and city-walking video, and the probe, prototype and RPE tests are on two MIND subsets rendered in one engine (Sec. 4.1, p. 5); how many distinct games the MiraData categories cover isn't stated (reader's inference)
- A2: The two domains share one 8-action label space (4 navigation keys, 4 look directions), and in-domain and cross-domain probes use these labels (Sec. 4.1, p. 5)
- A3: A frozen self-supervised video encoder gives a feature change that reflects the semantic effect of the action, and the dominant change in a clip comes from the controlled action (Sec. 1, p. 2, Sec. 3.1, p. 4)
- A4: Temporal modelling in the reference encoder matters: video encoders (V-JEPA 2, VideoMAEv2) work, the image encoder DINOv3 does not (Sec. 4.5.1, pp. 8–9, Table 5)
- A5: Actions are camera or locomotion-style controls whose visual effect is a consistent change of view or position; "look" actions are less aligned across viewpoints because the same control gives different motion in first- and third-person (reader's inference, Sec. 4.2.2, p. 6, Fig. 5)
- A6: The latent action dimension is 32, the clip window K = 16 and λ = 0.02, β = 2×10⁻⁴ (Sec. 4.1, p. 5, App. B.1, p. 15)
- A7: Adaptation gets action labels (0, 1 or 50 videos, up to 120 min) and a LoRA of rank 16 unless stated (Sec. 4.4.1, p. 7, Sec. 4.5.2, p. 9)
- A8: One run per configuration with no seeds or error bars reported (reader's inference, Table 1, p. 5, Table 5, p. 9)

## Claims

- C1: Standard step-wise LAM training (reconstruction plus β-VAE KL) can fix the same reconstruction loss under context-dependent reparameterisations of the latent (Proposition A.1), so a latent inferred in one context need not mean the same action in another; the diagonal-Gaussian KL leaves only signed-permutation symmetries, which can still vary by context [] (App. A.2–A.3, p. 14, App. A, p. 15)
- C2: Cross-domain linear probe (8 actions, trained on one MIND subset, zero-shot on the other; Macro-F1): Olaf-World 0.8138 / 0.6250 / 0.8256 / 0.5904 for 1st→1st / 1st→3rd / 3rd→3rd / 3rd→1st versus AdaWorld 0.6004 / 0.4820 / 0.4827 / 0.4999 [A1, A2, A6, A8] (Sec. 4.2.1, p. 5, Table 1, Fig. 4)
- C3: Even for Olaf-World, cross-domain F1 is clearly below in-domain F1 (0.6250 vs 0.8138 and 0.5904 vs 0.8256); the gain is largest on the harder 3rd-P subset where AdaWorld saturates at low F1 [A1, A2, A6, A8] (Sec. 4.2.1, p. 6, Table 1)
- C4: Cosine similarity between per-action latent prototypes of 1st-P and 3rd-P is high everywhere for AdaWorld (different actions look similar across domains) but more diagonal-dominant for Olaf-World; the look-left/right actions remain the most confused [A1, A2, A5, A6] (Sec. 4.2.2, p. 6, Fig. 5)
- C5: Zero-shot action-sequence transfer (a latent sequence taken from a reference clip drives generation from another initial frame) is qualitatively better preserved by Olaf-World, while AdaWorld shows temporal wash-out, character or scale loss and drift toward generic motion; no quantitative metric is given for this [A1, A3, A6] (Sec. 4.3, pp. 6–7, Fig. 6)
- C6: After adaptation with 0, 1 or 50 labelled videos, Olaf-World has the lowest relative pose error (RPE-trans, RPE-rot) on both subsets with visual quality comparable to AdaWorld; e.g. 1st-P with 50 videos RPE-trans 0.0230, RPE-rot 0.3785 vs AdaWorld 0.0263 / 0.3834 and DirectAct 0.0351 / 0.4527. At 0 videos, 1st-P: 0.0387 / 0.8773 vs AdaWorld 0.0470 / 1.0844; AdaWorld has the better visual metric in some cells and DirectAct the best image quality at 0 videos [A1, A2, A6, A7, A8] (Sec. 4.4.1, p. 7, Table 2)
- C7: After adaptation to the 1st-P action space on 50 OOD initial frames (anime, oil painting, objects), Olaf-World has the lowest RPE (0.0478 / 1.2221 vs AdaWorld 0.0482 / 1.7063, DirectAct 0.0547 / 1.2343); the RPE-trans gap to AdaWorld is 0.0004, and image quality is not best (0.6274 vs 0.6322 DirectAct) [A1, A2, A7, A8] (Sec. 4.4.2, p. 8, Table 3)
- C8: Removing Δ (align static features instead of feature differences) lowers cross-domain Macro-F1 (1st→3rd 0.5287 vs 0.6250; 3rd→1st 0.4823 vs 0.5904); removing the ℓ2 normalisation (MSE instead of cosine) gives 0.5311 / 0.5934 cross-domain, with 3rd→1st at or slightly above the full objective (0.5934 vs 0.5904) and lower in-domain scores [A1, A2, A3, A6, A8] (Sec. 4.5.1, p. 8, Fig. 9, Table 4)
- C9: Reference encoder (10-epoch runs, Macro-F1 1st→1st / 1st→3rd / 3rd→3rd / 3rd→1st): V-JEPA 2 0.7044 / 0.2972 / 0.5845 / 0.4704; VideoMAEv2 0.6626 / 0.4697 / 0.6616 / 0.5377; DINOv3 0.2334 / 0.1782 / 0.2869 / 0.1910; no alignment (AdaWorld) 0.1849 / 0.1382 / 0.2105 / 0.1593; the authors conclude temporal modelling in the reference encoder matters [A1, A2, A4, A6, A8] (Sec. 4.5.1, pp. 8–9, Table 5)
- C10: Action accuracy improves with more labelled adaptation videos (0 to 50; RPE-trans 0.0387 to 0.0230, RPE-rot 0.8773 to 0.3785, steepest gain at low budgets, non-monotone at 1–5 videos) and generally with LoRA rank (rank 16 to full: RPE-trans 0.0230 to 0.0185; rank 64 is worse than 16 at 0.0251), while video quality stays roughly flat [A1, A2, A6, A7, A8] (Sec. 4.5.2, p. 9, Fig. 10)
- C11: Linear interpolation and extrapolation between a no-op prototype and an action prototype (α from 0 to 2) gives progressively stronger motion in the intended direction (qualitative, left/backward/right on one image) [A1, A6] (App. D, p. 17, Eq. 14, Fig. 11, p. 16)
- C12: Failure cases: transferred action that would collide in the target scene (the model alters the scene instead), blurry completion under large reveals (zoom out), and event-like actions (an entering character) realised as relative motion instead of the event [A1, A3] (App. D, p. 17, Fig. 12)

## Implementation notes

- LAM: causal spatio-temporal encoder, spatial-only decoder, 16 blocks, 1024 dim, 16 heads, clips of 16 frames at 272×480; head LayerNorm→3-layer MLP (Linear-SiLU) to D = 1408; teacher V-JEPA 2 ViT-Giant/16 (384); AdamW lr 2.5×10⁻⁵, wd 10⁻², batch 32 on 8×H200, β = 2×10⁻⁴, λ = 0.02, 100 epochs (~146k steps, ~4.5 days), colour jitter off (source: App. B.1, p. 15)
- World model: SkyReels-V2 I2V 1.3B DiT, 540p, 97-frame clips (25 latent frames), latent 32→1536 injected into the timestep stream with a learned gain initialised to 2.0 (source: Sec. 4.1, p. 5, App. B.2, p. 15)
- World model pretraining: 10k steps, lr 5×10⁻⁵, wd 10⁻³, 4×H200, batch 4 per device; adaptation LoRA rank 16 on attn q,k,v,o and ffn 0,2, lr 10⁻⁴, no weight decay (source: App. B.2, p. 15)
- Probe: linear classifier, SGD momentum 0.9, wd 10⁻⁶, 12 epochs, StepLR, focal loss γ = 2, best in-domain validation Macro-F1 checkpoint (source: App. C.1.1, p. 15)
- RPE protocol: camera poses estimated with ViPE, Sim(3) Umeyama alignment, VBench Imaging Quality and Temporal Consistency (source: App. C.2, p. 16)
- Code is at github.com/showlab/Olaf-World per the arXiv abstract page, not read (source: arXiv listing)
- Not stated: number of seeds, the MIND subset sizes and the number of clips used for probes and prototypes, the pooling used for per-frame descriptors beyond "spatially pool", how latents are time-aligned to the frame rate (App. E.1 says one latent per frame at 16 FPS), the pretraining data size (source: assumption after a full read)
- Applying SeqΔ-REPA needs a clip-level (multi-frame, K = 16) LAM plus a frozen ViT-g V-JEPA 2 teacher, not AdaWorld's two-frame encoder; the paper's LAM took about 4.5 days on 8×H200, so cost on a single consumer GPU is unmeasured (source: App. B.1, p. 15, reader's inference)

## Limitations / open questions

- Authors: the single effect signal mixes ego motion, agent motion, other agents and environment events; factorising them is future work; the latent is step-level, a hierarchy of skills is future work (App. E.1, p. 18)
- Authors: failures on control-physics mismatch, large reveals and event-driven actions (App. D, p. 17, Fig. 12)
- Authors: the gap to DirectAct/AdaWorld may narrow with larger adaptation capacity than rank-16 LoRA (Sec. 4.4.1, p. 7)
- No quantitative test across different games or engines: probes and RPE are on two MIND subsets of one engine with a shared action space; cross-game transfer is shown only qualitatively (Fig. 6, p. 6), and no policy is trained on the latents (reader's observation, Sec. 4.1, p. 5)
- The comparison is against AdaWorld retrained by the authors on MiraData with their backbone and budgets, not the released checkpoint pretrained on Retro/Procgen games; [[gao2025adaworld#C3]] reports same-action clustering across Habitat, Minecraft and DMLab with that checkpoint, a setting not tested here; no seeds or error bars (reader's observation, Sec. 4.1, p. 5, Tables 1–3)
- The paper's own cross-domain F1 (0.59–0.63) shows decodability does not carry over fully even with the new objective (Table 1, p. 5)
- Open question: does SeqΔ-REPA still help in 2D side-scrolling games, where V-JEPA 2 feature differences may be dominated by sprites and scrolling rather than the agent's action?

## Related

- extends [[gao2025adaworld]]: β-VAE LAM objective like AdaWorld's but with a causal 16-frame clip encoder (not two frames), plus a sequence-level alignment to frozen video-encoder feature changes; compared with a retrained AdaWorld on cross-domain probing, transfer and adaptation; mildly in tension with [[gao2025adaworld#C3]]: retrained AdaWorld latents show weak cross-viewpoint action consistency (Fig. 5a), where the released model's UMAP shows same-action clusters across environments, though the settings differ (retrained on MiraData, MIND views vs Habitat/Minecraft/DMLab)
- related [[hafner2025scalable]]: also pretrains an action-conditioned world model on mostly unlabelled video, using a null action and a small labelled set
- related [[baker2022video]]: the labelled inverse-dynamics alternative to latent actions
- related [[magne2026nitrogen]]: recovers real action labels from gameplay video, whereas this work infers latents
- related [[wang2025gametars]]: unified action space across games by design rather than learned latents
- Decodable, Not Transferable (NeurIPS 2026, not in the KB: PDF unreachable) evaluates AdaWorld and Olaf-World with probes across 193 Stable Retro games; its abstract reports weak linear separability and a large drop on unseen environments

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
