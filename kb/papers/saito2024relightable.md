---
id: saito2024relightable
title: Relightable Gaussian Codec Avatars
authors: [Shunsuke Saito, Gabriel Schwartz, Tomas Simon, Junxuan Li, Giljoo Nam]
year: 2024
venue: CVPR 2024 conference paper
url: https://arxiv.org/abs/2312.03704
arxiv: 2312.03704
doi:
pdf_sha256: 1107ed865bb19c165e5f7ad7a77b716abbc905fd0d6afab52ddab4771ec44cdd
tags: [gaussian-splatting, avatars, neural-rendering, relighting, radiance-transfer, cvae, real-time-rendering]
status: read
read: 2026-11-20
---

# Relightable Gaussian Codec Avatars

<!-- Read from arXiv v2 (2024-05-28, CVPR 2024 oral; v1 2023-12-06), 15 PDF pages:
pp. 1-8 paper, pp. 9-12 references, pp. 13-15 appendices A-E. -->

## TL;DR

RGCA: a CVAE expression code (z ∈ R²⁵⁶) and gaze drive 2D-CNN decoders that output one 3D
Gaussian per texel of a 1024² UV map, with learned radiance transfer (SH diffuse, spherical
Gaussian specular), relit in real time; beats MVP/Linear/EyeNeRF combos on most metrics, ~31 ms/frame.

## Problem

Relighting quality is bounded by both geometry and appearance (Abstract, p. 1). Mesh and
volumetric avatars blur hair strands and other sub-millimetre structure; neural relighting
models fed with the lighting as input (point lights or grouped lights from a light stage) need
teacher-student distillation or a linear bottleneck for environment maps, and none achieve
all-frequency reflections on hair and eyes (Sec. 1, p. 2; Sec. 2, p. 3). Physically based
light-stage pipelines don't extend easily to dynamic capture, hair or eyes (p. 2).

## Method

- **Data** (Sec. 3.1, p. 4): 110 cameras at 4096×2668, 460 white LEDs, 90 Hz; ~144,000 frames per
  person of expressions, sentences and gaze. Time-multiplexed lighting: every third frame fully
  lit (for tracking), the rest lit by grouped or random sets of 5 lights. Coarse topologically
  consistent mesh tracking, head pose stabilization, and gaze estimation on fully lit frames,
  interpolated to the partially lit ones.
- **Geometry** (Sec. 3.2, pp. 4-5): standard 3DGS Gaussians (Eqs. 1-2), but one Gaussian per texel
  of a shared UV map of a coarse template mesh. A CVAE encoder E takes tracked mesh vertices V and
  the unwrapped average texture T and gives μ, σ of z ∈ R²⁵⁶ (Eq. 3). The geometry decoder
  D_g(z, gaze e_l, e_r) outputs per Gaussian a position offset δt, rotation, scale and opacity;
  position = barycentric interpolation of the decoded coarse mesh V' = D_v(z) at the texel + δt
  (Eqs. 3-4). V' lets the avatar be driven from a headset or by latent manipulation.
- **Appearance: learnable radiance transfer** (Sec. 3.3, pp. 5-6). Colour = diffuse + specular.
  Diffuse (Eq. 5): static per-Gaussian albedo ρ times the dot product of the light's SH
  coefficients with decoded per-Gaussian transfer coefficients: RGB up to 3rd order plus
  monochrome 4th to 8th order, for sharper shadows within GPU memory. The transfer pre-convolves
  visibility, subsurface and multi-bounce scattering. Specular (Eqs. 6-8): an angle-based
  normalized spherical Gaussian with lobe axis = reflection of the view about a decoded,
  view-conditioned normal, roughness σ, times a learned view-dependent visibility v ∈ (0,1)
  that absorbs Fresnel, occlusion and geometric attenuation (derivation from the microfacet
  rendering equation in App. B, Eqs. 17-22, p. 13). Both terms are linear in the light, so
  point lights and environment maps work without retraining; with a prefiltered environment
  map, specular costs one mipmap lookup per Gaussian (p. 6).
- **Decoders** (Eq. 9, p. 6; App. A, p. 13): view-independent D_ci(z, e) → d^c, d^m, σ (merged
  with D_g into one network) and view-dependent D_cv(z, e, ω_o) → δn, v. z goes through a linear
  layer + leaky-ReLU to 256×8×8; each gaze through a linear layer to 16×2×2, placed at the
  eyeballs' UV locations (zero elsewhere); the camera-to-head-centre direction through a linear
  layer to an 8-d feature tiled spatially. Both decoders are stacks of 4×4 stride-2 transposed
  convolutions + leaky-ReLU, channels (272, 256, 128, 128, 64, 32, 16, 125) and (280, 256, 128,
  128, 64, 32, 16, 4), i.e. 8×8 up to 1024×1024, with weight normalization and untied biases.
- **Explicit eyes** (Sec. 3.4, p. 6; App. A, p. 13): each eyeball is two blended spheres (eyeball,
  cornea) rotated by gaze, parameters refined end to end. Eye Gaussians have frozen positions on
  the sphere and normals fixed to it (glints from point lights are single-pixel and otherwise
  fall into poor minima), and a view-conditioned albedo for corneal refraction. Separate eye
  decoders take gaze, relative head position and rotation (Eq. 10), from a 4×4 start, with an
  eyeball visibility mask in UV for eyelid shadows.
- **Training** (Sec. 3.5, pp. 6-7): L = L_rec + L_reg + λ_kl L_kl; L_rec = 10 L1 + 0.2 D-SSIM + 10
  L2 on V'. L_reg: scale range penalty (Eq. 14, s < 0.1 or s > 10), negative diffuse colour
  penalty (Eq. 15), and three terms keeping eye Gaussians opaque and small (Eq. 16), weights 1e-2
  and 1e-4; λ_kl = 2e-3. Adam, lr 5e-4, 4 A100s, batch 16, 200k iterations.
- **Embedding, for the survey lens**: no transformer. The learned embedding is a per-person CVAE
  latent of facial expression (from tracked geometry + texture), decoded by CNNs on a UV grid;
  gaze, view and lighting enter outside the latent.

## Claims

- C1: On held-out expression segments, 3D Gaussians + explicit eyes + the proposed appearance reach PSNR 36.501 / SSIM 0.943 / LPIPS 0.110; with the same geometry, EyeNeRF appearance gives 34.550 / 0.939 / 0.115; with MVP geometry the proposed appearance gives 35.789 / 0.943 / 0.134 (Table 1, Sec. 4.2, p. 7).
- C2: On held-out lights, the full model reaches PSNR 34.042 / SSIM 0.858 / LPIPS 0.148, vs. 30.7976 / 0.828 / 0.162 with EyeNeRF appearance; MVP geometry + the proposed appearance has the best SSIM (0.877) but lower PSNR (33.778) and worse LPIPS (0.168) (Table 2, Sec. 4.2, p. 7).
- C3: With Gaussian geometry (no explicit eyes), the proposed appearance beats the Linear model of Yang et al. on held-out segments (36.529 vs. 33.831 PSNR, LPIPS 0.111 vs. 0.184) and held-out lights (33.845 vs. 32.829 PSNR, LPIPS 0.148 vs. 0.202), though Linear has the higher held-out-light SSIM (0.870 vs. 0.831); Linear relights blurrily, and view-dependent SH (EyeNeRF) overfits and flickers in animation (Table 1, Table 2, Fig. 5, pp. 7-8).
- C4: Gaussians resolve eyelashes, pores and hair strands better than MVP with the same appearance model, and the explicit eye model gives convincing eye glints (Fig. 4, Sec. 4.2, pp. 7-8).
- C5: Removing the 4th-8th order monochrome SH lowers held-out-light PSNR from 34.042 to 33.762 (LPIPS 0.148 → 0.152) and blurs shadows (Table 3, Fig. 6, App. C, pp. 13-14).
- C6: Removing view conditioning of the normal does not clearly hurt (PSNR 33.927, SSIM 0.864, LPIPS 0.148 vs. 34.042 / 0.858 / 0.148); the authors keep it as more geometrically correct for fibre-like Gaussians and expect it to matter under continuous environments (Table 3, App. C, p. 14).
- C7: The conventional cosine spherical Gaussian is comparable on metrics (33.778 / 0.855 / 0.147) but recovers sharp eye glints worse than the angle-based one (Table 3, Fig. 7, App. C, p. 14).
- C8: A generic head template with only rigid head pose and gaze, without L_geo, still reconstructs surprisingly well, with occasional missing eye glints and blurrier extreme expressions (Fig. 8, App. C, p. 14).
- C9: 32 cameras give good results (16 show artifacts around the eyes, 8 degrade noticeably, more than 32 gives diminishing returns), and 10-20% of the light conditions give acceptable relighting (Fig. 9, Fig. 10, App. C, p. 15).
- C10: Per-frame inference takes 31 ms for the full model (35 with EyeNeRF appearance), 18 ms without explicit eyes, 34 ms with MVP geometry, and 6 ms for the Linear appearance; splatting alone takes 12.84 ms for 1024² = 1 Mi Gaussians and 6.40 ms for 512² on an A100, with no pruning (Table 4, App. D, p. 15).
- C11: A 512² (256 Ki Gaussian) version is relit in real time on a tethered consumer VR headset, and avatars can be driven live from head-mounted-camera images (Fig. 1, p. 1, App. D, p. 15).
- C12: Gaussian-based models converge within 3 days; MVP-based ones need twice the iterations (400k) (App. D, p. 15).

## Implementation notes

- What the decoder outputs per frame, per Gaussian (1024² of them): 125 view-independent channels and 4 view-dependent ones. The 125 match δt (3) + quaternion (4) + scale (3) + opacity (1) + RGB SH to order 3 (16×3 = 48) + monochrome SH orders 4-8 (81−16 = 65) + roughness (1); the 4 match δn (3) + visibility (1) (source: App. A, p. 13, Eqs. 3, 5, 9; the per-channel split is the reader's arithmetic, an assumption).
- The eye decoders' 122 and 7 output channels: 122 matches the head's 125 minus δt; how the 7 view-dependent channels split (albedo 3 + visibility 1 + 3 more) is not stated (source: App. A, p. 13; assumption).
- For a Vulkan splatting renderer: shading is per Gaussian before splatting (SH dot product with the light's 81 SH coefficients, one prefiltered environment-map mip lookup for the specular SG), and the authors note it scales with Gaussian and avatar count and suggest moving it to a per-pixel fragment shader (source: Eqs. 5, 8, p. 6; Sec. 5, p. 8). That a standard 3DGS rasterizer fed precomputed per-Gaussian colours then suffices is the reader's inference (source: assumption).
- How the SG width σ maps to a mip level of the prefiltered environment map is not given (source: p. 6, citing [23, 47]; assumption).
- Encoder E's architecture is not described; only the decoders are (source: App. A, p. 13).
- Camera count is 110 in Sec. 3.1 but "the full 149 cameras" in Fig. 9 (source: pp. 4, 15); unresolved.
- L_geo is an L2 loss on the decoded coarse geometry V′; that its target is the tracked mesh V is the reader's reading (source: Eq. 12, p. 6; assumption).

## Limitations / open questions

- Needs coarse mesh and gaze tracking as preprocessing, sensitive to tracking failures; end-to-end learning with tracking is future work (Sec. 5, p. 8).
- In-the-wild capture is hard without known illumination (Sec. 5, p. 8).
- Relighting is per Gaussian and scales linearly with the number of avatars (Sec. 5, p. 8).
- Specular is monochrome, so coloured highlights (dielectrics, multi-bounce specular) are not modelled (App. B, p. 13).
- The expression latent is personalized per subject: one model per consenting, densely captured person (App. E, p. 15).
- Open question (survey lens): the latent is learned from tracked meshes and textures, not from raw frames; whether a codec-style CVAE latent of this kind transfers across identities is not tested here.
- Open question: inference time (Table 4) presumably includes decoding plus splatting at 1024²; the split between CNN decoding and splatting is only given for splatting.

## Related

- none

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
