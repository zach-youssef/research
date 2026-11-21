---
id: zielonka2025drivable
title: Drivable 3D Gaussian Avatars
authors: [Wojciech Zielonka, Timur Bagautdinov, Shunsuke Saito, Michael Zollhöfer, Justus Thies, Javier Romero]
year: 2025
venue: 3DV 2025 conference paper
url: https://arxiv.org/abs/2311.08581
arxiv: 2311.08581
doi:
pdf_sha256: 0a8fcf2d0156999f0c85526e29dbefe6004f7617a0768188ddd45111100f3f12
tags: [gaussian-splatting, avatars, neural-rendering, cage-deformation, drivable, real-time-rendering]
status: read
read: 2026-11-20
---

# Drivable 3D Gaussian Avatars

<!-- Read from arXiv v2 (2025-02-10, the 3DV 2025 version; v1 2023-11-14), 16 PDF pages:
pp. 1-9 paper, pp. 9-12 references, pp. 13-16 supplement (App. A-C). -->

## TL;DR

D3GA: full-body avatars as 3D Gaussians embedded in tetrahedral cages (body, face, garments),
deformed by LBS plus small pose-conditioned MLPs, with each Gaussian's covariance warped by its
tetrahedron's deformation gradient; best PSNR/SSIM vs. MVP/BD and MLP 3DGS avatars, ~26 FPS on a V100.

## Problem

Drivable photoreal avatars for telepresence are driven by several signals (body pose, facial
expression, garments) that a single-layer model handles poorly; garments slide over the body
(Sec. 1, p. 1). Mesh-attached volumetric primitives (MVP) fail when the tracked mesh is imprecise,
CNN decoders fix the number of primitives and can't decompose garments or condition parts
locally, some state-of-the-art avatars need dense driving inputs (RGB-D, multi-view) at test
time, and NeRF/3DGS avatars warp with LBS, whose few degrees of freedom can't express stretching
or other non-linear motion (pp. 1-2). CNN-regressed Gaussian maps reach ~1 GiB models and ~10 FPS
(Sec. 2, p. 3).

## Method

- **Base renderer**: 3DGS (Eqs. 1-3, Sec. 3.1, p. 3); alpha compositing as in 3DGS, but per-Gaussian
  colour and opacity come from an MLP instead of spherical harmonics (Eq. 10, Sec. 3.5, p. 5).
- **Cages** (Sec. 3.2, pp. 3-4): images are segmented (EfficientNet + PointRend, trained on
  similar captures) into body/upper/lower garment; labels are projected onto a low-resolution LBS
  body mesh fitted to one 3D scan and to 3D keypoints. Per-part templates are unposed with LBS,
  garments inflated 1-3 cm along normals, voxelized, re-meshed with marching cubes and
  tetrahedralized with TetGen. Garment cages are hollow shells; the body cage is solid; the face
  cage is the body tetrahedra containing face triangles. Cage nodes get LBS weights from the
  nearest template vertex.
- **Gaussians in tetrahedra** (Sec. 3.3-3.4, pp. 4-5): each Gaussian's mean is a learnable
  barycentric combination of its tetrahedron's 4 canonical vertices (Eq. 5); after LBS posing
  of the nodes, the same barycentrics give the posed mean. The deformation gradient
  J = Ê E⁻¹ (edge matrices of the posed and canonical tetrahedron, Eqs. 6-7) warps the covariance,
  Σ̂ = J Σ Jᵀ (Eq. 8), so Gaussians rotate, stretch and shear with the cage. Gaussians are
  initialised on the template surface, first two axes in the triangle plane, scale from
  inter-point distances as in 3DGS; densification is possible but the count is fixed by default.
- **GaussianNet per part** (Eq. 9, p. 5), three MLPs, each 3 hidden layers of 128 with ReLU:
  - Ψ: pose φ + positional encoding of canonical cage vertices → cage node offsets Δv (like SMPL
    pose correctives);
  - Π: pose + each Gaussian's canonical barycentrics b (R⁴), rotation q (R⁴), scale s (R³) →
    corrections Δb, Δq, Δs;
  - Γ (shading): pose + view direction encoded on 4 SH bands + an auto-decoded per-Gaussian
    feature h_i (DeepSDF-style) + f_j → colour c_i and opacity o_i; pose-dependent colour models
    self-shadows and wrinkles. (f_j appears in Eq. 9 but is not defined in the text.)
- **Driving codes** (p. 5): body and garments take joint angles φ; the face takes an embedding κ
  that a small auxiliary MLP regresses from 150 3D face keypoints normalized by training mean/std.
  A per-frame learned time embedding (NeRF-W-style) absorbs what the inputs can't explain (cloth
  dynamics); at test time the mean training embedding is used. No transformer or VAE anywhere:
  the "embeddings" are joint angles, a keypoint-regressed face code, auto-decoded per-Gaussian
  features and a per-frame latent.
- **Objective** (Eqs. 11-13, p. 5): L = ν L_Color + ν L_Garment + τ L_Neo with ν = 10, τ = 0.005.
  L_Color = (1-ω) L1 + ω D-SSIM + ζ VGG, ω = 0.2, ζ = 0.005 after 400k iterations (0 before).
  L_Garment is L1 between a render with constant per-part colours and the ground-truth
  segmentation. L_Neo is a Neo-Hookean energy on each tetrahedron's J (volume and shear),
  regularizing poorly observed tetrahedra.
- **Training** (p. 5): Adam, lr 5e-4, multi-step decay 0.33, batch 1, 700k steps (own data) or
  400k (ActorsHQ), one V100, 1024×667 images. On ActorsHQ (no ground-truth poses) poses regressed
  from keypoints are refined during training and at test time, optionally projected on a
  training-set PCA basis.
- **Data** (Sec. 4, pp. 5-6): 9 subjects, 200 cameras, 4096×2668 downsampled to 1024×667; 12,000
  training frames (10 FPS) and 1,500 test frames (30 FPS); 4 subjects public as Goliath-4.
  ActorsHQ with 40 cameras and SMPL-X fits from OpenPose + scan-to-mesh (Sec. 5.1, p. 6).

## Claims

- C1: On the authors' dataset D3GA gets PSNR 30.634 / LPIPS 0.054 / SSIM 0.964, vs. MVP 28.795 / 0.051 / 0.955 and BodyDecoder 29.918 / 0.044 / 0.959: best PSNR and SSIM, worst LPIPS of the three (Table 1, Sec. 5.1, p. 6, p. 8).
- C2: On ActorsHQ D3GA gets PSNR 26.562 / LPIPS 0.065 / SSIM 0.944, vs. GaussianAvatar 24.731 / 0.088 / 0.933, 3DGS-Avatar 21.709 / 0.082 / 0.915 and Animatable Gaussians 26.454 / 0.055 / 0.937; AG is sharper (better LPIPS), which the authors attribute to its much larger CNN (StyleUNet) (Table 1, Fig. 5, Sec. 5.1, pp. 6-8).
- C3: D3GA has 9M parameters (45 MiB), vs. GaussianAvatar 7M (59 MiB), 3DGS-Avatar 6M (57 MiB) and Animatable Gaussians 232M (862 MiB) (Table 2, p. 8).
- C4: Rendering at 1024×667 on a V100 runs at 28 / 26 / 23 / 20 FPS with 25k / 100k / 200k / 300k Gaussians, while PSNR stays within 29.825-29.938 and SSIM at 0.960; the authors call 100k-200k the best speed/quality compromise (Table 3, App. A, p. 14).
- C5: Replacing tetrahedra with triangles (barycentrics in R³, no deformation gradient) gives holes and artifacts and worsens LPIPS from 0.0648 to 0.0703 (Fig. 8, Sec. 5.2, pp. 6-8).
- C6: Without L_Garment, PSNR/LPIPS/SSIM are 30.140 / 0.057 / 0.961 vs. 29.825 / 0.056 / 0.960 with it (PSNR and SSIM slightly better without, LPIPS slightly worse), but garment boundaries become blurry and semi-transparent; a single-layer avatar (29.740 / 0.057 / 0.959) over-smooths the T-shirt/jeans boundary and models sliding poorly (Table 4, Fig. 7, Fig. 15, p. 8, pp. 14-15).
- C7: Disabling the corrective networks and deforming only the body triangle mesh with LBS gives test-time artifacts from wrongly oriented Gaussians (Fig. 11, App. A, p. 13).
- C8: L_Neo keeps tetrahedra from shearing, flipping or losing volume, mostly visible in regions without supervision such as under garments (Fig. 13, App. A, pp. 13-15).
- C9: Avatars can be driven by poses of other subjects (motion transfer) and decomposed into independently controlled garment layers (Fig. 6, Fig. 9, Fig. 16, p. 8, p. 13, p. 15).
- C10: Discussing NPC's nearest-neighbour searches during training, the paper contrasts "12 hours vs. 30 minutes for our model"; it doesn't say what the 30 minutes cover (Sec. 2, p. 2).
- C11: Long garments (dresses, long skirts) are a common failure for pose-conditioned methods; MLP-based methods including D3GA give stable but incorrect results on the SMPL template, while Animatable Gaussians needs a per-frame tracked garment template and fails completely without one (Fig. 12, App. A, pp. 14-15).

## Implementation notes

- Per frame the decoder must output, per part, cage node offsets (Ψ), per-Gaussian Δb/Δq/Δs (Π) and per-Gaussian colour + opacity (Γ); LBS of the cage nodes, J per tetrahedron and J Σ Jᵀ per Gaussian then feed a standard 3DGS rasterizer as precomputed colours, no SH (source: Eqs. 5-10, pp. 4-5).
- The supplement's Eq. 17 writes J = V Ṽ⁻¹, the inverse of Eq. 7's Ê E⁻¹ (edges of posed times inverse of canonical); Eq. 7 is the one consistent with Eq. 16, so use J = Ê E⁻¹ (source: Eqs. 7, 16-17, pp. 4, 14).
- Table 3 doesn't say whether its FPS includes the MLP decoding, and the Gaussian count behind Table 1 isn't given (source: Table 3, p. 14). Table 3/4's "Ours" (PSNR 29.825) differs from Table 1's (30.634), so they are probably a different subset or setting (source: assumption).
- Neo-Hookean Lamé parameters λ, μ are not given (source: Eq. 12, p. 5); take them from Macklin and Müller 2021 or tune (source: assumption).
- f_j in Γ's input and the dimension of h_i, κ and the time embedding are not given (source: Eq. 9, p. 5); authors' code not checked (source: assumption).
- Multi-step scheduler milestones are not given, only the decay rate 0.33 (source: p. 5).
- Training wall-clock time is not given beyond the 30-minute remark (C10), which seems short for 700k steps at batch 1 on one V100; treat it as unverified (source: pp. 2, 5; assumption).
- Inputs needed at test time: joint angles (and 150 face keypoints for the face); the mean training time embedding (source: p. 5).

## Limitations / open questions

- Less sharp than CNN decoders; high-frequency patterns like stripes blur; the authors suggest a VAE regressing Gaussians per texel of a guide mesh as in AG/MVP (Sec. 6, p. 8).
- Self-collisions of loose garments remain hard despite L_Garment; sparse driving signals don't carry enough information about complex wrinkles or self-shadowing (Sec. 6, p. 8; Fig. 12, p. 15).
- Appearance isn't relightable; the authors name a relightable appearance model as future work (Sec. 6, p. 8).
- Person-specific: needs a dense multi-view capture of a consenting subject (Sec. 6, p. 8).
- Open question: the per-frame decoder is small MLPs over every Gaussian (~100k+) and 3 networks per part; is MLP cost per frame included in the Table 3 FPS, and how does it compare with a Vulkan compute pass doing the same?
- Open question (survey lens): the driving state is a hand-designed low-dimensional code (joint angles, keypoints) rather than a learned visual embedding; only the face code κ is learned, from keypoints.

## Related

- related [[saito2024relightable]]: same lab's head counterpart, a CVAE decoding Gaussians per UV texel; D3GA cites it as a CNN-based Gaussian-map method and names relighting as future work.

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
