---
id: example2000jittered
title: "Jittered Grids: Fast Sampling with a Minimum Distance"
authors: [Ada Example, "Ben Sample, Jr."]
year: 2000
venue: Fictional Workshop on Sampling 2000
url: https://example.org/papers/jittered-grids
arxiv:
doi: 10.0000/example.2000.1
pdf_sha256: 0000000000000000000000000000000000000000000000000000000000000000
tags: [sampling, blue-noise, example]
status: read
read: 2026-10-25
---

# Jittered Grids: Fast Sampling with a Minimum Distance

<!-- A format example: this paper is fictional. See docs/knowledge-base.md, "Examples". -->

## TL;DR

Places one random point per cell of a grid whose cell size is the minimum distance, and rejects
points too close to a neighbour. Claims linear time in the number of points, and a minimum
distance that always holds, at the cost of fewer points than dart throwing.

## Problem

Dart throwing gives well-spaced samples but slows down sharply as the domain fills, because
most candidates are rejected (Sec. 1, p. 1). The paper wants a sampler whose cost grows
linearly with the number of points.

## Method

- A grid with cell size r covers the domain; each cell gets one uniform candidate (Sec. 2, p. 2).
- Cells are visited in scan order; a candidate closer than r to an accepted point in the 3×3
  neighbourhood is rejected (Alg. 1, p. 2).
- The acceptance test is the squared distance against r² (Eq. 2, Sec. 2, p. 2).

## Assumptions

- A1: The dimension is fixed, so a cell's neighbourhood has a constant number of cells (Sec. 3, p. 3).
- A2: The domain is the unit square in 2D (Sec. 4, p. 4).
- A3: Candidates in different cells are drawn independently (reader's inference, Alg. 1, p. 2).

## Claims

- C1: Run time is linear in the number of grid cells, for a fixed dimension [A1] (Sec. 3, p. 3).
- C2: Every pair of accepted points is at least r apart [] (Sec. 3, Eq. 3, p. 3).
- C3 [withdrawn: the paper reports 0.6 of dart throwing's count, not 0.9; see C4]: Accepts about 0.9 as many points as dart throwing (Table 1, p. 4).
- C4: Accepts 0.6 ± 0.05 times as many points as dart throwing, over 10 runs [A2, A3] (Table 1, p. 4).

## Implementation notes

- Candidates use a uniform generator; the paper doesn't name one (source: assumption).
- The domain is the unit square, r = 0.01 for Table 1 (source: Sec. 4, p. 4).
- Ties at distance exactly r are accepted (source: Eq. 2, p. 2, which uses ≥).

## Limitations / open questions

- The point count is lower than dart throwing's (Table 1, p. 4).
- Open question: does the scan order bias the point density towards the first rows?
- Open question: C2 restates [[sample1999grid#C2]]; is the proof here any tighter?

## Related

- extends [[sample1999grid]]: replaces the regular grid with one jittered candidate per cell.
- uses [[sample2001hashing]]: the hash grid for storing accepted points (Sec. 2, p. 2).

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
