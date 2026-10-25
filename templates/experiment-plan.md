---
id: <YYYY-MM-DD>-<slug>
title: <one line>
claims: [<paper id>#C<n>, <paper id>#C<n>]
status: proposed
gpu: <yes | no>
budget_wall_min: <minutes, all runs>
budget_downloads_mb: <MB>
ci_smoke: <true | false>
approved:
---

# <title>

## Hypothesis

<What we expect and why, in a few lines, citing the claims.>

## Success criteria

- <paper id>#C<n>: supports if <metric> <comparison>; contradicts if <metric> <comparison>; otherwise inconclusive.
- tolerance <metric>: <number, or number%>

## Method

<Inputs, parameters, seeds, repetitions, what is measured and how. Domain-repo code needed, if
any. Anything vendored, with its URL, commit and license.>

## Budget

<How the wall-clock and download numbers were estimated, and why the GPU is or isn't needed.>
