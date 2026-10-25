# <experiment title>: results

Experiment `<YYYY-MM-DD>-<slug>`, run on <YYYY-MM-DD> at research `<short sha>` (code id `inputs.src` `<short id>`). Plan:
[plan.md](plan.md). Machine-readable results: [results.json](results.json).

## Verdicts

| Claim | Verdict | Evidence |
|---|---|---|
| <paper id>#C<n> | <supports \| contradicts \| inconclusive> | <metrics and values, against the criterion> |

## Metrics

| Metric | Value | Unit | n | CI |
|---|---|---|---|---|
| <name> | <value> | <unit> | <n> | <[low, high] at level, or blank> |

## Deviations from the plan

- <what was done differently and why, or none>

## Figures

- <![<what it shows>](figures/<name>.png), or none>

## Reproduce

```sh
experiments/<YYYY-MM-DD>-<slug>/run.sh
```

<Expected wall clock, and the tolerances from plan.md's success criteria: `<metric>` within <number or number%>. A rerun reproduces when every criterion's verdict is unchanged and each toleranced metric is within its tolerance.>

## Notes

- <anything surprising, and experiments worth running next, or none>
