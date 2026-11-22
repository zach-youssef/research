# Recent-work scan

How to check for papers published since the last scan that the KB doesn't cover. The output is a
candidate list for the user: nothing is fetched or read until they pick (docs/intake.md, "Who
picks papers").

## Procedure

1. Run the web searches and the arXiv queries below with the window starting at the last scan's
   date (the log at the end). Add queries for topics the KB has grown into since.
2. Drop candidates already in `kb/INDEX.md` or already listed as queued or optional in the
   survey plan.
3. For each remaining candidate, confirm the arXiv id against its title (one `id_list` call),
   and note the venue if it was published.
4. Show the user each candidate with its id, year, title, one line on what it adds to the KB
   and a recommendation (read, optional, skip). Record the abstracts of those marked optional
   so they can be revisited without another search.
5. Append a line to the log below.

## Web searches

Run them in extended mode: recent papers are what this scan is for, and standard results lag.
Replace the years with the current window.

- `generalist game-playing agent transformer many games 2025 arXiv`
- `cross-domain game state embedding transfer reinforcement learning unseen games 2025 2026`
- `world model pretraining across many video games action-agnostic latent actions 2025`
- `vision-language-action model video games benchmark 2025 2026 arXiv`
- `object-centric slot representation reinforcement learning Atari generalization 2025`
- `latent action policies learning from unlabeled gameplay videos Genie LAPO 2024 2025 cross-game`

Then search each promising hit by name (standard mode) to find its arXiv id and venue, e.g.
`Game-TARS unified action space keyboard mouse generalist agent arXiv`.

## arXiv API queries

The same topics as arXiv API searches, so the results can be repeated and filtered by date.
Set `FROM` and `TO` (`YYYYMMDDHHMM`), keep at least 3 s between calls, and use only the
`research-kb-reader/0.1` User-Agent (docs/intake.md).

```sh
FROM=202505210000; TO=202611212359
for q in \
  'abs:generalist AND abs:game AND abs:agent' \
  'abs:"cross-domain" AND abs:game AND abs:transfer' \
  'abs:"world model" AND abs:"video games"' \
  'abs:"latent action" AND (abs:game OR abs:video)' \
  'abs:"vision-language-action" AND abs:game' \
  'abs:"object-centric" AND abs:reinforcement AND abs:Atari' \
  'abs:"unseen games" OR abs:"held-out games"'
do
  curl -s -A research-kb-reader/0.1 --get https://export.arxiv.org/api/query \
    --data-urlencode "search_query=($q) AND submittedDate:[$FROM TO $TO]" \
    --data-urlencode max_results=50 --data-urlencode sortBy=submittedDate |
    grep -E '^\s*<(id|title|published)>' | sed 's/^ *//'
  sleep 3
done
```

Searching by title words finds papers whose ids you don't know, e.g.
`ti:latent AND ti:actions AND ti:worlds`. A paper that isn't on arXiv (some conference-only
papers) won't turn up; ask the user for a PDF or link.

## Log

- 2026-11-21: first scan, web searches only, over 2025-26. Queued NitroGen (2601.02427),
  Game-TARS (2510.23691), Dreamer 4 (2509.24527), AdaWorld (2503.18938), ORCA (2610.28197) and
  "Decodable, Not Transferable: Latent Actions Across Worlds" (NeurIPS 2026, not on arXiv);
  recorded SIMA 2, GATN, OneLA, VideoGameBench, GameVerse and the 2605.09965 survey as optional.
