# Knowledge base

The knowledge base (KB) is markdown under `kb/`: one note per paper, optional topic syntheses,
and a generated index. The markdown is the source of truth; everything generated from it can be
rebuilt with `kb.py index`.

```
kb/INDEX.md            generated: one line per paper, plus the experiments table
kb/papers/<id>.md      one note per paper
kb/papers/_*.md        format examples (not papers; see "Examples" below)
kb/topics/<topic>.md   optional hand-written syntheses across papers
```

## Paper ids

`<firstauthor><year><keyword>`, all lowercase ASCII, matching `^[a-z]+[0-9]{4}[a-z0-9]+$`:
- `firstauthor`: the first author's family name, with diacritics folded and spaces, hyphens and
  apostrophes dropped (`Müller-Brockmann` → `mullerbrockmann`);
- `year`: the year of the version read (the venue's year, or the arXiv year for a preprint);
- `keyword`: the most distinctive word of the title, not necessarily the first
  (`kerbl2023gaussian`, `bridson2007poisson`).

If an id is taken, pick a different keyword. An id never changes once it's on `main`: other
notes, experiments and results refer to it.

## Note format

A note is `kb/papers/<id>.md`: frontmatter, a `# <title>` heading, then these sections, all
present and in this order. `templates/paper-note.md` is the starting point, and
`kb/papers/_example.md` shows a filled-in note.

| Section | Contents |
|---|---|
| `## TL;DR` | At most 3 non-empty source lines (wrap at about 100 columns): what the paper does and its main result. It feeds `INDEX.md`. |
| `## Problem` | What problem it solves, and what was lacking before. |
| `## Method` | How it works, with references (`Eq. 5, Sec. 4, p. 4`). |
| `## Claims` | The paper's testable claims, one per line, with stable ids (below). |
| `## Implementation notes` | What an implementation needs that the paper doesn't state, each with its source. |
| `## Limitations / open questions` | Limitations the paper admits, and the reader's open questions (marked as such). |
| `## Related` | Links to other notes, each with a relation. |
| `## Experiments` | Generated block: the experiments that test this paper's claims. |

A section with nothing to say holds `- none`. A `skimmed` note may hold `- not read yet` in
`Problem`, `Method`, `Implementation notes` and `Limitations / open questions`; `TL;DR` and
`Claims` are always filled in.

Keep notes under about 250 lines (`kb.py check` warns above that): the KB is searched, not read
end to end. Paraphrase; quote only short passages where the wording matters, in double quotes
with a reference. Don't copy figures (see `docs/intake.md`).

### Frontmatter

Between `---` lines at the top of the file. Every key is present in every note, in this order:

| Key | Value |
|---|---|
| `id` | The paper id; equals the file name without `.md`. |
| `title` | The paper's title, as printed. |
| `authors` | Inline list, `Given Family` form: `[Robert Bridson]`. |
| `year` | Four digits. |
| `venue` | Short venue name and kind: `SIGGRAPH 2007 sketch`, `arXiv preprint`. |
| `url` | The canonical landing page (publisher, project page or arXiv abstract). |
| `arxiv` | arXiv id without version (`2308.04079`), or empty. |
| `doi` | DOI without `https://doi.org/`, or empty. |
| `pdf_sha256` | SHA-256 of the PDF that was read (`papers/<id>/paper.pdf`), lowercase hex; empty only if no PDF was read. |
| `tags` | Inline list of lowercase kebab-case tags: `[sampling, blue-noise]`. |
| `status` | `skimmed`, `read` or `verified` (below). |
| `read` | Date of the last reading, `YYYY-MM-DD`. |

Statuses:
- `skimmed`: TL;DR and the main claims only; the other sections may say `- not read yet`.
- `read`: the whole paper was read; every section is filled in and every claim is referenced.
  This is what a reader task produces.
- `verified`: a `read` note at least one of whose claims (not withdrawn) has a `supports`
  latest verdict. Per claim, the latest verdict counts: among the `results.json` files with
  `exploratory: false` that are not under an `abandoned` plan, the one with the latest `date`
  (ties broken by run id) decides. So a later `contradicts` or `inconclusive` removes
  `verified` and the note goes back to `read`. All verdicts show in the note's `Experiments`
  block and in `INDEX.md`.

`verified` is derived, and `kb.py check` enforces it exactly: a note whose status is `read` or
`verified` must say `verified` if and only if such a latest verdict exists. So:
- the experimenter sets `verified` on the `read` notes that gain a qualifying latest verdict,
  in the commit that records the results;
- a re-read writes `verified` when a qualifying verdict exists (the reader checks the note's
  `Experiments` block) and `read` otherwise, so it never downgrades a verified note;
- a `skimmed` note stays `skimmed` whatever its verdicts; when it is later read in full, the
  reader sets `verified` or `read` by the same rule;
- when the last qualifying verdict goes away (a later non-supporting verdict, results removed or
  abandoned, or the claim withdrawn), the note goes back to `read`, and `kb.py check` fails until it does.

The frontmatter is a restricted YAML subset, so the stdlib-only tools can parse it without a
YAML library (and the same subset is used by experiment plans, `docs/experiments.md`):
- one `key: value` per line; keys are lowercase `[a-z0-9_]+`; no nesting, no multi-line values;
- the value is the rest of the line after the first `:`, trimmed; `key:` with nothing after it
  is an empty value;
- a value wrapped in double quotes has them removed (`\"` inside is a literal quote); quote a
  value that starts with `[` or `"` but isn't a list;
- a value starting with `[` and ending with `]` is a list; items are separated by commas and
  trimmed, and an item containing a comma is double-quoted; `[]` is the empty list;
- there are no comments: `#` is an ordinary character (`example2000jittered#C1`).

### Claims

One claim per line under `## Claims`:
```
- C1: <the claim, paraphrased with its numbers and conditions> (<reference>)
```
The line ends with the reference group, optionally followed by a single `.`.
- **Ids** are `C1`, `C2`, … in the order assigned. Once a note is on `main`, an id is never
  renumbered, reused or deleted; a re-read adds new claims after the last id. A claim that turns
  out wrong in the note (not in the paper) is kept and marked `- C4 [withdrawn: <why>]: …`;
  `<why>` must not contain `]`.
- **Full claim ids** are `<paper id>#C<n>` (`example2000jittered#C1`). Experiments and other
  notes use them; within a note, `C1` is enough.
- **The reference** is the last parenthesized group on the line (it contains no parentheses
  itself; the claim text before it may) and holds at least a page or a section. Its parts are
  separated by commas, from: `p. N`, `pp. N–M` (or `N-M`), `Sec. X`, `Eq. X`, `Fig. X`, `Table X`,
  `Alg. X`, `App. X`. `kb.py check` rejects a claim without one. Page numbers `N` and `M` are
  digits. A label `X` is written as the paper prints it: arabic or Roman numerals and capital
  letters, joined by `.` or `-`, with an optional lowercase sub-label (`Sec. 3.2.1`,
  `Sec. VI-E`, `Sec. III-C7`, `Table III`, `Table 1c`, `Fig. 9a`, `App. A.2`).
- **Page numbers** are PDF pages counted from 1, the numbers the Read tool's `pages` parameter
  takes, not the page numbers printed on the paper; a reviewer must be able to open the page
  directly.
- A claim is something the paper asserts and an experiment could check: a bound, a measured
  result, a property of the method. Background and definitions go in `Method`.

### Implementation notes

What an implementation needs that the paper leaves out or leaves vague, one per line, each with
its source:
```
- <value or decision> (source: <reference in this paper | [[other-id]] | authors' code <url>@<commit> | assumption>)
```
A value with `source: assumption` is the reader's guess and is labelled as one.

### Links and Related

`[[id]]` links a paper note, and `[[id#C2]]` one of its claims. Links can appear in any
hand-written section; every link must resolve to an existing note (and claim), which
`kb.py check` enforces. `## Related` lists the notes this one is related to, one per line:
```
- <relation> [[id]]: <one line on how>
```
Relations, from this paper's point of view: `extends` (builds on and improves it), `uses` (uses
its method or results as a component), `contradicts` (reports results or claims that conflict
with it), `related` (anything else worth following). Reverse links aren't written by hand:
`kb.py show` lists backlinks.

### Generated blocks

`kb.py index` writes the content between these markers; never edit it by hand:
```
<!-- kb:begin experiments -->
...
<!-- kb:end experiments -->
```
The `## Experiments` section holds only this block. Its content is one line per experiment
whose `plan.md` lists a claim of this paper, sorted by experiment id:
- with a `results.json`:
  `- [<experiment>](../../experiments/<experiment>/results.md): C1 supports, C3 inconclusive`
- without one:
  `- [<experiment>](../../experiments/<experiment>/plan.md) (<plan status>): tests C1, C3`
- with no experiments: `- none`

Claims within a line are in id order, and the verdicts are the ones in `results.json`.

## INDEX.md

`kb/INDEX.md` is generated by `kb.py index` and committed, so a session can read it in one call.
It contains, in order:
1. the line `<!-- Generated by kb.py index; do not edit. -->` and a `# Knowledge base index`
   heading;
2. `## Papers`: a table `| id | year | title | tags | status | TL;DR |`, one row per note, sorted
   by id. Tags are comma-separated; the TL;DR lines are joined with spaces; `|` in a cell is
   escaped as `\|`;
3. `## Experiments`: a table `| experiment | status | claims | verdicts |`, one row per
   experiment, sorted by id. `claims` lists full claim ids; `verdicts` lists
   `<claim id> <verdict>` pairs from `results.json`, or is empty.

`_*.md` examples are left out of both tables.

**Conflicts.** Parallel reader or experiment branches all touch `INDEX.md`, and an experiment
touches the `Experiments` blocks of the notes it tests. When a rebase conflicts in generated
content, take either side, run `kb.py index`, and `git add` the result. CI runs
`kb.py index --check`, so a stale merge fails.

## Topics

`kb/topics/<topic>.md` is an optional synthesis across papers (`<topic>` lowercase kebab-case).
Frontmatter: `id` (the file name), `title`, `tags`, `updated` (`YYYY-MM-DD`). The body is free
markdown; its `[[id]]` and `[[id#C2]]` links are checked like a note's, and any statement about
a paper's results cites the claim.

## Examples

Files in `kb/papers/` whose names start with `_` are format examples, not papers.
`kb/papers/_example.md` describes a fictional paper: `kb.py check` validates it like a note,
except that its id needn't match the file name and its links needn't resolve; `kb.py index`,
`kb.py search` and `kb.py show`'s backlinks skip it. The tests use it as a known-good note.

## Tools

`tools/kb` (entry point `kb.py` at the repo root) is stdlib-only Python for 3.9 and later. It is
not written yet; until it is, follow these formats by hand. Planned commands:
- `kb.py check`: validates the frontmatter, ids, links, claim references and their stability
  (against `origin/main`), note length, the `verified` rule, experiment plans and results
  (including their input fingerprints, `docs/experiments.md`), and that generated content is
  current;
- `kb.py index [--check]`: regenerates `INDEX.md` and the `Experiments` blocks; `--check` exits
  non-zero when they are stale;
- `kb.py new paper <id>`, `kb.py new experiment <slug>`: scaffold from `templates/`;
- `kb.py search <query> [--tag t] [--status s] [--limit n]`: prints `id §section L<line>: snippet`;
- `kb.py show <id>`: frontmatter, heading outline with line numbers, claims and backlinks,
  without the body.

To find something without loading notes whole: read `INDEX.md`, then `kb.py search`, then
`kb.py show`, then only the line ranges you need (Grep works as a fallback).
