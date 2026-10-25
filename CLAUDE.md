# research

A knowledge base of research-paper notes (`kb/`), experiments that test their claims
(`experiments/`), and the stdlib-only Python tools that check and search them (`tools/`).

Read the docs for the area you touch:

- [docs/knowledge-base.md](docs/knowledge-base.md): before writing or changing a note, searching
  the KB, or changing `tools/kb`. Note format, ids, claims, links, generated blocks, `INDEX.md`.
- [docs/intake.md](docs/intake.md): before fetching or reading a paper. Who picks papers, where
  PDFs live, reading big PDFs, licensing.
- [docs/experiments.md](docs/experiments.md): before proposing, running or reviewing an
  experiment. Layout, `plan.md`, `results.json`, reproducibility, budgets, the GPU slot.

Templates for new notes, plans and results are in `templates/`; `kb/papers/_example.md` is a
filled-in note.

Rules that apply everywhere:

- Never commit PDFs, page renders or figures from papers; PDFs live in the main checkout's
  git-ignored `papers/<id>/`.
- `kb/INDEX.md` and the notes' `Experiments` blocks are generated: run `kb.py index`, never
  edit them by hand, and resolve rebase conflicts in them by regenerating.
- Claim ids (`<paper id>#C<n>`) are never renumbered, reused or deleted once on `main`.
- Branches: `paper-<id>` (or `papers-<topic>`) for notes, `exp-<slug>` for experiments.
- Tools target Python 3.9 (the system `python3`) with the standard library only; experiments
  pin their own dependencies.

Open items: `tools/kb`, `tools/fetch_paper.py`, `tools/pdf_pages.swift`, `tools/lab` and CI are
not written yet. Until they are, follow the formats by hand.
