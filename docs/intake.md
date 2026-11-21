# Paper intake

How papers get into the repo: who picks them, where the PDFs live, and how to read big ones.

## Choosing papers

The user picks every paper that is read:
- they supply arXiv ids, URLs or local PDFs; or
- they ask for a search, and pick from the candidates it lists.

Agents never fetch or read papers nobody picked. A reader ends its report with "papers worth
reading next" for the user to choose from; it doesn't fetch them. Before reading a paper, check
`kb/INDEX.md`: a paper already in the KB is re-read only when asked, and a re-read keeps its
claim ids (`docs/knowledge-base.md`).

## Storing PDFs

PDFs are **never committed**: their licenses vary and they are large. They live git-ignored in
the **main checkout**, not a worktree, because worktrees are removed after a merge:
```
<main>/papers/<id>/paper.pdf     the PDF that was read
<main>/papers/<id>/pages/pN.txt  per-page text (N counts PDF pages from 1)
<main>/papers/<id>/pages/pN.png  page renders, made on request
```
`<main>` is the main checkout, found from any worktree as the parent of
`git rev-parse --path-format=absolute --git-common-dir`; tools resolve it that way and never
hard-code a path. Wherever the docs write `papers/<id>/…` (in `pdf_sha256`, in the fetch tool),
they mean `<main>/papers/<id>/…`, even when run from a worktree.
Supplementary material goes next to it (`supp.pdf`, …). The note's `pdf_sha256` is the SHA-256
of `paper.pdf` (`shasum -a 256`), so a later reader or reviewer can tell it has the same file;
when re-fetching gives a different hash, the paper changed version, and the note says which
version it describes.

## Fetching

`tools/fetch_paper.py` (not written yet) will:
- take an arXiv id, a URL or a local PDF; download (or copy) it to `papers/<id>/paper.pdf` and
  record `pdf_sha256`;
- look up arXiv metadata through the export.arxiv.org API, with the User-Agent
  `research-kb-reader/0.1` and at least 3 s between calls, as arXiv's API terms ask. Never put
  the user's name, email or other identity in a request's headers or URL;
- scaffold the note with `kb.py new paper` and fill in the frontmatter;
- with `--search "<query>"`, list candidates (id, year, title, authors) and download nothing.

Until it exists, download by hand into the layout above and fill the frontmatter from the
paper's landing page. Prefer the arXiv or the authors' version when the publisher's is
paywalled, and record which one was read in `url`. Requests made by hand follow the same rules:
use `curl -A research-kb-reader/0.1` (a fetch tool that can't set the User-Agent is fine as long
as nothing identifying is added), no identity, and 3 s between calls to arXiv.

## Reading

- **Read tool first.** Read the PDF with the `pages` parameter, at most 20 pages per call, so
  figures, tables and equations are seen as printed.
- **Big or scanned PDFs.** When Read can't take the PDF (the 117 MB 3DGS PDF was too big) or a
  page has no text layer, `tools/pdf_pages.swift` (not written yet; it uses macOS PDFKit, so it
  needs no install) writes `pages/pN.txt` for every page and, on request, `pages/pN.png`. Read
  the PNGs to see the page, and Grep the text to find where something is said. PDFKit's text
  for math and multi-column layouts is rough: check equations and numbers against the page
  image before writing them into a note.
- Read appendices and the supplementary material the brief points to; the reader's report says
  what it didn't read.

## Licensing

- Notes paraphrase. Quote only short passages where the exact wording matters, in double
  quotes with a page reference.
- Don't commit figures, tables or page images from papers, even cropped. Committed figures are
  produced by our own experiments (`docs/experiments.md`).
- Code from a paper's repository is used under its license; an experiment that vendors any
  records the URL, commit and license in its `plan.md` method.
