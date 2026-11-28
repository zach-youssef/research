"""JOWA's released weights: which files, their SHA-256s, and fetching them one at a time.

The GitHub repo's ``checkpoints/JOWA/*.pt`` are Git LFS pointers and the repository has LFS
disabled (batch API 403), so the weights come from the README's Google Drive folder. The
40M file's SHA-256 equals the LFS pointer's oid in the repo (experiment
``2026-11-24-jowa-mac-smoke``). The weights' licence is unknown (README.md here): they are
never committed or uploaded.

    python -m gthbaselines.jowa_weights DIR JOWA_40M [JOWA_150M]

fetches the tokenizer and the named sizes into DIR (a few hundred MB: 75 MB tokenizer, 76 MB
40M, 511 MB 150M), skipping files already there with the right size, and checks every hash.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

DRIVE_FOLDER = "1YHaCemhobchJWE5zt28TUrKp2tp8yysF"
"""The Google Drive folder JOWA's README links to."""

FILES = {  # name: (SHA-256 of the official Drive file, bytes)
    "tokenizer.pt": ("4f624acf7ae82cfdeb6bc3b11651d073bc168fa5300f70a3b8ba029b07c68a4a", 75162490),
    "JOWA_40M.pt": ("c1c3d8242141725a2a4bfa61e899e18ff8153a2600a2803d7aa98405366fd771", 76485274),
    "JOWA_150M.pt": ("3f60776688ead94a684b359f23709327600af5b20a109b48e649cedaee37c5b1", 511212618),
}
SIZES = ("JOWA_40M", "JOWA_150M")
"""The sizes with released weights (the paper's 70M has none in the folder the adapter uses)."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(weights_dir: Path, name: str) -> tuple[Path, str]:
    """``weights_dir/name`` and its SHA-256, after checking it is the official file; raises
    ValueError naming the file otherwise."""
    path = Path(weights_dir) / name
    if not path.is_file():
        raise ValueError(
            f"{path}: missing (python -m gthbaselines.jowa_weights {weights_dir} ...)"
        )
    got = sha256(path)
    if got != FILES[name][0]:
        raise ValueError(f"{path}: SHA-256 {got}, the official file's is {FILES[name][0]}")
    return path, got


def fetch(weights_dir: Path, sizes: list[str]) -> None:
    """Downloads the tokenizer and ``sizes`` into ``weights_dir`` (files of the right size
    are kept) and checks every hash."""
    names = ["tokenizer.pt"] + [f"{size}.pt" for size in sizes]
    weights_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        n
        for n in names
        if not (weights_dir / n).is_file() or (weights_dir / n).stat().st_size != FILES[n][1]
    ]
    if missing:
        import gdown

        listing = gdown.download_folder(
            id=DRIVE_FOLDER, skip_download=True, quiet=True, use_cookies=False
        )
        ids = {Path(entry.path).name: entry.id for entry in listing}
        for name in missing:
            gdown.download(id=ids[name], output=str(weights_dir / name), quiet=True)
    for name in names:
        verified(weights_dir, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("weights_dir", type=Path)
    parser.add_argument("sizes", nargs="+", choices=SIZES)
    args = parser.parse_args(argv)
    fetch(args.weights_dir, args.sizes)
    print(f"{args.weights_dir}: tokenizer.pt and {', '.join(args.sizes)} verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
