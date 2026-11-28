"""The JOWA weights' checks (no model loading)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gthbaselines import jowa_weights


def test_refuses_weights_with_another_hash(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.pt").write_bytes(b"not the tokenizer")
    with pytest.raises(ValueError, match="SHA-256"):
        jowa_weights.verified(tmp_path, "tokenizer.pt")
    with pytest.raises(ValueError, match="missing"):
        jowa_weights.verified(tmp_path, "JOWA_40M.pt")
