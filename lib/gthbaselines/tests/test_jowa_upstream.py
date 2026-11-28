"""Upstream JOWA loaded from the pinned clone, with the real JOWA-40M weights (check.sh sets
the GTHB_* variables)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gthbaselines import _jowa_upstream, jowa_weights


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is unset: run lib/gthbaselines/check.sh")
    return value


def test_refuses_a_clone_at_another_commit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a git checkout"):
        _jowa_upstream.check_clone(tmp_path)


def test_builds_and_loads_jowa_40m() -> None:
    jowa = Path(env("GTHB_JOWA_DIR"))
    weights = Path(env("GTHB_JOWA_WEIGHTS"))
    _jowa_upstream.install(jowa)
    tokenizer, model = _jowa_upstream.build(jowa, "JOWA_40M")
    skipped = _jowa_upstream.load(
        tokenizer,
        model,
        jowa_weights.verified(weights, "tokenizer.pt")[0],
        jowa_weights.verified(weights, "JOWA_40M.pt")[0],
    )
    assert skipped and all(name.endswith(".attn.mask") for name in skipped)
    model.eval()
    # functorch's ensemble keeps the Q heads' MLPs in training mode through .eval().
    ensembles = [m for m in model.modules() if hasattr(m, "vmap")]
    assert all(e.vmap.__wrapped__.stateless_model.training for e in ensembles)
    assert _jowa_upstream.set_q_dropout(model, False) == 2
    assert not any(e.vmap.__wrapped__.stateless_model.training for e in ensembles)
