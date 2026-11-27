"""Shared helpers for the correctness checks: import path, Hypothesis settings, tiny models."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis import settings  # noqa: E402

import torch  # noqa: E402

FAST = settings(max_examples=int(os.environ.get("X7_CHECK_EXAMPLES", "12")), derandomize=True, deadline=None,
                database=None)
ATOL = 1e-5


def close(a, b, what):
    if not torch.allclose(a, b, atol=ATOL, rtol=0):
        raise AssertionError(f"{what}: max abs diff {(a - b).abs().max().item():.3g}")
