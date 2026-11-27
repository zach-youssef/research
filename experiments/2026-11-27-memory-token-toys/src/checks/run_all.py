"""Run every check; exit non-zero on the first failure."""
import importlib
import sys
import time

import _util  # noqa: F401

from common import setup_torch

MODULES = ["perm_invariance", "prefix_mask", "causality", "labels_not_inputs", "oracles"]


def main():
    setup_torch()
    for name in MODULES:
        t0 = time.time()
        try:
            msg = importlib.import_module(name).run()
        except Exception as e:  # noqa: BLE001
            print(f"CHECK FAILED {name}: {type(e).__name__}: {e}", flush=True)
            sys.exit(1)
        print(f"check {name}: ok ({time.time() - t0:.1f}s): {msg}", flush=True)


if __name__ == "__main__":
    main()
