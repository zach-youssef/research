#!/usr/bin/env bash
# JAT Mac smoke test. Usage: run.sh [--smoke] [--devices cpu,mps] [--finalize]
# Everything big goes under runs/<id>/ (git-ignored) or the HF cache (outside the checkout).
set -euo pipefail

EXP_ID=2026-11-24-jat-mac-smoke
JAT_COMMIT=2e1ac33216adf48910bb6bc3bbb27bf9a94132aa
UV_VERSION=0.13.9
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RUNS="$REPO/runs/$EXP_ID"
mkdir -p "$RUNS"

# 1. Environment: uv (pinned) bootstraps CPython 3.11 and the pinned requirements.
STAMP="$RUNS/.venv/.reqs-$(shasum "$HERE/requirements.txt" | cut -c1-12)"
if [ ! -x "$RUNS/.venv/bin/python" ] || [ ! -f "$STAMP" ]; then
  if [ ! -x "$RUNS/.boot/bin/uv" ]; then
    /usr/bin/python3 -m venv "$RUNS/.boot"
    "$RUNS/.boot/bin/pip" install --quiet "uv==$UV_VERSION"
  fi
  "$RUNS/.boot/bin/uv" venv --quiet --clear --python 3.11 "$RUNS/.venv"
  "$RUNS/.boot/bin/uv" pip install --quiet --python "$RUNS/.venv/bin/python" -r "$HERE/requirements.txt"
  touch "$STAMP"
fi
PY="$RUNS/.venv/bin/python"

# 2. Upstream repo at the pinned commit (not vendored); only its env wrappers are imported.
if [ ! -d "$RUNS/jat/.git" ]; then
  git clone --quiet https://github.com/huggingface/jat.git "$RUNS/jat"
fi
git -C "$RUNS/jat" checkout --quiet "$JAT_COMMIT"
test "$(git -C "$RUNS/jat" rev-parse HEAD)" = "$JAT_COMMIT"

# 3. Run from a scratch cwd; PYTHONPATH carries the upstream repo.
export PYTHONPATH="$RUNS/jat"
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$RUNS/work" "$RUNS/logs"
cd "$RUNS/work"
"$PY" "$HERE/src/smoke.py" --runs-dir "$RUNS" --exp-dir "$HERE" --jat-dir "$RUNS/jat" "$@" 2>&1 \
  | tee "$RUNS/logs/$(date +%Y%m%dT%H%M%S).log"
exit "${PIPESTATUS[0]}"
