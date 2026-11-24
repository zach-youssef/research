#!/usr/bin/env bash
# JOWA Mac smoke test. Usage: run.sh [--smoke] [--devices cpu,mps] [--sizes JOWA_40M,JOWA_150M] [--no-write]
#        run.sh --diagnose   (CPU only: action-set check, tokenizer reconstructions, greedy action histograms)
# Big things live under runs/<id>/ (git-ignored) or ~/.cache/jowa (weights, outside the checkout).
set -euo pipefail

EXP_ID=2026-11-24-jowa-mac-smoke
JOWA_COMMIT=8cb02c5e6768a948c566a9012ac8a4fc420c6970
UV_VERSION=0.13.9
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RUNS="$REPO/runs/$EXP_ID"
mkdir -p "$RUNS"

# 1. Environment: pinned uv bootstraps CPython 3.8 (upstream's version) and the pinned requirements.
STAMP="$RUNS/.venv/.reqs-$(shasum "$HERE/requirements.txt" | cut -c1-12)"
if [ ! -x "$RUNS/.venv/bin/python" ] || [ ! -f "$STAMP" ]; then
  if [ ! -x "$RUNS/.boot/bin/uv" ]; then
    /usr/bin/python3 -m venv "$RUNS/.boot"
    "$RUNS/.boot/bin/pip" install --quiet "uv==$UV_VERSION"
  fi
  "$RUNS/.boot/bin/uv" venv --quiet --clear --python 3.8 "$RUNS/.venv"
  "$RUNS/.boot/bin/uv" pip install --quiet --python "$RUNS/.venv/bin/python" -r "$HERE/requirements.txt"
  touch "$STAMP"
fi
PY="$RUNS/.venv/bin/python"

# 2. Upstream JOWA at the pinned commit (not vendored; GPL-3.0). Shallow fetch of that commit only.
if [ ! -d "$RUNS/JOWA/.git" ]; then
  git init --quiet "$RUNS/JOWA"
  git -C "$RUNS/JOWA" remote add origin https://github.com/CJReinforce/JOWA.git
fi
if [ "$(git -C "$RUNS/JOWA" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$JOWA_COMMIT" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git -C "$RUNS/JOWA" fetch --quiet --depth 1 origin "$JOWA_COMMIT"
  git -C "$RUNS/JOWA" checkout --quiet --force "$JOWA_COMMIT"
fi
test "$(git -C "$RUNS/JOWA" rev-parse HEAD)" = "$JOWA_COMMIT"

# 3. Run (cwd is a scratch dir; upstream modules are imported by smoke.py via sys.path).
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RUNS/work" "$RUNS/logs"
cd "$RUNS/work"
SCRIPT=smoke.py
if [ "${1:-}" = "--diagnose" ]; then SCRIPT=diagnose.py; shift; fi
"$PY" "$HERE/src/$SCRIPT" --runs-dir "$RUNS" --exp-dir "$HERE" --jowa-dir "$RUNS/JOWA" "$@" 2>&1 \
  | tee "$RUNS/logs/$(date +%Y%m%dT%H%M%S).log"
exit "${PIPESTATUS[0]}"
