#!/usr/bin/env bash
# OCCAM/HackAtari Mac smoke test. Usage: run.sh [--smoke | --mps-only | --finalize-only]
#   (none)          CPU evaluation, videos, results (includes mps_bench.json if one exists)
#   --mps-only      only the MPS forward-pass microbenchmark (needs the supervisor's GPU slot)
#   --finalize-only rebuild results.json/results.md from the saved episodes and benchmark
# Big things live under runs/<id>/ (git-ignored) or ~/.cache/occam-hackatari (outside the checkout).
set -euo pipefail

EXP_ID=2026-11-24-occam-hackatari-mac-smoke
UV_VERSION=0.13.9
WRAPPERS_URL=https://github.com/VanillaWhey/OCAtariWrappers.git
WRAPPERS_COMMIT=057afc4e53f66ae1c12eaa72c321575be05534f0
HACKATARI_URL=https://github.com/k4ntz/HackAtari.git
HACKATARI_COMMIT=3b5fc3209c3ae7ff7d14a1338477963d4353b78b
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RUNS="$REPO/runs/$EXP_ID"
CACHE="${OCCAM_CACHE:-$HOME/.cache/occam-hackatari}"
EXTRA=()
for a in "$@"; do case "$a" in --smoke|--mps-only|--finalize-only) EXTRA+=("$a");; *) echo "unknown arg $a" >&2; exit 2;; esac; done
mkdir -p "$RUNS" "$CACHE"

# 1. Environment: uv (pinned) bootstraps CPython 3.11 and the pinned requirements.
TAG="$RUNS/.venv/.reqs-$(shasum "$HERE/requirements.txt" | cut -c1-12)"
if [ ! -x "$RUNS/.venv/bin/python" ] || [ ! -f "$TAG" ]; then
  if [ ! -x "$RUNS/.boot/bin/uv" ]; then
    /usr/bin/python3 -m venv "$RUNS/.boot"
    "$RUNS/.boot/bin/pip" install --quiet "uv==$UV_VERSION"
  fi
  "$RUNS/.boot/bin/uv" venv --quiet --clear --python 3.11 "$RUNS/.venv"
  "$RUNS/.boot/bin/uv" pip install --quiet --python "$RUNS/.venv/bin/python" -r "$HERE/requirements.txt"
  touch "$TAG"
fi
PY="$RUNS/.venv/bin/python"

# 2. Upstream code and weights at pinned commits (not vendored). Weights come from the
#    OCAtariWrappers repo itself: sparse checkout of the 18 Pong/Freeway/Breakout files.
W="$CACHE/OCAtariWrappers"
if [ ! -d "$W/.git" ]; then
  git clone --quiet --filter=blob:none --no-checkout "$WRAPPERS_URL" "$W"
  git -C "$W" sparse-checkout init --no-cone
  git -C "$W" sparse-checkout set '/*' '!/models/*' '/models/Pong/0/*' '/models/Freeway/0/*' '/models/Breakout/0/*'
fi
git -C "$W" checkout --quiet "$WRAPPERS_COMMIT"
test "$(git -C "$W" rev-parse HEAD)" = "$WRAPPERS_COMMIT"
H="$CACHE/HackAtari"
if [ ! -d "$H/.git" ]; then
  git clone --quiet --filter=blob:none --no-checkout "$HACKATARI_URL" "$H"
fi
git -C "$H" checkout --quiet "$HACKATARI_COMMIT"
test "$(git -C "$H" rev-parse HEAD)" = "$HACKATARI_COMMIT"

# 3. Run (cwd is a scratch dir; the upstream trees are put on PYTHONPATH, not installed).
export PYTHONPATH="$H:$W:$W/scripts"
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export SDL_VIDEODRIVER=dummy
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=1
mkdir -p "$RUNS/work" "$RUNS/logs"
cd "$RUNS/work"
ARGS=(--runs-dir "$RUNS" --exp-dir "$HERE" --wrappers-dir "$W" --hackatari-dir "$H" --cache-dir "$CACHE")
ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})
"$PY" "$HERE/src/run_agents.py" "${ARGS[@]}" 2>&1 | tee "$RUNS/logs/$(date +%Y%m%dT%H%M%S).log"
exit "${PIPESTATUS[0]}"
