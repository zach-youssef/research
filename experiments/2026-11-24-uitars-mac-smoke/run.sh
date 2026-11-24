#!/usr/bin/env bash
# UI-TARS-1.5-7B (4-bit MLX) 2048 smoke test. Usage: run.sh [--smoke]
# Everything big goes under runs/<id>/ (git-ignored), the HF cache and Playwright's cache (outside the checkout).
set -euo pipefail

EXP_ID=2026-11-24-uitars-mac-smoke
UITARS_COMMIT=582f3a7ea5d285ee8ed9e2e84048d1ab01453c49
G2048_COMMIT=478b6ec346e3787f589e4af751378d06ded4cbbc
UV_VERSION=0.13.9
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RUNS="$REPO/runs/$EXP_ID"
mkdir -p "$RUNS/logs"

# 1. Environment: uv (pinned) bootstraps CPython 3.12 and the pinned requirements.
STAMP="$RUNS/.venv/.reqs-$(shasum "$HERE/requirements.txt" | cut -c1-12)"
if [ ! -x "$RUNS/.venv/bin/python" ] || [ ! -f "$STAMP" ]; then
  if [ ! -x "$RUNS/.boot/bin/uv" ]; then
    /usr/bin/python3 -m venv "$RUNS/.boot"
    "$RUNS/.boot/bin/pip" install --quiet "uv==$UV_VERSION"
  fi
  "$RUNS/.boot/bin/uv" venv --quiet --clear --python 3.12 "$RUNS/.venv"
  "$RUNS/.boot/bin/uv" pip install --quiet --python "$RUNS/.venv/bin/python" -r "$HERE/requirements.txt"
  touch "$STAMP"
fi
PY="$RUNS/.venv/bin/python"
"$RUNS/.venv/bin/playwright" install --only-shell chromium >/dev/null

# 2. Upstream code at pinned commits (not vendored).
clone() { # url dir commit
  [ -d "$2/.git" ] || git clone --quiet "$1" "$2"
  git -C "$2" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "$2" checkout --quiet "$3"
  test "$(git -C "$2" rev-parse HEAD)" = "$3"
}
clone https://github.com/bytedance/UI-TARS.git "$RUNS/UI-TARS" "$UITARS_COMMIT"
clone https://github.com/gabrielecirulli/2048.git "$RUNS/2048" "$G2048_COMMIT"

# 3. Run (cwd is a scratch dir; the script needs PYTHONPATH for ui_tars).
export PYTHONPATH="$RUNS/UI-TARS/codes"
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$RUNS/work"
cd "$RUNS/work"
"$PY" "$HERE/src/play.py" --runs-dir "$RUNS" --exp-dir "$HERE" "$@" 2>&1 \
  | tee "$RUNS/logs/$(date +%Y%m%dT%H%M%S).log"
