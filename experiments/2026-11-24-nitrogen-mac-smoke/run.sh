#!/usr/bin/env bash
# NitroGen Mac smoke test. Usage: run.sh [--smoke] [--device-only cpu|mps] [--prepare]
# Everything big goes under runs/<id>/ (git-ignored) or the HF cache (outside the checkout).
set -euo pipefail

EXP_ID=2026-11-24-nitrogen-mac-smoke
NITROGEN_COMMIT=32608444660950ffda95e1e57c79632ad65bea10
UV_VERSION=0.13.9
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RUNS="$REPO/runs/$EXP_ID"
mkdir -p "$RUNS"

# 1. Environment: uv (pinned) bootstraps CPython 3.12 and the pinned requirements.
if [ ! -x "$RUNS/.venv/bin/python" ] || [ ! -f "$RUNS/.venv/.reqs-$(shasum "$HERE/requirements.txt" | cut -c1-12)" ]; then
  if [ ! -x "$RUNS/.boot/bin/uv" ]; then
    /usr/bin/python3 -m venv "$RUNS/.boot"
    "$RUNS/.boot/bin/pip" install --quiet "uv==$UV_VERSION"
  fi
  "$RUNS/.boot/bin/uv" venv --quiet --clear --python 3.12 "$RUNS/.venv"
  "$RUNS/.boot/bin/uv" pip install --quiet --python "$RUNS/.venv/bin/python" -r "$HERE/requirements.txt"
  touch "$RUNS/.venv/.reqs-$(shasum "$HERE/requirements.txt" | cut -c1-12)"
fi
PY="$RUNS/.venv/bin/python"

# 2. NitroGen code at the pinned commit (not vendored).
if [ ! -d "$RUNS/NitroGen/.git" ]; then
  git clone --quiet https://github.com/MineDojo/NitroGen.git "$RUNS/NitroGen"
fi
git -C "$RUNS/NitroGen" fetch --quiet --depth 1 origin "$NITROGEN_COMMIT" 2>/dev/null || true
git -C "$RUNS/NitroGen" checkout --quiet "$NITROGEN_COMMIT"
test "$(git -C "$RUNS/NitroGen" rev-parse HEAD)" = "$NITROGEN_COMMIT"

# 3. Run. -I is not used: the script needs PYTHONPATH for NitroGen; cwd is a scratch dir.
export PYTHONPATH="$RUNS/NitroGen:$HERE/src"
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RUNS/work" "$RUNS/logs"
cd "$RUNS/work"
"$PY" "$HERE/src/smoke.py" --runs-dir "$RUNS" --exp-dir "$HERE" --nitrogen-dir "$RUNS/NitroGen" "$@" 2>&1 \
  | tee "$RUNS/logs/$(date +%Y%m%dT%H%M%S).log"
