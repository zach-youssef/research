#!/usr/bin/env bash
# The checks of lib/gthbaselines (README.md, "Testing"). The research repo has no CI, so this
# is what every commit that touches lib/gthbaselines runs, from a clean tree:
#
#   lib/gthbaselines/check.sh            set up (first run: ~10 min, ~1.5 GB) and run the tests
#   lib/gthbaselines/check.sh --setup    set up only
#
# Everything it builds lives in runs/gthbaselines/ (git-ignored): a pinned uv, the Python 3.12
# venv from requirements.txt, upstream JOWA and the harness at their pinned commits, the
# harness's extension module (release) and test core. The JOWA weights live outside the
# checkout, in $JOWA_WEIGHTS_DIR (default ~/.cache/jowa); the tokenizer and JOWA-40M (151 MB)
# are fetched there when missing.
set -euo pipefail

JOWA_COMMIT=8cb02c5e6768a948c566a9012ac8a4fc420c6970     # _jowa_upstream.JOWA_COMMIT
HARNESS_COMMIT=b0bbb955a82a4987e53c83b0c3bc3ec0204af1bd  # main: Policy protocol, jowa2025, atari_full
UV_VERSION=0.13.10
HARNESS_URL="${GTH_HARNESS_URL:-https://github.com/zach-youssef/game-transformer-harness.git}"
WEIGHTS="${JOWA_WEIGHTS_DIR:-$HOME/.cache/jowa}"

HERE="$(cd "$(dirname "$0")" && pwd -P)"
REPO="$(cd "$HERE/../.." && pwd -P)"
RUNS="$REPO/runs/gthbaselines"
mkdir -p "$RUNS"

MODE=test
case "${1:-}" in
  "") ;;
  --setup) MODE=setup ;;
  *) echo "usage: $0 [--setup]" >&2; exit 2 ;;
esac

# 1. uv, pinned, bootstrapped with the system Python.
UV="$RUNS/.boot/bin/uv"
if [ ! -x "$UV" ] || [ "$("$UV" --version | cut -d' ' -f2)" != "$UV_VERSION" ]; then
  /usr/bin/python3 -m venv --clear "$RUNS/.boot"
  "$RUNS/.boot/bin/pip" install --quiet --disable-pip-version-check "uv==$UV_VERSION"
fi

# 2. The venv, rebuilt when requirements.txt changes.
VENV="$RUNS/.venv"
PY="$VENV/bin/python"
STAMP="$VENV/.reqs-$(shasum "$HERE/requirements.txt" | cut -c1-12)"
if [ ! -x "$PY" ] || [ ! -f "$STAMP" ]; then
  "$UV" venv --quiet --clear --python 3.12 "$VENV"
  "$UV" pip sync --quiet --require-hashes --python "$PY" "$HERE/requirements.txt"
  touch "$STAMP"
fi

# 3. Upstream code at pinned commits: a shallow fetch of that commit only, never edited.
checkout() {  # dir url commit
  if [ ! -d "$1/.git" ]; then
    git init --quiet "$1"
    git -C "$1" remote add origin "$2"
  fi
  if [ "$(git -C "$1" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$3" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git -C "$1" fetch --quiet --depth 1 origin "$3"
    git -C "$1" checkout --quiet --force "$3"
  fi
  test "$(git -C "$1" rev-parse HEAD)" = "$3"
  test -z "$(git -C "$1" status --porcelain)"
}
checkout "$RUNS/JOWA" https://github.com/CJReinforce/JOWA.git "$JOWA_COMMIT"
checkout "$RUNS/harness" "$HARNESS_URL" "$HARNESS_COMMIT"

# 4. The harness: its extension module into the venv (editable, so gtharness finds the
#    clone's games/ and protocols/) and its test core, in a target dir of their own.
export CARGO_TARGET_DIR="$RUNS/harness-target"
BUILT="$VENV/.harness-$HARNESS_COMMIT"
if [ ! -f "$BUILT" ]; then
  (cd "$RUNS/harness" && VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$RUNS/.boot/bin:$PATH" \
    "$VENV/bin/maturin" develop --quiet --release --uv)
  touch "$BUILT"
fi
TESTCORE="$(cd "$RUNS/harness" && PATH="$VENV/bin:$PATH" cargo build --quiet --release \
  --locked -p gth-testcore --message-format=json \
  | "$PY" -c 'import json, sys
for line in sys.stdin:
    m = json.loads(line) if line.startswith("{") else {}
    for f in m.get("filenames", []) if m.get("reason") == "compiler-artifact" else []:
        if "gth_testcore." in f.rsplit("/", 1)[-1]:
            print(f)')"
test -f "$TESTCORE"

# 5. The weights the tests use (the tokenizer and JOWA-40M), checked by SHA-256.
export PYTHONPATH="$REPO/lib"
"$PY" -m gthbaselines.jowa_weights "$WEIGHTS" JOWA_40M

[ "$MODE" = setup ] && exit 0

# 6. The tests (CPU).
export GTHB_JOWA_DIR="$RUNS/JOWA" GTHB_JOWA_WEIGHTS="$WEIGHTS" GTHB_TESTCORE="$TESTCORE"
export PYTHONDONTWRITEBYTECODE=1
"$PY" -m pytest -q -p no:cacheprovider "$HERE/tests"
