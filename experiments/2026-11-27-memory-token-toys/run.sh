#!/usr/bin/env bash
# X7 memory-token toys. Usage: run.sh [--smoke] [--part a|b|c]
#   (no flag)  full run: checks, X7a/X7b/X7c as three parallel CPU processes, then aggregation
#              into results.json, results.md and figures/ in this directory.
#   --smoke    the same at toy size, writing only under runs/<id>/smoke/ (under a minute).
#   --part P   development: one part at full size into runs/<id>/dev/, no aggregation.
set -euo pipefail

EXP_ID=2026-11-27-memory-token-toys
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RUNS="$REPO/runs/$EXP_ID"
mkdir -p "$RUNS"

MODE=full; PART=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) MODE=smoke ;;
    --part) PART="$2"; MODE=dev; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done

# 1. Environment: the system python3 (3.9) and the pinned requirements.
STAMP="$RUNS/.venv/.reqs-$(shasum "$HERE/requirements.txt" | cut -c1-12)"
if [ ! -x "$RUNS/.venv/bin/python" ] || [ ! -f "$STAMP" ]; then
  /usr/bin/python3 -m venv --clear "$RUNS/.venv"
  "$RUNS/.venv/bin/pip" install --quiet --no-deps -r "$HERE/requirements.txt"
  touch "$STAMP"
fi
PY="$RUNS/.venv/bin/python"

export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export X7_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
OUT="$RUNS/$MODE"
mkdir -p "$OUT"
START=$(date +%s)
SMOKE_FLAG=(); [ "$MODE" = smoke ] && SMOKE_FLAG=(--smoke)

# 2. Correctness checks first; a failure stops the run.
"$PY" "$HERE/src/checks/run_all.py" 2>&1 | tee "$OUT/checks.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "checks failed" >&2; exit 1; }

# 3. The parts, in parallel.
PARTS=(a b c); [ -n "$PART" ] && PARTS=("$PART")
PIDS=()
for p in "${PARTS[@]}"; do
  ( cd "$HERE/src" && "$PY" "x7$p.py" --out "$OUT/part_x7$p.json" "${SMOKE_FLAG[@]+"${SMOKE_FLAG[@]}"}" ) \
    > "$OUT/x7$p.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then echo "part x7${PARTS[$i]} failed; see $OUT/x7${PARTS[$i]}.log" >&2; FAIL=1; fi
done
[ "$FAIL" -eq 0 ] || exit 1
[ "$MODE" = dev ] && exit 0

# 4. Aggregate.
WALL=$(( $(date +%s) - START ))
"$PY" "$HERE/src/aggregate.py" --parts-dir "$OUT" --exp-dir "$HERE" --wall-s "$WALL" "${SMOKE_FLAG[@]+"${SMOKE_FLAG[@]}"}"
echo "done in ${WALL}s"
