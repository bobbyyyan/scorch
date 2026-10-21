#!/usr/bin/env bash
# One step of the runtime grid, for a host where each step is its own job (README.md).
#
# usage: run_step.sh capture <tree> <variants> <pairs.json> <out dir> [parts]
#        run_step.sh time <process index> <new tree> <out dir>
#        run_step.sh analyze <old tree> <new tree> <out dir>
#
#   capture   capture.py on <tree> for <variants> ("shipped" on the old tree, "shipped,new" on
#             the new one), [parts] processes at once, each with the tree's own src on
#             PYTHONPATH; one kernel build directory per tree under <out dir>.  Run it on the
#             host that will time the kernels: they are compiled with -march=native.
#   time      one timing process over every cell.  Never two at once on one host.
#   analyze   analyze.py -> <out dir>/report.txt.
# A step that did not finish exits non-zero and says which part did not.
set -u
STEP="${1:?step}"; shift
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"

header() {  # <out dir> <text>: one block of provenance per step
  {
    echo "== $STEP: $(date)"
    echo "host: $(hostname)"
    echo "harness: $(cat "$HERE"/capture.py "$HERE"/time_cells.py "$HERE"/analyze.py | (shasum -a 256 2>/dev/null || sha256sum))"
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} OMP_PROC_BIND=${OMP_PROC_BIND:-unset} OMP_PLACES=${OMP_PLACES:-unset}"
    "$PY" -c "import torch; print('torch', torch.__version__, 'threads', torch.get_num_threads())"
    echo "$2"
  } >> "$1/run.txt"
}

case "$STEP" in
  capture)
    TREE="$(cd "${1:?tree}" && pwd)"; VARIANTS="${2:?variants}"
    PAIRS="$(cd "$(dirname "${3:?pairs.json}")" && pwd)/$(basename "$3")"
    mkdir -p "${4:?out dir}"; OUT="$(cd "$4" && pwd)"; PARTS="${5:-8}"
    NAME="$(basename "$TREE")"
    header "$OUT" "tree: $TREE variants: $VARIANTS parts: $PARTS
pairs: $PAIRS $(shasum -a 256 "$PAIRS" 2>/dev/null || sha256sum "$PAIRS")"
    mkdir -p "$OUT/ext_$NAME"
    for part in $(seq 0 $((PARTS - 1))); do
      PYTHONHASHSEED=0 PYTHONPATH="$TREE/src" TORCH_EXTENSIONS_DIR="$OUT/ext_$NAME" \
        "$PY" "$HERE/capture.py" "$TREE" "$PAIRS" "$OUT" "$VARIANTS" "$part" "$PARTS" \
        > "$OUT/capture_$NAME.$part.log" 2>&1 &
    done
    wait
    count=0
    for f in "$OUT"/capture.*_"$NAME".*.jsonl; do
      [ -f "$f" ] || continue
      count=$((count + 1))
      grep -q '"part_done"' "$f" || { echo "capture did not finish: $f" >&2; exit 2; }
    done
    [ "$count" -eq "$PARTS" ] || { echo "expected $PARTS capture files for $NAME, found $count" >&2; exit 2; }
    ;;
  time)
    PROCESS="${1:?process index}"; NEW="$(cd "${2:?new tree}" && pwd)"; OUT="$(cd "${3:?out dir}" && pwd)"
    header "$OUT" "process: $PROCESS types from: $NEW rounds: ${GRID_ROUNDS:-21}"
    "$PY" "$HERE/time_cells.py" "$OUT" "$PROCESS" "$NEW" > "$OUT/time.$PROCESS.log" 2>&1
    grep -q '"part_done"' "$OUT/times.$PROCESS.0.jsonl" || {
      echo "timing process $PROCESS did not finish (see $OUT/time.$PROCESS.log)" >&2; exit 3; }
    ;;
  analyze)
    OLD="$(cd "${1:?old tree}" && pwd)"; NEW="$(cd "${2:?new tree}" && pwd)"; OUT="$(cd "${3:?out dir}" && pwd)"
    header "$OUT" "old: $OLD new: $NEW"
    "$PY" "$HERE/analyze.py" "$OUT" "$OLD" "$NEW" > "$OUT/report.txt" || exit 4
    tail -n 8 "$OUT/report.txt"
    ;;
  *)
    echo "unknown step: $STEP" >&2; exit 1 ;;
esac
