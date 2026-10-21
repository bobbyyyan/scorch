#!/usr/bin/env bash
# Runtime grid for kernels whose C++ changed: capture on two trees, then time side by side.
#
# usage: run_host.sh <old tree> <new tree> <pairs.json> <out dir> [capture parts] [processes]
#
# Every step in order on one host (run_step.sh holds each step, for a host where a step is a
# job of its own):
#   1. capture on <old tree> (variant shipped), then on <new tree> (variants shipped and new);
#   2. <processes> timing processes, one after another (never two at once, so no timing shares
#      the machine with another), each timing every cell;
#   3. the analysis -> <out dir>/report.txt.
# A step that does not finish stops the script and says which did not.
set -u
OLD="$(cd "${1:?old tree}" && pwd)"
NEW="$(cd "${2:?new tree}" && pwd)"
PAIRS="${3:?pairs.json}"
OUT="${4:?out dir}"
PARTS="${5:-8}"
PROCESSES="${6:-3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
[ "$(basename "$OLD")" != "$(basename "$NEW")" ] || {
  echo "the trees need different directory names: analyze.py tells them apart by name" >&2; exit 1; }
# The timing process registers the kernels' C++ result types from <new tree>'s scorch_ops
# (time_cells.py), so both trees must define them identically.
types() { sed -n '/^typedef struct {/,/^} Tensor;/p' "$1/src/scorch/csrc/header.h"; }
[ -n "$(types "$NEW")" ] && [ "$(types "$OLD")" = "$(types "$NEW")" ] || {
  echo "the trees define the kernels' result structs differently (csrc/header.h)" >&2; exit 1; }

bash "$HERE/run_step.sh" capture "$OLD" shipped "$PAIRS" "$OUT" "$PARTS" || exit $?
bash "$HERE/run_step.sh" capture "$NEW" shipped,new "$PAIRS" "$OUT" "$PARTS" || exit $?
for process in $(seq 0 $((PROCESSES - 1))); do
  bash "$HERE/run_step.sh" time "$process" "$NEW" "$OUT" || exit $?
done
bash "$HERE/run_step.sh" analyze "$OLD" "$NEW" "$OUT"
