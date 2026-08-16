#!/usr/bin/env bash
# Phase-0 attribution driver. Takes the tree root as $1 so no ledger path is baked in.
#
#   run_phase0.sh <TREE_ROOT> <OUT_DIR> [CELLS_FILE]
#
# One `perf stat` per (cell, arm), counters gated to the timed region by the control
# fifo. Cells run one at a time: two SpMM benchmarks sharing this box would poison
# each other's counters as thoroughly as they poison each other's timings.
set -u
TREE="${1:?tree root}"
OUT="${2:?out dir}"
CELLS="${3:-}"
mkdir -p "$OUT"

PY="${PY:-python}"
EVENTS="${EVENTS:-cycles,instructions,r412e,r4f2e,r0e12,r0820}"
REPS="${REPS:-7}"

# matrix:N:arm
DEFAULT_CELLS=$(cat <<'EOF'
gcn:reddit 16 sc_off
gcn:reddit 16 mkl32
gcn:reddit 32 sc_off
gcn:reddit 32 mkl32
gcn:reddit 64 sc_off
gcn:reddit 64 mkl32
gcn:reddit 128 sc_off
gcn:reddit 128 mkl32
gcn:reddit 128 sc_balanced
syn:band16 128 sc_off
syn:band16 128 mkl32
ss:inline_1 32 sc_off
ss:inline_1 32 mkl32
gcn:pubmed 32 sc_off
gcn:pubmed 32 mkl32
ss:bcsstk17 32 sc_off
ss:bcsstk17 32 mkl32
syn:scatter200 32 sc_off
syn:scatter200 32 mkl32
ss:mouse_gene 512 sc_off
ss:mouse_gene 512 mkl32
ss:mouse_gene 512 sc_balanced
EOF
)

if [ -n "$CELLS" ]; then CELL_LIST=$(cat "$CELLS"); else CELL_LIST="$DEFAULT_CELLS"; fi

CTL="$OUT/perf_ctl.fifo"; ACK="$OUT/perf_ack.fifo"
JSONL="$OUT/phase0.jsonl"
: > "$JSONL"

echo "$CELL_LIST" | while read -r MAT N ARM CREPS; do
  [ -z "${MAT:-}" ] && continue
  R="${CREPS:-$REPS}"; [ -z "$R" ] && R="$REPS"
  tag="$(echo "${MAT}_${N}_${ARM}" | tr ':/' '__')"
  rm -f "$CTL" "$ACK"; mkfifo "$CTL" "$ACK"
  echo "=== $tag ===" >&2
  PYTHONPATH="$TREE/src" \
  PERF_CTL_FIFO="$CTL" PERF_ACK_FIFO="$ACK" \
  perf stat -x, -o "$OUT/$tag.perf" -D -1 --control "fifo:$CTL,$ACK" \
      -e "$EVENTS" -- \
      "$PY" "$TREE/bench/phase0_attrib.py" --matrix "$MAT" --n "$N" --arm "$ARM" \
      --reps "$R" > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  rc=$?
  rm -f "$CTL" "$ACK"
  if [ $rc -ne 0 ]; then echo "FAILED $tag rc=$rc" >&2; tail -5 "$OUT/$tag.err" >&2; continue; fi
  "$PY" - "$OUT/$tag.out" "$OUT/$tag.perf" "$tag" >> "$JSONL" <<'PYEOF'
import json, sys
out, perf, tag = sys.argv[1], sys.argv[2], sys.argv[3]
rec = None
for line in open(out):
    if line.startswith("ATTRIB "):
        rec = json.loads(line[len("ATTRIB "):])
if rec is None:
    sys.exit(0)
ctr, mux = {}, {}
for line in open(perf):
    line = line.rstrip("\n")
    if not line or line.startswith("#"):
        continue
    f = line.split(",")
    if len(f) < 3:
        continue
    val, ev = f[0], f[2]
    try:
        ctr[ev] = float(val)
    except ValueError:
        ctr[ev] = None
    # field 5 is the % of time the event was actually counted (multiplexing)
    if len(f) > 4 and f[4]:
        try:
            mux[ev] = float(f[4])   # % of region the event was actually counted
        except ValueError:
            pass
rec["counters"] = ctr
rec["counter_pct"] = mux
rec["tag"] = tag
print(json.dumps(rec))
PYEOF
done
echo "wrote $JSONL" >&2
