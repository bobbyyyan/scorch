#!/usr/bin/env bash
# Thread-scaling probe: does the small/cache-resident loss come from launching too
# many threads (and the wrong kind) for the work?
#
#   run_scaling.sh <TREE_ROOT> <OUT_DIR> [CELLS_FILE]
#
# Affinity is set with taskset because libgomp latches its available-CPU count at
# process start: omp_get_num_procs() — which is what scorch_nthreads caps against —
# only follows a mask that was already in place before the interpreter came up.
# torch's thread count is set to the mask size so MKL is measured at the same width.
#
# CPU map on this host: 0-15 = 8 P-cores (SMT pairs), 16-31 = 16 E-cores.
set -u
TREE="${1:?tree root}"
OUT="${2:?out dir}"
CELLS="${3:-}"
mkdir -p "$OUT"
PY="${PY:-python}"

CONFIGS="${CONFIGS:-p1:0 p2:0,2 p4:0,2,4,6 p8:0,2,4,6,8,10,12,14 p16smt:0-15 e16:16-31 p16smt_e8:0-23 all32:0-31}"

DEFAULT_CELLS=$(cat <<'EOF'
gcn:cora 32 200
gcn:pubmed 32 60
ss:bcsstk17 32 60
ss:inline_1 32 15
syn:scatter200 32 30
gcn:reddit 32 7
EOF
)
if [ -n "$CELLS" ]; then CELL_LIST=$(cat "$CELLS"); else CELL_LIST="$DEFAULT_CELLS"; fi

JSONL="$OUT/scaling.jsonl"
: > "$JSONL"

echo "$CELL_LIST" | while read -r MAT N REPS; do
  [ -z "${MAT:-}" ] && continue
  for cfg in $CONFIGS; do
    name="${cfg%%:*}"; mask="${cfg#*:}"
    nc=$($PY -c "
import sys
s=sys.argv[1]; n=0
for part in s.split(','):
    if '-' in part:
        a,b=part.split('-'); n+=int(b)-int(a)+1
    else:
        n+=1
print(n)" "$mask")
    for ARM in sc_off mkl32; do
      tag="$(echo "${MAT}_${N}_${ARM}_${name}" | tr ':/' '__')"
      PYTHONPATH="$TREE/src" taskset -c "$mask" \
        "$PY" "$TREE/bench/phase0_attrib.py" --matrix "$MAT" --n "$N" --arm "$ARM" \
        --reps "$REPS" --threads "$nc" \
        --extra "{\"cfg\":\"$name\",\"ncpu\":$nc,\"mask\":\"$mask\"}" \
        > "$OUT/$tag.out" 2> "$OUT/$tag.err"
      if [ $? -ne 0 ]; then echo "FAILED $tag" >&2; tail -3 "$OUT/$tag.err" >&2; continue; fi
      grep '^ATTRIB ' "$OUT/$tag.out" | sed 's|^ATTRIB ||' >> "$JSONL"
    done
  done
  echo "done $MAT N=$N" >&2
done
echo "wrote $JSONL" >&2
