#!/usr/bin/env bash
# A/B two built trees on the same cells.
#
# The two arms live in different .so files, so they cannot be interleaved inside one
# process the way same-build arms are. Instead each tree's process also times the MKL
# arm, which is byte-identical in both trees: |mkl_base/mkl_cand - 1| is therefore the
# cross-process noise floor for that cell, and nothing smaller than it counts.
# Trees alternate base,cand,base,cand,... so thermal drift hits both equally.
set -u
BASE="${1:?base tree}"
CAND="${2:?candidate tree}"
OUT="${3:?out dir}"
ROUNDS="${ROUNDS:-3}"
PY="${PY:-python}"
CELLS_FILE="${4:-}"
mkdir -p "$OUT"
JSONL="$OUT/ab.jsonl"
: > "$JSONL"

DEFAULT_CELLS=$(cat <<'EOF'
gcn:cora 32 200
gcn:cora 512 60
gcn:pubmed 32 60
gcn:pubmed 128 40
gcn:pubmed 512 15
ss:bcsstk17 32 60
ss:bcsstk17 128 40
ss:bcsstk17 512 15
syn:band16 128 30
syn:scatter200 32 30
syn:scatter200 128 15
syn:scatter200 512 7
ss:inline_1 32 15
ss:inline_1 512 5
ss:mouse_gene 512 5
gcn:reddit 16 5
gcn:reddit 32 5
gcn:reddit 128 3
EOF
)
LIST="$DEFAULT_CELLS"
[ -n "$CELLS_FILE" ] && LIST=$(cat "$CELLS_FILE")

for r in $(seq 1 "$ROUNDS"); do
  echo "$LIST" | while read -r MAT N REPS; do
    [ -z "${MAT:-}" ] && continue
    for TREE_NAME in base cand; do
      if [ "$TREE_NAME" = base ]; then T="$BASE"; else T="$CAND"; fi
      for ARM in sc_off sc_balanced mkl32; do
        PYTHONPATH="$T/src" "$PY" "$T/bench/phase0_attrib.py" \
          --matrix "$MAT" --n "$N" --arm "$ARM" --reps "$REPS" \
          --extra "{\"tree\":\"$TREE_NAME\",\"round\":$r}" 2>/dev/null \
          | grep '^ATTRIB ' | sed 's/^ATTRIB //' >> "$JSONL"
      done
    done
    echo "round $r done $MAT N=$N" >&2
  done
done
echo "wrote $JSONL" >&2
