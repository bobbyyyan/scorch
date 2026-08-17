#!/usr/bin/env bash
# Bit-identity + accuracy check across two trees.
#
# The change under test touches only input validation and where the int32 narrowing
# happens, so every output bit must be unchanged. Any difference here is a bug, not a
# tolerance question — hence an exact hash rather than allclose.
set -u
BASE="${1:?base tree}"
CAND="${2:?candidate tree}"
PY="${PY:-python}"

CELLS=$(cat <<'EOF'
ss:bcsstk17 32
gcn:pubmed 32
gcn:cora 32
syn:scatter200 32
syn:band16 128
ss:inline_1 512
ss:mouse_gene 512
gcn:reddit 32
EOF
)

nid=0; ndiff=0
echo "$CELLS" | while read -r MAT N; do
  [ -z "${MAT:-}" ] && continue
  for LVL in off balanced; do
    b=$(PYTHONPATH="$BASE/src" "$PY" "$BASE/bench/result_hash.py" "$MAT" "$N" "$LVL" 2>&1 | grep '^HASH' || true)
    c=$(PYTHONPATH="$CAND/src" "$PY" "$CAND/bench/result_hash.py" "$MAT" "$N" "$LVL" 2>&1 | grep '^HASH' || true)
    bs=$(printf '%s' "$b" | sed -n 's/.*sha=\([a-f0-9]*\).*/\1/p')
    cs=$(printf '%s' "$c" | sed -n 's/.*sha=\([a-f0-9]*\).*/\1/p')
    if [ -z "$bs" ] || [ -z "$cs" ]; then
      echo "ERROR       $MAT N=$N $LVL"
      echo "    base: ${b:-<no output>}"
      echo "    cand: ${c:-<no output>}"
    elif [ "$bs" = "$cs" ]; then
      echo "IDENTICAL   $MAT N=$N $LVL  sha=$bs  $(printf '%s' "$c" | sed -n 's/.*\(relerr=[^ ]*\).*/\1/p')"
    else
      echo "*** DIFFERS $MAT N=$N $LVL  base=$bs cand=$cs"
    fi
  done
done
