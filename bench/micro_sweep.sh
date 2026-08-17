#!/usr/bin/env bash
# Kernel-variant sweep over a matrix x N x threads grid.
#
# A row-kernel change ships only if it is neutral-or-better across the whole space,
# so this is a grid rather than the handful of cells that first suggested the change.
# Every cell carries its own A/A control (the `aa` arm is `base` under another name),
# so a per-cell noise floor comes out of the same run.
#
# Note on what the mask change can even touch: for N > 32 the wide path works in
# 64-wide k-tiles that never need a mask, so at N in {64,128,256,512} `nomask` differs
# from `base` only through the prefetch. The mask itself only matters for the narrow
# path, N <= 32 — the GCN feature-width regime.
set -u
OUT="${1:?out file}"
THREADS="${THREADS:-32}"
VARIANTS="${VARIANTS:-base,aa,nopf,pfT0,d16T0,d32T0,nomask,nm_d16,ilp4d16}"
: > "$OUT"

# matrix                 N list                     reps
run() {
  local m="$1" ns="$2" reps="$3"
  for n in $ns; do
    echo "### $m N=$n threads=$THREADS" >> "$OUT"
    ./spmm_micro "mtx/$m.bin" "$n" "$reps" "$THREADS" "$VARIANTS" >> "$OUT" 2>&1
  done
}

run gcn__cora        "8 16 32 64 128 256 512" 200
run gcn__pubmed      "8 16 32 64 128 256 512" 60
run ss__bcsstk17     "8 16 32 64 128 256 512" 60
run syn__band16      "8 16 32 64 128 256 512" 30
run syn__scatter200  "8 16 32 64 128 256"     20
run ss__crankseg_1   "8 16 32 64 128 256"     15
run ss__nd24k        "8 16 32 64 128"         10
run ss__mouse_gene   "8 16 32 64 128"         10
run ss__inline_1     "8 16 32 64 128"         10
run ss__audikw_1     "8 16 32 64"             7
run gcn__reddit      "8 16 32 64 128"         5
echo "wrote $OUT" >&2
