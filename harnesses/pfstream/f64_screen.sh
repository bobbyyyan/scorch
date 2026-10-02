#!/bin/bash
# The float64 four-arm screen, as a host-portable driver.
#
# WHY float64 k=2..16 IS THE WHOLE QUESTION. scorch_simd<double>::lanes is 4 and the register
# block kernel is specialized NVEC 1..4, so float64's register-block range is exactly k=1..16.
# Below it, scorch_spmm_row_narrow_exact owns double k=1..3 (exact_cap_ = 3), so k=2 is a width at
# which SCORCH_SPMM_REGBLOCK_PFD provably cannot change executed code -- a NEGATIVE CONTROL rather
# than a data point. The recorded harm from removing the prefetch is at width 32, which in float64
# needs NVEC 8 and is therefore not reachable by this kernel at all; that harm is float32's.
#
# THE LATIN SQUARE is identical to the first host's: every build leads once and occupies each
# position once, so pass position is balanced across builds.
set -u

ROOT=${1:-}
[ -n "$ROOT" ] || { echo "usage: f64_screen.sh <root> [ks] [reps]"; exit 1; }
[ -d "$ROOT" ] || { echo "REFUSE: root $ROOT is not a directory"; exit 1; }
KS=${2:-2,4,8,16}
REPS=${3:-9}

TREE=${F64_TREE:-$ROOT/scorch-beat-mkl}
PFS=${F64_PFS:-$ROOT/pfs}
CACHE=${F64_CACHE:-$ROOT/mklsweep/csrcache/ss}
KEYS=${F64_KEYS:-$ROOT/f64_keys_screen.txt}
OUT=${F64_OUT:-$ROOT/csv64}
ACTIVATE=${F64_ACTIVATE:-$ROOT/pfs-activate.sh}
THREADS=${F64_THREADS:-32}

for spec in "TREE=$TREE" "PFS=$PFS" "CACHE=$CACHE"; do
  n=${spec%%=*}; p=${spec#*=}
  [ -d "$p" ] || { echo "REFUSE: $n=$p is not a directory"; exit 1; }
done
for spec in "KEYS=$KEYS" "ACTIVATE=$ACTIVATE"; do
  n=${spec%%=*}; p=${spec#*=}
  [ -f "$p" ] || { echo "REFUSE: $n=$p is not a file"; exit 1; }
done
H=$TREE/harnesses/pfstream
[ -f "$H/mrgate.py" ] || { echo "REFUSE: $H/mrgate.py absent"; exit 1; }

( source "$ACTIVATE" >/dev/null 2>&1 ) \
  || { echo "REFUSE: $ACTIVATE fails when sourced (probed in a subshell, which contains both a"
       echo "        nonzero return and an outright exit)"; exit 1; }
source "$ACTIVATE" || { echo "REFUSE: $ACTIVATE failed on the real source"; exit 1; }
python -c "import torch,sys; sys.stderr.write('torch %s at %s\n'%(torch.__version__,sys.executable))" \
  || { echo "REFUSE: the activated environment cannot import torch"; exit 1; }

# Every arm must exist before the first measurement, so a missing build fails now rather than
# three hours in with a partial Latin square on disk.
BUILDS="ship shpb strm nopf"
for t in $BUILDS; do
  [ -f "$PFS/$t/src/scorch_ops.so" ] || [ -n "$(ls "$PFS/$t/src"/scorch_ops*.so 2>/dev/null)" ] \
    || { echo "REFUSE: build $t has no scorch_ops .so under $PFS/$t/src"; exit 1; }
done

mkdir -p "$OUT" || exit 1
NK=$(wc -l < "$KEYS" | tr -d ' ')
echo "host $(hostname)  pool $(python -c 'import torch;print(torch.get_num_threads())')  threads $THREADS  $(date +%H:%M:%S)"
echo "corpus $NK matrices, dtype float64, widths $KS, reps $REPS"
echo "cache $CACHE"
echo "builds $BUILDS"

ORDERS=( "ship shpb strm nopf" "shpb strm nopf ship" "strm nopf ship shpb" "nopf ship shpb strm" )
failed=0
for p in 1 2 3 4; do
  order=${ORDERS[$((p-1))]}
  echo "== pass p$p order: $order"
  for tag in $order; do
    out=$OUT/ss64_${tag}_p${p}.csv
    echo "  --- p$p $tag start $(date +%H:%M:%S)"
    OMP_NUM_THREADS=$THREADS PYTHONPATH="$PFS/$tag/src" \
      python "$H/mrgate.py" --cache "$CACHE" --keys "$KEYS" --csv "$out" \
        --dtype float64 --ks "$KS" --reps "$REPS" --target-ms 15 \
        --pass-tag "p$p" --build "$tag" --slice all > "$OUT/ss64_${tag}_p${p}.log" 2>&1
    rc=$?
    rows=$(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 ))
    echo "  --- p$p $tag done  $(date +%H:%M:%S)  rc=$rc  ($rows rows)"
    if [ "$rc" != "0" ]; then
      failed=$(( failed + 1 ))
      echo "REFUSE: p$p $tag exited $rc"
      tail -4 "$OUT/ss64_${tag}_p${p}.log"
      exit 1
    fi
  done
done
echo "F64_SCREEN_ALL_PASSES_DONE $(date +%H:%M:%S)  nonzero rc $failed"
printf "passes 4\nbuilds 4\nnonzero_rc %s\ncsvs %s\nks %s\nkeys %s\n" \
  "$failed" "$(ls "$OUT"/ss64_*.csv 2>/dev/null | wc -l | tr -d ' ')" "$KS" "$NK" > "$OUT/f64screen.done"
cat "$OUT/f64screen.done"
