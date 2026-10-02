#!/bin/bash
# The four-arm prefetch screen, host- and dtype-portable.
#
# THE WIDTH SET IS DTYPE-DEPENDENT AND THAT IS THE WHOLE DESIGN. The register-block kernel is
# specialized NVEC 1..4 with NVEC = ceil(k / V::lanes), and scorch_simd<float>::lanes is 8 while
# scorch_simd<double>::lanes is 4. So the reachable register-block range is k=1..32 for float32
# and k=1..16 for float64, and "width 32" is a float32-only question -- in float64 it would need
# NVEC 8 and the kernel cannot reach it.
#
# Below the range, scorch_spmm_row_narrow_exact owns float k=1..7 and double k=1..3
# (exact_cap_ = exact_f32_ ? 7 : 3), so a width inside that span is one at which
# SCORCH_SPMM_REGBLOCK_PFD provably cannot change executed code -- a NEGATIVE CONTROL rather than
# a data point, and the only thing that bounds a single-object arm's layout offset for free.
#
# A GRID MUST COVER BOTH TAIL STATES. The tail vector is full when k % lanes == 0 and masked
# otherwise, and that is a distinct branch in the kernel. A width set of 2,4,8,16,32 reaches
# neither NVEC 3 nor any masked tail for float32, which is the gap this project recorded against
# its own first screen. Defaults below are chosen per dtype to cover NVEC 1..4 and both tails.
#
# THE LATIN SQUARE is identical to the first host's: every build leads once and occupies each
# position once, so pass position is balanced across builds.
set -u

ROOT=${1:-}
[ -n "$ROOT" ] || { echo "usage: arm_screen.sh <root> [dtype] [ks] [reps]"; exit 1; }
[ -d "$ROOT" ] || { echo "REFUSE: root $ROOT is not a directory"; exit 1; }
DTYPE=${2:-float64}
case "$DTYPE" in
  float32) DEFAULT_KS=2,8,12,16,20,24,32 ;;   # control 2; nvec 1,2,3,4; masked tails at 12 and 20
  float64) DEFAULT_KS=2,4,6,8,10,12,16 ;;     # control 2; nvec 1,2,3,4; masked tails at 6 and 10
  *) echo "REFUSE: dtype $DTYPE is not float32 or float64, and the width defaults below are"
     echo "        derived from the lane count, so there is no safe default for it"; exit 1 ;;
esac
KS=${3:-$DEFAULT_KS}
REPS=${4:-9}


# The harness directory is overridable on its own, because the two hosts lay it out differently:
# one has a checked-out tree, the other keeps the producer files flat beside the builds. Deriving
# it only from a tree root would mean fabricating a tree on the second host to satisfy a path.
TREE=${F64_TREE:-$ROOT/scorch-beat-mkl}
PFS=${F64_PFS:-$ROOT/pfs}
CACHE=${F64_CACHE:-$ROOT/mklsweep/csrcache/ss}
KEYS=${F64_KEYS:-$ROOT/f64_keys_screen.txt}
OUT=${F64_OUT:-$ROOT/csv_$DTYPE}
ACTIVATE=${F64_ACTIVATE:-$ROOT/pfs-activate.sh}
THREADS=${F64_THREADS:-32}

HARNESS=${F64_HARNESS:-$TREE/harnesses/pfstream}
for spec in "HARNESS=$HARNESS" "PFS=$PFS" "CACHE=$CACHE"; do
  n=${spec%%=*}; p=${spec#*=}
  [ -d "$p" ] || { echo "REFUSE: $n=$p is not a directory"; exit 1; }
done
for spec in "KEYS=$KEYS" "ACTIVATE=$ACTIVATE"; do
  n=${spec%%=*}; p=${spec#*=}
  [ -f "$p" ] || { echo "REFUSE: $n=$p is not a file"; exit 1; }
done
H=$HARNESS
for f in mrgate.py csrresolve.py; do
  [ -f "$H/$f" ] || { echo "REFUSE: $H/$f absent"; exit 1; }
done

( source "$ACTIVATE" >/dev/null 2>&1 ) \
  || { echo "REFUSE: $ACTIVATE fails when sourced (probed in a subshell, which contains both a"
       echo "        nonzero return and an outright exit)"; exit 1; }
source "$ACTIVATE" || { echo "REFUSE: $ACTIVATE failed on the real source"; exit 1; }
python -c "import torch,sys; sys.stderr.write('torch %s at %s\n'%(torch.__version__,sys.executable))" \
  || { echo "REFUSE: the activated environment cannot import torch"; exit 1; }

# THE GRID CHECK RUNS HERE, NOT EARLIER. It needs a python interpreter, and on one of the two
# hosts there is no `python` on PATH until the environment is activated -- placed above the
# activation it failed with "python: command not found", which is a refusal for the wrong reason
# and says nothing about the grid. A check has to be able to reach its own subject.
# The grid is checked against the dtype rather than trusted: every width must be reachable, and
# NVEC 1..4 and both tail states must each appear, or the screen cannot support a claim about the
# macro's whole range.
case "$DTYPE" in float32) LANES=8 ;; float64) LANES=4 ;; esac
python - "$KS" "$LANES" "$DTYPE" <<'PY' || exit 1
import sys
ks = [int(x) for x in sys.argv[1].split(",") if x]
lanes, dtype = int(sys.argv[2]), sys.argv[3]
exact_hi = 7 if lanes == 8 else 3
hi = 4 * lanes
print("  GRID CHECK for %s (%d lanes): regblock reaches k=1..%d, exact kernel owns k=1..%d"
      % (dtype, lanes, hi, exact_hi))
over = [k for k in ks if k > hi]
if over:
    sys.exit("REFUSE: widths %s exceed k=%d, which needs NVEC>4 and is not instantiated" % (over, hi))
ctrl = [k for k in ks if k <= exact_hi]
arm = [k for k in ks if k > exact_hi]
nvec = sorted({-(-k // lanes) for k in arm})
full = sorted({k for k in arm if k % lanes == 0})
mask = sorted({k for k in arm if k % lanes != 0})
print("    negative-control widths (macro inert): %s" % ctrl)
print("    arm widths %s -> nvec %s" % (arm, nvec))
print("    full tail %s   masked tail %s" % (full, mask))
missing = []
if not ctrl: missing.append("no negative control (no width at or below k=%d)" % exact_hi)
if nvec != [1, 2, 3, 4]: missing.append("nvec coverage is %s, not 1..4" % nvec)
if not full: missing.append("no full-tail width")
if not mask: missing.append("no masked-tail width")
if missing:
    sys.exit("REFUSE: the grid does not cover the kernel's branch axes: " + "; ".join(missing))
print("    COVERS nvec 1..4, both tail states, and a negative control")
PY

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
echo "corpus $NK matrices, dtype $DTYPE, widths $KS, reps $REPS"
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
        --dtype "$DTYPE" --ks "$KS" --reps "$REPS" --target-ms 15 \
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
printf "passes 4\nbuilds 4\nnonzero_rc %s\ncsvs %s\nks %s\nkeys %s\ndtype %s\n" \
  "$failed" "$(ls "$OUT"/ss64_*.csv 2>/dev/null | wc -l | tr -d ' ')" "$KS" "$NK" "$DTYPE" > "$OUT/f64screen.done"
cat "$OUT/f64screen.done"
