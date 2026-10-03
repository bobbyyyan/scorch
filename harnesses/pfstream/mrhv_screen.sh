#!/bin/bash
# The four-arm prefetch screen, host- and dtype-portable.
#
# THE WIDTH SET IS DTYPE-DEPENDENT AND THAT IS THE WHOLE DESIGN. The register-block kernel is
# specialized NVEC 1..4 with NVEC = ceil(k / V::lanes), and scorch_simd<float>::lanes is 8 while
# scorch_simd<double>::lanes is 4. So the reachable register-block range is k=1..32 for float32
# and k=1..16 for float64, and "width 32" is a float32-only question -- in float64 it would need
# NVEC 8 and the kernel cannot reach it.
#
# Below the range, scorch_spmm_row_narrow_exact owns k=1..3 for BOTH dtypes, so a width inside
# that span is one at which
# SCORCH_SPMM_REGBLOCK_PFD provably cannot change executed code -- a NEGATIVE CONTROL rather than
# a data point, and the only thing that bounds a single-object arm's layout offset for free.
#
# NOT k=1..7 for float32, which these lines said until this correction. `exact_cap_ =
# exact_f32_ ? 7 : 3` only CLAMPS; the value deciding what the exact kernel owns is
# SCORCH_NARROWK_EXACT_HI, which ships at 3 with both the _F32 and _F64 aliases pointing at it.
# The executable grid check below uses exact_hi = 3 and explains why at line ~99. Reading 7 off
# this header and passing it as the $3 width override would offer k=5, 6 or 7 as float32
# "negative controls" when they are live register-block widths.
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
# OUTPUT NAMES CARRY THE CORPUS AND DTYPE. They used to be ss64_<build>_p<n>.csv and
# f64screen.done regardless of what ran, so a float32 DLMC screen wrote files named for a float64
# SuiteSparse one. The sentinel carries a dtype line, so the DATA was self-describing while the
# FILENAMES were not -- and a filename is what survives into a manifest, where only provenance
# separates one run from another. Two screens writing byte-identically-named outputs for
# different dtypes over different corpora is the setup in which a mix-up is invisible.
CORPUS=${F64_CORPUS:-$(basename "$CACHE")}
PREFIX=${F64_PREFIX:-${CORPUS}_${DTYPE}}

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
raw = [int(x) for x in sys.argv[1].split(",") if x]
dupes = sorted({k for k in raw if raw.count(k) > 1})
if dupes:
    sys.exit("REFUSE: width(s) %s appear more than once in the grid %s. A repeated width is "
             "counted twice by the category lists below while the partition compares against "
             "the distinct count, so the partition would refuse with an empty unplaced list and "
             "point a reader at the partition instead of at the duplicate." % (dupes, raw))
ks = sorted(set(raw))
lanes, dtype = int(sys.argv[2]), sys.argv[3]
# THE SHIPPED BOUND, NOT THE INSTANTIATION CAP. spmm.h reads
#   int exact_hi_ = narrowk_exact_hi;
#   const int exact_cap_ = exact_f32_ ? 7 : 3;
#   if (exact_hi_ > exact_cap_) exact_hi_ = exact_cap_;
# so 7 only CLAMPS; it is the number of instantiated float widths. The value that decides what
# the exact kernel owns is SCORCH_NARROWK_EXACT_HI, which ships at 3, and both
# SCORCH_NARROWK_EXACT_HI_F32 and _F64 are defined to it. So the exact kernel owns k=1..3 for
# BOTH dtypes. Using 7 here would let k=5, 6 or 7 be accepted as a float32 "negative control"
# when they are live register-block widths (nvec 1, masked tail), and would drop them from the
# nvec and tail census -- a latent hole in a fail-closed guard. It is invisible on the float64
# side because min(3, 3) makes the two bounds agree.
exact_hi = 3
# float32 additionally routes k == lanes/2 to the half-vector kernel (halfvec_exact is
# B1_size == LH, LH = 4 for float, and SCORCH_SPMM_HALFVEC_F32 ships at 1), so that one width is
# neither the exact kernel nor a plain register block. SCORCH_SPMM_HALFVEC_F64 ships at 0.
halfvec = lanes // 2 if lanes == 8 else None
hi = 4 * lanes
print("  GRID CHECK for %s (%d lanes): regblock reaches k=1..%d, exact kernel owns k=1..%d"
      % (dtype, lanes, hi, exact_hi))
over = [k for k in ks if k > hi]
if over:
    sys.exit("REFUSE: widths %s exceed k=%d, which needs NVEC>4 and is not instantiated" % (over, hi))
ctrl = [k for k in ks if k <= exact_hi]
arm = [k for k in ks if k > exact_hi and k != halfvec]
nvec = sorted({-(-k // lanes) for k in arm})
full = sorted({k for k in arm if k % lanes == 0})
mask = sorted({k for k in arm if k % lanes != 0})
print("    negative-control widths (macro inert, exact kernel owns k=1..%d): %s" % (exact_hi, ctrl))
if halfvec is not None and halfvec in ks:
    print("    NOTE k=%d routes to the half-vector kernel on this dtype, so it is neither the"
          % halfvec)
    print("         exact kernel nor a plain register block; it is excluded from the tail census")
print("    arm widths %s -> nvec %s" % (arm, nvec))
print("    full tail %s   masked tail %s" % (full, mask))
missing = []
if not ctrl: missing.append("no negative control (no width at or below k=%d)" % exact_hi)
if nvec != [1, 2, 3, 4]: missing.append("nvec coverage is %s, not 1..4" % nvec)
if not full: missing.append("no full-tail width")
if not mask: missing.append("no masked-tail width")
# PARTITION ASSERTION. ctrl, arm and the half-vector width must account for every width in the
# grid. They do not automatically: with exact_hi = 3 a float32 k=4 is excluded from ctrl (4 > 3)
# and from arm (k == halfvec), so it would sit in the grid and in no printed category, visible
# only through a conditional note. Same principle as the analyser's width partition, one axis
# down.
accounted = len(ctrl) + len(arm) + (1 if halfvec is not None and halfvec in ks else 0)
if accounted != len(set(ks)):
    unplaced = [k for k in set(ks) if k not in ctrl and k not in arm and k != halfvec]
    missing.append("ctrl %d + arm %d + half-vector %d accounts for %d of %d widths; unplaced %s"
                   % (len(ctrl), len(arm),
                      1 if halfvec is not None and halfvec in ks else 0,
                      accounted, len(set(ks)), unplaced))
if missing:
    sys.exit("REFUSE: the grid does not cover the kernel's branch axes: " + "; ".join(missing))
print("    PARTITION: ctrl %s + arm %s + half-vector %s = %d of %d widths, all accounted"
      % (len(ctrl), len(arm), 1 if halfvec is not None and halfvec in ks else 0,
         accounted, len(set(ks))))
print("    COVERS nvec 1..4, both tail states, and a negative control")
PY

# Every arm must exist before the first measurement, so a missing build fails now rather than
# three hours in with a partial Latin square on disk.
BUILDS="mr0 mr0b mrhv mrhv4"
for t in $BUILDS; do
  [ -f "$PFS/$t/src/scorch_ops.so" ] || [ -n "$(ls "$PFS/$t/src"/scorch_ops*.so 2>/dev/null)" ] \
    || { echo "REFUSE: build $t has no scorch_ops .so under $PFS/$t/src"; exit 1; }
done

mkdir -p "$OUT" || exit 1
NK=$(wc -l < "$KEYS" | tr -d ' ')
# THE POOL MUST MATCH WHAT WAS ASKED FOR. A batch submission without a cpus-per-task request
# gets the scheduler's default and torch reads a pool of 2, while OMP_NUM_THREADS still says 32 --
# the cgroup caps it and nothing in the output looks wrong. That run completes and its timings are
# comparable to nothing. Refusing here costs one second and catches it before the first cell.
POOL=$(python -c 'import torch;print(torch.get_num_threads())')
echo "host $(hostname)  pool $POOL  threads $THREADS  $(date +%H:%M:%S)"
case "$POOL" in ''|*[!0-9]*) echo "REFUSE: could not read the torch thread pool (got '$POOL')"; exit 1 ;; esac
MINPOOL=${F64_MINPOOL:-$(( THREADS / 2 ))}
if [ "$POOL" -lt "$MINPOOL" ]; then
  echo "REFUSE: torch reports a thread pool of $POOL but this screen asked for $THREADS threads"
  echo "        (floor $MINPOOL). A pool far below the request means the allocation is smaller"
  echo "        than intended -- a batch job without a cpus-per-task request reads 2 here -- and"
  echo "        the timings would not be comparable to any other run of this screen."
  echo "        Set F64_MINPOOL to override if a small pool is deliberate."
  exit 1
fi
echo "corpus $NK matrices, dtype $DTYPE, widths $KS, reps $REPS"
echo "output prefix $PREFIX, sentinel $OUT/screen.done"
echo "cache $CACHE"
echo "builds $BUILDS"

ORDERS=( "mr0 mr0b mrhv mrhv4" "mr0b mrhv mrhv4 mr0" "mrhv mrhv4 mr0 mr0b" "mrhv4 mr0 mr0b mrhv" )
failed=0
for p in 1 2 3 4; do
  order=${ORDERS[$((p-1))]}
  echo "== pass p$p order: $order"
  for tag in $order; do
    out=$OUT/${PREFIX}_${tag}_p${p}.csv
    echo "  --- p$p $tag start $(date +%H:%M:%S)"
    OMP_NUM_THREADS=$THREADS PYTHONPATH="$PFS/$tag/src" \
      python "$H/mrgate.py" --cache "$CACHE" --keys "$KEYS" --csv "$out" \
        --dtype "$DTYPE" --ks "$KS" --reps "$REPS" --target-ms 15 \
        --pass-tag "p$p" --build "$tag" --slice all > "$OUT/${PREFIX}_${tag}_p${p}.log" 2>&1
    rc=$?
    rows=$(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 ))
    echo "  --- p$p $tag done  $(date +%H:%M:%S)  rc=$rc  ($rows rows)"
    if [ "$rc" != "0" ]; then
      failed=$(( failed + 1 ))
      echo "REFUSE: p$p $tag exited $rc"
      tail -4 "$OUT/${PREFIX}_${tag}_p${p}.log"
      exit 1
    fi
  done
done
echo "F64_SCREEN_ALL_PASSES_DONE $(date +%H:%M:%S)  nonzero rc $failed"
# POOL IS RECORDED, not only compared to a threshold. The guard above refuses a pool below half
# the request, but comparability with another run of this screen wants the pool EQUAL to that
# run's, not merely above a floor -- and our kernel and MKL scale differently with thread count,
# so an undersized allocation that passes the floor moves the ratio by an unknown amount.
# Recording it turns a threshold judgement into an exact cross-run comparison a reader can make.
printf "passes 4\nbuilds 4\nnonzero_rc %s\ncsvs %s\nks %s\nkeys %s\ndtype %s\npool %s\nthreads %s\n" \
  "$failed" "$(ls "$OUT"/${PREFIX}_*.csv 2>/dev/null | wc -l | tr -d ' ')" "$KS" "$NK" "$DTYPE" \
  "$POOL" "$THREADS" > "$OUT/screen.done"
cat "$OUT/screen.done"
