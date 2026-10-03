#!/bin/bash
# The row-as-vector screen: three builds, both full collections, ours-vs-MKL per cell.
#
# THE WIDTH ROLES ARE THE INVERSE OF THE PREFETCH SCREENS'. arm_screen.sh uses k<=3 as its
# NEGATIVE CONTROL, because that span is owned by scorch_spmm_row_narrow_exact and the
# register-block prefetch define provably cannot change executed code there. SCORCH_NARROWK_EXACT_ROWVEC
# changes that exact kernel and nothing else, so here k<=3 is the LIVE set and every width above
# it is the control. That inversion is also why k=2 is absent from both prefetch campaigns'
# tables: it was their control.
#
# WHAT THE CONTROL BUYS. The define replaces template instantiations, so the whole object moves:
# rvec's .text is 983314 bytes against ship's 1073586. A width the define cannot reach therefore
# measures this object's layout offset for free, and an effect at k=2 is only an effect if it
# exceeds what k=8 shows.
#
# KEYS ARE DERIVED FROM THE CACHE, not from a corpus list. csrresolve.py exists because a flat
# corpus list silently dropped 50 of 298 keys against MKT's group-qualified cache; keys taken as
# paths relative to the cache resolve 'exact' by construction and cannot drop any.
set -u

ROOT=${1:-}
[ -n "$ROOT" ] || { echo "usage: rvec_screen.sh <root> [ks] [reps] [dtype]"; exit 1; }
[ -d "$ROOT" ] || { echo "REFUSE: root $ROOT is not a directory"; exit 1; }
KS=${2:-2,3,8}
REPS=${3:-7}
DTYPE=${4:-float32}
TARGET_MS=${RVEC_TARGET_MS:-15.0}

BUILDS=${RVEC_BUILDS:-/scr/bobbyy/rvec}
# FLAT, not a tree. MKT keeps the producer files beside the builds rather than in a checked-out
# worktree, which arm_screen.sh records as the reason the harness directory is overridable at all.
# Pointing this at .../harnesses/pfstream gets "No such file or directory" from python, which a
# screen would otherwise discover several hours into a job.
HARNESS=${RVEC_HARNESS:-/matx/u/bobbyy/pfsstage}
CACHEDIR=${RVEC_CACHEDIR:-/scr/bobbyy/mklsweep/csrcache}
OUT=${RVEC_OUT:-$ROOT/csv}
ACTIVATE=${RVEC_ACTIVATE:-/scr/bobbyy/venv/bin/activate}
THREADS=${RVEC_THREADS:-32}

for d in "$BUILDS" "$HARNESS" "$CACHEDIR"; do
  [ -d "$d" ] || { echo "REFUSE: $d is not a directory"; exit 1; }
done
[ -f "$ACTIVATE" ] || { echo "REFUSE: ACTIVATE=$ACTIVATE is not a file"; exit 1; }
for f in mrgate.py csrresolve.py; do
  [ -f "$HARNESS/$f" ] || { echo "REFUSE: $HARNESS/$f absent"; exit 1; }
done
mkdir -p "$OUT" || exit 1

( source "$ACTIVATE" >/dev/null 2>&1 ) || { echo "REFUSE: $ACTIVATE fails when sourced"; exit 1; }
source "$ACTIVATE" || { echo "REFUSE: $ACTIVATE failed on the real source"; exit 1; }
python -c "import torch,sys; sys.stderr.write('torch %s at %s\n'%(torch.__version__,sys.executable))" \
  || { echo "REFUSE: the activated environment cannot import torch"; exit 1; }

echo "host $(hostname)  $(date +%H:%M:%S)  dtype $DTYPE  ks $KS  reps $REPS  target ${TARGET_MS}ms"
echo "builds $BUILDS  out $OUT  threads $THREADS"

# THE GRID CHECK. Unlike the prefetch screens this needs a live width and a control, and nothing
# about nvec coverage -- the kernel under test is reached by width alone.
python - "$KS" <<'PY' || exit 1
import sys
raw = [int(x) for x in sys.argv[1].split(",") if x]
d = sorted({k for k in raw if raw.count(k) > 1})
if d:
    sys.exit("REFUSE: width(s) %s appear more than once; a repeated width is counted twice by "
             "the two category lists while the partition compares against the distinct count" % d)
ks = sorted(set(raw))
# SCORCH_NARROWK_EXACT_HI ships at 3 and clamps at 7 for float; 3 is the value that decides
# what the exact kernel owns, so it is the value that decides what this define can reach.
EXACT_HI = 3
live = [k for k in ks if k <= EXACT_HI]
ctrl = [k for k in ks if k > EXACT_HI]
print("  GRID CHECK: SCORCH_NARROWK_EXACT_ROWVEC reaches the widths the exact kernel owns, k<=%d" % EXACT_HI)
print("    live widths (define CAN change executed code):  %s" % live)
print("    control widths (define provably CANNOT):        %s" % ctrl)
print("    PARTITION: live %d + control %d = %d of %d widths, all accounted"
      % (len(live), len(ctrl), len(live) + len(ctrl), len(ks)))
miss = []
if not live: miss.append("no live width (none at or below k=%d), so the screen tests nothing" % EXACT_HI)
if not ctrl: miss.append("no control width (none above k=%d), so a whole-object layout shift "
                         "cannot be told from an effect" % EXACT_HI)
if miss:
    sys.exit("REFUSE: " + "; ".join(miss))
PY

export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS

for corpus in ss dlmc; do
  CACHE="$CACHEDIR/$corpus"
  [ -d "$CACHE" ] || { echo "REFUSE: cache $CACHE absent"; exit 1; }
  K="$OUT/keys_$corpus.txt"
  find "$CACHE" -name '*.npz' | sed "s|^$CACHE/||; s|\.npz$||" | sort > "$K"
  n=$(wc -l < "$K")
  echo "corpus $corpus: $n keys derived from the cache"
  [ "$n" -gt 0 ] || { echo "REFUSE: $corpus produced 0 keys, so every number below would be vacuous"; exit 1; }

  # THREE SLICES, build order rotated by one each slice. Each build leads exactly once and
  # occupies each position exactly once, so pass position is balanced across builds -- an
  # unbalanced rotation loads the later arms with whatever drifts during the run.
  split -n l/3 -d "$K" "$OUT/sl_${corpus}_"
  for s in 0 1 2; do
    SK="$OUT/sl_${corpus}_0$s"
    [ -s "$SK" ] || { echo "REFUSE: slice $s of $corpus is empty"; exit 1; }
    case $s in
      0) ORDER="ship shpb rvec" ;;
      1) ORDER="shpb rvec ship" ;;
      2) ORDER="rvec ship shpb" ;;
    esac
    echo "  slice $s ($(wc -l < "$SK") keys) order: $ORDER"
    for b in $ORDER; do
      SO=$(ls "$BUILDS/$b"/src/scorch_ops*.so 2>/dev/null | head -1)
      [ -n "$SO" ] || { echo "REFUSE: build $b has no object"; exit 1; }
      C="$OUT/${corpus}_${DTYPE}_${b}_s${s}.csv"
      echo "    $b -> $(basename "$C")  $(date +%H:%M:%S)"
      PYTHONPATH="$BUILDS/$b/src" python "$HARNESS/mrgate.py" \
        --cache "$CACHE" --keys "$SK" --csv "$C" --dtype "$DTYPE" \
        --ks "$KS" --reps "$REPS" --target-ms "$TARGET_MS" \
        --build "$b" --slice "s$s" --pass-tag "p1" \
        >> "$OUT/${corpus}_${b}_s${s}.log" 2>&1
      rc=$?
      echo "      rc=$rc  rows=$( [ -f "$C" ] && wc -l < "$C" || echo 0 )"
      [ "$rc" = "0" ] || { echo "      last log lines:"; tail -4 "$OUT/${corpus}_${b}_s${s}.log" | sed 's/^/        /'; }
    done
  done
done
echo "SCREEN DONE $(date +%H:%M:%S)"
touch "$OUT/rvec_screen.done"
