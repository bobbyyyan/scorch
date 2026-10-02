#!/usr/bin/env bash
# Build the three objects the ADOPT_GRAIN measurement needs, at EQUAL-LENGTH paths.
#
# THE LEVER. scorch_policy.h ships SCORCH_SPMM_ADOPT_GRAIN at 0, which is off. The drop-in SpMM
# adopts the caller's pool so a pipeline does not reshape its team at every op boundary, and that
# adoption currently has no work term at all: it takes min(pool, rows/16) whatever the arithmetic
# is. The knob caps the adopted count so each worker gets a grain of REAL arithmetic (nnz*k, not
# the cache-line-floored nnz*max(k,16)), never below one -- so it can only make an adopted team
# narrower, never decline the adoption, and the pipeline still gets one shared team.
#
# WHY THREE AND NOT TWO. adg0 and adgb are the SAME configuration built twice at paths of equal
# length. That pair is the floor, and it has to be two separate OBJECTS: the arm here is a
# compile-time constant that changes emitted code, and a floor taken from one binary bounds only
# the timing, not the placement. Equal-length paths because path length changes .text, so unequal
# names would put a real codegen difference inside the floor's own pair.
#
# WHAT THIS IS NOT INERT ON, stated because the obvious assumption is wrong. The cap binds
# wherever work_true/grain < pool, which includes WIDE k on small matrices -- at 32530 nonzeros
# and k=16 it takes 6 workers to 3. Those are the widths we already beat MKL at (0.6584 at k=32),
# so the grid must carry the wide-k small-nonzero corner or the measurement cannot see its own
# worst case. Do not treat k>=16 as a negative control here; it is an exposed surface.
set -u
STAGE=${ADG_STAGE:-$(cd "$(dirname "$0")/../.." && pwd)}
ROOT=${ADG_ROOT:?set ADG_ROOT to the build root}
ACTIVATE=${ADG_ACTIVATE:-}

# bash 3.2 has no `declare -A`, and under it the subscript is evaluated ARITHMETICALLY: every
# lookup errors, the error does not stop the script even under `set -u`, and a digest loop written
# that way prints a pass. A `case` needs no associative array and runs on both, so there is
# nothing to guard.
defs_for() {
  case "$1" in
    adg0) echo "" ;;
    adgb) echo "" ;;
    adgG) echo "-DSCORCH_SPMM_ADOPT_GRAIN=150000" ;;
    *)    echo "REFUSE-UNKNOWN-TAG" ;;
  esac
}
TAGS="adg0 adgb adgG"

if [ -n "$ACTIVATE" ]; then
  [ -f "$ACTIVATE" ] || { echo "REFUSE: ADG_ACTIVATE=$ACTIVATE is not a file"; exit 1; }
  # Probed in a subshell first: `source` runs in THIS shell, so a shim spelled `exit 1` would
  # terminate this script at that line with no output at all and never reach the `||` branch.
  ( source "$ACTIVATE" >/dev/null 2>&1 ) || {
      echo "REFUSE: ADG_ACTIVATE=$ACTIVATE exists but does not bring up an environment"; exit 1; }
  source "$ACTIVATE" || { echo "REFUSE: ADG_ACTIVATE=$ACTIVATE failed on the real source"; exit 1; }
fi
for f in src pyproject.toml scorch_build.py; do
  [ -e "$STAGE/$f" ] || { echo "REFUSE: ADG_STAGE=$STAGE has no $f -- wrong stage tree"; exit 1; }
done
python -c "import torch, sys; sys.stderr.write('torch %s at %s\n' % (torch.__version__, sys.executable))" \
  || { echo "REFUSE: the activated environment cannot import torch"; exit 1; }
mkdir -p "$ROOT" || exit 1
echo "stage $STAGE  root $ROOT  host $(hostname)  $(date +%H:%M:%S)"

len=""
for tag in $TAGS; do
  d="$ROOT/$tag"
  rm -rf "$d"; mkdir -p "$d"
  cp -r "$STAGE"/src "$STAGE"/pyproject.toml "$STAGE"/scorch_build.py "$d"/ || exit 1
  echo "  staged $tag  (path len ${#d})  defines='$(defs_for "$tag")'"
  if [ -z "$len" ]; then len=${#d}; elif [ "$len" -ne "${#d}" ]; then
    echo "REFUSE: path lengths differ ($len vs ${#d}); path length changes .text and would put a"
    echo "        codegen difference inside the floor pair."; exit 1
  fi
done

n_amb=$(env | grep -c '^SCORCH_' || true)
echo "ambient SCORCH_* at build time: ${n_amb:-0}"
[ "${n_amb:-0}" -eq 0 ] || env | grep '^SCORCH_' | sort | sed 's/^/  /'

for tag in $TAGS; do
  ( cd "$ROOT/$tag" || exit 1
    d=$(defs_for "$tag")
    [ "$d" != "REFUSE-UNKNOWN-TAG" ] || { echo 1 > rc.txt; exit 1; }
    # SCORCH_BUILD_DEFINES alone does not force a rebuild -- changing only a define relinks the
    # old objects -- and each tag is a fresh copy here, so there is nothing stale to relink.
    { echo "defines: $d"; echo "---- build output ----"; } > build.log
    SCORCH_BUILD_DEFINES="$d" python -c "import setuptools; setuptools.setup()" \
      build_ext --inplace >> build.log 2>&1
    echo "$?" > rc.txt ) &
done
wait

echo
echo "BUILD RESULTS"
fail=0
for tag in $TAGS; do
  rc=$(cat "$ROOT/$tag/rc.txt" 2>/dev/null || echo 99)
  so=$(ls "$ROOT/$tag"/src/scorch_ops*.so 2>/dev/null | head -1)
  if [ "$rc" != "0" ] || [ -z "$so" ]; then
    echo "  $tag  rc=$rc  NO OBJECT"; tail -5 "$ROOT/$tag/build.log" 2>/dev/null | sed 's/^/      /'
    fail=1; continue
  fi
  echo "  $tag  rc=0  $(wc -c < "$so" | tr -d ' ') bytes  so_sha $(shasum -a 256 "$so" | cut -c1-16)"
done
[ "$fail" -eq 0 ] || { echo "REFUSE: a build failed"; exit 1; }

echo
echo "BYTE DIGESTS (one-sided: a difference here proves nothing about decisions, and"
echo "              agreement is impossible by construction -- each object embeds its own"
echo "              staging path, which is why the path-LENGTH clause above exists. The"
echo "              floor pair's 'agree' clause is therefore asserted on DECISIONS below.)"
a=$(shasum -a 256 "$ROOT"/adg0/src/scorch_ops*.so | cut -d' ' -f1)
b=$(shasum -a 256 "$ROOT"/adgb/src/scorch_ops*.so | cut -d' ' -f1)
g=$(shasum -a 256 "$ROOT"/adgG/src/scorch_ops*.so | cut -d' ' -f1)
echo "  adg0 ${a:0:16}   adgb ${b:0:16}   adgG ${g:0:16}"
[ "$a" != "$g" ] || { echo "REFUSE: adgG is byte-identical to adg0, so the define never reached the"
                      echo "        compiler and every reading would be a floor."; exit 1; }
[ "$a" != "$b" ] || { echo "REFUSE: adg0 and adgb are byte-identical. Each object embeds its own staging"
                      echo "        path, so identical bytes mean one build was SKIPPED and both tags are"
                      echo "        reading one object -- the copied-.so-newer-than-source hazard."; exit 1; }
echo "  ASSERTED: all three digests are distinct."

echo
echo "HOOKS CHECK (a hooks object charges ~19us of getenv a call and inverts a short cell vs MKL)"
for tag in $TAGS; do
  PYTHONPATH="$ROOT/$tag/src" python -c "
import sys, scorch_ops as so
h = so.scorch_tune_hooks()
print('  $tag  spmm_tune_hooks=%s' % h)
sys.exit(1 if h else 0)" || { echo "REFUSE: $tag is a hooks build"; exit 1; }
done

echo
echo "DECISION CHECK -- the grid each object resolves, asked of the object itself."
echo "The pool comes from the object's own process (torch.get_num_threads), not a"
echo "constant, so this table is host-shaped. The row counts are DLMC's eight real"
echo "ones; the degrees are a synthetic span of the collection's range, so a grid"
echo "point need not be a matrix in the corpus."
for tag in $TAGS; do
  PYTHONPATH="$ROOT/$tag/src" python -c "
import torch, scorch_ops as so
pool = torch.get_num_threads()
print('pool %d' % pool)
for rows in (64, 128, 256, 512, 1000, 1024, 2048, 33288):
    for deg in (4, 16, 64):
        nnz = rows * deg
        for k in (2, 8, 16, 32):
            print('rows=%-5d deg=%-2d nnz=%-8d k=%-2d -> %d'
                  % (rows, deg, nnz, k,
                     so.scorch_spmm_nthreads(nnz*max(k,16), rows, pool, nnz*k, nnz)))
" > "$ROOT/decisions.$tag.txt" || { echo "REFUSE: $tag could not answer the decision grid."; exit 1; }
  echo "  $tag  $(head -1 "$ROOT/decisions.$tag.txt")  $(( $(wc -l < "$ROOT/decisions.$tag.txt") - 1 )) grid points"
done

# THE FLOOR PAIR'S AGREEMENT CLAUSE, ASSERTED. Stating "adg0/adgb must agree, adgG must
# differ" and then only testing the difference lets a floor pair whose configuration had
# silently diverged print the divergence and pass -- the unasserted conjunct is always the
# one about two things AGREEING, because its violation looks like ordinary output.
if ! diff -q "$ROOT/decisions.adg0.txt" "$ROOT/decisions.adgb.txt" >/dev/null; then
  echo "REFUSE: the floor pair adg0/adgb resolves DIFFERENT counts, so it is not a floor."
  diff "$ROOT/decisions.adg0.txt" "$ROOT/decisions.adgb.txt" | head -20
  exit 1
fi
echo "  ASSERTED: adg0 and adgb agree on every grid point (the floor pair is a floor)."

ndiff=$(diff "$ROOT/decisions.adg0.txt" "$ROOT/decisions.adgG.txt" | grep -c '^<' || true)
[ "${ndiff:-0}" -ge 1 ] || { echo "REFUSE: adgG resolves the same count as adg0 at every one of the grid points,"
                             echo "        so the define is inert here and the arm cannot be measured."; exit 1; }
total=$(( $(wc -l < "$ROOT/decisions.adg0.txt") - 1 ))
echo "  ASSERTED: adgG differs from adg0 on $ndiff of $total grid points (the arm reaches the decision)."
echo "  Grids kept at $ROOT/decisions.*.txt"