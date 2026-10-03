#!/bin/bash
# Select a layout-floor pair for the ROWVEC comparison, BEFORE spending any job time on timing.
#
# THE QUANTITY AND WHY IT HAS NO PRIOR CALIBRATION. An arm that changes code needs two different
# controls, and they are not the same pair:
#   routing inertness -- any cell the define provably cannot reach. One pair, any width.
#   layout floor at W -- a pair with DIFFERENT .text that routes IDENTICALLY at W.
# The ROWVEC screen has the first (k=8) and cannot have the second: ship/shpb are byte-identical,
# and ship/rvec, ship/hi01, rvec/hi01 all route differently at k=2. chain185 was read as supplying
# the second and does not: its only identical-.text pair is c0b/c0, which w1fix_body.sh:118 states
# outright ("NEED=\"c0 and c0b: identical .text"), so it is the NOISE tier. Its w1s8/w1pg rows are
# candidate arms -- the width-2 one carries verdict FASTER and is the single cell its own
# twohost.txt annotates HOSTS DISAGREE. So the layout tier is unmeasured in this line.
#
# WHY THIS SCRIPT ONLY SELECTS. A _F64-only knob leaves the float32 code's CONTENT identical and
# perturbs only its PLACEMENT. Whether the linker actually moves it depends on emission order,
# which no amount of source reading settles -- if the float32 symbols land at unchanged addresses
# the pair measures nothing. That is checkable from nm alone, so the selection is a measurement
# rather than a guess, and it costs no timing.
#
# THE CANDIDATES, ordered by the strength of their inertness argument, not by convenience:
#
#   hv64  -DSCORCH_SPMM_HALFVEC_F64=1        (default 0, scorch_policy.h:1515)
#   mr64  -DSCORCH_SPMM_MULTIROW_ROWS_F64=2  (default 0 via :620/:626)
#   both  both of the above, since knobs on this axis compose
#         STRONGEST inertness: every per-dtype constant is selected by a
#         std::is_same<scalar_t,float> ternary (spmm.h:4139, :4144, :4158, :4163, :4356, :4364),
#         constexpr in the non-hooks branch, so a _F64 knob cannot alter the float32
#         instantiation's emitted code at ANY width, on ANY matrix, under ANY EXACT_HI, MINDEG or
#         EXACT_UNROLL. spmm.h:4350-4353 asserts the property in the mirror direction: "a dtype
#         whose row count is 0 folds multirow_ok -- and the whole dispatch that reads it -- to
#         nothing. That is what keeps the float64 object byte-identical while float32 gains the
#         kernel."
#
#   hv00  -DSCORCH_SPMM_HALFVEC_F32=0        (default 1, :1512)
#         Inert at every float32 width EXCEPT 4, by width and by short-circuit: :5004 opens with
#         `if (spmm_halfvec > 0)`, :5010's halfvec_under needs `spmm_halfvec >= 2` against a
#         shipped 1 so it dies BEFORE !exact_width is evaluated, and halfvec_exact is
#         `B1_size == LH` (4 for float). Weaker than the _F64 knobs because it changes float32
#         CONTENT, but it is the only candidate that gives a same-decision pair at k=2 while
#         perturbing the float32 code itself.
#
# RULED OUT, recorded so no build is spent on them:
#   EXACT_HI       changes what is CALLED, never what is emitted. The exact-kernel instantiations
#                  come from the compile-time switch(exact_width) at :5122 and :5294, which is why
#                  hi01 held ship's full 15-symbol set at the same digest.
#   REGBLOCK_PFD=0 inert at k=2 ONLY via exact_width (:5087 gates every regblock case, :4528 gates
#                  the workspace path), so its inertness needs EXACT_UNROLL != 0 AND exact_lo_ <= 2
#                  AND exact_hi_ >= 2 AND MINDEG == 0 -- the same conjunction as the HALFVEC=2
#                  vacuity. Under MINDEG > 0 it goes LIVE at k=2 on exactly the matrices below
#                  that mean degree, so it would contaminate a spread on a subset of the corpus
#                  instead of failing visibly.
#
# UNPROVEN BUILD, flagged rather than assumed: spmm.h:4570 records that
# -DSCORCH_SPMM_MULTIROW_ROWS=2 "failed outright on `sp_row_units was not declared`". The repair
# is at :4573 (`constexpr long sp_row_units = 0;`), so mr64 should build -- but by that comment's
# own account no release object has ever been built with it. A build failure here is a finding.
set -u

STAGE=${PFS_STAGE:-/matx/u/bobbyy/pfsstage}
ROOT=${PFS_ROOT:-/scr/bobbyy/rvlf}
ACTIVATE=${PFS_ACTIVATE:-/scr/bobbyy/venv/bin/activate}
# The perturbation this must bracket: rvec's .text against ship's, from the ROWVEC build.
TARGET_DELTA=${RVLF_TARGET_DELTA:-90272}

[ -f "$ACTIVATE" ] || { echo "REFUSE: PFS_ACTIVATE=$ACTIVATE is not a file"; exit 1; }
( source "$ACTIVATE" >/dev/null 2>&1 ) || { echo "REFUSE: $ACTIVATE fails when sourced"; exit 1; }
source "$ACTIVATE" || { echo "REFUSE: $ACTIVATE failed on the real source"; exit 1; }
for f in src pyproject.toml scorch_build.py; do
  [ -e "$STAGE/$f" ] || { echo "REFUSE: PFS_STAGE=$STAGE has no $f -- wrong stage tree"; exit 1; }
done
python -c "import torch,sys; sys.stderr.write('torch %s at %s\n'%(torch.__version__,sys.executable))" \
  || { echo "REFUSE: the activated environment cannot import torch"; exit 1; }

# A CONCURRENT BUILD PERTURBS A LIVE MEASUREMENT, and this script exists to measure a floor, so
# running it beside a timing job would put the contention inside the number it produces.
if squeue -u "$(whoami)" -h -o '%i %T %j' 2>/dev/null | grep -qE 'RUNNING.*(rvec_screen|rvlf)'; then
  echo "REFUSE: a screen job of this family is RUNNING. Building now contends for the cores and"
  echo "        the memory bandwidth it is timing on. Wait for it, then re-run."
  squeue -u "$(whoami)" -h -o '  %i %T %M %j' | sed 's/^/  /'
  exit 1
fi

mkdir -p "$ROOT" || exit 1
echo "stage $STAGE  root $ROOT"
echo "host $(hostname)  g++ $(g++ -dumpversion)  nproc $(nproc)  $(date +%H:%M:%S)"
echo "bracketing target: rvec moved .text by $TARGET_DELTA bytes against ship"

declare -A DEFS
DEFS[base]=""
DEFS[hv64]="-DSCORCH_SPMM_HALFVEC_F64=1"
DEFS[mr64]="-DSCORCH_SPMM_MULTIROW_ROWS_F64=2"
DEFS[both]="-DSCORCH_SPMM_HALFVEC_F64=1 -DSCORCH_SPMM_MULTIROW_ROWS_F64=2"
DEFS[hv00]="-DSCORCH_SPMM_HALFVEC_F32=0"
TAGS="base hv64 mr64 both hv00"

for tag in $TAGS; do
  d="$ROOT/$tag"
  rm -rf "$d"; mkdir -p "$d"
  cp -r "$STAGE"/src "$STAGE"/pyproject.toml "$STAGE"/scorch_build.py "$d"/ || exit 1
  echo "  staged $tag  (path len ${#d})"
done

for tag in $TAGS; do
  (
    cd "$ROOT/$tag" || exit 1
    n_amb=$(env | grep -c '^SCORCH_' || true)
    { echo "AMBIENT SCORCH_* AT BUILD TIME: $n_amb"
      [ "${n_amb:-0}" -gt 0 ] && env | grep '^SCORCH_' | sort | sed 's/^/  /'
      echo "SCORCH_BUILD_DEFINES for this tag: ${DEFS[$tag]}"
      echo "---- build output ----"; } > build.log
    SCORCH_BUILD_DEFINES="${DEFS[$tag]}" \
      python -c "import setuptools; setuptools.setup()" build_ext --inplace >> build.log 2>&1
    echo "$? $tag" > rc.txt
  ) &
done
wait

echo
echo "BUILD RESULTS"
fail=0
for tag in $TAGS; do
  rc=$(cut -d' ' -f1 "$ROOT/$tag/rc.txt" 2>/dev/null)
  so=$(ls "$ROOT/$tag"/src/scorch_ops*.so 2>/dev/null | head -1)
  if [ "$rc" != "0" ] || [ -z "$so" ]; then
    echo "  $tag  BUILD FAILED rc=$rc"
    tail -6 "$ROOT/$tag/build.log" 2>/dev/null | sed 's/^/      /'
    [ "$tag" = "base" ] && fail=1
    continue
  fi
  echo "  $tag  rc=0  $(stat -c %s "$so") bytes"
done
[ "$fail" = "0" ] || { echo "REFUSE: the base object failed; nothing below has a reference"; exit 1; }

# Hooks must be off, or a _F64 knob changes an immediate instead of removing code (spmm.h:4235
# reads SCORCH_SPMM_HALFVEC from the environment under SCORCH_TUNE_HOOKS, and :4160 records that
# the value is read early "so the dispatch below is a compile-time constant when the hook is off").
echo
echo "HOOKS CHECK -- the instrument DEPENDS on this, it is not a convenience"
for tag in $TAGS; do
  so=$(ls "$ROOT/$tag"/src/scorch_ops*.so 2>/dev/null | head -1); [ -n "$so" ] || continue
  read -r v mod <<<"$(PYTHONPATH="$ROOT/$tag/src" python -c \
    'import torch, scorch_ops as S; print(getattr(S,"spmm_tune_hooks","ABSENT"), S.__file__)' 2>/dev/null)"
  case "$mod" in "$ROOT/$tag"/*) ;; *) echo "REFUSE: $tag answered from ${mod:-<none>}"; exit 1 ;; esac
  [ "$v" = "False" ] || { echo "REFUSE: $tag reports spmm_tune_hooks=$v"; exit 1; }
  echo "  $tag  spmm_tune_hooks=False  from this object"
done

echo
echo "SELECTION. Two numbers decide it, and the second is the one that can be zero:"
echo "  .text delta   -- the size of the object perturbation"
echo "  f32 displaced -- how many float32 SpMM symbols changed ADDRESS. A _F64 knob leaves the"
echo "                   float32 code's CONTENT identical, so if the linker does not move it the"
echo "                   pair measures nothing and the .text delta is irrelevant."
BSO=$(ls "$ROOT/base"/src/scorch_ops*.so | head -1)
objcopy -O binary --only-section=.text "$BSO" "$ROOT/base/text.bin" 2>/dev/null
BT=$(stat -c %s "$ROOT/base/text.bin")
nm -C --defined-only "$BSO" 2>/dev/null | grep -E "scorch_spmm_row_(narrow_exact|narrow_rowvec|regblock|gather)<(float|scalar_t)|spmm_csr_v2_core<float" \
  | awk '{print $1, $3, $4, $5, $6, $7, $8, $9}' | sort -k2 > "$ROOT/base/f32syms.txt"
NB=$(wc -l < "$ROOT/base/f32syms.txt")
echo "  base: .text $BT bytes, $NB float32 SpMM symbols tracked"
[ "$NB" -gt 0 ] || { echo "REFUSE: zero float32 symbols tracked, so 'displaced' could only read 0"; exit 1; }
echo
printf "  %-5s %12s %12s %10s %8s  %s\n" tag text_bytes delta f32_displ frac verdict
for tag in $TAGS; do
  [ "$tag" = "base" ] && continue
  so=$(ls "$ROOT/$tag"/src/scorch_ops*.so 2>/dev/null | head -1); [ -n "$so" ] || { printf "  %-5s %12s\n" "$tag" "NO OBJECT"; continue; }
  objcopy -O binary --only-section=.text "$so" "$ROOT/$tag/text.bin" 2>/dev/null
  T=$(stat -c %s "$ROOT/$tag/text.bin")
  nm -C --defined-only "$so" 2>/dev/null | grep -E "scorch_spmm_row_(narrow_exact|narrow_rowvec|regblock|gather)<(float|scalar_t)|spmm_csr_v2_core<float" \
    | awk '{print $1, $3, $4, $5, $6, $7, $8, $9}' | sort -k2 > "$ROOT/$tag/f32syms.txt"
  # A symbol is displaced if its name is present in both and its address differs.
  D=$(join -j 2 -o 1.1,2.1 \
        <(awk '{a=$1; $1=""; sub(/^ /,""); print a, $0}' "$ROOT/base/f32syms.txt" | sort -k2) \
        <(awk '{a=$1; $1=""; sub(/^ /,""); print a, $0}' "$ROOT/$tag/f32syms.txt" | sort -k2) \
        2>/dev/null | awk '$1 != $2' | wc -l)
  AD=$((T - BT)); [ "$AD" -lt 0 ] && AD=$((-AD))
  FR=$(awk -v d="$D" -v n="$NB" 'BEGIN{printf "%.2f", n?d/n:0}')
  if [ "$D" = "0" ]; then V="UNUSABLE -- float32 code did not move"
  elif [ "$AD" -ge "$TARGET_DELTA" ]; then V="BRACKETS FROM ABOVE"
  else V="brackets from below"; fi
  printf "  %-5s %12s %12s %10s %8s  %s\n" "$tag" "$T" "$((T-BT))" "$D" "$FR" "$V"
done
echo
echo "Keep the candidates whose verdict is not UNUSABLE and whose deltas STRADDLE $TARGET_DELTA."
echo "Those, paired against base, are the arms for the layout-floor job -- which measures the"
echo "tier at every width in the grid at once, and so also answers whether it varies with width."
