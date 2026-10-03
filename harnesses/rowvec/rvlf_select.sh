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
# FAILS CLOSED, which the first version did not. `squeue | grep -q` produces no match when
# squeue is absent, errors, or changes format, so a broken query was indistinguishable from "no
# such job" and the script would have built next to a live measurement -- the one thing it exists
# to prevent. The status is captured and a nonzero rc is a REFUSAL. It also matches any state, not
# just RUNNING: a PENDING job of this family can start mid-build, and a build that began legally
# does not become harmless when the measurement starts underneath it.
if ! q=$(squeue -u "$(whoami)" -h -o '%i %T %M %j' 2>&1); then
  echo "REFUSE: squeue failed, so this cannot tell 'no such job' from 'could not ask':"
  echo "$q" | sed 's/^/        /'
  exit 1
fi
if printf '%s\n' "$q" | grep -qE '(rvec_screen|rvlf)'; then
  echo "REFUSE: a job of this family is queued or running. Building now contends for the cores and"
  echo "        the memory bandwidth it is timing on, which would put this script's own"
  echo "        contention inside the floor it exists to measure."
  printf '%s\n' "$q" | grep -E '(rvec_screen|rvlf)' | sed 's/^/        /'
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

# THE AXIS PER ARM, because `displaced` does NOT mean the same thing on both and a single verdict
# label would misread one of them.
#   placement  a _F64 knob leaves the float32 code's CONTENT provably identical (every per-dtype
#              constant is a std::is_same<scalar_t,float> ternary, constexpr off the hooks path).
#              So displacement is the WHOLE of the difference the float32 side can see: zero
#              displacement means the float32 execution is bit-identical and the pair measures
#              nothing. This isolates a pure-placement floor, which is a LOWER bound on the
#              artefact a content-changing arm like ROWVEC incurs.
#   content    a float32 knob changes the emitted code as well as where it sits, so zero
#              displacement does NOT mean nothing changed and the pair is still informative.
#              Closer in kind to ROWVEC, which is why it is kept despite the weaker inertness
#              argument.
# Measured on the existing ROWVEC objects: ship vs hi01 has a 208-byte .text delta and displaces
# ZERO of 28 tracked float32 symbols, while ship vs rvec displaces 21 of 21 JOINED (22 tracked,
# one of which -- narrow_exact<float, 6, 2> -- is absent from ship, so it cannot join). So delta and
# displacement are close to independent, and selecting on delta alone would have been wrong.
declare -A AXIS
AXIS[hv64]="placement"
AXIS[mr64]="placement"
AXIS[both]="placement"
AXIS[hv00]="content"

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
# THE SYMBOL TABLE, KEYED ON THE WHOLE NAME. The first version keyed the join on field 2 of a
# whitespace-split line, which for a function template is the RETURN TYPE -- `void` for nearly
# every kernel here -- so `join` emitted the cross product and counted address pairs between
# DIFFERENT symbols. With n symbols sharing `void` the count was bounded below by n*(n-1), so
# `[ "$D" = 0 ]` was unreachable and the UNUSABLE verdict this script exists to produce could
# never fire. Measured on ship vs shpb, whose .text digests are IDENTICAL so the true displacement
# is exactly 0: the old pipeline read 514. Splitting address from name on nm's TYPE column also
# removes the $9 truncation, which cut signatures at seven tokens and dropped the type field.
#
# AND THE POPULATION IS NOW DEFINED BY EXCLUSION, NOT BY A LIST OF NAMES. A pattern enumerating
# kernel families cannot be trusted in a REFUSAL: the float32 execution path is wider than any
# list I thought to write. Checked rather than assumed -- `spmm_csr_float_v2_core` does not match
# the kernel pattern at all, and `spmm_csr_float_v2` is a live T symbol on that path. If an
# untracked function moves, a name-keyed metric reads zero and UNUSABLE asserts a bit-identity
# that was never tested.
#
# Counting over ALL symbols instead fails the other way: a _F64 knob is SUPPOSED to move the
# float64 code, so `all displaced` is never zero for it and UNUSABLE becomes unreachable again.
# So the population is every defined symbol MINUS the ones the define is allowed to change,
# identified by naming double/f64. Forgetting a float32 function name now leaves it in `rest` by
# default rather than dropping it silently, which is the fail-closed direction.
KPAT='scorch_spmm_row_(narrow_exact|narrow_rowvec|regblock|gather)<(float|scalar_t)|spmm_csr_v2_core<float'
# The symbol-table metric, the demangler choice and the two refusals live in rvlf_lib.sh, which
# the self-test also sources. They were duplicated here once and the copies drifted on the next
# change -- the script moved to a chained demangler, the self-test did not, and its CONTROL case
# then refused on clean input. A harness that restates the thing it checks is wrong silently; this
# one was wrong loudly, which was luck.
. "$(dirname "$0")/rvlf_lib.sh"
rvlf_pick_demangler || exit 1
echo "demangler: $RVLF_DEMANGLER"
KPAT=$RVLF_KPAT
XPAT=$RVLF_XPAT

textsize() { # $1 = .so, $2 = output bin path; echoes the size or returns nonzero
  objcopy -O binary --only-section=.text "$1" "$2" 2>/dev/null || return 1
  [ -s "$2" ] || return 1
  stat -c %s "$2"
}

BSO=$(ls "$ROOT/base"/src/scorch_ops*.so | head -1)
BT=$(textsize "$BSO" "$ROOT/base/text.bin") || {
  echo "REFUSE: objcopy could not extract .text from the base object, so every delta below"
  echo "        would be computed against nothing"; exit 1; }
rvlf_symtab "$BSO" "$ROOT/base/sym"
NA=$(wc -l < "$ROOT/base/sym.all"); NR=$(wc -l < "$ROOT/base/sym.rest"); NK=$(wc -l < "$ROOT/base/sym.kern")
# THE EXCLUSION'S OWN NON-VACUITY NUMBER. XPAT matches the STRING "double", but a symbol nm failed
# to demangle keeps its mangled form, where a double template parameter is encoded as `d`. Such a
# symbol matches none of XPAT's alternatives, stays in `rest`, and for a _F64 arm then moves
# legitimately -- inflating rest_displ and KEEPING an arm that should have been sent away, which is
# the direction the exclusion exists to prevent. Expected zero, printed rather than expected: the
# symbols that fail to demangle here are C-linkage ones and those are not double instantiations.
# THE ASSERTION, RUN PER OBJECT RATHER THAN ONLY ON BASE. A first version checked this on `base`
# alone, which is the one object that by construction CANNOT carry the symbols it was written to
# catch: new unresolvable names come from new float64 template code, and that is exactly what
# `mr64` and `both` introduce. The guard was on the side that cannot fail. Checked here for base
# and again inside the per-tag loop, naming the tag.
rvlf_unres_check "base" "$ROOT/base/sym" >/dev/null || { rvlf_unres_check "base" "$ROOT/base/sym"; exit 1; }
echo "  base: .text $BT bytes;  symbols all $NA, rest $NR, kernels $NK, unresolved 0"
echo
echo "  rest_displ is the verdict number: all symbols EXCEPT those the define may change."
echo "  kern_displ is the same question asked of the SpMM kernels only, reported as a detail."
echo "  ovx_mvd    is how many EXCLUDED symbols moved for this arm -- the bounded set in which an"
echo "             over-exclusion could matter. Names are in <root>/<tag>/overexcluded.txt; every"
echo "             member should be a float64 instantiation, and any float32-path name there is a"
echo "             finding. Measured on ship vs rvec: 31 of 31 are float64 instantiations."
echo
printf "  %-5s %-9s %12s %11s %6s %6s %10s %10s  %s\n" \
  tag axis text_bytes delta rest kern rest_displ kern_displ verdict
for tag in $TAGS; do
  [ "$tag" = "base" ] && continue
  so=$(ls "$ROOT/$tag"/src/scorch_ops*.so 2>/dev/null | head -1)
  [ -n "$so" ] || { printf "  %-5s %s\n" "$tag" "NO OBJECT -- build failed, see build.log"; fail=1; continue; }
  # A TOOL FAILURE MUST NOT ARRIVE WEARING A MEASUREMENT VERDICT. The base side was guarded and
  # the tag side was not: a failing nm left an empty table, the join yielded nothing, and the row
  # printed "UNUSABLE -- float32 code did not move".
  T=$(textsize "$so" "$ROOT/$tag/text.bin") || {
    printf "  %-5s %s\n" "$tag" "REFUSE: objcopy failed on this object"; fail=1; continue; }
  rvlf_symtab "$so" "$ROOT/$tag/sym"
  if ! rvlf_unres_check "$tag" "$ROOT/$tag/sym" >/dev/null; then rvlf_unres_check "$tag" "$ROOT/$tag/sym"; fail=1; continue; fi
  TR=$(wc -l < "$ROOT/$tag/sym.rest"); TK=$(wc -l < "$ROOT/$tag/sym.kern")
  if [ "$TR" -eq 0 ] || [ "$TK" -eq 0 ]; then
    printf "  %-5s %-9s %12s %11s %6s %6s  %s\n" "$tag" "${AXIS[$tag]:-content}" "$T" "$((T-BT))" \
      "$TR" "$TK" "REFUSE: nm tracked 0 in a population here -- a tool failure, not a null"
    fail=1; continue
  fi
  read -r JR DR <<<"$(rvlf_disp "$ROOT/base/sym.rest" "$ROOT/$tag/sym.rest")"
  read -r JK DK <<<"$(rvlf_disp "$ROOT/base/sym.kern" "$ROOT/$tag/sym.kern")"
  # OVER-EXCLUSION, MEASURED PER ARM. |excluded| is the wrong quantity -- an over-excluded symbol
  # that never moves changes no displacement count. The ones that can change a verdict are those
  # that MOVED for this arm, and that set is bounded by what the define perturbed, so it is short
  # enough to read. Written to a file rather than summarised, because the question it answers is
  # "is any of these a float32-path name", which only the names can answer.
  rvlf_overexcluded_moved "$ROOT/base/sym" "$ROOT/$tag/sym" > "$ROOT/$tag/overexcluded.txt"
  read -r OE OM OB <<<"$(head -1 "$ROOT/$tag/overexcluded.txt")"
  # THE INVARIANT THAT CATCHES A CROSS PRODUCT. With one row per name the join cannot exceed the
  # smaller table. Checked rather than assumed, because this exact failure has now occurred twice
  # in this script from two different causes and both times produced a plausible number.
  MR=$(( NR < TR ? NR : TR ))
  if [ "$JR" -gt "$MR" ]; then
    printf "  %-5s %s\n" "$tag" "REFUSE: join returned $JR rows against a smaller table of $MR -- a cross product"
    fail=1; continue
  fi
  AD=$((T - BT)); [ "$AD" -lt 0 ] && AD=$((-AD))
  AX=${AXIS[$tag]:-content}
  if [ "$JR" -eq 0 ]; then
    V="REFUSE: join matched 0 names in 'rest' -- not a null displacement"; fail=1
  elif [ "$DR" -eq 0 ] && [ "$AX" = "placement" ]; then
    # Content provably fixed on this axis AND nothing outside the excluded set moved.
    V="UNUSABLE -- content provably fixed and nothing moved: this pair is an A/A"
  elif [ "$DR" -eq 0 ]; then
    V="usable (content axis) -- unmoved, but the emitted float32 code differs"
  elif [ "$AD" -ge "$TARGET_DELTA" ]; then
    V="BRACKETS FROM ABOVE"
  else
    V="brackets from below"
  fi
  printf "  %-5s %-9s %12s %11s %6s %6s %10s %10s %8s  %s\n" \
    "$tag" "$AX" "$T" "$((T-BT))" "$JR" "$JK" "$DR" "$DK" "$OB" "$V"
done
echo
echo "Keep the candidates whose verdict is not UNUSABLE and whose deltas STRADDLE $TARGET_DELTA."
echo "Those, paired against base, are the arms for the layout-floor job -- which measures the"
echo "tier at every width in the grid at once, and so also answers whether it varies with width."
