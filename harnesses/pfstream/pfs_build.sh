#!/bin/bash
# Build the four objects the PFSTREAM measurement needs, at EQUAL-LENGTH paths.
#
# WHY FOUR AND NOT THREE. ship and shpb are the SAME configuration built twice at paths of equal
# length. That pair is the A/A floor, and it has to be a separate build rather than two runs of one
# object: the build floor's whole increment over the arm floor is reproduced by two byte-identical
# objects, so a floor taken from one binary understates what an arm-vs-arm comparison must clear.
# Equal-length paths because path length changes .text -- a longer directory name is itself a
# codegen difference, which would put a real artefact inside the floor's own pair.
#
# THE THREE ARMS, all on the SHIPPED register-block kernel (not the deep one):
#   ship  defaults: prefetch run-ahead 2 nonzeros, CLIPPED at the row boundary
#   strm  SCORCH_SPMM_REGBLOCK_PFSTREAM=1: the same run-ahead CLAMPED to the end of the nonzero
#         array instead, which is branch-free (compare + cmov against a loop-invariant bound)
#   nopf  SCORCH_SPMM_REGBLOCK_PFD=0: no prefetch at all
#
# The pre-registered prediction (scorch_policy.h:732) is a conjunction and can fail: strm must be
# at least as fast as nopf wherever B is cache-resident (the guard is gone) AND at least as fast as
# ship where the prefetch earns its keep (the coverage is extended, not removed). One object
# beating both arms. Failing either clause is a result.
set -u
# HOST PARAMETERISATION. The three paths below were hardcoded to MKT, which made this script
# unrunnable on the second confirmation host -- and a one-host perf claim is not a claim here.
# The MKT values are kept as the defaults so that path is byte-unchanged; redwood overrides them.
# PFS_ACTIVATE is SOURCED, so a conda host points it at a shim that does the conda activation
# rather than this script learning two environment managers.
STAGE=${PFS_STAGE:-/matx/u/bobbyy/pfsstage}
ROOT=${PFS_ROOT:-/scr/bobbyy/pfs}
ACTIVATE=${PFS_ACTIVATE:-/scr/bobbyy/venv/bin/activate}

# FAIL CLOSED ON EACH, with the value printed. A missing stage tree made `cp -r` fail per-tag and
# the build then ran against whatever was already in $ROOT/$tag from a previous run -- stale
# objects that pass every digest check in this file because they are internally consistent.
[ -f "$ACTIVATE" ] || { echo "REFUSE: PFS_ACTIVATE=$ACTIVATE is not a file"; exit 1; }
source "$ACTIVATE" || exit 1
for f in src pyproject.toml scorch_build.py; do
  [ -e "$STAGE/$f" ] || { echo "REFUSE: PFS_STAGE=$STAGE has no $f -- wrong stage tree"; exit 1; }
done
# AND THE ENVIRONMENT MUST ACTUALLY HAVE TORCH. Without this the activation can succeed while
# importing the wrong interpreter, and the failure surfaces 200 lines into a build log as a
# missing header rather than as a wrong environment.
python -c "import torch, sys; sys.stderr.write('torch %s at %s\n' % (torch.__version__, sys.executable))" \
  || { echo "REFUSE: the activated environment cannot import torch"; exit 1; }
mkdir -p "$ROOT" || exit 1
echo "stage $STAGE  root $ROOT  activate $ACTIVATE"
echo "host $(hostname)  g++ $(g++ -dumpversion)  nproc $(nproc)  $(date +%H:%M:%S)"

declare -A DEFS
DEFS[ship]=""
DEFS[shpb]=""
DEFS[strm]="-DSCORCH_SPMM_REGBLOCK_PFSTREAM=1"
DEFS[nopf]="-DSCORCH_SPMM_REGBLOCK_PFD=0"

for tag in ship shpb strm nopf; do
  d="$ROOT/$tag"
  rm -rf "$d"; mkdir -p "$d"
  cp -r "$STAGE"/src "$STAGE"/pyproject.toml "$STAGE"/scorch_build.py "$d"/ || exit 1
  echo "  staged $tag  (path len ${#d})"
done

# POSITIVE CONTROL for the ambient-environment section below. The section's failure mode is
# silence, and silence is what it also prints when nothing is set -- so prove the machinery emits
# a line when a matching variable IS present, before trusting an empty section to mean "empty".
# Greps the PRODUCTION pattern, not a probe-specific one. A probe matching
# '^SCORCH_PFS_BUILD_ENV_PROBE' would certify the env-to-grep-to-count plumbing while leaving a
# mistyped '^SCORCH_' in the section below undetected; since the probe variable starts with
# SCORCH_ anyway, using the real pattern is strictly tighter at the same cost and couples the
# guard to the thing it certifies.
probe=$(SCORCH_PFS_BUILD_ENV_PROBE=1 env | grep -c '^SCORCH_' || true)
if [ "${probe:-0}" -lt 1 ]; then
  echo "REFUSE: the ambient-environment probe cannot see a variable that is set, so an empty"
  echo "        SCORCH_* section in the build logs would carry no information."
  exit 1
fi
echo "ambient-environment probe: sees a set variable (count $probe), so an empty section is a fact"

# Build all four concurrently; each is one translation unit, so they do not contend on make jobs.
for tag in ship shpb strm nopf; do
  (
    cd "$ROOT/$tag" || exit 1
    # Record the AMBIENT SCORCH_* environment before building. scorch_build.py:103 keys
    # -DSCORCH_TUNE_HOOKS on SCORCH_BUILD_TUNE_HOOKS, a DIFFERENT variable from the
    # SCORCH_BUILD_DEFINES this script sets, and there is no `env -i` here. So whether these are
    # hooks builds depended on the ambient shell and nothing recorded it. A hooks build adds
    # ~54 getenv per call, 17-26us under Slurm, which manufactured a 1.57x MKL deficit on short
    # kernels once -- so it would make the ours/mkl columns unreadable, not merely the split
    # gate unanswerable. The emission check below cannot see this axis: ship == shpb and
    # strm != ship hold identically whether or not the hooks are compiled in.
    # The count is taken FIRST and branched on, rather than relying on `|| echo` after a
    # pipeline. A pipeline's exit status is its LAST command's, and `sed` succeeds on empty
    # input, so `env | grep ... | sed ... || echo "(none)"` never reaches the fallback: the
    # no-variables case prints NOTHING, which is indistinguishable from a truncated log, a
    # redirect that landed elsewhere, or this section not running at all -- in the one file
    # whose whole purpose is to record what the ambient environment was.
    n_amb=$(env | grep -c '^SCORCH_' || true)
    {
      echo "AMBIENT SCORCH_* ENVIRONMENT AT BUILD TIME: $n_amb variable(s)"
      if [ "${n_amb:-0}" -gt 0 ]; then
        env | grep '^SCORCH_' | sort | sed 's/^/  /'
      else
        echo "  (no SCORCH_* variables were set in the ambient environment)"
      fi
      echo "SCORCH_BUILD_DEFINES deliberately set for this tag: ${DEFS[$tag]}"
      echo "---- build output follows ----"
    } > build.log
    SCORCH_BUILD_DEFINES="${DEFS[$tag]}" \
      python -c "import setuptools; setuptools.setup()" build_ext --inplace \
      >> build.log 2>&1
    echo "$? $tag" > rc.txt
  ) &
done
wait

echo
echo "BUILD RESULTS"
for tag in ship shpb strm nopf; do
  d="$ROOT/$tag"
  rc=$(cut -d' ' -f1 "$d/rc.txt" 2>/dev/null)
  so=$(ls "$d"/src/scorch_ops*.so 2>/dev/null | head -1)
  if [ "$rc" != "0" ] || [ -z "$so" ]; then
    echo "  $tag  BUILD FAILED rc=$rc"; tail -5 "$d/build.log" 2>/dev/null | sed 's/^/      /'
    continue
  fi
  echo "  $tag  rc=0  $(stat -c %s "$so") bytes  so_sha $(sha256sum "$so" | cut -c1-16)"
done

# THE A/A PAIR MUST BE BYTE-IDENTICAL IN .text, and strm/nopf must NOT be. A floor built from a
# pair that silently differs is not a floor, and an arm that silently matches ship is not an arm --
# the define could have been dropped and every number would still look plausible.
#
# ASSERTED, NOT PRINTED. A first version emitted four digests and a byte count and left the script's
# exit status independent of every one of them, so "the floor pair is sound" was a human reading a
# printout. Its own comment named the failure it did not check for. Four digests, four assertions,
# and a missing object is a REFUSAL rather than a `continue`.
echo
echo "EMISSION CHECK (.text digests; the floor pair must AGREE, the arms must DIFFER)"
fail=0
declare -A TSHA
for tag in ship shpb strm nopf; do
  so=$(ls "$ROOT/$tag"/src/scorch_ops*.so 2>/dev/null | head -1)
  if [ -z "$so" ]; then
    echo "  $tag  REFUSE: no object, so its digest cannot be compared"; fail=1; continue
  fi
  objcopy -O binary --only-section=.text "$so" "$ROOT/$tag/text.bin" 2>/dev/null
  TSHA[$tag]=$(sha256sum "$ROOT/$tag/text.bin" | cut -d' ' -f1)
  echo "  $tag  text_sha $(echo "${TSHA[$tag]}" | cut -c1-24)  $(stat -c %s "$ROOT/$tag/text.bin") bytes"
done
[ "$fail" = "0" ] || { echo "REFUSE: an object is missing; no comparison below is meaningful"; exit 1; }

# The floor pair: same configuration, equal-length paths, so .text must be identical.
if [ "${TSHA[ship]}" != "${TSHA[shpb]}" ]; then
  echo "REFUSE: ship and shpb differ in .text. They are the same configuration at equal-length"
  echo "        paths, so a difference means the floor pair is not an A/A and the floor it"
  echo "        produces is not the floor an arm must clear."
  fail=1
fi
# The arms: each must differ from the floor pair AND from each other. An arm matching ship is a
# dropped define, which no timing can distinguish from a true null result.
for tag in strm nopf; do
  if [ "${TSHA[$tag]}" = "${TSHA[ship]}" ]; then
    echo "REFUSE: $tag has ship's .text, so its define did not take. Every number it produces"
    echo "        would look like a plausible null result."
    fail=1
  fi
done
if [ "${TSHA[strm]}" = "${TSHA[nopf]}" ]; then
  echo "REFUSE: strm and nopf have identical .text; they are different configurations"
  fail=1
fi
[ "$fail" = "0" ] || exit 1
echo "  ASSERTED: ship == shpb, and strm, nopf each differ from ship and from each other."

# THE HOOKS AXIS, WHICH THE DIGESTS ABOVE CANNOT SEE. Every .text assertion above passes whether
# or not SCORCH_TUNE_HOOKS was compiled in, because it is orthogonal to the PFSTREAM/PFD defines
# these arms differ by. The built object publishes the flag precisely so a harness can fail closed
# instead of reporting a null it manufactured (ops.cpp:595-600) -- and until now no screen in this
# directory had ever queried it. Asked of each object, not of the headers or the environment.
echo
echo "HOOKS CHECK (spmm_tune_hooks must be False on all four; a hooks build taxes every call)"
for tag in ship shpb strm nopf; do
  # Bind the answer to the object under $ROOT/$tag. The emission check above already refuses on
  # a missing object and line 36 rm -rf's the directory before staging, so an installed copy
  # winning is mostly closed by construction -- but "mostly" is not a check, and the sibling
  # instrument pfs_splitgate.py does bind it. Printing the module path and requiring the prefix
  # costs one expression.
  read -r v mod <<<"$(PYTHONPATH="$ROOT/$tag/src" python -c \
        'import torch, scorch_ops as S; print(getattr(S, "spmm_tune_hooks", "ABSENT"), S.__file__)' \
        2>/dev/null)"
  echo "  $tag  spmm_tune_hooks=${v:-IMPORT_FAILED}  from ${mod:-<no module>}"
  case "$mod" in
    "$ROOT/$tag"/*) ;;
    *) echo "REFUSE: $tag reported from '${mod:-<no module>}', not from $ROOT/$tag -- an"
       echo "        installed copy answered, so the flag is not this object's."
       fail=1; continue ;;
  esac
  case "$v" in
    False) ;;
    True)  echo "REFUSE: $tag is a HOOKS build. Its per-call getenv tax makes every ours/mkl"
           echo "        number in this screen unreadable, not just the row-split question."
           fail=1 ;;
    *)     echo "REFUSE: $tag did not report the flag (got '${v:-IMPORT_FAILED}'), so the check"
           echo "        is vacuous rather than passing."
           fail=1 ;;
  esac
done
[ "$fail" = "0" ] || exit 1
echo "  ASSERTED: all four objects report spmm_tune_hooks=False."
