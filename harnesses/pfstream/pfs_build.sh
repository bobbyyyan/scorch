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
STAGE=/matx/u/bobbyy/pfsstage
ROOT=/scr/bobbyy/pfs
source /scr/bobbyy/venv/bin/activate || exit 1
mkdir -p "$ROOT" || exit 1
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

# Build all four concurrently; each is one translation unit, so they do not contend on make jobs.
for tag in ship shpb strm nopf; do
  (
    cd "$ROOT/$tag" || exit 1
    SCORCH_BUILD_DEFINES="${DEFS[$tag]}" \
      python -c "import setuptools; setuptools.setup()" build_ext --inplace \
      > build.log 2>&1
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
echo
echo "EMISSION CHECK (.text digests; the floor pair must agree, the arms must differ)"
for tag in ship shpb strm nopf; do
  so=$(ls "$ROOT/$tag"/src/scorch_ops*.so 2>/dev/null | head -1)
  [ -n "$so" ] || { echo "  $tag  (no object)"; continue; }
  objcopy -O binary --only-section=.text "$so" "$ROOT/$tag/text.bin" 2>/dev/null
  echo "  $tag  text_sha $(sha256sum "$ROOT/$tag/text.bin" | cut -c1-24)  $(stat -c %s "$ROOT/$tag/text.bin") bytes"
done
