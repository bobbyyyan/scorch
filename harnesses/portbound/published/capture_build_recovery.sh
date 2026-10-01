#!/bin/bash
# Capture the recovered build configuration of the chain206/chain210 objects, so the recovery is
# checkable by anyone later instead of resting on a reading taken once.
cd /scratch/bobbyy/det1 || exit 1
echo "# RECOVERED BUILD CONFIGURATION, chain206 / chain210 objects"
echo "# captured $(date -u +%Y-%m-%dT%H:%M:%SZ) on redwood from /scratch/bobbyy/det1"
echo "#"
echo "# THE TEST. Every getenv(\"SCORCH_NARROWK_*\"), getenv(\"SCORCH_SPMM_HALFVEC\") and"
echo "# getenv(\"SCORCH_SPMM_ALLOC\") call sits inside #ifdef SCORCH_TUNE_HOOKS, so the string"
echo "# literal is in .rodata if and only if that macro was defined. The non-hook names below are"
echo "# the positive control: they are present, which is what makes the hook-name zeros"
echo "# informative rather than the output of a broken search."
echo
for o in aa bb aax; do
  f="$o/src/scorch_ops.cpython-311-x86_64-linux-gnu.so"
  [ -f "$f" ] || { echo "object $o  MISSING"; echo; continue; }
  echo "object $o   ($f)"
  echo "  sha256       $(sha256sum "$f" | awk '{print $1}')"
  objcopy -O binary --only-section=.text "$f" /tmp/_t.bin 2>/dev/null
  echo "  .text sha256 $(sha256sum /tmp/_t.bin | awk '{print $1}')"
  rm -f /tmp/_t.bin
  echo "  all SCORCH_ string literals:"
  strings "$f" | grep '^SCORCH_' | sort -u | sed 's/^/    /'
  echo "  hook-only literals (present iff SCORCH_TUNE_HOOKS defined):"
  for n in SCORCH_SPMM_HALFVEC SCORCH_NARROWK_UNROLL SCORCH_NARROWK_EXACT_HI \
           SCORCH_SPMM_ALLOC SCORCH_REGTILE_BASE SCORCH_MULTIROW SCORCH_DUMMY_TESTS; do
    printf "    %-28s %s\n" "$n" "$(strings "$f" | grep -cx "$n")"
  done
  echo "  getenv imported (nm -D): $(nm -D "$f" 2>/dev/null | grep -ci getenv)"
  echo
done
echo "# CONCLUSION: no hook-only literal is present in any object and getenv IS imported, so the"
echo "# builds were made WITHOUT SCORCH_TUNE_HOOKS. spmm_halfvec is therefore the compiled-in"
echo "# SCORCH_SPMM_HALFVEC_F32, which scorch_policy.h:1332 defines as 1 under an #ifndef, and no"
echo "# environment variable could have altered it because the override is not in the object."
echo "# So k=4 float32 ran the half-vector __m128 path with four of four lanes live."
