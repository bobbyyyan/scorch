#!/usr/bin/env bash
# Prove seal_ledger.sh's two properties on a synthetic tree, both directions.
#
# The exclusions have failed four times (see seal_ledger.sh's header), and every
# fix was checked by eye against the one tree that had just failed.  This checks
# them against a tree built to contain one instance of every shape that has ever
# slipped through, at a depth none of them were originally written for -- and
# then checks that the read-back can still FAIL, because a seal that cannot fail
# was the whole problem.
#
#   bash statics/seal_ledger_selftest.sh
set -euo pipefail

SEAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/seal_ledger.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# ---------------------------------------------------------------------------
# 1. Everything build-shaped is excluded, at any depth, and the evidence is not.
# ---------------------------------------------------------------------------
LEDGER="$WORK/ledger"
mkdir -p "$LEDGER"/{receipts/proofs,receipts/scratch/mypy,worktrees/w/src}
mkdir -p "$LEDGER"/{a/b/c/scratch,a/b/torch_cand,deep/x/y/tmp_7}
mkdir -p "$LEDGER"/deep/pytest-of-someone/pytest-3 "$LEDGER"/nested/.mypy_cache/3.11
echo evidence > "$LEDGER/CLOSEOUT.md"
echo evidence > "$LEDGER/receipts/proofs/report.json"
: > "$LEDGER/receipts/scratch/mypy/cache.data.json"   # section 67.10's 158 MB
: > "$LEDGER/a/b/c/scratch/kernel.so"                 # scratch below the root
: > "$LEDGER/a/b/torch_cand/build.ninja"              # section 66.9's suffix
: > "$LEDGER/deep/x/y/tmp_7/thing.o"                  # a numbered partition tmp
: > "$LEDGER/deep/pytest-of-someone/pytest-3/junk.dylib"
: > "$LEDGER/worktrees/w/src/mod.py"                  # git-backed scratch
: > "$LEDGER/nested/.mypy_cache/3.11/x.data.json"

bash "$SEAL" "$LEDGER" > "$WORK/seal.log"
sealed="$(awk '{print $2}' "$LEDGER/SHA256SUMS" | LC_ALL=C sort | tr '\n' ' ')"
expected="./CLOSEOUT.md ./receipts/proofs/report.json "
if [ "$sealed" != "$expected" ]; then
  echo "SELFTEST FAILED: sealed [$sealed], expected [$expected]"
  exit 1
fi
grep -q 'read back: 0 compiled or cache paths' "$WORK/seal.log"
echo "ok: 7 build-shaped paths excluded at four different depths, 2 evidence files sealed"

# ---------------------------------------------------------------------------
# 2. The read-back can still fail.  A compiled artifact sitting where no name
#    exclusion applies must be caught rather than sealed.
# ---------------------------------------------------------------------------
LEAKY="$WORK/leaky"
mkdir -p "$LEAKY/receipts"
echo evidence > "$LEAKY/CLOSEOUT.md"
: > "$LEAKY/receipts/kernel.so"

set +e
bash "$SEAL" "$LEAKY" > "$WORK/leaky.log" 2>&1
status=$?
set -e
if [ "$status" -ne 2 ]; then
  echo "SELFTEST FAILED: a leaked .so did not reject the seal (exit $status)"
  cat "$WORK/leaky.log"
  exit 1
fi
grep -q 'SEAL REJECTED' "$WORK/leaky.log"
echo "ok: a compiled artifact outside every name exclusion rejects the seal"
echo "seal_ledger.sh selftest PASSED"
