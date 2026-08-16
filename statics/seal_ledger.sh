#!/usr/bin/env bash
# Seal an evidence ledger: SHA256SUMS over every file it contains.
#
# Takes the ledger root as $1 and DEFAULTS TO ITS OWN LOCATION, which for this
# copy is the repository -- so a ledger seals with
#
#   bash statics/seal_ledger.sh ~/.cache/scorch-codex/<milestone>
#
# and never with a hardcoded path.  Hardcoding a ledger path has been caught
# three times on this branch.
#
# ---------------------------------------------------------------------------
# WHY THIS FILE IS IN THE REPOSITORY
# ---------------------------------------------------------------------------
# Because it has failed four times, each time in a DIFFERENT copy of itself, and
# each time by letting build output into a seal:
#
#   * review section 62.9 -- compiled kernels from the partitioned suite;
#   * section 63.8 -- the shared copy never got 62.9's exclusions;
#   * section 66.9 -- an extension cache named ``torch_cand`` slipped a pattern
#     anchored on ``torch``;
#   * section 67.10 -- 158 MB of mypy cache under ``receipts/scratch``, missed by
#     an exclusion anchored at the LEDGER ROOT.
#
# Every one of those was fixed in the copy that failed, and the next ledger
# started from a different copy.  Section 62.6 diagnosed exactly this class for
# the protected-file digests -- a thing that exists in a dozen copies has no
# owner -- and fixed it by putting the canonical copy in the tree, at
# ``statics/protected-hashes.txt``.  This is the same fix for the same reason.
#
# **Ledger copies are snapshots of this one from now on.**  The copies past
# ledgers were sealed with are deliberately NOT overwritten: a seal records what
# the script did at the time, and rewriting the script would invalidate it.
#
# ---------------------------------------------------------------------------
# WHY THE EXCLUSIONS ARE WRITTEN THE WAY THEY ARE
# ---------------------------------------------------------------------------
# The first three fixes each added another POSITION -- ``./receipts/*/torch/*``,
# then one more directory level, then a suffixed name.  The positions were never
# the pattern; the NAMES are.  ``find -path`` matches with fnmatch and no
# FNM_PATHNAME, so ``*`` matches ``/`` and ``*/name/*`` is depth-independent.
# Every exclusion below therefore anchors on a directory name at ANY depth.
#
# What is excluded and why: worktrees are git-backed scratch rather than
# evidence; per-partition ``TMPDIR``/``XDG_CACHE_HOME``/``TORCH_EXTENSIONS_DIR``
# trees are compiled output that no later session can or should reproduce (a
# five-partition suite run leaves about 680 MB of them, which once took a ledger
# from 344 files to 5,716); mypy's incremental cache is sqlite. The logs, the
# JUnit XML, the file lists, the exit codes, the receipts and ``provenance.json``
# are the evidence.
#
# ---------------------------------------------------------------------------
# AND IT READS ITS OWN OUTPUT BACK
# ---------------------------------------------------------------------------
# A seal that cannot fail was the whole problem, so this script refuses to
# complete if any compiled or cache path is in the SHA256SUMS it just wrote.
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT"

find . -type f \
     ! -name SHA256SUMS \
     ! -path './worktrees/*' \
     ! -path '*/worktrees/*' \
     ! -path '*/probe/*' \
     ! -path '*/scratch/*' \
     ! -path '*/tmp/*' \
     ! -path '*/tmp[-_0-9]*/*' \
     ! -path '*/mypy-cache*/*' \
     ! -path '*/.mypy_cache/*' \
     ! -path '*/xdg/*' \
     ! -path '*/torch/*' \
     ! -path '*/torch[-_]*/*' \
     ! -path '*/pytest-of-*/*' \
     ! -name '*.pyc' \
     ! -path '*/__pycache__/*' \
  | LC_ALL=C sort \
  | xargs shasum -a 256 > SHA256SUMS

leaked="$(grep -cE '\.(so|o|dylib|ninja)$|/(build\.ninja|\.ninja_log|\.ninja_deps)$|/mypy-cache|/\.mypy_cache|/pytest-of-' SHA256SUMS || true)"
if [ "${leaked:-0}" -ne 0 ]; then
  echo "SEAL REJECTED: $leaked compiled or cache paths are in SHA256SUMS"
  grep -nE '\.(so|o|dylib|ninja)$|/(build\.ninja|\.ninja_log|\.ninja_deps)$|/mypy-cache|/\.mypy_cache|/pytest-of-' SHA256SUMS | head -20
  exit 2
fi
echo "read back: 0 compiled or cache paths in the seal"

echo "sealed $(wc -l < SHA256SUMS) files under $ROOT"
shasum -a 256 SHA256SUMS
