#!/usr/bin/env bash
# Verify a ledger directory whose live PROVENANCE.md has been revised, without touching bytes.
#
# THE PROBLEM THIS SOLVES. A revision scheme that keeps the current prose at PROVENANCE.md and
# copies the old one aside leaves every superseded manifest permanently unverifiable in place:
# the manifest records a digest under the name PROVENANCE.md, that name now resolves to the
# CURRENT revision, and `shasum -c` reports `PROVENANCE.md: FAILED` plus rc=1 forever. The
# directory is intact; the check reports the signature of corruption. Measured on
# chain212-dlmc-f32-p1: three retained manifests, each rc=1 with exactly one FAILED line.
#
# Two fixes were available. Rewriting each retained manifest so its filename column names the
# renamed copy makes it verify rc=0 in place -- at the cost of no longer being byte-identical to
# what was sealed, which is the only property a manifest has. This script takes the other: the
# sealed bytes are never modified, and the name is remapped at verification time instead.
#
# Harness convention: the ledger directory is "$1". Never hardcoded.
set -u

DIR=${1:-}
[ -n "$DIR" ] || { echo "usage: verify_ledger_revisions.sh <ledger-dir>"; exit 2; }
[ -d "$DIR" ] || { echo "REFUSE: $DIR is not a directory"; exit 2; }

# A retained manifest's revision is read from its own filename, and the prose copy it refers to
# is derived from that. An unrecognised manifest name is a refusal, not a skip: silently passing
# over a manifest is how a directory reports fewer problems than it has.
prose_for() {
  case "$1" in
    *.v1)             echo "PROVENANCE.v1.md" ;;
    *.v2)             echo "PROVENANCE.v2.md" ;;
    *.rev3.*)         echo "__LOST__" ;;
    *.rev4.*)         echo "PROVENANCE.rev4.md" ;;
    *.rev6*)          echo "PROVENANCE.rev6.md" ;;
    *) echo "" ;;
  esac
}

fail=0
live=0
checked=0
for m in "$DIR"/SHA256SUMS*; do
  b=$(basename "$m")
  case "$b" in
    SHA256SUMS.sha256) continue ;;
    SHA256SUMS)
      out=$(cd "$DIR" && shasum -a 256 -c "$b" 2>&1); rc=$?
      n_ok=$(printf '%s\n' "$out" | grep -c ': OK$')
      printf "  %-34s CURRENT   rc=%d  %d OK\n" "$b" "$rc" "$n_ok"
      [ "$rc" -eq 0 ] || { echo "    REFUSE: the current manifest must verify as-is"; fail=1; }
      live=1; checked=$((checked+1))
      continue ;;
  esac
  p=$(prose_for "$b")
  if [ -z "$p" ]; then
    echo "  REFUSE: $b has no known revision mapping; add one rather than skipping it"
    fail=1; continue
  fi
  checked=$((checked+1))
  if [ "$p" = "__LOST__" ]; then
    # The prose this manifest sealed does not survive. The expected result is one FAILED line on
    # PROVENANCE.md and OK on everything else -- a TRUE POSITIVE, and the reason a transcribed
    # manifest is worth more than a reconstructed one: a reconstruction recomputed from surviving
    # files cannot report a loss, because it agrees with whatever is there now.
    out=$(cd "$DIR" && shasum -a 256 -c "$b" 2>&1)
    n_ok=$(printf '%s\n' "$out" | grep -c ': OK$')
    n_bad=$(printf '%s\n' "$out" | grep -c ': FAILED$')
    which=$(printf '%s\n' "$out" | grep ': FAILED$' | sed 's/: FAILED//' | tr '\n' ' ')
    printf "  %-34s PROSE LOST  %d OK  %d FAILED (%s)\n" "$b" "$n_ok" "$n_bad" "${which% }"
    if [ "$n_bad" -ne 1 ] || [ "${which% }" != "PROVENANCE.md" ]; then
      echo "    REFUSE: expected exactly one FAILED line, on PROVENANCE.md"; fail=1
    fi
    continue
  fi
  [ -f "$DIR/$p" ] || { echo "  REFUSE: $b maps to $p, which is absent"; fail=1; continue; }
  # Remap the sealed manifest's PROVENANCE.md line onto the renamed copy. The sealed file is read
  # only; the remap exists for the duration of one check.
  tmp=$(mktemp)
  sed "s|  PROVENANCE.md\$|  $p|" "$m" > "$tmp"
  n_remap=$(grep -c "  $p\$" "$tmp")
  if [ "$n_remap" -ne 1 ]; then
    echo "  REFUSE: $b remapped $n_remap lines onto $p, expected exactly 1"
    rm -f "$tmp"; fail=1; continue
  fi
  out=$(cd "$DIR" && shasum -a 256 -c "$tmp" 2>&1); rc=$?
  n_ok=$(printf '%s\n' "$out" | grep -c ': OK$')
  n_bad=$(printf '%s\n' "$out" | grep -c ': FAILED$')
  printf "  %-34s remapped  rc=%d  %d OK  %d FAILED\n" "$b" "$rc" "$n_ok" "$n_bad"
  [ "$rc" -eq 0 ] || { printf '%s\n' "$out" | grep ': FAILED$' | sed 's/^/    /'; fail=1; }
  rm -f "$tmp"
done

echo "  checked $checked manifests; live manifest present: $live"
[ "$live" -eq 1 ] || { echo "REFUSE: no current SHA256SUMS found, so nothing certifies the directory now"; fail=1; }
[ "$checked" -ge 2 ] || { echo "REFUSE: only $checked manifest(s) examined; this script exists for multi-revision directories"; fail=1; }
if [ "$fail" -ne 0 ]; then echo "VERIFY FAILED"; exit 1; fi
echo "VERIFY OK -- every retained manifest accounted for, no sealed bytes modified"
