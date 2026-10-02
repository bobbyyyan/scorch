#!/usr/bin/env bash
# Verify a ledger directory whose live PROVENANCE.md has been revised, without touching bytes.
#
# THE PROBLEM. A revision scheme that keeps the current prose at PROVENANCE.md and copies the old
# one aside leaves every superseded manifest permanently unverifiable in place: it records a
# digest under the name PROVENANCE.md, that name now resolves to the CURRENT revision, and
# `shasum -c` reports `PROVENANCE.md: FAILED` with rc=1 forever while the directory is intact.
# Rewriting each retained manifest to name the renamed copy would fix that at the cost of
# byte-identity with what was sealed, which is the only property a manifest has. So the sealed
# bytes are never modified here and the name is remapped at verification time.
#
# Harness convention: the ledger directory is "$1". Never hardcoded.
#
# FOUR DEFECTS THIS VERSION FIXES, ALL FOUND BY PLANTING AGAINST THE PREVIOUS ONE.
# (The header said THREE while enumerating four, in the file whose job is counting.)
#
# 1. It skipped SHA256SUMS.sha256 and therefore reported VERIFY OK, exit 0, on a directory whose
#    live prose AND live manifest had both been rewritten -- which is what a tamper does and what
#    an ordinary edit does. The only artefact that detects that is the seal-of-seal, and it was
#    the one thing not checked. Now checked first, and a failure is fatal.
# 2. The revision mapping was a hand-written case statement: it knew only the names thought of in
#    advance, and a branch was added for a prose file whose manifest did not exist, so it read as
#    coverage while covering nothing. The mapping is now DERIVED by extracting tags from the
#    filenames present and set-differencing them, which also makes a half-preserved revision --
#    prose without a manifest, or a manifest without prose -- visible rather than invisible.
# 3. `shasum` reports an absent file on stderr as "No such file or directory", not as a FAILED
#    line, so a missing file gave rc=1 with a FAILED count of 0 and no reason printed. Missing
#    lines are now counted and shown.
# 4. Nothing checked that the live manifest COVERS the directory. A retained manifest added after
#    the last reseal is named by neither the live manifest nor the seal-of-seal, so no digest
#    binds its bytes -- and a dropped DATA line in an unbound manifest passes every check here,
#    because a manifest's own "rc=0, N OK" says that every line it still contains resolves and
#    nothing about a line it no longer contains. Completeness is a property only something else
#    holding its digest can supply. Checked by set-differencing the directory against the live
#    manifest's filename column, not by comparing two counts.
#
# WHAT A PASS CANNOT MEAN. If prose, live manifest and seal-of-seal are rewritten together the
# directory is self-consistent and no check inside it can tell. That is inherent to a
# self-contained seal chain. The only defence is a digest recorded outside the directory before
# the replacement, so the success message says "internally consistent", not "unmodified".
set -u

DIR=${1:-}
[ -n "$DIR" ] || { echo "usage: verify_ledger_revisions.sh <ledger-dir>"; exit 2; }
[ -d "$DIR" ] || { echo "REFUSE: $DIR is not a directory"; exit 2; }
cd "$DIR" || exit 2

fail=0

# --- the seal-of-seal, first, because it is the only thing that can detect a rewritten live
# --- manifest. Its rc is captured directly; piping it into anything replaces the status.
if [ -f SHA256SUMS.sha256 ]; then
  out=$(shasum -a 256 -c SHA256SUMS.sha256 2>&1); rc=$?
  if [ "$rc" -ne 0 ] || printf '%s\n' "$out" | grep -q ': FAILED$'; then
    echo "  SHA256SUMS.sha256                  SEAL-OF-SEAL  rc=$rc  FAILED"
    printf '%s\n' "$out" | sed 's/^/    /'
    echo "    REFUSE: the live manifest does not match its own seal, so the current revision's"
    echo "            prose and digests cannot be trusted. Every check below is downstream of this."
    fail=1
  else
    echo "  SHA256SUMS.sha256                  SEAL-OF-SEAL  rc=0  OK"
  fi
else
  echo "  REFUSE: no SHA256SUMS.sha256; a rewritten live manifest would be undetectable"
  fail=1
fi

# --- derive the revision tags mechanically from the filenames present
prose_tags=$(ls PROVENANCE.*.md 2>/dev/null | sed 's/^PROVENANCE\.//; s/\.md$//' | sort -u)
man_tags=$(ls SHA256SUMS.* 2>/dev/null | grep -v '\.sha256$' \
           | sed 's/^SHA256SUMS\.//; s/\..*$//' | sort -u)
echo "  revision tags -- prose: $(echo $prose_tags | tr '\n' ' ')| manifests: $(echo $man_tags | tr '\n' ' ')"
for t in $prose_tags; do
  echo "$man_tags" | grep -qx "$t" || { echo "  REFUSE: revision '$t' has prose but no retained manifest"; fail=1; }
done
for t in $man_tags; do
  echo "$prose_tags" | grep -qx "$t" || echo "  NOTE: revision '$t' has a manifest but no prose (a recorded loss)"
done

checked=0
n_recon=0
n_evid=0
n_unknown=0
for m in SHA256SUMS.*; do
  case "$m" in *.sha256) continue ;; esac
  # THE SUFFIX IS KEPT, NOT STRIPPED. `s/\..*$//` discarded `.RECONSTRUCTED` before any check
  # saw it, so a reconstruction's OK lines were printed in the same format as an attestation's.
  # They are not the same thing: a reconstruction's digests were recomputed from the surviving
  # files, so they agree with whatever is present and CANNOT report loss. On chain212 that is 9
  # of the OK lines guaranteed by construction, reported indistinguishably from 13 that are
  # evidence.
  full=$(echo "$m" | sed 's/^SHA256SUMS\.//')
  t=$(echo "$full" | sed 's/\..*$//')
  sfx=$(echo "$full" | sed 's/^[^.]*//; s/^\.//')
  case "$sfx" in
    RECONSTRUCTED) cls="RECONSTRUCTED -- its OK lines are guaranteed, NOT evidence"
                   n_recon=$((n_recon+1)) ;;
    TRANSCRIBED)   cls="TRANSCRIBED -- can report loss, so its OK lines are evidence"
                   n_evid=$((n_evid+1)) ;;
    "")            cls="NO SUFFIX -- class UNDETERMINED by this tool, see below"
                   n_unknown=$((n_unknown+1)) ;;
    *)             cls="SUFFIX '$sfx' UNKNOWN to this tool -- treated as undetermined"
                   n_unknown=$((n_unknown+1)) ;;
  esac
  echo "  provenance class of $m: $cls"
  checked=$((checked+1))
  p="PROVENANCE.$t.md"
  if [ ! -f "$p" ]; then
    # Prose recorded as lost. Expect exactly one FAILED line, on PROVENANCE.md -- a TRUE
    # POSITIVE. A transcription can report this; a reconstruction recomputed from surviving
    # files cannot, which is why the two are named differently.
    out=$(shasum -a 256 -c "$m" 2>&1)
    n_ok=$(printf '%s\n' "$out" | grep -c ': OK$')
    n_bad=$(printf '%s\n' "$out" | grep -c ': FAILED$')
    n_miss=$(printf '%s\n' "$out" | grep -c 'No such file')
    which=$(printf '%s\n' "$out" | grep ': FAILED$' | sed 's/: FAILED//' | tr '\n' ' ')
    printf "  %-34s PROSE LOST  %d OK  %d FAILED  %d MISSING (%s)\n" "$m" "$n_ok" "$n_bad" "$n_miss" "${which% }"
    if [ "$n_bad" -ne 1 ] || [ "${which% }" != "PROVENANCE.md" ] || [ "$n_miss" -ne 0 ]; then
      echo "    REFUSE: expected exactly one FAILED line on PROVENANCE.md and no missing files"
      printf '%s\n' "$out" | grep -E ': FAILED$|No such file' | sed 's/^/      /'
      fail=1
    fi
    continue
  fi
  tmp=$(mktemp)
  sed "s|  PROVENANCE.md\$|  $p|" "$m" > "$tmp"
  n_remap=$(grep -c "  $p\$" "$tmp")
  if [ "$n_remap" -ne 1 ]; then
    echo "  REFUSE: $m remapped $n_remap lines onto $p, expected exactly 1"
    rm -f "$tmp"; fail=1; continue
  fi
  out=$(shasum -a 256 -c "$tmp" 2>&1); rc=$?
  n_ok=$(printf '%s\n' "$out" | grep -c ': OK$')
  n_bad=$(printf '%s\n' "$out" | grep -c ': FAILED$')
  n_miss=$(printf '%s\n' "$out" | grep -c 'No such file')
  printf "  %-34s remapped  rc=%d  %d OK  %d FAILED  %d MISSING\n" "$m" "$rc" "$n_ok" "$n_bad" "$n_miss"
  if [ "$rc" -ne 0 ]; then
    printf '%s\n' "$out" | grep -E ': FAILED$|No such file' | sed 's/^/    /'
    fail=1
  fi
  rm -f "$tmp"
done

# --- does the live manifest COVER the directory? Derived by set difference, because comparing a
# --- file count to an entry count only says the sizes agree and names no uncovered file.
if [ -f SHA256SUMS ]; then
  present=$(ls -A | grep -vx -e SHA256SUMS -e SHA256SUMS.sha256 | sort)
  named=$(sed 's/^[0-9a-f]*  //' SHA256SUMS | sort)
  uncovered=$(comm -23 <(printf '%s\n' "$present") <(printf '%s\n' "$named"))
  phantom=$(comm -13 <(printf '%s\n' "$present") <(printf '%s\n' "$named"))
  printf "  %-34s COVERAGE  %d files (excl. itself + its seal)  %d entries\n" "SHA256SUMS" \
         "$(printf '%s\n' "$present" | grep -c .)" "$(printf '%s\n' "$named" | grep -c .)"
  if [ -n "$uncovered" ]; then
    printf '%s\n' "$uncovered" | sed 's/^/    UNBOUND: /'
    echo "    REFUSE: no digest anywhere binds the bytes of the file(s) above. A dropped data line"
    echo "            in an unbound retained manifest passes every other check in this script."
    fail=1
  fi
  if [ -n "$phantom" ]; then
    printf '%s\n' "$phantom" | sed 's/^/    NAMED BUT ABSENT: /'
    fail=1
  fi
fi

if [ -f SHA256SUMS ]; then
  out=$(shasum -a 256 -c SHA256SUMS 2>&1); rc=$?
  printf "  %-34s CURRENT   rc=%d  %d OK\n" "SHA256SUMS" "$rc" "$(printf '%s\n' "$out" | grep -c ': OK$')"
  [ "$rc" -eq 0 ] || { printf '%s\n' "$out" | grep -E ': FAILED$|No such file' | sed 's/^/    /'; fail=1; }
else
  echo "  REFUSE: no current SHA256SUMS, so nothing certifies the directory now"; fail=1
fi

echo "  checked $checked retained manifests"
[ "$checked" -ge 2 ] || { echo "REFUSE: only $checked retained manifest(s); this script is for multi-revision directories"; fail=1; }
if [ "$fail" -ne 0 ]; then echo "VERIFY FAILED"; exit 1; fi
echo "VERIFY OK -- seal-of-seal, coverage, every retained manifest and the live manifest all check."
echo "  This is INTERNAL consistency. A simultaneous rewrite of prose + manifest + seal is"
echo "  self-consistent and undetectable from inside the directory; only a digest recorded"
echo "  elsewhere beforehand can rule that out. No sealed bytes were modified by this script."
echo
echo "  WHAT 'ALL CHECK' DOES AND DOES NOT MEAN, by provenance class:"
echo "    RECONSTRUCTED manifests: $n_recon. Their digests were recomputed from surviving files,"
echo "      so they agree with what is present BY CONSTRUCTION and cannot report loss. Their OK"
echo "      lines are not evidence of integrity and are excluded from any such claim."
echo "    TRANSCRIBED manifests:   $n_evid. These can report loss, so their OK lines are evidence."
echo "    UNDETERMINED (no suffix): $n_unknown. THIS TOOL CANNOT CLASSIFY THESE and does not"
echo "      guess. A missing suffix does NOT mean original: chain212's SHA256SUMS.v2 is itself"
echo "      reconstructed (PROVENANCE.md:125) and carries no suffix, but was digest-tested"
echo "      against a value read BEFORE the replacement, so it is reconstructed-AND-ANCHORED."
echo "      The real taxonomy is three-way -- original / reconstructed-and-anchored /"
echo "      reconstructed-and-unanchored -- and two of those three are spelled identically on"
echo "      disk, so no filename rule can separate them. The authoritative source is the"
echo "      directory's own prose table, which this script deliberately does not parse: a"
echo "      verifier that inferred provenance from prose would be trusting the artefact it is"
echo "      checking. Read it yourself before quoting any OK count as attestation."
