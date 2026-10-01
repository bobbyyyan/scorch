#!/bin/bash
# Seal the chain210 ledger. Runs AFTER collect_c210.sh has staged the 12 CSVs and 3 logs and
# after an_c210.py has been run, and BEFORE anything else touches the directory.
#
# It adds the four artefacts the collector deliberately does not carry, because the collector's
# manifest is pinned at exactly 15 entries and was committed before the data existed:
#   ERRATA.md                   - from harnesses/ERRATA_c210.md, written pre-data
#   portbound_boundaries.txt    - chain210's own /proc/stat snapshots, so the occupancy figures can
#                                 be paired with THIS chain's CSVs rather than another chain's
#   REPS_WITNESS_c210.txt       - the --reps attestation, captured from the live twelfth pass. No
#                                 sealed ledger records this parameter and it determines the
#                                 interpretation of every timed-share figure.
#   DISJOINTNESS.txt            - errata 8d: both trees' realpaths and both twelve-CSV digest sets
#                                 side by side, so a later hand re-run cannot repeat the identity
#                                 mistake even without having read erratum 8.
# PROVENANCE.md is written by hand afterwards, because it quotes results.
#
# WHAT IT REFUSES ON: a destination that is not a collected chain210 ledger; a missing input; a
# manifest that is not SHA-256 on every line; any verification short of OK==entries and FAILED==0.
set -u
DEST=${1:?usage: seal_c210.sh <ledger dir>}
H=/Users/bobby/scorch-beat-mkl/harnesses
C207=$HOME/.cache/scorch-codex/spmm-beat-mkl/chain207-dlmc-redwood
SNAP=${2:-/private/tmp/claude-501/-Users-bobby-scorch/f10fe439-c6ee-4604-8a84-403f68c79be7/scratchpad/occ/portbound_boundaries.txt}

[ -d "$DEST" ] || { echo "REFUSE: $DEST is not a directory"; exit 1; }
n=$(ls -1 "$DEST"/dl_*_float32_p*.csv 2>/dev/null | wc -l | tr -d ' ')
[ "$n" -eq 12 ] || { echo "REFUSE: $DEST holds $n object-pass CSVs, not 12 -- run collect_c210.sh first"; exit 1; }
[ -f "$DEST/c210_stage.log" ] || { echo "REFUSE: no c210_stage.log in $DEST"; exit 1; }
[ -f "$DEST/SHA256SUMS.producing" ] || { echo "REFUSE: no SHA256SUMS.producing -- this is not a collected ledger"; exit 1; }
# A SEALED LEDGER OF ANOTHER CHAIN MUST NEVER REACH THE `rm -f SHA256SUMS*` BELOW. The three
# checks above happen to refuse chain207 (it has no c210_stage.log), but that is luck, not design:
# pointed at a sealed tree this script would delete its manifest and rewrite it over the wrong file
# set. Ledger artefacts are sealed evidence and are read-only for a chain that is not this one.
other=$(ls -1 "$DEST" 2>/dev/null | grep -E '^c[0-9]+_(stage|run)\.log$' | grep -v '^c210_' | tr '\n' ' ')
[ -z "$other" ] || { echo "REFUSE: $DEST carries another chain's logs ($other) -- it is not chain210's ledger and its seal is not mine to touch"; exit 1; }
for f in "$H/ERRATA_c210.md" "$H/REPS_WITNESS_c210.txt" "$SNAP"; do
  [ -f "$f" ] || { echo "REFUSE: input missing: $f"; exit 1; }
done

cp -p "$H/ERRATA_c210.md" "$DEST/ERRATA.md"
cp -p "$H/REPS_WITNESS_c210.txt" "$DEST/REPS_WITNESS.txt"
cp -p "$SNAP" "$DEST/portbound_boundaries.txt"

# COMMIT BINDING. ERRATA.md and REPS_WITNESS.txt are copies from a live worktree, renamed, so the
# sealed digest pins the copy and not a revision -- the worktree file can move afterwards with
# nothing recording what was sealed. Record the revision and whether it was clean.
{
  echo "=== SOURCE BINDING for the artefacts copied in from the worktree"
  echo "worktree: /Users/bobby/scorch-beat-mkl"
  echo "HEAD:     $(git -C /Users/bobby/scorch-beat-mkl rev-parse HEAD)"
  echo "branch:   $(git -C /Users/bobby/scorch-beat-mkl rev-parse --abbrev-ref HEAD)"
  echo "ERRATA.md           <- harnesses/ERRATA_c210.md      sha256 $(shasum -a 256 "$H/ERRATA_c210.md" | cut -d' ' -f1)"
  echo "REPS_WITNESS.txt    <- harnesses/REPS_WITNESS_c210.txt sha256 $(shasum -a 256 "$H/REPS_WITNESS_c210.txt" | cut -d' ' -f1)"
  echo "an_c210.py          sha256 $(shasum -a 256 "$DEST/an_c210.py" | cut -d' ' -f1)"
  d=$(git -C /Users/bobby/scorch-beat-mkl status --porcelain -- harnesses/ERRATA_c210.md harnesses/REPS_WITNESS_c210.txt harnesses/an_c210.py)
  if [ -z "$d" ]; then echo "worktree state: all three tracked files CLEAN at the HEAD above"
  else echo "worktree state: UNCOMMITTED changes present, so HEAD does not reproduce these bytes:"; echo "$d" | sed 's/^/  /'; fi
} > "$DEST/SOURCE_BINDING.txt"
echo "OK  source binding recorded: HEAD $(git -C /Users/bobby/scorch-beat-mkl rev-parse --short HEAD)"

# ---- FIX: the baseline must EXIST and be POPULATED before any comparison. Without this,
# `cd "$C207"` fails, `b` is the empty string, `comm -12` finds nothing and the verdict reads
# DISJOINT -- so a moved, renamed, pruned or not-yet-populated baseline prints the registered
# headline. My own note says a copy check can be vacuous and to size-filter first; I planted a
# duplicate (which the check catches) and left the empty operand open one level up.
[ -d "$C207" ] || { echo "REFUSE: baseline $C207 is not a directory -- a comparison against nothing reads as DISJOINT"; exit 1; }
NB=$( (cd "$C207" && ls -1 dl_*_float32_p*.csv 2>/dev/null) | wc -l | tr -d ' ')
[ "$NB" -eq 12 ] || { echo "REFUSE: baseline holds $NB object-pass CSVs, not 12 -- an undersized baseline weakens the disjointness claim silently"; exit 1; }

# ---- errata 8d: attest disjointness in the artefact, not at read time.
{
  echo "=== DISJOINTNESS ATTESTATION (errata item 8d)"
  echo "WHY. an_c210.py:34 hardcodes chain207 as its baseline and NOTHING in it refuses argv[1]"
  echo "BEING that baseline. Pointed at the baseline's own CSVs it passes every guard and prints"
  echo "Jaccard 1.0000 with 'survivors 10 of 10' -- which is chain207's registered headline, so the"
  echo "failure mode and the hoped-for result are the same output. Per-input guards cannot reach a"
  echo "defect in the RELATION between inputs. realpath inequality is not sufficient (it misses"
  echo "symlinks and copies); digest set-disjointness is."
  echo
  echo "chain210 realpath: $(cd "$DEST" && pwd -P)"
  echo "chain207 realpath: $(cd "$C207" && pwd -P)"
  echo
  echo "--- chain210's twelve object-pass digests"
  (cd "$DEST" && shasum -a 256 dl_*_float32_p*.csv)
  echo "--- chain207's twelve object-pass digests"
  (cd "$C207" && shasum -a 256 dl_*_float32_p*.csv)
  echo
  a=$(cd "$DEST" && shasum -a 256 dl_*_float32_p*.csv | awk '{print $1}' | sort)
  b=$(cd "$C207" && shasum -a 256 dl_*_float32_p*.csv | awk '{print $1}' | sort)
  shared=$(comm -12 <(echo "$a") <(echo "$b") | wc -l | tr -d ' ')
  echo "digests in common: $shared of 12"
  if [ "$shared" -eq 0 ]; then
    echo "VERDICT: DISJOINT -- these are two distinct runs, not one run read twice."
  else
    echo "VERDICT: REFUSE -- $shared digest(s) shared; the two trees are not independent."
  fi
} > "$DEST/DISJOINTNESS.txt"
grep -q "^VERDICT: DISJOINT" "$DEST/DISJOINTNESS.txt" || { echo "REFUSE: digest sets are not disjoint (see $DEST/DISJOINTNESS.txt)"; exit 1; }
# The guard reads its OWN artefact, because the tell for a vacuous pass is already written there:
# an empty baseline realpath and zero digest lines under the baseline heading.
grep -qE '^chain207 realpath: /' "$DEST/DISJOINTNESS.txt" || { echo "REFUSE: DISJOINTNESS.txt records no baseline realpath -- the comparison had an empty operand"; exit 1; }
for side in chain210 chain207; do
  nd=$(awk "/^--- $side's twelve object-pass digests\$/{f=1;next} /^--- /{f=0} f && /^[0-9a-f]{64} /{n++} END{print n+0}" "$DEST/DISJOINTNESS.txt")
  [ "$nd" -eq 12 ] || { echo "REFUSE: DISJOINTNESS.txt lists $nd digests for $side, not 12"; exit 1; }
done
echo "OK  disjointness attested: 12 vs 12 digests listed, 0 shared with chain207"

# ---- the manifest. Identify a real manifest by its digest FIELD WIDTH, not its name: 9 of 25
# ledger manifests in this project are CRC-32, against which `shasum -c` prints zero OK and zero
# FAILED -- a vacuous pass. 64 hex chars is the check.
cd "$DEST" || exit 1
# CONTAINMENT. shasum follows symlinks, so a sealed name whose bytes live outside the directory
# verifies clean forever while the real file is unsealed and mutable. attest.py has contained()
# for exactly this; the seal is where it matters most.
nl=$(find . -type l | wc -l | tr -d ' ')
[ "$nl" -eq 0 ] || { echo "REFUSE: $nl symlink(s) in $DEST -- a sealed name must be a sealed file:"; find . -type l | sed 's/^/  /'; exit 1; }
# DOTFILES. The enumeration below is `ls -1`, which omits them, so the one class of file this
# sealer cannot see is an rsync/scp temporary -- `.name.XXXXXX` -- and that is precisely the file
# whose presence means the transfer did not finish and the directory must NOT be sealed yet. The
# blindness is in DETECTION, not in what gets sealed: a dotfile should block the seal, not join
# the manifest, so this refuses rather than widening the enumeration (which would also change
# every manifest line's format from `name` to `./name`).
dots=$(find . -maxdepth 1 -type f -name '.*' | sed 's|^\./||' | tr '\n' ' ')
[ -z "$dots" ] || { echo "REFUSE: dotfile(s) present, which the ls -1 enumeration below cannot see: $dots"; echo "  An rsync temporary means the transfer is incomplete. Resolve it before sealing."; exit 1; }
# The refusal above can only fire if find sees this directory at all; a wrong CWD would make it
# vacuous, so assert the non-dot count is the one already verified above.
nf=$(find . -maxdepth 1 -type f | wc -l | tr -d ' ')
[ "$nf" -ge "$n" ] || { echo "REFUSE: find sees $nf files but $n object-pass CSVs were counted -- the dotfile check ran somewhere else"; exit 1; }
rm -f SHA256SUMS.sha256
# NUL-delimited into xargs: never let the shell word-split a file list, and zsh would not
# split it at all, so `shasum $(echo "$SET")` is wrong in two different ways depending on shell.
# EXACT name, not a prefix. `grep -v '^SHA256SUMS'` also excluded SHA256SUMS.producing --
# the producing-host manifest, which cannot attest itself and whose ONLY binding is this seal.
# Left unsealed it could be edited afterwards undetected, breaking the chain of custody from
# redwood through the tarball to here, which is the single thing it exists for.
ls -1 | grep -vx 'SHA256SUMS.sha256' | sort | tr '\n' '\0' | xargs -0 shasum -a 256 > SHA256SUMS.sha256 \
  || { echo "REFUSE: could not write the manifest"; exit 1; }
awk '{ if (length($1) != 64) { print "  line " NR " digest field is " length($1) " chars, not 64"; bad=1 } } END { exit bad+0 }' SHA256SUMS.sha256 \
  || { echo "REFUSE: manifest is not SHA-256 on every line"; exit 1; }
NE=$(wc -l < SHA256SUMS.sha256 | tr -d ' ')
V=$(shasum -a 256 -c SHA256SUMS.sha256 2>&1)
OK=$(echo "$V" | grep -c ': OK$'); FA=$(echo "$V" | grep -c ': FAILED')
[ "$FA" -eq 0 ] && [ "$OK" -eq "$NE" ] || { echo "REFUSE: verification short -- $OK OK, $FA FAILED, $NE entries"; exit 1; }
echo "OK  manifest SHA256SUMS.sha256: $NE entries, $OK OK, $FA FAILED"
echo "    manifest digest: $(shasum -a 256 SHA256SUMS.sha256 | cut -d' ' -f1)"
echo
echo "SEALED: $DEST"
ls -1 | sed 's/^/  /'
echo
echo "REMAINING BY HAND: write PROVENANCE.md (it quotes results), then re-run this script so the"
echo "manifest covers it. Seal history must be recorded in PROVENANCE.md so no version is silently"
echo "replaced."
