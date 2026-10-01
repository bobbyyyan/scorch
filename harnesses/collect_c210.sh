#!/bin/bash
# chain210 -- collect the DLMC replication run off redwood and stage it, producing-host first.
#
# WHY A SCRIPT AND NOT A HAND SEQUENCE. Three staging decisions in this collection are degrees of
# freedom that an_c210.py's output depends on, and all three were exercised wrongly earlier:
#   1. c210_stage.log lives at /scratch/bobbyy/c210_stage.log, ONE LEVEL ABOVE the CSV directory,
#      because the stage script's stdout was redirected there at launch. an_c210.py reads it from
#      os.path.join(argv[1], "c210_stage.log") and EXITS without it. Two sessions already produced
#      two different analysis digests (0990bb8f vs d239c553) for no reason other than one
#      symlinking chain207's stage log and the other synthesising a three-line substitute.
#   2. Digests must be computed on the PRODUCING host before any transfer, or they attest the copy.
#   3. MTIMES.txt is computed on redwood and included in the producing manifest. chain207 added it
#      after its first seal and had to recompute the manifest twice.
#
# THE ATTESTATION CHAIN, END TO END, with the gap the first version left open:
#   SHA256SUMS.producing attests the 15 files but CANNOT ATTEST ITSELF. Its only binding is the
#   archive digest -- and the first version of this script held that digest in /tmp and deleted the
#   archive, so after collection nothing in the sealed tree attested the manifest. COLLECTION.log
#   now records the archive size and digest AS READ ON BOTH HOSTS, inside $DEST, written before the
#   archive is removed. The final seal (written at analysis time) then covers SHA256SUMS.producing
#   and COLLECTION.log, so the manifest is bound by a value that was checked on two hosts.
#   Found by the peer session; verified here by reading lines 86-89 of the previous version.
#
# WHAT IT REFUSES ON. No done marker; a stage log without 12 start and 12 done lines; a missing
# object-pass, NAMED; more than the 12 declared CSVs present; any object-pass whose line count is
# not 57301 (57300 rows + header = 3820 keys x 5 widths x 3 arms); an archive whose size or digest
# differs across the transfer; a manifest with ANY line whose digest field is not 64 hex chars; any
# file failing the producing-host manifest; a destination that already exists. It does NOT run the
# analyzer -- separate step, so a refusal here cannot be mistaken for a null result there.
set -u
DEST=${1:-$HOME/.cache/scorch-codex/spmm-beat-mkl/chain210-dlmc-replicate-redwood}

[ -e "$DEST" ] && { echo "REFUSE: $DEST already exists -- a seal is never overwritten in place"; exit 1; }

echo "=== 1. producing host: completeness, then digests"
mkdir -p "$DEST"
ssh -o BatchMode=yes redwood bash -s > "$DEST/COLLECTION.log" 2>&1 <<'REMOTE'
set -u
R=/scratch/bobbyy/c210
STAGE=/scratch/bobbyy/c210_stage.log
# Derived from $R, not hardcoded: a planted test of this block must not be able to write the
# production archive path. My first run of the R1-R6 battery did exactly that, and I cleaned it.
ARCH=$(dirname "$R")/$(basename "$R")_final.tgz
cd "$R" || { echo "RFAIL no $R"; exit 1; }
grep -q RW_C210_ALL_PASSES_DONE "$STAGE" || { echo "RFAIL no done marker in $STAGE -- run incomplete"; exit 1; }
# an_c210.py's TIMING block scores H1/H2/H3 only `if len(v) == 12`, where v comes from the stage
# log's start/done pairs. A short or truncated stage log therefore yields NO verdict and NO
# complaint -- a null result indistinguishable from a failure. The estimator is pre-registered and
# is not amended; the refusal goes here instead.
sd=$(grep -cE '^    --- p[1-6] (aa|bb) done ' "$STAGE")
ss=$(grep -cE '^    --- p[1-6] (aa|bb) start' "$STAGE")
[ "$sd" -eq 12 ] && [ "$ss" -eq 12 ] || { echo "RFAIL stage log has $ss start / $sd done lines, not 12 each -- the H1/H2/H3 block would skip silently"; exit 1; }

# ENUMERATE FROM THE DECLARING FACTORS, {aa,bb} x {p1..p6}, and resolve each to a file. The first
# version globbed dl_*_float32_p*.csv in five places and relied on a count of 12 to backstop it,
# which fails closed but refuses with "11 CSVs" instead of naming the absent object-pass -- and the
# commit message claimed enumeration that the code did not do. The count is kept, as the guard for
# the OTHER axis: an extra file matching the pattern.
CSVS=""; miss=""
for b in aa bb; do for p in 1 2 3 4 5 6; do
  f=dl_${b}_float32_p${p}.csv
  if [ -f "$f" ]; then CSVS="$CSVS $f"; else miss="$miss $f"; fi
done; done
[ -z "$miss" ] || { echo "RFAIL missing object-pass(es):$miss"; exit 1; }
nall=$(ls -1 dl_*.csv 2>/dev/null | wc -l)
[ "$nall" -eq 12 ] || { echo "RFAIL $nall files match dl_*.csv, but only the 12 declared object-passes may be sealed"; exit 1; }

bad=0
for f in $CSVS; do
  l=$(wc -l < "$f") || { echo "RFAIL wc failed on $f"; exit 1; }
  [ "$l" -eq 57301 ] || { echo "RFAIL $f has $l lines, expected 57301"; bad=1; }
done
[ "$bad" -eq 0 ] || exit 1

cp -p "$STAGE" "$R/c210_stage.log" || { echo "RFAIL could not copy $STAGE into $R"; exit 1; }
SEAL="$CSVS c210_run.log c210_stage.log MTIMES.txt"
{ echo "# chain210 per-file mtimes, recorded ON REDWOOD before transfer, so the pre-registration"
  echo "# boundary is checkable from the artefacts rather than from a commit clock."
  echo "# an_c210.py's final digest 9c42d96123ce3560 entered 23e9700 at 00:29:27 PDT 2026-10-01."
  echo
  for f in $CSVS c210_run.log c210_stage.log; do
    date -r "$f" "+%Y-%m-%d %H:%M:%S  $f" || exit 1
  done
} > MTIMES.txt || { echo "RFAIL could not write MTIMES.txt"; exit 1; }
sha256sum $SEAL > SHA256SUMS.producing || { echo "RFAIL sha256sum failed over the declared file set"; exit 1; }
ne=$(grep -c . SHA256SUMS.producing)
[ "$ne" -eq 15 ] || { echo "RFAIL manifest has $ne entries, expected 15 (12 object-passes + 3 logs)"; exit 1; }
tar czf "$ARCH" $SEAL SHA256SUMS.producing || { echo "RFAIL tar failed"; exit 1; }
echo "ROK manifest_sha=$(sha256sum SHA256SUMS.producing | awk '{print $1}')"
echo "ROK entries=$ne"
echo "ROK tgz_bytes=$(stat -c %s "$ARCH")"
echo "ROK tgz_path=$ARCH"
echo "ROK tgz_sha=$(sha256sum "$ARCH" | cut -d' ' -f1)"
REMOTE
cat "$DEST/COLLECTION.log"
grep -q '^RFAIL' "$DEST/COLLECTION.log" && { echo "REFUSE: producing host refused (above)"; exit 1; }
RB=$(sed -n 's/^ROK tgz_bytes=//p' "$DEST/COLLECTION.log")
RS=$(sed -n 's/^ROK tgz_sha=//p' "$DEST/COLLECTION.log")
RP=$(sed -n 's/^ROK tgz_path=//p' "$DEST/COLLECTION.log")
RM=$(sed -n 's/^ROK manifest_sha=//p' "$DEST/COLLECTION.log")
[ -n "$RB" ] && [ -n "$RS" ] && [ -n "$RP" ] && [ -n "$RM" ] || { echo "REFUSE: producing host did not report the archive path, size, digest and manifest digest"; exit 1; }

echo "=== 2. transfer, verified by archive size AND digest, both values kept in COLLECTION.log"
scp -q "redwood:$RP" "$DEST/c210_final.tgz" || { echo "REFUSE: scp failed"; exit 1; }
LB=$(stat -f %z "$DEST/c210_final.tgz"); LS=$(shasum -a 256 "$DEST/c210_final.tgz" | cut -d' ' -f1)
{ echo "LOCAL tgz_bytes=$LB"; echo "LOCAL tgz_sha=$LS"; } >> "$DEST/COLLECTION.log"
echo "  redwood $RB $RS"
echo "  local   $LB $LS"
[ "$LB" = "$RB" ] && [ "$LS" = "$RS" ] || { echo "REFUSE: archive differs across the transfer"; exit 1; }

echo "=== 3. extract and verify every file against the producing-host manifest"
tar xzf "$DEST/c210_final.tgz" -C "$DEST" || { echo "REFUSE: extract failed"; exit 1; }
rm -f "$DEST/c210_final.tgz"
cd "$DEST" || exit 1
# EVERY line's digest field, not the first. A manifest whose first line is SHA-256 and whose rest
# are CRC-32 passes an NR==1 check. sha256sum cannot emit that today, which is exactly why a
# first-line check would not fail until the day it mattered. Peer's catch.
awk '{ if (length($1) != 64) { print "  line " NR " has a " length($1) "-char digest field"; bad=1 } }
     END { exit bad }' SHA256SUMS.producing \
  || { echo "REFUSE: manifest is not SHA-256 on every line -- a CRC-32 manifest verifies vacuously"; exit 1; }
# BIND THE ARCHIVE TO THE DIRECTORY ITS MANIFEST CAME FROM. Checking tar's exit status closes the
# instance where a stale archive gets digested; it does not close the class, because an archive can
# fail to be the archive of this directory for reasons other than tar returning non-zero. Without
# this, a stale archive at the expected path would be digested on redwood, compared against itself
# across the transfer, extracted, and verified 15 OK 0 FAILED against its OWN manifest -- a clean
# verification of planted data, with the completeness guards having already passed on the live
# directory. Nothing bound the two. Peer's catch; the one-line form is theirs.
EM=$(shasum -a 256 SHA256SUMS.producing | awk '{print $1}')
echo "MANIFEST extracted=$EM redwood=$RM" >> "$DEST/COLLECTION.log"
[ "$EM" = "$RM" ] || { echo "REFUSE: the extracted manifest is not the one computed on redwood"; echo "  redwood   $RM"; echo "  extracted $EM"; exit 1; }
echo "  manifest bound: $EM computed on redwood over the live directory, same bytes here"
shasum -a 256 -c SHA256SUMS.producing > "$DEST/VERIFY.log" 2>&1
NE=$(grep -c . SHA256SUMS.producing)
OK=$(grep -c ': OK$' "$DEST/VERIFY.log"); FA=$(grep -c ': FAILED' "$DEST/VERIFY.log")
echo "  $OK OK, $FA FAILED against $NE producing-host entries"
[ "$FA" -eq 0 ] && [ "$OK" -eq "$NE" ] || { echo "REFUSE: verification short"; exit 1; }

cp -p /Users/bobby/scorch-beat-mkl/harnesses/an_c210.py . || { echo "REFUSE: could not stage the analyzer"; exit 1; }
echo "=== 4. staged. NOT sealed: PROVENANCE.md needs the result, so sealing follows the analysis."
echo "  the final seal must cover SHA256SUMS.producing and COLLECTION.log, which is what binds the"
echo "  manifest to an archive digest that was read on both hosts."
echo "  dir: $DEST"
ls -la
echo
echo "NEXT: python3 harnesses/an_c210.py $DEST   (digest 9c42d96123ce3560)"
