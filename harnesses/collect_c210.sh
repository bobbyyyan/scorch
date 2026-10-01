#!/bin/bash
# chain210 -- collect the DLMC replication run off redwood and seal it, producing-host first.
#
# WHY A SCRIPT AND NOT A HAND SEQUENCE. Three staging decisions in this collection are degrees of
# freedom that an_c210.py's own output depends on, and all three were exercised wrongly earlier in
# this project:
#   1. c210_stage.log lives at /scratch/bobbyy/c210_stage.log, ONE LEVEL ABOVE the CSV directory,
#      because the stage script's stdout was redirected there at launch. an_c210.py reads it from
#      os.path.join(argv[1], "c210_stage.log") and EXITS without it. Two sessions already produced
#      two different analysis digests (0990bb8f vs d239c553) for no reason other than one
#      symlinking chain207's stage log and the other synthesising a three-line substitute. This
#      script copies the real file and seals its digest, so the staging is named, not improvised.
#   2. Digests must be computed on the PRODUCING host before any transfer, or they attest the copy
#      rather than the run.
#   3. MTIMES.txt is computed on redwood and included in the producing manifest. chain207 added it
#      after its first seal and had to recompute the manifest twice; doing it before the manifest
#      means one seal.
#
# WHAT IT REFUSES ON. No done marker; a CSV count other than 12; any object-pass whose line count
# is not 57301 (57300 rows + header, = 3820 keys x 5 widths x 3 arms); an archive whose size or
# digest differs across the transfer; any file failing the producing-host manifest; a target
# directory that already exists. It does NOT run the analyzer -- that is a separate, deliberate step
# so that a refusal here can never be mistaken for a null result there.
set -u
R=/scratch/bobbyy/c210
STAGE=/scratch/bobbyy/c210_stage.log
DEST=${1:-$HOME/.cache/scorch-codex/spmm-beat-mkl/chain210-dlmc-replicate-redwood}
EXPECT_LINES=57301
EXPECT_CSV=12

[ -e "$DEST" ] && { echo "REFUSE: $DEST already exists -- a seal is never overwritten in place"; exit 1; }

echo "=== 1. producing host: completeness, then digests"
ssh -o BatchMode=yes redwood bash -s <<'REMOTE' > /tmp/c210_remote.out 2>&1
set -u
R=/scratch/bobbyy/c210
STAGE=/scratch/bobbyy/c210_stage.log
cd "$R" || { echo "RFAIL no $R"; exit 1; }
grep -q RW_C210_ALL_PASSES_DONE "$STAGE" || { echo "RFAIL no done marker in $STAGE -- run incomplete"; exit 1; }
# an_c210.py's TIMING block scores H1/H2/H3 only `if len(v) == 12`, where v comes from the stage
# log's start/done pairs. A short or truncated stage log therefore yields NO verdict and NO
# complaint -- a null result indistinguishable from a failure. The estimator is pre-registered and
# is not amended; the refusal goes here instead.
sd=$(grep -cE '^    --- p[1-6] (aa|bb) done ' "$STAGE")
ss=$(grep -cE '^    --- p[1-6] (aa|bb) start' "$STAGE")
[ "$sd" -eq 12 ] && [ "$ss" -eq 12 ] || { echo "RFAIL stage log has $ss start / $sd done lines, not 12 each -- the H1/H2/H3 block would skip silently"; exit 1; }
n=$(ls -1 dl_*_float32_p*.csv 2>/dev/null | wc -l)
[ "$n" -eq 12 ] || { echo "RFAIL $n CSVs, expected 12"; exit 1; }
bad=0
for f in dl_*_float32_p*.csv; do
  l=$(wc -l < "$f")
  [ "$l" -eq 57301 ] || { echo "RFAIL $f has $l lines, expected 57301"; bad=1; }
done
[ "$bad" -eq 0 ] || exit 1
cp -p "$STAGE" "$R/c210_stage.log"
{ echo "# chain210 per-file mtimes, recorded ON REDWOOD before transfer, so the pre-registration"
  echo "# boundary is checkable from the artefacts rather than from a commit clock."
  echo "# an_c210.py's final digest 9c42d96123ce3560 entered 23e9700 at 00:29:27 PDT 2026-10-01."
  echo
  for f in dl_*_float32_p*.csv c210_run.log c210_stage.log; do
    echo "$(date -r "$f" '+%Y-%m-%d %H:%M:%S')  $f"
  done
} > MTIMES.txt
sha256sum dl_*_float32_p*.csv c210_run.log c210_stage.log MTIMES.txt > SHA256SUMS.producing
tar czf /scratch/bobbyy/c210_final.tgz dl_*_float32_p*.csv c210_run.log c210_stage.log MTIMES.txt SHA256SUMS.producing
echo "ROK entries=$(grep -c . SHA256SUMS.producing)"
echo "ROK tgz_bytes=$(stat -c %s /scratch/bobbyy/c210_final.tgz)"
echo "ROK tgz_sha=$(sha256sum /scratch/bobbyy/c210_final.tgz | cut -d' ' -f1)"
REMOTE
cat /tmp/c210_remote.out
grep -q '^RFAIL' /tmp/c210_remote.out && { echo "REFUSE: producing host refused (above)"; exit 1; }
RB=$(sed -n 's/^ROK tgz_bytes=//p' /tmp/c210_remote.out)
RS=$(sed -n 's/^ROK tgz_sha=//p' /tmp/c210_remote.out)
[ -n "$RB" ] && [ -n "$RS" ] || { echo "REFUSE: producing host did not report archive size and digest"; exit 1; }

echo "=== 2. transfer, verified by archive size AND digest"
mkdir -p "$DEST"
scp -q redwood:/scratch/bobbyy/c210_final.tgz "$DEST/c210_final.tgz" || { echo "REFUSE: scp failed"; exit 1; }
LB=$(stat -f %z "$DEST/c210_final.tgz"); LS=$(shasum -a 256 "$DEST/c210_final.tgz" | cut -d' ' -f1)
echo "  redwood $RB $RS"
echo "  local   $LB $LS"
[ "$LB" = "$RB" ] && [ "$LS" = "$RS" ] || { echo "REFUSE: archive differs across the transfer"; exit 1; }

echo "=== 3. extract and verify every file against the producing-host manifest"
tar xzf "$DEST/c210_final.tgz" -C "$DEST" || { echo "REFUSE: extract failed"; exit 1; }
rm -f "$DEST/c210_final.tgz"
cd "$DEST" || exit 1
# shasum -c, not cksum: the manifest's digest field is 64 hex chars, which is what makes it SHA-256.
W=$(awk 'NR==1{print length($1)}' SHA256SUMS.producing)
[ "$W" -eq 64 ] || { echo "REFUSE: manifest digest field is $W chars, not 64 -- that is not SHA-256"; exit 1; }
shasum -a 256 -c SHA256SUMS.producing > /tmp/c210_verify.out 2>&1
OK=$(grep -c ': OK$' /tmp/c210_verify.out); FA=$(grep -c ': FAILED' /tmp/c210_verify.out)
echo "  $OK OK, $FA FAILED against $(grep -c . SHA256SUMS.producing) producing-host entries"
[ "$FA" -eq 0 ] && [ "$OK" -eq "$(grep -c . SHA256SUMS.producing)" ] || { echo "REFUSE: verification short"; exit 1; }

cp -p /Users/bobby/scorch-beat-mkl/harnesses/an_c210.py .
echo "=== 4. staged. NOT sealed: PROVENANCE.md needs the result, so sealing follows the analysis."
echo "  dir: $DEST"
ls -la
echo
echo "NEXT: python3 harnesses/an_c210.py $DEST   (digest 9c42d96123ce3560)"
