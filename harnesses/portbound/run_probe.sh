#!/bin/bash
# Compile, ATTEST, and run the port-boundedness probe, binding the timings to the binary that
# produced them.
#
# WHY THE BINDING EXISTS. The registered refutation accepts when t(V5) is at t(V1). A V5 whose ALU
# ops were deleted by -O3 IS a V1 with different labels, so it lands exactly where the refutation
# says to look -- the strongest result of the probe and its most dangerous failure mode are the same
# reading. Three attempts at V4/V5 were deleted or spilled before the disassembly gate passed, and
# one of those was precisely a vacuous V5 reading +0.00 ALU ops. A gate that caught earlier attempts
# is not an attestation of the binary that produced a number. So: one compile, its sha256 recorded,
# asmcheck run against THAT binary's disassembly, the digest written into every CSV, and the
# analyzer refuses unless the CSV's digest matches the attested one.
set -u
D=/scratch/bobbyy/portbound
cd "$D" || exit 1
N=${1:-4096}; REPS=${2:-61}; NB=${3:-8192}; SEED=${4:-12345}

g++ -O3 -march=native -funroll-loops -std=c++17 -o probe probe.cpp || { echo "REFUSE: compile failed"; exit 1; }
BIN=$(sha256sum probe | cut -d' ' -f1)
SRC=$(sha256sum probe.cpp | cut -d' ' -f1)
objdump -d --no-show-raw-insn probe > probe.asm || { echo "REFUSE: objdump failed"; exit 1; }
python3 asmcheck.py probe.asm > asmcheck.txt 2>&1
AC=$?
{
  echo "binary_sha256 $BIN"
  echo "source_sha256 $SRC"
  echo "compiler $(g++ --version | head -1)"
  echo "flags -O3 -march=native -funroll-loops -std=c++17"
  echo "asmcheck_exit $AC"
  sed -n '/^PER-NONZERO/,/^$/p' asmcheck.txt
} > "ATTEST-${BIN:0:16}.txt"

# WHY THE ATTESTATION IS NAMED BY THE BINARY AND NEVER OVERWRITTEN.
# This script used to write a single ATTEST.txt in place. A re-run then clobbered the previous
# run's attestation while leaving the previous run's CSVs on disk untouched -- the one combination
# that makes an ALREADY PUBLISHED number unverifiable after the fact, because the CSV still names
# a binary whose emitted counts and gate verdict no longer exist anywhere. It happened: the
# attestation for binary ae683b29, which produced every P-core figure in commits 56b2350 and
# a058bba, is gone. Found by the peer session reading the directory, not by this script.
# So: one attestation file per binary digest, an append-only log, and the disassembly and source
# kept under the same digest -- the disassembly because a later question about symbol PLACEMENT
# across builds is answerable only if both builds' symbol addresses survive, and the previous
# build's probe.asm did not.
cp -f probe.asm "probe-${BIN:0:16}.asm"
cp -f probe.cpp "probe-${BIN:0:16}.cpp"
printf '%s  binary %s  source %s  asmcheck_exit %s  argv "%s %s %s %s"\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$BIN" "$SRC" "$AC" "$N" "$REPS" "$NB" "$SEED" >> ATTEST_LOG.txt
# ATTEST.txt remains as a convenience pointer to the newest attestation. It is NOT the record:
# the analyzer resolves each CSV's own digest to ATTEST-<digest>.txt and refuses if that file is
# absent, so a stale CSV can never be read against a newer binary's attestation.
cp -f "ATTEST-${BIN:0:16}.txt" ATTEST.txt

if [ "$AC" -ne 0 ]; then
  echo "REFUSE: asmcheck rejected the emitted code; no timing is read."
  cat asmcheck.txt | tail -6
  exit 1
fi
echo "ATTESTED binary $BIN"
for core in 0 16; do
  taskset -c "$core" ./probe "$N" "$REPS" "$NB" "$SEED" > "pb_core${core}.raw" || { echo "REFUSE: probe failed on cpu$core"; exit 1; }
  # The digest goes INTO the data file, so a CSV can never be read against a different binary.
  # THE DATA IS DIGEST-NAMED FOR THE SAME REASON THE ATTESTATION IS, and I fixed only half of
  # this problem the first time: ATTEST.txt was made per-binary while the CSVs kept their fixed
  # names, so a re-run still destroyed the DATA behind a published number even though its
  # attestation now survived. It happened within the hour -- the 526c0e97 E-core CSV, the
  # published half of the latency refutation, was overwritten by the next run and survived only
  # because a copy had already been fetched. Fixing one half of a two-part custody problem and
  # calling it done is the error; the generic name stays as a convenience pointer only.
  { echo "# binary_sha256 $BIN"; cat "pb_core${core}.raw"; } > "pb_core${core}-${BIN:0:16}.csv"
  cp -f "pb_core${core}-${BIN:0:16}.csv" "pb_core${core}.csv"
  rm -f "pb_core${core}.raw"
  echo "cpu$core -> pb_core${core}-${BIN:0:16}.csv ($(wc -l < pb_core${core}.csv) lines)"
done
echo "freq during a P-core block, for the cycles conversion:"
( taskset -c 0 ./probe "$N" 9 "$NB" "$SEED" > /dev/null & \
  for i in 1 2 3; do sleep 1; echo "  cpu0 $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) kHz"; done; wait ) 2>/dev/null
( taskset -c 16 ./probe "$N" 9 "$NB" "$SEED" > /dev/null & \
  for i in 1 2 3; do sleep 1; echo "  cpu16 $(cat /sys/devices/system/cpu/cpu16/cpufreq/scaling_cur_freq) kHz"; done; wait ) 2>/dev/null
