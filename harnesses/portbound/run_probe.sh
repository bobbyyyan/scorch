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
} > ATTEST.txt
if [ "$AC" -ne 0 ]; then
  echo "REFUSE: asmcheck rejected the emitted code; no timing is read."
  cat asmcheck.txt | tail -6
  exit 1
fi
echo "ATTESTED binary $BIN"
for core in 0 16; do
  taskset -c "$core" ./probe "$N" "$REPS" "$NB" "$SEED" > "pb_core${core}.raw" || { echo "REFUSE: probe failed on cpu$core"; exit 1; }
  # The digest goes INTO the data file, so a CSV can never be read against a different binary.
  { echo "# binary_sha256 $BIN"; cat "pb_core${core}.raw"; } > "pb_core${core}.csv"
  rm -f "pb_core${core}.raw"
  echo "cpu$core -> pb_core${core}.csv ($(wc -l < pb_core${core}.csv) lines)"
done
echo "freq during a P-core block, for the cycles conversion:"
( taskset -c 0 ./probe "$N" 9 "$NB" "$SEED" > /dev/null & \
  for i in 1 2 3; do sleep 1; echo "  cpu0 $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) kHz"; done; wait ) 2>/dev/null
( taskset -c 16 ./probe "$N" 9 "$NB" "$SEED" > /dev/null & \
  for i in 1 2 3; do sleep 1; echo "  cpu16 $(cat /sys/devices/system/cpu/cpu16/cpufreq/scaling_cur_freq) kHz"; done; wait ) 2>/dev/null
