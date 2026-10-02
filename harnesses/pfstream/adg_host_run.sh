#!/usr/bin/env bash
# Build the three ADOPT_GRAIN objects and run both measurements on ONE host.
#
# IT REFUSES WHILE ANOTHER MEASUREMENT IS TIMING. Compiling three objects and then timing
# thousands of sub-30-microsecond cells next to a running screen perturbs both: the screen's
# readings pick up our compiler and our readings pick up its kernels. The guard greps for the
# producer by script name. Its pattern cannot match the guard's own process, because the pattern
# names the producer scripts and this line does not contain those names as a contiguous string --
# a pgrep pattern that matches its own command line is a guard that always fires.
#
# ARGUMENTS, all from the environment so one script serves hosts with different layouts:
#   ADG_STAGE     tree with src/, pyproject.toml, scorch_build.py   (verified by adg_build.sh)
#   ADG_ROOT      where the three objects are staged and built
#   ADG_ACTIVATE  sourced to bring up the environment
#   ADG_CACHE     corpus root: a DLMC .smtx tree or a cache of .npz
#   ADG_OUT       where CSVs go
#   ADG_PERCORE   this host's measured per-worker team cost in microseconds, for the analysis
set -u
: "${ADG_STAGE:?}"; : "${ADG_ROOT:?}"; : "${ADG_CACHE:?}"; : "${ADG_OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)

# COUNTED BY LINES, NOT BY `pgrep -c`. On macOS pgrep has no -c: it exits 2, prints nothing, and
# `$(pgrep -fc ... || echo 0)` turns that into a count of zero -- so the guard reported "no
# producer running" while one was running. Fail-OPEN, in the guard whose whole job is to refuse.
# Verified against a live long-running process: `pgrep -f mrgate` returns the pid on both
# platforms, `pgrep -fc` returns rc=2 here and a count on Linux.
PATS='mrgate arm_screen ntoracle adggate'
for pat in $PATS; do
  n=$(pgrep -f "$pat" 2>/dev/null | grep -c . || true)
  if [ "${n:-0}" -gt 0 ]; then
    echo "REFUSE: $n process(es) matching '$pat' are running. Building and timing next to a live"
    echo "        measurement perturbs both directions. Wait for it, or name a different host."
    exit 1
  fi
done
# POSITIVE CONTROL ON EACH PATTERN, not just on the mechanism. An earlier version probed with
# "sleep 8", which is none of the four patterns below: it proved `pgrep -f | grep -c .` works on
# this platform -- covering the `pgrep -c` defect -- and proved nothing about the patterns
# themselves. A misspelt or invalid pattern yields n=0 through the same pipeline, which is
# indistinguishable from an idle machine and is the original defect's exact signature. So each
# pattern gets a process whose command line contains it, and the probe's OWN pid must appear --
# counting matches would be satisfied by any unrelated process that happens to match.
# FIRST, A CHECK THAT IS NOT CIRCULAR. The probe loop below creates a file named after each
# pattern, so it matches its own purpose-built process and would pass for a pattern naming no
# real producer -- it establishes that pgrep and this regex WORK (which does catch an invalid
# regex, since an invalid one fails to match even its own probe) and nothing about whether the
# pattern corresponds to anything that runs. What IS checkable without the producer: every
# pattern must name a producer script that exists in this directory. Derived by matching against
# the files on disk rather than asserted, so adding a producer under a new name and forgetting to
# list it here is visible.
for pat in $PATS; do
  hits=$(ls "$HERE" 2>/dev/null | grep -c "^${pat}\." || true)
  if [ "${hits:-0}" -lt 1 ]; then
    echo "REFUSE: guard pattern '$pat' names no producer script in $HERE, so it can never match"
    echo "        a real process and its silence would carry no information."
    exit 1
  fi
done
echo "guard: all $(echo $PATS | wc -w | tr -d ' ') patterns name a producer script that exists"

PROBE_DIR=$(mktemp -d) || exit 1
for pat in $PATS; do
  printf 'sleep 6\n' > "$PROBE_DIR/${pat}_probe.sh"
  bash "$PROBE_DIR/${pat}_probe.sh" &
  ppid=$!
  sleep 0.3
  if ! pgrep -f "$pat" 2>/dev/null | grep -qx "$ppid"; then
    echo "REFUSE: the guard's pattern '$pat' cannot see a live process whose command line"
    echo "        contains it (probe pid $ppid). A pattern that matches nothing reads exactly"
    echo "        like an idle machine, which is the failure this guard exists to avoid."
    kill "$ppid" 2>/dev/null; rm -rf "$PROBE_DIR"; exit 1
  fi
  kill "$ppid" 2>/dev/null
  wait "$ppid" 2>/dev/null
done
rm -rf "$PROBE_DIR"
echo "guard: all $(echo $PATS | wc -w | tr -d ' ') patterns verified against a live process of"
echo "       their own, matched by pid, and no producer is running"

bash "$HERE/adg_build.sh" || exit 1

if [ -n "${ADG_ACTIVATE:-}" ]; then source "$ADG_ACTIVATE" || exit 1; fi
mkdir -p "$ADG_OUT" || exit 1
KEYS="$HERE/adg_keys.txt"
echo "keys digest: $( (sha256sum "$KEYS" 2>/dev/null || shasum -a 256 "$KEYS") | cut -c1-16)"

# Three passes, order rotated as a balanced Latin square so no build sits systematically later
# in the run than another and reads a warmer machine for it.
for pass in p1 p2 p3; do
  case "$pass" in
    p1) ORDER="adg0 adgb adgG" ;;
    p2) ORDER="adgG adg0 adgb" ;;
    p3) ORDER="adgb adgG adg0" ;;
  esac
  echo "=== pass $pass  order $ORDER  $(date +%H:%M:%S) ==="
  for tag in $ORDER; do
    PYTHONPATH="$ADG_ROOT/$tag/src" python "$HERE/adggate.py" \
      --root "$ADG_CACHE" --keys "$KEYS" --csv "$ADG_OUT/${tag}_${pass}.csv" \
      --dtype float32 --ks 2,4,8,12,16,32 --reps 9 --target-ms 15 \
      --pass-tag "$pass" --build "$tag" 2>&1 | tail -3
  done
done

# The oracle sweep needs only ONE object -- it varies the override, not the build.
echo "=== oracle sweep  $(date +%H:%M:%S) ==="
PYTHONPATH="$ADG_ROOT/adg0/src" python "$HERE/ntoracle.py" \
  --root "$ADG_CACHE" --keys "$KEYS" --csv "$ADG_OUT/oracle.csv" \
  --dtype float32 --ks 2,4,8,12,16,20,24,32 --reps 5 --target-ms 10 2>&1 | tail -3

echo "=== analysis  $(date +%H:%M:%S) ==="
python "$HERE/an_adg.py" --csvdir "$ADG_OUT" 2>&1 | tail -60
python "$HERE/an_ntoracle.py" --csv "$ADG_OUT/oracle.csv" \
  --percore-us "${ADG_PERCORE:?set ADG_PERCORE to the measured per-worker cost for this host}" \
  2>&1 | tail -60
echo "DONE $(date +%H:%M:%S)"
