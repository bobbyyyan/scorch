#!/bin/bash
# Does V6's 21% survive at PRODUCTION row lengths? Sweep the row degree on the ALREADY ATTESTED
# binary -- no recompile, so every point in the sweep is bound to one gate verdict.
#
# WHY THIS IS THE DECIDING MEASUREMENT. The 11-arm result establishes V6 at -21.259% (P) and
# -4.032% (E) on a single row of 4096 nonzeros, where the per-row setup and the accumulator
# fold-down amortise to essentially nothing. chain210's own deg column over the 3820 DLMC keys
# reads p10 10.24, p25 25.60, median 102.40, p75 204.80 -- so the probe's row is 40x the median
# production row and 400x the 10th percentile. V6 trades a per-NONZERO gain for a per-ROW cost:
# four accumulator chains fold in two dependent levels where two chains fold in one. Below some
# degree the trade inverts and V6 is a REGRESSION, which under the project's performance
# convention means it cannot ship unguarded however large the asymptotic win is.
#
# THE ESTIMATOR, registered here and not after seeing the numbers. The probe reports
# nspn = t_call / (inner * n), so t_call = nspn * n. Per arm, fit
#
#     t_call(n) = a + b*n
#
# b is the per-nonzero cost -- the asymptotic figure the 4096 run already measured -- and a is
# the per-call constant: function-call overhead PLUS the fold. The function-call overhead is
# identical across arms, so it CANCELS in the difference of intercepts:
#
#     delta_a = a(V6) - a(V0)  =  the extra fold cost alone
#     break-even degree n* = delta_a / (b(V0) - b(V6))
#
# That is the overhead-free break-even, which is what production needs, because production has
# no per-row function call -- it has an inlined row loop. Reporting the raw measured ratio at
# each n INSTEAD would understate V6's low-degree penalty, since the shared call overhead sits
# in both numerator and denominator and pulls the ratio toward 1 exactly where the penalty is.
# So both are printed: the raw ratio per n, which needs no model, and the fit, which removes an
# overhead production does not pay.
#
# PREDICTION, recorded before the run.
#   V6 folds 4 chains in 2 dependent vaddps levels; V0 folds 2 chains in 1. One extra level at
#   4 cycles is 0.70 ns on the P-core at 5.7 GHz and 0.91 ns on the E-core at 4.4 GHz. V7 folds
#   8 chains in 3 levels, so two extra levels: roughly double.
#   With b(V0)-b(V6) = 0.074384 ns/nnz on P and 0.027403 on E:
#     P-core break-even  n* = 0.70 / 0.074384 =  9.4   (to 18.8 if the fold costs two levels)
#     E-core break-even  n* = 0.91 / 0.027403 = 33.2
#   So V6 is predicted to REGRESS below degree ~9-19 on the P-core and below ~33 on the E-core.
#   Against the DLMC distribution that is the bottom 5-15% of keys on P and the bottom ~25% on
#   E (p25 = 25.60). If the break-even lands there, V6 ships only behind a degree guard; if it
#   lands below degree 4, V6 ships unguarded. A break-even ABOVE the median 102.40 would mean
#   the asymptotic win is unreachable in production and the lever is closed for this corpus.
set -u
D=/scratch/bobbyy/portbound
cd "$D" || exit 1
REPS=${1:-61}; NB=${2:-8192}; SEED=${3:-12345}
LADDER="4 6 8 12 16 24 32 48 64 128 512 4096"

BIN=$(sha256sum probe | cut -d' ' -f1)
ATT="ATTEST-${BIN:0:16}.txt"
if [ ! -f "$ATT" ]; then
  # The single-slot era left only ATTEST.txt; accept it ONLY if it records this exact digest.
  if [ -f ATTEST.txt ] && grep -qx "binary_sha256 $BIN" ATTEST.txt; then
    cp -f ATTEST.txt "$ATT"
    echo "adopted ATTEST.txt as $ATT (it records this binary's digest)"
  else
    echo "REFUSE: no attestation for binary ${BIN:0:16}; recompile via run_probe.sh first."; exit 1
  fi
fi
grep -qx "asmcheck_exit 0" "$ATT" || { echo "REFUSE: $ATT does not record asmcheck_exit 0."; exit 1; }
[ -f "probe-${BIN:0:16}.asm" ] || cp -f probe.asm "probe-${BIN:0:16}.asm" 2>/dev/null
echo "sweeping degree on attested binary ${BIN:0:16}; ladder: $LADDER"

for n in $LADDER; do
  for core in 0 16; do
    out="pbdeg_n${n}_core${core}.csv"
    taskset -c "$core" ./probe "$n" "$REPS" "$NB" "$SEED" > "$out.raw" \
      || { echo "REFUSE: probe failed at n=$n on cpu$core"; exit 1; }
    { echo "# binary_sha256 $BIN"; cat "$out.raw"; } > "$out"
    rm -f "$out.raw"
  done
  printf '%s  degree-sweep n=%s reps=%s binary %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$n" "$REPS" "$BIN" >> ATTEST_LOG.txt
  echo "  n=$n done ($(grep -vc '^#' pbdeg_n${n}_core0.csv) rows per core)"
done
echo "SWEEP DONE"
