# Two corrections to the latency diagnosis, both recorded before the E-core data existed

Written 2026-10-01 04:52 PDT. The E-core arm of the attested 11-variant run was in flight at
that moment (pid 2099219, `./probe 4096 61 8192 12345` pinned to cpu16, `pb_core16.raw` still
zero bytes). The P-core arm had landed. So correction 2 below changes a prediction whose test
had not yet produced a number, which is the only time a prediction may be changed.

Both were found by the peer session. Neither is a quibble: the first withdraws a precision I
published, the second withdraws half of a registered falsifier.

## 1. The frequency read is quantized to 100 MHz, so "0.3%" is six times finer than the instrument

`run_probe.sh:47,49` read `/sys/devices/system/cpu/cpu{0,16}/cpufreq/scaling_cur_freq`. Under
`intel_pstate` that field is the P-state **request**, on a 100 MHz grid -- not an achieved
average. The three consecutive reads per core returned 5.700 GHz and 4.400 GHz with no variation,
and those are exactly redwood's spec max turbo for the two core types. A field that returns the
datasheet value to four figures three times running is a ceiling, not a measurement; an achieved
average would wander.

So the resolution of the ns -> cycles conversion is +/-100 MHz: **+/-1.75% on the P-core and
+/-2.27% on the E-core.** I published agreement to 0.3%. That agreement is real but cannot be
stated at that precision, and the defensible form is:

  **the measured 1.994 and 2.991 cycles/nnz are consistent with the predicted 2.00 and 3.00
  within the resolution of the frequency read.**

Two details sharpen this rather than soften it.

- The two misses are -0.28% and -0.30%, nearly equal. That is the signature of a common
  multiplicative factor -- both reads pinned at their ceiling with the true clock slightly under
  -- rather than two independent agreements.
- **The error is one-sided.** A sustained frequency can only be at or below the requested one,
  and a lower frequency makes the measured cycles/nnz *smaller*, moving it further below the
  prediction, never toward it. At 5.6 GHz the P-core reads 1.959, a -2.03% miss; at 5.5 GHz,
  -3.78%. So the model cannot be rescued by a better frequency figure, only damaged by one. The
  agreement is therefore a genuine constraint and not an artefact of a flattering conversion,
  but its precision belongs to the field's grid and not to the timing.

redwood has `perf_event_paranoid = 4`, so a core-cycle counter is unavailable. That is a reason
to state the resolution, not a reason to go hunting for a better instrument.

## 2. Gracemont has 2 load ports, not 3, which deletes half of the registered V7 falsifier

`probe.cpp:279-281` recorded, before measuring: V6 (4 chains) gives 1.00 cycles/nnz on P and
1.50 on E, both 2.0x; V7 (8 chains) shows **no further gain on P** because the 3-loads/3-ports
floor is 1.00, "while still gaining on E, whose load width is narrower but whose latency is
longer"; and "if V7 beats V6 on P, the load-port model is wrong."

The clause about the E-core is wrong **by my own model**, because I wrote "narrower" and then
used the P-core's port count anyway. Golden Cove issues 3 loads/cycle; Gracemont issues 2. With
the right count the two cores' tables are:

    P-core (3 load ports)  chains 2/4/8   latency 2.00/1.00/0.50   port floor 1.00   binding 2.00/1.00/1.00
    E-core (2 load ports)  chains 2/4/8   latency 3.00/1.50/0.75   port floor 1.50   binding 3.00/1.50/1.50

The E-core port floor of 3 ops / 2 ports = **1.50 cycles/nnz is exactly the V6 latency
prediction.** So V6 already sits at the E-core's port floor, and V7 cannot gain on the E-core
either. "No gain on P, still gaining on E" had no discriminating power; the model predicts no
gain on **either** core, and my falsifier's complement was never there.

**The corrected registered prediction, stated before the E-core number:** V6 reaches 1.00
cycles/nnz on P and 1.50 on E, both 2.0x over the shipped two chains; V7 gains over V6 on
neither core. A V7 that beats V6 on **either** core refutes the port model. This is one-sided
where I had claimed two-sided, which is weaker -- but it is still falsifiable, and the two cores
remain independent tests because their floors differ (1.00 against 1.50) and arise from
different port counts.

**The load-port counts are an assumption, not a measurement.** They come from published
microarchitecture descriptions, not from this host. Since the whole V7 arm now rests on the two
cores having *different* floors, the count has to be measured here: a loop of independent loads
with no dependent chain, reading loads/cycle directly, should return ~3 on cpu0 and ~2 on cpu16.
That is queued as a probe arm. Until it runs, every V7 statement carries "assuming 3 and 2 load
ports".

## 3. The diagnosis is asymptotic in row length; production rows are 40x shorter

The probe is a single row of 4096 nonzeros with one `_mm_add_ps` fold at the end, so per-row
setup and fold-down amortise to essentially nothing. chain210's own `deg` column over the 3820
DLMC keys: min 0.01, p10 10.24, p25 25.60, **median 102.40**, p75 204.80, p90 307.20, max 2304
-- 42.1% of keys below degree 64 and 81.3% below 256.

So 2.00 cycles/nnz is established for the long-row limit. Whether the shipped kernel reaches it
at degree 102, or 25, is a separate measurement. I had already attached this qualification to the
V6/V7 *proposal*, because the fold-down grows from one add to three or seven; the peer's point is
that it attaches equally to the **diagnosis**, and the diagnosis is what "the kernel is
accumulator-latency bound" rests on. A degree sweep settles both at once: it tells whether the
bound holds at production row lengths, and it locates the degree below which V6's extra folds
stop paying -- which is the number that decides whether any of this ships. Queued.

## What the peer's first message got right, and the one thing it had stale

The structural defect is real and is fixed separately: `run_probe.sh` wrote `ATTEST.txt` in
place, so a re-run clobbered the previous run's attestation while leaving the previous run's CSVs
intact. That is the one combination that makes an already-published number unverifiable, and it
did happen -- the attestation for binary `ae683b29`, which produced every P-core figure in
`56b2350` and `a058bba`, no longer exists on disk.

Their reading of the directory's *contents* was a snapshot of a transient, and the transient has
moved on. At 04:52 PDT `ATTEST.txt` holds binary `526c0e97` from source `b3beb018` with
`asmcheck_exit 0` and all eleven arms, `probe.cpp` on disk digests to `b3beb018`, and the
`probe` binary digests to `526c0e97` -- consistent, and the gate passed. `pb_core0.csv` carries
`526c0e97` with 11 arms x 61 reps and inner 138468. `pb_core16.csv` is the **stale 9-arm file**
from the previous run, still carrying `ae683b29`, and is being overwritten as this is written. So
the analyzer would refuse the pair on the digest mismatch, correctly, and for a reason that
disappears when the live run closes -- but the single-slot attestation that let it happen is a
defect whether or not it bit this time.
