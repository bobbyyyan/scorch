r"""Pool a mass SpMM sweep and find the cells that are actually below MKL parity.

The sweep's job is breadth; this script's job is to not be fooled by it. Over tens of
thousands of cells, a few hundred will read below 1.0 from noise alone, so a raw
"cells below parity" count is close to meaningless. Three things separate a finding
from a fluctuation here:

  * The A/A control. Every cell timed the same scorch code twice; ``aa_ratio`` is
    what a ratio of 1.000 actually looks like at that cell's size and rep count.
    The screen uses a *pooled* quantile of |aa_ratio - 1| taken over cells of
    comparable cost, not each cell's own control -- a cell whose own control happens
    to read 1.00 is not thereby more trustworthy, and judging it by that is how you
    manufacture confirmations.
  * Two estimators, and the control picks between them. The sweep records a median
    and a minimum over samples for every arm. On a hybrid-core host the median is
    contaminated by which cores the OpenMP team happened to land on; the minimum
    takes the uncontended placement for every arm. Neither is right a priori, so
    both ratios are reported with their own A/A floor and the tighter floor wins.
  * Primary matrices versus auxiliary files. A SuiteSparse tarball holds the matrix
    and also its right-hand-side vectors -- shapes like 1447360x1 with 440
    nonzeros. Those are worth timing and they expose real behaviour, but a headline
    that pools them is reporting a library's behaviour on operators using
    measurements of its behaviour on vectors.
  * Attribution. ``vs_mkl`` times the whole ``scorch.matmul`` call and
    ``vs_mkl_kernel`` times only the kernel. A cell that loses on the first and wins
    on the second is losing to fixed per-call cost, which no kernel change can move;
    a cell that loses on both is a kernel finding.
  * Correctness first. A fast wrong answer is not a win, so the ratio columns are
    only reported for cells whose float64 relative error is under --relerr-max.

Everything it prints is a count or a quantile over a stated subset; there is no
single headline number, because "is scorch slower than MKL" does not have one.

usage:
  python bench/analyze_spmm_sweep.py sweep_redwood.csv --host redwood
  python bench/analyze_spmm_sweep.py rw.csv mkt.csv --candidates cand.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd


def geo(x):
    x = np.asarray([v for v in x if np.isfinite(v) and v > 0], dtype=float)
    return float(np.exp(np.mean(np.log(x)))) if x.size else float("nan")


def bucket(series, edges, labels):
    return pd.cut(series, bins=edges, labels=labels, right=False)


def describe(name, sub, col="vs_mkl"):
    v = sub[col].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return f"  {name:<26s}      n=0"
    return (
        f"  {name:<26s} n={v.size:6d}  geomean {geo(v):6.3f}  "
        f"p05 {np.quantile(v, .05):6.3f}  median {np.median(v):6.3f}  "
        f"p95 {np.quantile(v, .95):7.3f}  min {v.min():6.3f}  "
        f"below 1.0: {int((v < 1).sum()):5d} ({100 * (v < 1).mean():4.1f}%)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--relerr-max", type=float, default=1e-4)
    ap.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="drop cells whose shared batch was below this. A cell at batch 1 has "
        "each arm timed as one isolated call, which on a two-socket host makes the "
        "arm pay an OpenMP team wake-up that the A/A control cannot see. Use 2 or "
        "more to exclude those unless the run was made with --settle.",
    )
    ap.add_argument(
        "--aa-quantile",
        type=float,
        default=0.95,
        help="pooled |aa-1| quantile that sets the parity screen",
    )
    ap.add_argument(
        "--candidates",
        default=None,
        help="write below-parity candidates here for a confirm pass",
    )
    a = ap.parse_args()

    frames = []
    for p in a.csv:
        d = pd.read_csv(p, low_memory=False)
        d["source"] = p
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    # A SuiteSparse key is derived from the file's path below the scan root, and
    # the two hosts do not lay the collection out the same way: one has
    # <group>/<name>/<name>.mtx throughout, the other has a flat <name>/<name>.mtx
    # shared cache alongside a group-qualified tree of its own. Joining the hosts
    # on `key` therefore matched only the minority that agree -- 50022 of 83310
    # comparable cells, silently. The matrix name is what is portable.
    d["join_key"] = [
        ("ss:" + k.rsplit("/", 1)[-1]) if str(k).startswith("ss:") else k for k in d.key
    ]
    if a.min_batch > 1:
        before = len(d)
        d = d[pd.to_numeric(d.get("batch"), errors="coerce").fillna(0) >= a.min_batch]
        print(f"--min-batch {a.min_batch}: kept {len(d)} of {before} rows")
    num = [
        "k",
        "nnz",
        "rows",
        "cols",
        "density",
        "mean_row",
        "std_row",
        "max_row",
        "vs_mkl",
        "vs_mkl_kernel",
        "aa_ratio",
        "relerr_sc",
        "relerr_mkl",
        "mkl32_ms",
        "mkl64_ms",
        "sc_ms",
        "aa_ms",
        "sc_kernel_ms",
        "reps",
        "batch_sc",
        "batch_mkl",
    ]
    for c in num:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["collection"] = d.key.str.split(":").str[0]
    if "primary" not in d or d.primary.isna().all():
        # older sweeps predate the flag; recover it from SuiteSparse's own layout
        parts = d.key.str.split(":", n=1).str[1].str.split("/")
        d["primary"] = np.where(
            d.collection == "ss", (parts.str[-1] == parts.str[-2]).astype(float), 1.0
        )
    d["is_primary"] = d.primary.fillna(1) > 0

    print("=" * 78)
    print("SWEEP INVENTORY")
    print("=" * 78)
    print(d.status.value_counts().to_string())
    print(f"\nmatrices seen : {d.key.nunique()}")
    print(f"k values      : {sorted(int(x) for x in d.k.dropna().unique())}")
    print("\nby collection:")
    print(
        d.groupby("collection")
        .agg(
            cells=("status", "size"),
            ok=("status", lambda s: (s == "ok").sum()),
            matrices=("key", "nunique"),
        )
        .to_string()
    )

    sk = d[d.status == "skipped"]
    if len(sk):
        print("\nwhat was skipped, and why (top reasons):")
        why = sk.note.astype(str).str.replace(
            r"\d+(\.\d+)?([eE][+-]?\d+)?", "N", regex=True
        )
        for r, n in why.value_counts().head(12).items():
            print(f"  {n:6d}  {r[:96]}")
    er = d[d.status == "error"]
    if len(er):
        print(f"\nERRORS: {len(er)} cells")
        why = er.note.astype(str).str.slice(0, 110)
        for r, n in why.value_counts().head(12).items():
            print(f"  {n:6d}  {r}")

    ok = d[d.status == "ok"].copy()
    if not len(ok):
        print("\nno completed cells")
        return

    print("\n" + "=" * 78)
    print("CORRECTNESS (float64 scipy reference, random row sample)")
    print("=" * 78)
    for c, lbl in (("relerr_sc", "scorch"), ("relerr_mkl", "MKL int32")):
        v = ok[c].dropna()
        print(
            f"  {lbl:<10s} median {v.median():.2e}  p99 {v.quantile(.99):.2e}  "
            f"max {v.max():.2e}   cells over {a.relerr_max:.0e}: "
            f"{int((v > a.relerr_max).sum())}"
        )
    bad = ok[ok.relerr_sc > a.relerr_max]
    if len(bad):
        print(f"\n  scorch above the correctness cap on {len(bad)} cells:")
        print(
            bad[["key", "k", "nnz", "relerr_sc", "relerr_mkl"]]
            .head(20)
            .to_string(index=False)
        )
    ok = ok[ok.relerr_sc <= a.relerr_max]

    print("\n" + "=" * 78)
    print("A/A CONTROL -- what a ratio of 1.000 looks like here")
    print("=" * 78)
    ok["aa_ratio_min"] = ok.sc_min_ms / ok.aa_min_ms
    for lbl, col in (("median ", "aa_ratio"), ("minimum", "aa_ratio_min")):
        dv = (ok[col] - 1).abs().dropna()
        if not len(dv):
            continue
        print(
            f"  {lbl} |aa-1|  median {dv.median():.4f}  p75 {dv.quantile(.75):.4f}  "
            f"p90 {dv.quantile(.90):.4f}  p95 {dv.quantile(.95):.4f}  "
            f"p99 {dv.quantile(.99):.4f}  max {dv.max():.4f}"
        )
    # Neither estimator is right a priori. The A/A control is the only thing that can
    # say which one is measuring, so let it choose, and say that it did.
    est, aacol = "vs_mkl", "aa_ratio"
    if ok.aa_ratio_min.notna().any():
        med95 = (ok.aa_ratio - 1).abs().quantile(0.95)
        min95 = (ok.aa_ratio_min - 1).abs().quantile(0.95)
        if min95 < med95:
            est, aacol = "vs_mkl_min", "aa_ratio_min"
        print(
            f"\n  the tighter control is the "
            f"{'median' if est == 'vs_mkl' else 'minimum'} "
            f"({min(med95, min95):.4f} vs {max(med95, min95):.4f} at p95), so the "
            f"screen below uses {est}"
        )
    # The floor depends on cell cost, so set it per cost decile rather than globally:
    # a 20 us cell and a 200 ms cell do not deserve the same threshold.
    ok["cost_bin"] = pd.qcut(ok.sc_ms, 10, duplicates="drop")
    floor = ok.groupby("cost_bin", observed=True)[aacol].apply(
        lambda s: (s - 1).abs().quantile(a.aa_quantile)
    )
    ok["aa_floor"] = ok.cost_bin.map(floor).astype(float)
    print(f"\n  per-cost-decile floor at the p{int(a.aa_quantile * 100)} quantile:")
    for iv, f in floor.items():
        n = int((ok.cost_bin == iv).sum())
        print(f"    sc_ms in {str(iv):<24s} n={n:6d}  floor {f:.4f}")

    print("\n" + "=" * 78)
    print("SCORCH vs BEST MKL (whole scorch.matmul call vs the faster MKL arm)")
    print("=" * 78)
    print(f"  (the estimator is {est}; both are printed for ALL CELLS)")
    print(describe("ALL CELLS, median", ok, "vs_mkl"))
    print(describe("ALL CELLS, minimum", ok, "vs_mkl_min"))
    print("\nPRIMARY matrices only -- the headline:")
    prim = ok[ok.is_primary]
    print(describe("primary, all", prim, est))
    for c, sub in prim.groupby("collection"):
        print(describe("primary, " + c, sub, est))
    print(describe("AUXILIARY files", ok[~ok.is_primary], est))
    print("\nby host:")
    for host, sub in prim.groupby("source"):
        print(describe("  " + host.split("/")[-1], sub, est))
    ok = prim
    print("\nby k:")
    for kk, sub in ok.groupby("k"):
        print(describe(f"k={int(kk)}", sub))
    print("\nby nnz:")
    edges = [0, 1e4, 1e5, 1e6, 1e7, np.inf]
    labels = ["<1e4", "1e4-1e5", "1e5-1e6", "1e6-1e7", ">1e7"]
    ok["nnz_bin"] = bucket(ok.nnz, edges, labels)
    for b, sub in ok.groupby("nnz_bin", observed=True):
        print(describe(str(b), sub))
    print("\nby mean nonzeros per row:")
    ok["deg_bin"] = bucket(
        ok.mean_row,
        [0, 2, 4, 8, 16, 32, 64, np.inf],
        ["<2", "2-4", "4-8", "8-16", "16-32", "32-64", ">64"],
    )
    for b, sub in ok.groupby("deg_bin", observed=True):
        print(describe(str(b), sub))
    print("\nby rows-per-nonzero -- an SpMM writes rows*k output whether or not a")
    print("row has any nonzeros, so a mostly-empty matrix is an output-writing job:")
    ok["rpn"] = ok.rows / ok.nnz.clip(lower=1)
    for b, sub in ok.groupby(
        bucket(
            ok.rpn,
            [0, 0.05, 0.2, 1, 10, 1e3, np.inf],
            ["<0.05", "0.05-0.2", "0.2-1", "1-10", "10-1e3", ">1e3"],
        ),
        observed=True,
    ):
        print(describe(str(b), sub))
    print("\nby fraction of rows that are empty:")
    ok["ef"] = ok.empty_rows / ok.rows.clip(lower=1)
    for b, sub in ok.groupby(
        bucket(
            ok.ef,
            [0, 0.001, 0.05, 0.25, 0.75, 1.01],
            ["~0", "<5%", "5-25%", "25-75%", ">75%"],
        ),
        observed=True,
    ):
        print(describe(str(b), sub))
    print("\nby selector route:")
    for b, sub in ok.groupby(ok.route.fillna("(none)"), observed=True):
        print(describe(str(b), sub))

    print("\n" + "=" * 78)
    print("KERNEL ONLY (scorch's eval_time vs the same best-MKL whole call)")
    print("=" * 78)
    print("  MKL's number includes its own dispatch, so this comparison is generous")
    print("  to scorch by exactly the cost MKL pays and scorch's kernel does not.")
    print(describe("ALL CELLS", ok, "vs_mkl_kernel"))
    for kk, sub in ok.groupby("k"):
        print(describe(f"k={int(kk)}", sub, "vs_mkl_kernel"))

    print("\n" + "=" * 78)
    print("BELOW PARITY, SCREENED AGAINST THE A/A FLOOR")
    print("=" * 78)
    lose = ok[ok.vs_mkl < 1 - ok.aa_floor].copy()
    print(
        f"  cells below 1.0 at all              : {int((ok.vs_mkl < 1).sum())} "
        f"of {len(ok)} ({100 * (ok.vs_mkl < 1).mean():.1f}%)"
    )
    print(
        f"  cells below the A/A floor           : {len(lose)} "
        f"({100 * len(lose) / len(ok):.2f}%)"
    )
    if len(lose):
        kern_ok = lose.vs_mkl_kernel >= 1
        print(
            f"    of those, kernel already >= MKL   : {int(kern_ok.sum())} "
            f"-- fixed per-call cost, not the kernel"
        )
        print(
            f"    of those, kernel also below MKL   : {int((~kern_ok).sum())} "
            f"-- a kernel finding if it confirms"
        )
        print(
            f"  worst cell: {lose.vs_mkl.min():.3f}   "
            f"median of losers: {lose.vs_mkl.median():.3f}"
        )
        print("\n  losers by k:")
        print(
            lose.groupby("k")
            .agg(
                n=("vs_mkl", "size"), worst=("vs_mkl", "min"), med=("vs_mkl", "median")
            )
            .to_string()
        )
        print("\n  losers by collection:")
        print(
            lose.groupby("collection")
            .agg(n=("vs_mkl", "size"), worst=("vs_mkl", "min"))
            .to_string()
        )
        print("\n  losers by scorch call time (is this a small-cell effect?):")
        lose["ms_bin"] = bucket(
            lose.sc_ms,
            [0, 0.05, 0.1, 0.2, 0.5, 1, 5, np.inf],
            ["<50us", "50-100us", "0.1-0.2ms", "0.2-0.5ms", "0.5-1ms", "1-5ms", ">5ms"],
        )
        print(
            lose.groupby("ms_bin", observed=True)
            .agg(
                n=("vs_mkl", "size"), worst=("vs_mkl", "min"), med=("vs_mkl", "median")
            )
            .to_string()
        )
        cols = [
            "key",
            "k",
            "nnz",
            "mean_row",
            "sc_ms",
            "sc_kernel_ms",
            "mkl32_ms",
            "mkl64_ms",
            "vs_mkl",
            "vs_mkl_kernel",
            "aa_ratio",
            "aa_floor",
            "route",
            "reps",
            "source",
        ]
        print("\n  worst 30 cells:")
        print(lose.nsmallest(30, "vs_mkl")[cols].to_string(index=False))
        if a.candidates:
            lose.sort_values("vs_mkl")[cols].to_csv(a.candidates, index=False)
            print(f"\n  {len(lose)} candidates -> {a.candidates}")
    else:
        print("  no cell is below MKL parity by more than its own noise floor")

    if ok.source.nunique() > 1:
        print("\n" + "=" * 78)
        print("CROSS-HOST: cells that lose on one host and not the other")
        print("=" * 78)
        # A handful of SuiteSparse names occur in more than one group, so the
        # portable join key is not unique for them; they cannot be matched across
        # hosts without guessing which group is which, so drop them and say so.
        dup = ok.groupby(["source", "join_key", "k"]).size()
        ambiguous = {j for (_, j, _), n in dup.items() if n > 1}
        if ambiguous:
            print(
                f"  {len(ambiguous)} matrix name(s) occur in more than one "
                f"SuiteSparse group and are excluded from the cross-host join: "
                f"{', '.join(sorted(x.split(':')[-1] for x in ambiguous))}"
            )
        xh = ok[~ok.join_key.isin(ambiguous)]
        piv = xh.pivot_table(index=["join_key", "k"], columns="source", values=est)
        piv = piv.dropna()
        cols = list(piv.columns)
        if len(cols) == 2:
            both = ((piv[cols[0]] < 1) & (piv[cols[1]] < 1)).sum()
            only0 = ((piv[cols[0]] < 1) & (piv[cols[1]] >= 1)).sum()
            only1 = ((piv[cols[0]] >= 1) & (piv[cols[1]] < 1)).sum()
            print(f"  cells present on both hosts : {len(piv)}")
            print(f"  below 1.0 on both           : {both}")
            print(f"  below 1.0 only on {cols[0].split('/')[-1]:<20s}: {only0}")
            print(f"  below 1.0 only on {cols[1].split('/')[-1]:<20s}: {only1}")
            print(
                f"  correlation of the two ratios: "
                f"{piv[cols[0]].corr(piv[cols[1]]):.3f}"
            )


if __name__ == "__main__":
    sys.exit(main())
