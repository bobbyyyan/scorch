#!/usr/bin/env python3
r"""train_autotune_model.py — Phase-2 offline training for the `learned` autotune level.

Input : bench/bench_results/autotune_train_<machine>.csv  (one row per (matrix,N,candidate)
        with cheap features -> measured median time/GFLOP/s; from collect_autotune_data.py)
Output: autotune-levels/models/model_<machine_id>.json  (dependency-free tree walker the
        runtime predictor loads; numpy-only inference, no sklearn at runtime)

Model : predict-runtime per candidate -> argmin over the probe's candidate set
        (the confirmed Phase-2 framing). A classic sklearn GradientBoostingRegressor on
        y = log(time_s); its .tree_ arrays export to a trivial numpy walker (raw-feature
        thresholds, no binning) — cleaner to ship dep-free than HistGradientBoosting.

Eval  : at the CELL level (each (matrix,N)): predict every candidate's time, argmin ->
        predicted winner; regret = gflops[pred] / gflops[oracle] in (0,1]. Report the
        geomean regret on MATRIX-GROUPED held-out splits (never test on a matrix seen in
        train) vs the ANALYTIC baseline (reproduce tiling's cost-model pick on the same
        held-out cells) — learned must beat analytic (M5 ~0.826 / redwood ~0.977).

Ablations prove the two design hypotheses carry signal:
    full  vs  (- degree_cv)  vs  (- analytic-bytes)  vs  analytic-only(no ML).

Usage: python bench/train_autotune_model.py [csv_path]   (default: pick by platform)
"""
from __future__ import annotations
import os
import sys
import json
import math
import platform

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
# CANONICAL feature/walker definitions live in the shipped package -> import them so
# training and the tiling.py runtime predictor are byte-identical (no train/serve drift).
from scorch.tiling import (  # noqa: E402
    _FEATURES, _featurize, _build_stacked, _walker_predict)

MODELS_DIR = os.path.join(REPO, "autotune-levels", "models")

# tiling constants for the analytic baseline (mirror src/scorch/tiling.py defaults)
DEG_FLOOR = float(os.environ.get("SCORCH_TILING_DEG_FLOOR", "64"))
NIJK_MIN = int(os.environ.get("SCORCH_TILING_NIJK_MIN", "512"))
LOC_MIN = float(os.environ.get("SCORCH_TILING_LOC_MIN", "0.3"))


# --------------------------------------------------------------- features -----
FEATURES = list(_FEATURES)                 # canonical order (from scorch.tiling)
# feature groups for ablation (prove each carries signal)
DEGREE_CV_FEATS = {"f_degree_cv"}
ANALYTIC_BYTES_FEATS = {"f_log_cand_bytes", "f_cand_over_output"}


def featurize_row(r):
    """r: dict of CSV columns. Delegates to the CANONICAL scorch.tiling._featurize so
    the trainer's features are byte-identical to the runtime predictor's."""
    vec = _featurize(r["M"], r["J"], r["nnz"], r["N"], r["C_llc"],
                     r["locality"], r["degree_cv"], r["kind"],
                     r["cand_Jc"], r["cand_Nc"])
    return dict(zip(FEATURES, vec))


# ------------------------------------------------------------ csv loading -----
def load_rows(path):
    import csv as csvmod
    rows = []
    with open(path) as f:
        for r in csvmod.DictReader(f):
            for k in ("M", "J", "nnz", "N", "degree", "locality", "degree_cv", "C_llc",
                      "operand", "thrash_ratio", "output_bytes", "A_bytes",
                      "cand_Jc", "cand_Nc", "cand_P", "cand_nk", "cand_bytes",
                      "time_s", "gflops", "is_v2", "is_tilej", "is_tileijk", "is_oracle"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def cell_key(r):
    return (r["matrix"], int(r["N"]))


# --------------------------------------------------- analytic baseline pick ---
def analytic_pick(cell_rows):
    """Reproduce tiling.py's analytic (cost-model, no probe) decision for a cell, and
    return the matching candidate row (so we can read its measured gflops). Mirrors
    _eligible + _scattered + _panel_width + _ijk_beats_tilej_bytes."""
    r0 = cell_rows[0]
    J = r0["J"]; N = r0["N"]; M = r0["M"]; nnz = r0["nnz"]
    C = r0["C_llc"]; operand = r0["operand"]; deg = r0["degree"]; loc = r0["locality"]
    by_kind = {}
    for r in cell_rows:
        by_kind.setdefault(r["kind"], []).append(r)

    def v2_row():
        return by_kind["v2"][0]

    # gate: eligible?
    eligible = (operand > C) and (deg > max(DEG_FLOOR, 2.0 * operand / C))
    if not eligible or loc <= LOC_MIN:
        return v2_row()
    # tile-j @ base (the analytic width)
    base = max(256, int(C // (4 * N)))
    tj = [r for r in by_kind.get("tilej", []) if r["cand_Jc"] == min(J, base)]
    if not tj:  # nearest available Jc to base
        cands = by_kind.get("tilej", [])
        tj = [min(cands, key=lambda r: abs(r["cand_Jc"] - base))] if cands else []
    # tile-ijk if wide N and byte model says it's cheaper
    if N >= NIJK_MIN and by_kind.get("tileijk"):
        ijk = by_kind["tileijk"][0]
        BN = 4.0 * N; Cwr = M * BN; A = 8.0 * nnz
        Jc_tj = min(J, base); P = max(1, -(-int(J) // max(1, int(Jc_tj))))
        tj_bytes = J * BN + P * 2 * Cwr + A + P * M * 4
        nk = max(1, -(-int(N) // max(1, int(ijk["cand_Nc"]))))
        ijk_bytes = J * BN + Cwr + A * nk + 2 * J * BN
        if ijk_bytes < tj_bytes:
            return ijk
    return tj[0] if tj else v2_row()


# --------------------------------------------------------- model + export -----
def train_gbr(X, y, seed=0):
    from sklearn.ensemble import GradientBoostingRegressor
    m = GradientBoostingRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=8, random_state=seed)
    m.fit(X, y)
    return m


def export_gbr(model, feature_names):
    """Classic GBR -> dep-free JSON. predict = init + lr * sum_t tree_t(x)."""
    init = float(np.ravel(model.init_.constant_)[0])
    trees = []
    for est in model.estimators_[:, 0]:
        t = est.tree_
        trees.append({
            "feature": t.feature.astype(int).tolist(),
            "threshold": t.threshold.astype(float).tolist(),
            "left": t.children_left.astype(int).tolist(),
            "right": t.children_right.astype(int).tolist(),
            "value": t.value[:, 0, 0].astype(float).tolist(),
        })
    return {"kind": "sklearn_gbr", "feature_names": list(feature_names),
            "init": init, "learning_rate": float(model.learning_rate), "trees": trees}


# The stacked-array builder + fast walker are the CANONICAL scorch.tiling versions
# (imported), so the parity check below validates the EXACT runtime algorithm.
build_stacked = _build_stacked
walker_predict = _walker_predict


# ---------------------------------------------------------------- eval --------
def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


MARGIN = 0.03   # matches tiling._LEARNED_MARGIN default (v2 floor)
WIDEN = os.environ.get("ATD_WIDEN", "1") == "1"        # model the widened gate (default)
CONFIRM_ALL = os.environ.get("ATD_CONFIRM_ALL", "0") == "1"  # confirm every tiled pick


def eval_cells(rows, predict_fn, feat_names):
    """Faithfully model the SHIPPED learned dispatch per (matrix,N) cell: predict each
    candidate's log-time, apply the v2 floor (only leave v2 for a tiled kernel predicted
    to beat it by MARGIN), then in the WIDENED-ONLY region (analytic gate would reject)
    a tiled pick is CONFIRMED against v2 (keep the faster) -- exactly maybe_dispatch's
    learned branch. Returns (learned_regret, analytic_regret, oracle_agree, n_cells,
    per_cell_records=(family,N,lreg,areg))."""
    logfloor = math.log(max(1e-9, 1.0 - MARGIN))
    cells = {}
    for r in rows:
        cells.setdefault(cell_key(r), []).append(r)
    lreg, areg, agree, recs = [], [], 0, []
    for key, cr in cells.items():
        feats = np.array([[featurize_row(r)[fn] for fn in feat_names] for r in cr])
        preds = predict_fn(feats)
        v2i = next(i for i, r in enumerate(cr) if r["is_v2"] == 1.0)
        tiled = [i for i in range(len(cr)) if i != v2i]
        pick = v2i
        if tiled:
            bt = min(tiled, key=lambda i: preds[i])
            if preds[bt] < preds[v2i] + logfloor:
                pick = bt
        oidx = int(np.argmax([r["gflops"] for r in cr]))
        og = cr[oidx]["gflops"]
        ap = analytic_pick(cr)
        analytic_ok = ap["kind"] != "v2"          # analytic gate admits this shape
        if not analytic_ok:
            # analytic gate rejects. keep-gate (WIDEN=0): learned routes v2 (== analytic).
            # widened (WIDEN=1): a tiled pick is confirmed vs v2 (keep faster) -> catches
            # analytic false-negatives (mid-degree scattered) without regressing.
            if WIDEN and pick != v2i:
                gf = max(cr[pick]["gflops"], cr[v2i]["gflops"])
            else:
                gf = cr[v2i]["gflops"]
        else:
            # eligible: model decides (v2 floor). CONFIRM_ALL also confirms here ->
            # provably >= v2 (max(pick, v2)); else trust the model pick (measurement-free).
            if CONFIRM_ALL and pick != v2i:
                gf = max(cr[pick]["gflops"], cr[v2i]["gflops"])
            else:
                gf = cr[pick]["gflops"]
        lr = gf / og if og else 1.0
        ar = ap["gflops"] / og if og else 1.0
        lreg.append(lr); areg.append(ar); agree += int(pick == oidx)
        recs.append((cr[0]["family"], int(cr[0]["N"]), lr, ar))
    return lreg, areg, agree, len(cells), recs


def group_split(rows, held_frac=0.25, seed=0):
    """Split by MATRIX (a matrix's rows never straddle train/test)."""
    mats = sorted(set(r["matrix"] for r in rows))
    rng = np.random.default_rng(seed); rng.shuffle(mats)
    n_test = max(1, int(len(mats) * held_frac))
    test_mats = set(mats[:n_test])
    train = [r for r in rows if r["matrix"] not in test_mats]
    test = [r for r in rows if r["matrix"] in test_mats]
    return train, test, sorted(test_mats)


def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = os.path.join(REPO, "bench", "bench_results",
                                f"autotune_train_{platform.system().lower()}.csv")
    rows = load_rows(csv_path)
    machine_id = rows[0]["machine_id"]
    print(f"[load] {csv_path}: {len(rows)} rows, machine={machine_id}, "
          f"{len(set(r['matrix'] for r in rows))} matrices, "
          f"{len(set(cell_key(r) for r in rows))} cells", flush=True)

    # ------- ablation via matrix-grouped CV (average over several splits) -------
    def feats_for(mode):
        drop = set()
        if mode == "no_degree_cv":
            drop = DEGREE_CV_FEATS
        elif mode == "no_analytic_bytes":
            drop = ANALYTIC_BYTES_FEATS
        return [f for f in FEATURES if f not in drop]

    modes = ["full", "no_degree_cv", "no_analytic_bytes"]
    n_splits = 4
    print("\n===== ABLATION (matrix-grouped held-out; geomean regret vs oracle) =====", flush=True)
    analytic_scores = []
    full_recs = []
    for mode in modes:
        fn = feats_for(mode)
        lregs = []
        for s in range(n_splits):
            train, test, _ = group_split(rows, seed=s)
            Xtr = np.array([[featurize_row(r)[k] for k in fn] for r in train])
            ytr = np.log(np.array([r["time_s"] for r in train]))
            m = train_gbr(Xtr, ytr, seed=s)
            lreg, areg, agree, ncell, recs = eval_cells(test, lambda X: m.predict(X), fn)
            lregs.extend(lreg)
            if mode == "full":
                analytic_scores.extend(areg)
                full_recs.extend(recs)
        print(f"  {mode:20s}: learned held-out geomean = {geomean(lregs):.4f} "
              f"({len(lregs)} cells)", flush=True)
    print(f"  {'analytic (no ML)':20s}: geomean = {geomean(analytic_scores):.4f} "
          f"  <- the baseline to beat", flush=True)

    # per-family held-out breakdown (where does learned beat analytic?)
    print("\n  --- held-out regret by family (learned vs analytic; count) ---", flush=True)
    fams = {}
    for fam, N, lr, ar in full_recs:
        fams.setdefault(fam, ([], []))
        fams[fam][0].append(lr); fams[fam][1].append(ar)
    for fam in sorted(fams):
        L, A = fams[fam]
        print(f"    {fam:14s}: learned {geomean(L):.3f}  analytic {geomean(A):.3f} "
              f"  ({len(L)} cells)", flush=True)
    # the wide-N tail (N>=512) specifically — where tile-ijk + the Jc tail live
    wide = [(lr, ar) for fam, N, lr, ar in full_recs if N >= 512]
    if wide:
        print(f"    {'N>=512 (wide)':14s}: learned {geomean([w[0] for w in wide]):.3f}  "
              f"analytic {geomean([w[1] for w in wide]):.3f}  ({len(wide)} cells)", flush=True)

    # ------- final model on ALL rows (for shipping) + walker parity check -------
    print("\n===== FINAL model on all rows =====", flush=True)
    Xall = np.array([[featurize_row(r)[k] for k in FEATURES] for r in rows])
    yall = np.log(np.array([r["time_s"] for r in rows]))
    final = train_gbr(Xall, yall, seed=0)
    spec = export_gbr(final, FEATURES)
    stacked = build_stacked(spec)
    # parity: FAST numpy walker == sklearn.predict
    samp = Xall[np.random.default_rng(0).integers(0, len(Xall), size=min(500, len(Xall)))]
    d = np.abs(walker_predict(stacked, samp) - final.predict(samp)).max()
    print(f"  walker-vs-sklearn max abs diff = {d:.2e} (must be ~0); "
          f"maxdepth={stacked['maxdepth']}", flush=True)
    # in-sample cell agreement (sanity, optimistic)
    lreg, areg, agree, ncell, _ = eval_cells(rows, lambda X: walker_predict(stacked, X), FEATURES)
    print(f"  in-sample: learned geomean={geomean(lreg):.4f}  analytic={geomean(areg):.4f}  "
          f"oracle-agree={agree}/{ncell}", flush=True)

    spec.update(version=1, machine_id=machine_id, platform=platform.system(),
                n_train_rows=len(rows),
                heldout_geomean=round(geomean([r for r in lreg]), 4))
    os.makedirs(MODELS_DIR, exist_ok=True)
    out = os.path.join(MODELS_DIR, f"model_{machine_id}.json")
    with open(out, "w") as f:
        json.dump(spec, f)
    print(f"\n[write] {out}  ({os.path.getsize(out)/1024:.0f} KB, {len(spec['trees'])} trees)", flush=True)


if __name__ == "__main__":
    main()
