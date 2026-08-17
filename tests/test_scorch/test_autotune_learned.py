"""Tests for the `learned` autotune level (Phase 2): the offline-trained cost model
runtime predictor in src/scorch/tiling.py.

Covers: the dependency-free numpy tree walker == sklearn.predict (train/serve parity);
canonical featurize determinism; model-absent -> analytic fallback (the level is always
safe); the WIDENED gate (operand>C admits low-degree/products-class shapes, but the 99%
operand<=C stays byte-neutral v2); the v2 floor (route a tiled kernel only when the model
predicts it beats v2 by the margin); bit-correctness of learned dispatch vs a torch
reference; and the one-shot v2-confirm opt-in. Synthetic hand-built models make these run
on ANY machine regardless of whether a real per-machine model is present.
"""
import json

import numpy as np
import pytest
import torch

import scorch
import scorch.tiling as T
from scorch import STensor


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Restore global level/LLC/memo, redirect caches to tmp, and reset the lazily
    loaded learned model around every test."""
    prev_level = T._global_level
    prev_llc = T._llc_bytes
    monkeypatch.setenv("SCORCH_AUTOTUNE_CACHE", str(tmp_path / "autotune.json"))
    # default: no learned model (hermetic) — tests that want one override this env.
    monkeypatch.setenv("SCORCH_AUTOTUNE_MODEL", "0")
    T._cache_loaded = False
    T._persist_cache = None
    T._decision.clear()
    T._reset_learned_model_cache()
    yield
    T._global_level = prev_level
    T._llc_bytes = prev_llc
    T._decision.clear()
    T._tls.level = None
    T._cache_loaded = False
    T._persist_cache = None
    T._reset_learned_model_cache()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_eligible(M=800, J=800, deg=70, N=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.zeros(M, J, dtype=torch.float32)
    for r in range(M):
        cols = torch.randperm(J, generator=g)[:deg]
        A[r, cols] = torch.randn(deg, generator=g)
    B = torch.randn(J, N, generator=g, dtype=torch.float32)
    Acsr = A.to_sparse_csr()
    A_st = STensor.from_torch(torch.sparse_csr_tensor(
        Acsr.crow_indices(), Acsr.col_indices(), Acsr.values(), size=A.shape))
    return A_st, STensor.from_torch(B), A @ B


def _st(A):
    Acsr = A.to_sparse_csr()
    return STensor.from_torch(torch.sparse_csr_tensor(
        Acsr.crow_indices(), Acsr.col_indices(), Acsr.values(), size=A.shape))


def _leaf(value):
    return dict(feature=[-2], threshold=[-2.0], left=[-1], right=[-1], value=[value])


def _split_on(feat_name, thr, lo_left, val_right):
    """A depth-1 tree: if X[feat] <= thr -> leaf lo_left else leaf val_right."""
    fi = list(T._FEATURES).index(feat_name)
    return dict(feature=[fi, -2, -2], threshold=[thr, -2.0, -2.0],
               left=[1, -1, -1], right=[2, -1, -1], value=[0.0, lo_left, val_right])


def _write_model(tmp_path, monkeypatch, trees, init=0.0, lr=1.0):
    spec = dict(kind="sklearn_gbr", version=T._LEARNED_VERSION,
                machine_id=T._machine_id(), feature_names=list(T._FEATURES),
                init=init, learning_rate=lr, trees=trees)
    path = tmp_path / "learned_model.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setenv("SCORCH_AUTOTUNE_MODEL", str(path))
    T._reset_learned_model_cache()
    return spec


# force tile-j to look fast, everything else slow -> learned routes tile-j
_PREFER_TILEJ = [_split_on("f_is_tilej", 0.5, +1.0, -1.0)]
# a flat model: every candidate equal -> v2 floor keeps v2
_ALL_EQUAL = [_leaf(0.0)]


# --------------------------------------------------------------------------- #
# 1. dependency-free walker == sklearn (train/serve parity)
# --------------------------------------------------------------------------- #
def test_walker_matches_sklearn():
    skl = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(0)
    F = len(T._FEATURES)
    X = rng.standard_normal((400, F))
    y = X[:, 0] * 1.3 - X[:, 4] * 0.7 + rng.standard_normal(400) * 0.1
    gbr = skl.GradientBoostingRegressor(n_estimators=60, max_depth=3,
                                        learning_rate=0.1, random_state=0).fit(X, y)
    spec = dict(kind="sklearn_gbr", feature_names=list(T._FEATURES),
                init=float(np.ravel(gbr.init_.constant_)[0]),
                learning_rate=float(gbr.learning_rate),
                trees=[dict(feature=e.tree_.feature.astype(int).tolist(),
                            threshold=e.tree_.threshold.astype(float).tolist(),
                            left=e.tree_.children_left.astype(int).tolist(),
                            right=e.tree_.children_right.astype(int).tolist(),
                            value=e.tree_.value[:, 0, 0].astype(float).tolist())
                       for e in gbr.estimators_[:, 0]])
    st = T._build_stacked(spec)
    Xt = rng.standard_normal((32, F))
    assert np.abs(T._walker_predict(st, Xt) - gbr.predict(Xt)).max() < 1e-9


# --------------------------------------------------------------------------- #
# 2. featurize is deterministic + correctly shaped
# --------------------------------------------------------------------------- #
def test_featurize_shape_and_determinism():
    a = T._featurize(1000, 1000, 200000, 256, 16 << 20, 0.9, 1.1, "tilej", 4096, 0)
    b = T._featurize(1000, 1000, 200000, 256, 16 << 20, 0.9, 1.1, "tilej", 4096, 0)
    assert len(a) == len(T._FEATURES)
    assert a == b
    # tile-j vs v2 differ only in the candidate features
    v = T._featurize(1000, 1000, 200000, 256, 16 << 20, 0.9, 1.1, "v2", 0, 0)
    i = list(T._FEATURES).index("f_is_tilej")
    assert a[i] == 1.0 and v[i] == 0.0


# --------------------------------------------------------------------------- #
# 3. model absent -> analytic fallback (always safe)
# --------------------------------------------------------------------------- #
def test_model_absent_falls_back_to_analytic(monkeypatch):
    monkeypatch.setenv("SCORCH_AUTOTUNE_MODEL", "0")   # disable -> no model
    T._reset_learned_model_cache()
    assert T._load_learned_model() is None
    T._llc_bytes = 131072
    A_st, B_st, ref = _make_eligible()
    # with no model, learned uses the ANALYTIC gate (not widened) and the analytic pick
    scorch.set_autotune("learned")
    out = scorch.matmul(A_st, B_st)
    out = out if isinstance(out, torch.Tensor) else out.to_torch()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    # analytic gate: eligible scattered shape -> the analytic COST MODEL picks tiled,
    # then the one-shot v2-confirm has the last word. On this tiny synthetic shape v2
    # wins, so assert the model's pick separately from the confirm's verdict rather
    # than asserting a route the confirm is entitled to overrule.
    kinds = {v[0] for v in T._decision.values()}
    assert kinds <= {"tilej", "tileijk", "v2"}
    T._decision.clear()
    monkeypatch.setattr(T, "_CONFIRM_TILED", False)
    scorch.matmul(A_st, B_st)
    unconfirmed = {v[0] for v in T._decision.values()}
    assert unconfirmed and unconfirmed <= {"tilej", "tileijk"}, T._decision


def test_foreign_machine_model_rejected(tmp_path, monkeypatch):
    _write_model(tmp_path, monkeypatch, _ALL_EQUAL)
    # corrupt the machine id -> must be rejected -> analytic fallback
    spec = json.loads((tmp_path / "learned_model.json").read_text())
    spec["machine_id"] = "deadbeef"
    (tmp_path / "learned_model.json").write_text(json.dumps(spec))
    T._reset_learned_model_cache()
    assert T._load_learned_model() is None


# --------------------------------------------------------------------------- #
# 4. widened gate: operand>C admits low-degree shapes, but 99% stays byte-neutral
# --------------------------------------------------------------------------- #
def test_widened_gate_admits_low_degree_but_keeps_operand_floor(tmp_path, monkeypatch):
    _write_model(tmp_path, monkeypatch, _ALL_EQUAL)
    assert T._load_learned_model() is not None
    T._llc_bytes = 4096  # tiny LLC so a small operand still exceeds it

    # low-degree (deg ~5 << DEG_FLOOR) but operand>C: analytic REJECTS, learned ADMITS.
    A_lo, B_lo, _ = _make_eligible(M=300, J=300, deg=5, N=16)
    assert T.is_candidate(A_lo, B_lo, level="analytic") is False
    assert T.is_candidate(A_lo, B_lo, level="learned") is True

    # operand<=C (the 99%): learned must NOT admit it (byte-neutral v2) even with a model.
    T._llc_bytes = 64 << 20   # 64 MB, huge vs a tiny operand
    A_sm, B_sm, _ = _make_eligible(M=64, J=64, deg=8, N=16)
    assert T.is_candidate(A_sm, B_sm, level="learned") is False


def test_widened_gate_needs_a_model(tmp_path, monkeypatch):
    # learned WITHOUT a model must not widen (falls back to the analytic gate).
    monkeypatch.setenv("SCORCH_AUTOTUNE_MODEL", "0")
    T._reset_learned_model_cache()
    T._llc_bytes = 4096
    A_lo, B_lo, _ = _make_eligible(M=300, J=300, deg=5, N=16)
    assert T.is_candidate(A_lo, B_lo, level="learned") is False  # analytic gate


# --------------------------------------------------------------------------- #
# 5. v2 floor + routing
# --------------------------------------------------------------------------- #
def test_v2_floor_keeps_v2_when_no_predicted_advantage(tmp_path, monkeypatch):
    _write_model(tmp_path, monkeypatch, _ALL_EQUAL)   # all candidates equal
    T._llc_bytes = 131072
    A_st, B_st, ref = _make_eligible()
    scorch.set_autotune("learned")
    out = scorch.matmul(A_st, B_st)
    out = out if isinstance(out, torch.Tensor) else out.to_torch()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    # flat model -> v2 floor -> route v2 (recorded as ("v2", None))
    assert all(v[0] == "v2" for v in T._decision.values()), T._decision


def test_learned_routes_tilej_when_model_predicts_it(tmp_path, monkeypatch):
    """The model's ARGMIN must be tile-j. Measured separately from the confirm, which
    is entitled to overrule it — and does here, since these shapes are tiny."""
    _write_model(tmp_path, monkeypatch, _PREFER_TILEJ)
    monkeypatch.setattr(T, "_CONFIRM_TILED", False)
    T._llc_bytes = 131072
    A_st, B_st, ref = _make_eligible()
    scorch.set_autotune("learned")
    out = scorch.matmul(A_st, B_st)
    out = out if isinstance(out, torch.Tensor) else out.to_torch()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    kinds = {v[0] for v in T._decision.values()}
    assert kinds == {"tilej"}, T._decision


def test_learned_confirms_inside_the_analytic_gate(tmp_path, monkeypatch):
    """The confirm must fire even for a pick the ANALYTIC gate would also have made.

    Restricting it to the widened-only region left a hole exactly where the model was
    confident and wrong: inline_1 at N=512 is eligible and scattered, so no confirm ran
    and the model's tile-ijk pick shipped at 0.385x of untiled. Over a 236-cell grid
    that hole was 3 of learned's 4 regressions. Here the shape passes the ordinary
    gate, the model insists on tile-j, and the confirm must still overrule it to v2."""
    _write_model(tmp_path, monkeypatch, _PREFER_TILEJ)
    T._llc_bytes = 131072
    A_st, B_st, _ = _make_eligible()
    assert T._eligible(int(A_st.shape[1]),
                       int(A_st.storage._mode_indices[1][1].numel()),
                       int(B_st.shape[1]), T.query_llc())
    assert T._scattered(A_st, int(A_st.shape[1]))
    T._decision.clear()
    scorch.set_autotune("learned")
    scorch.matmul(A_st, B_st)
    assert {v[0] for v in T._decision.values()} == {"v2"}, T._decision


# --------------------------------------------------------------------------- #
# 6. bit-correctness at the learned level across a small grid (with a model)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("N", [16, 64, 128])
def test_learned_bit_correct_grid(tmp_path, monkeypatch, N):
    _write_model(tmp_path, monkeypatch, _PREFER_TILEJ)
    T._llc_bytes = 131072
    A_st, B_st, ref = _make_eligible(N=N)
    scorch.set_autotune("learned")
    out = scorch.matmul(A_st, B_st)
    out = out if isinstance(out, torch.Tensor) else out.to_torch()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


# --------------------------------------------------------------------------- #
# 7. one-shot v2-confirm opt-in stays correct
# --------------------------------------------------------------------------- #
def test_confirm_opt_in_bit_correct(tmp_path, monkeypatch):
    _write_model(tmp_path, monkeypatch, _PREFER_TILEJ)
    monkeypatch.setenv("SCORCH_AUTOTUNE_CONFIRM", "1")
    monkeypatch.setattr(T, "_LEARNED_CONFIRM", True)
    T._llc_bytes = 131072
    A_st, B_st, ref = _make_eligible()
    scorch.set_autotune("learned")
    out = scorch.matmul(A_st, B_st)
    out = out if isinstance(out, torch.Tensor) else out.to_torch()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


# --------------------------------------------------------------------------- #
# 8. cheap structural features are sane
# --------------------------------------------------------------------------- #
def test_degree_cv_and_locality_sane():
    # scattered high-degree uniform matrix: high locality, low degree_cv
    A_sc, _, _ = _make_eligible(M=400, J=400, deg=60, N=16)
    assert T._locality_ratio(A_sc, 400) > 0.3
    assert T._degree_cv(A_sc) < 0.3           # uniform degree -> low skew

    # banded matrix: near-zero locality
    band = torch.zeros(400, 400)
    for i in range(400):
        lo, hi = max(0, i - 3), min(400, i + 4)
        band[i, lo:hi] = torch.randn(hi - lo)
    A_bd = _st(band)
    assert T._locality_ratio(A_bd, 400) < 0.1
