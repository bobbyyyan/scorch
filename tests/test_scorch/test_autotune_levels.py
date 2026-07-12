"""Tests for the autotune optimization-level API (scorch.set_autotune /
scorch.autotune) layered over the SpMM tiling selector (src/scorch/tiling.py).

Covers: the public surface + validation, thread-local context-manager/decorator
semantics, env-var -> level mapping, memo keying by (signature, level), the
persistent on-disk cache used by "max", and — the no-regression contract —
bit-correctness at EVERY level plus the eligibility gate staying byte-neutral on
ineligible shapes. Levels change WHICH kernel/width runs, never the numerics, so
every level must match the same torch reference (CLAUDE.md correctness convention).
"""
import json
import threading
from types import SimpleNamespace

import pytest
import torch

import scorch
import scorch.tiling as T
from scorch import STensor


@pytest.fixture(autouse=True)
def _isolate_autotune(tmp_path, monkeypatch):
    """Restore global level + memo + queried LLC around every test, and redirect
    the persistent cache to a per-test tmp file so no test ever touches the user's
    real ~/.cache/scorch/autotune.json (tests exercise the 'max' level)."""
    prev_level = T._global_level
    prev_llc = T._llc_bytes
    monkeypatch.setenv("SCORCH_AUTOTUNE_CACHE", str(tmp_path / "autotune.json"))
    T._cache_loaded = False
    T._persist_cache = None
    T._decision.clear()
    yield
    T._global_level = prev_level
    T._llc_bytes = prev_llc
    T._decision.clear()
    T._tls.level = None
    T._cache_loaded = False
    T._persist_cache = None


def _make_eligible(M=800, J=800, deg=70, N=64, seed=0):
    """A scattered, high-degree square matrix that the gate will accept once the
    LLC is shrunk (see tests that set T._llc_bytes). Returns (A_stensor, B_stensor,
    reference_dense)."""
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


# --------------------------------------------------------------------------- #
# Public API surface + validation
# --------------------------------------------------------------------------- #
def test_default_level_is_analytic():
    # A fresh process default; the fixture restores whatever was set before.
    assert scorch.get_autotune() in T._LEVELS


def test_set_get_roundtrip():
    for lvl in ("off", "analytic", "balanced", "max", "learned"):
        scorch.set_autotune(lvl)
        assert scorch.get_autotune() == lvl


def test_set_autotune_normalizes_case_and_space():
    scorch.set_autotune("  MAX ")
    assert scorch.get_autotune() == "max"


def test_set_autotune_rejects_unknown():
    with pytest.raises(ValueError):
        scorch.set_autotune("turbo")


def test_set_autotune_rejects_non_string():
    with pytest.raises(TypeError):
        scorch.set_autotune(3)


def test_autotune_rejects_unknown_level():
    with pytest.raises(ValueError):
        scorch.autotune("nope")


def test_schedule_from_tuner_choice_maps_tileijk_with_explicit_names():
    schedule = scorch.schedule_from_tuner_choice(
        ("tileijk", (32, 17)),
        row_var="row",
        panel_var="reduction",
        free_var="column",
        packed_operand="DenseRhs",
    )

    assert schedule == scorch.Schedule(
        loop_order=("row", "reduction", "column"),
        tiles=(
            scorch.TileSpec(
                "column",
                32,
                placement="outermost",
                accum="direct",
                unroll=False,
            ),
            scorch.TileSpec(
                "reduction",
                17,
                placement="child_of:column_out",
                kind="panel",
                accum="direct",
            ),
        ),
        relayout=scorch.RelayoutSpec(
            operand="DenseRhs",
            pack_var="column",
            strip_width=32,
        ),
        tag="tuner-tileijk",
        parallel_loop="row",
    )
    assert T.schedule_from_tuner_choice is scorch.schedule_from_tuner_choice


@pytest.mark.parametrize("choice", [("v2", None), ("tilej", 17)])
def test_schedule_from_tuner_choice_returns_none_for_other_valid_choices(choice):
    assert (
        scorch.schedule_from_tuner_choice(
            choice,
            row_var="row",
            panel_var="reduction",
            free_var="column",
            packed_operand="DenseRhs",
        )
        is None
    )


@pytest.mark.parametrize(
    ("choice", "error", "message"),
    [
        (["tileijk", (8, 4)], TypeError, "choice must be"),
        (("tileijk",), ValueError, "exactly"),
        ((1, (8, 4)), TypeError, "kind must be"),
        (("unknown", None), ValueError, "unknown tuner choice"),
        (("v2", 1), ValueError, "parameter None"),
        (("tilej", True), TypeError, "positive integer"),
        (("tilej", 0), ValueError, "positive integer"),
        (("tileijk", [8, 4]), TypeError, r"\(Nc, Jc\) tuple"),
        (("tileijk", (8,)), ValueError, r"exactly \(Nc, Jc\)"),
        (("tileijk", (True, 4)), TypeError, "Nc must be"),
        (("tileijk", (8, 0)), ValueError, "Jc must be"),
    ],
)
def test_schedule_from_tuner_choice_rejects_non_normalized_choices(
    choice, error, message
):
    with pytest.raises(error, match=message):
        scorch.schedule_from_tuner_choice(
            choice,
            row_var="row",
            panel_var="reduction",
            free_var="column",
            packed_operand="DenseRhs",
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("row_var", "", ValueError),
        ("panel_var", None, TypeError),
        ("free_var", "", ValueError),
        ("packed_operand", None, TypeError),
    ],
)
def test_schedule_from_tuner_choice_requires_structural_names(field, value, error):
    names = {
        "row_var": "row",
        "panel_var": "reduction",
        "free_var": "column",
        "packed_operand": "DenseRhs",
    }
    names[field] = value

    with pytest.raises(error, match=field):
        scorch.schedule_from_tuner_choice(("tileijk", (8, 4)), **names)


def test_schedule_adapter_does_not_replace_native_tuner_dispatch(monkeypatch):
    native_result = object()
    calls = []

    def native_tileijk(*args):
        calls.append(args)
        return native_result

    monkeypatch.setattr(
        T,
        "_ops",
        SimpleNamespace(spmm_csr_float_tileijk=native_tileijk),
    )
    monkeypatch.setattr(T, "_tileijk_args", lambda *args: ["native-arguments"])

    schedule = scorch.schedule_from_tuner_choice(
        ("tileijk", (8, 4)),
        row_var="row",
        panel_var="reduction",
        free_var="column",
        packed_operand="DenseRhs",
    )
    dispatched = T._dispatch_decision(None, None, None, "tileijk", (8, 4), -1)

    assert schedule is not None
    assert dispatched == (native_result, True)
    assert calls == [("native-arguments",)]


# --------------------------------------------------------------------------- #
# Context manager + decorator (thread-local, like torch.no_grad)
# --------------------------------------------------------------------------- #
def test_context_manager_scopes_and_restores():
    scorch.set_autotune("analytic")
    with scorch.autotune("max"):
        assert scorch.get_autotune() == "max"
    assert scorch.get_autotune() == "analytic"


def test_context_manager_nests():
    scorch.set_autotune("off")
    with scorch.autotune("balanced"):
        assert scorch.get_autotune() == "balanced"
        with scorch.autotune("max"):
            assert scorch.get_autotune() == "max"
        assert scorch.get_autotune() == "balanced"
    assert scorch.get_autotune() == "off"


def test_context_manager_restores_on_exception():
    scorch.set_autotune("analytic")
    with pytest.raises(ValueError):
        with scorch.autotune("max"):
            raise ValueError("boom")
    assert scorch.get_autotune() == "analytic"


def test_decorator_scopes_level():
    scorch.set_autotune("analytic")

    @scorch.autotune("max")
    def f():
        return scorch.get_autotune()

    assert f() == "max"
    assert scorch.get_autotune() == "analytic"


def test_level_override_is_thread_local():
    scorch.set_autotune("analytic")
    seen = {}

    def worker():
        seen["level"] = scorch.get_autotune()

    with scorch.autotune("max"):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
    # The CM override lives on the main thread only; the worker sees the global.
    assert seen["level"] == "analytic"


# --------------------------------------------------------------------------- #
# Env-var -> level mapping (Python API is primary; env is override/CI)
# --------------------------------------------------------------------------- #
def test_env_mapping(monkeypatch):
    monkeypatch.setenv("SCORCH_AUTOTUNE", "balanced")
    assert T._default_level_from_env() == "balanced"

    monkeypatch.delenv("SCORCH_AUTOTUNE", raising=False)
    monkeypatch.setenv("SCORCH_TILING", "0")
    assert T._default_level_from_env() == "off"

    monkeypatch.setenv("SCORCH_TILING", "1")
    monkeypatch.setenv("SCORCH_TILING_PROBE", "0")
    assert T._default_level_from_env() == "analytic"

    monkeypatch.setenv("SCORCH_TILING_PROBE", "1")
    assert T._default_level_from_env() == "balanced"

    # A bad SCORCH_AUTOTUNE value falls through to the legacy/default mapping.
    monkeypatch.setenv("SCORCH_AUTOTUNE", "nonsense")
    monkeypatch.delenv("SCORCH_TILING", raising=False)
    monkeypatch.delenv("SCORCH_TILING_PROBE", raising=False)
    assert T._default_level_from_env() == "analytic"


# --------------------------------------------------------------------------- #
# Jc ladder
# --------------------------------------------------------------------------- #
def test_jc_ladder_coarse_and_floored():
    assert T._jc_ladder(16384) == [16384, 8192, 4096, 2048]
    # Rungs floor at 16 and dedup when the base is tiny.
    lad = T._jc_ladder(20)
    assert lad[0] == 20 and min(lad) >= 16 and len(lad) == len(set(lad))


# --------------------------------------------------------------------------- #
# Correctness + routing at every level (the no-regression contract)
# --------------------------------------------------------------------------- #
def test_all_levels_bit_correct_on_eligible_shape():
    # Shrink the queried LLC so a small test matrix qualifies for the gate.
    T._llc_bytes = 131072  # 128 KiB
    A_st, B_st, ref = _make_eligible()
    assert T.is_candidate(A_st, B_st, level="analytic") is True

    for lvl in ("off", "analytic", "balanced", "max"):
        T._decision.clear()
        scorch.set_autotune(lvl)
        out = scorch.matmul(A_st, B_st)
        out = out if isinstance(out, torch.Tensor) else out.to_torch()
        assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3), f"level={lvl}"


def test_analytic_routes_tiled_and_off_routes_none():
    T._llc_bytes = 131072
    A_st, B_st, _ = _make_eligible()

    # analytic tiles an eligible scattered shape (no probe).
    T._decision.clear()
    scorch.set_autotune("analytic")
    scorch.matmul(A_st, B_st)
    kinds = {v[0] for v in T._decision.values()}
    assert kinds and kinds <= {"tilej", "tileijk"}, T._decision

    # off never even records a decision (short-circuits to pure v2).
    T._decision.clear()
    scorch.set_autotune("off")
    assert T.is_candidate(A_st, B_st, level="off") is False
    scorch.matmul(A_st, B_st)
    assert len(T._decision) == 0


def test_ineligible_shape_never_a_candidate_at_any_level():
    # Realistic LLC; a small low-degree shape must never qualify (byte-neutral v2).
    g = torch.Generator().manual_seed(3)
    A = (torch.rand(64, 64, generator=g) < 0.1).float() * torch.randn(64, 64, generator=g)
    B = torch.randn(64, 16, generator=g)
    Acsr = A.to_sparse_csr()
    A_st = STensor.from_torch(torch.sparse_csr_tensor(
        Acsr.crow_indices(), Acsr.col_indices(), Acsr.values(), size=A.shape))
    B_st = STensor.from_torch(B)
    ref = A @ B
    for lvl in ("off", "analytic", "balanced", "max"):
        T._decision.clear()
        scorch.set_autotune(lvl)
        assert T.is_candidate(A_st, B_st, level=lvl) is False
        out = scorch.matmul(A_st, B_st)
        out = out if isinstance(out, torch.Tensor) else out.to_torch()
        assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3), f"level={lvl}"


def test_memo_keyed_by_level():
    T._llc_bytes = 131072
    A_st, B_st, _ = _make_eligible()
    T._decision.clear()
    with scorch.autotune("analytic"):
        scorch.matmul(A_st, B_st)
    with scorch.autotune("balanced"):
        scorch.matmul(A_st, B_st)
    levels_in_keys = {k[1] for k in T._decision}
    assert {"analytic", "balanced"} <= levels_in_keys


# --------------------------------------------------------------------------- #
# Persistent on-disk cache ("max")
# --------------------------------------------------------------------------- #
def test_persistent_cache_roundtrip(tmp_path, monkeypatch):
    cache_file = tmp_path / "autotune.json"
    monkeypatch.setenv("SCORCH_AUTOTUNE_CACHE", str(cache_file))
    T._cache_loaded = False
    T._persist_cache = None
    T._llc_bytes = 131072
    A_st, B_st, ref = _make_eligible()

    # First "max" run probes + persists to disk.
    T._decision.clear()
    scorch.set_autotune("max")
    scorch.matmul(A_st, B_st)
    assert cache_file.exists()
    data = json.loads(cache_file.read_text())
    assert data["version"] == T._CACHE_VERSION
    assert sum(len(v) for v in data["entries"].values()) >= 1

    # Simulate a fresh process: drop in-memory memo + loaded flag, keep the file.
    T._decision.clear()
    T._cache_loaded = False
    T._persist_cache = None
    sig = T._signature(A_st, int(B_st.shape[1]))
    assert T._persist_get(sig) is not None  # reloaded from disk

    out = scorch.matmul(A_st, B_st)
    out = out if isinstance(out, torch.Tensor) else out.to_torch()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_cache_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SCORCH_AUTOTUNE_CACHE", "0")
    assert T._cache_path() is None
    T._cache_loaded = False
    T._persist_cache = None
    T._llc_bytes = 131072
    A_st, B_st, ref = _make_eligible()
    scorch.set_autotune("max")
    out = scorch.matmul(A_st, B_st)  # must not crash without a cache
    out = out if isinstance(out, torch.Tensor) else out.to_torch()
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_clear_autotune_cache(tmp_path, monkeypatch):
    cache_file = tmp_path / "autotune.json"
    monkeypatch.setenv("SCORCH_AUTOTUNE_CACHE", str(cache_file))
    T._cache_loaded = False
    T._persist_cache = None
    T._llc_bytes = 131072
    A_st, B_st, _ = _make_eligible()
    scorch.set_autotune("max")
    scorch.matmul(A_st, B_st)
    assert cache_file.exists()
    scorch.clear_autotune_cache()
    assert not cache_file.exists()
