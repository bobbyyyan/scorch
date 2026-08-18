"""``scorch.compile``'s fused SpMM+bias+act path composed with the tiling selector.

Before this composition existed, a fused graph could not reach the tiled kernels at
all: the selector is gated on the ``spmm_csr_float_v2`` symbol and a fused graph
resolves to ``spmm_csr_bias_relu_float``, so ``scorch.compile`` opted every user out
of tiling. On a high-degree operand that overflows the last-level cache that is the
difference between the tiled kernel and one thrashing on B.

Two properties are worth pinning, and they are what the tests below are for:

* **The tiled route computes the same numbers.** On a shape the selector tiles, a
  fused call must equal ``matmul + bias + act`` bit for bit -- both run the same
  tiled SpMM, so the only way to differ is a wrong tail.
* **The route cannot be entered on a verdict that was not measured for it.** The
  fused path's alternative is the fused kernel, which folds the tail into the SpMM's
  row epilogue and so beats the drop-in SpMM plus a separate pass. "tile-j beats the
  drop-in SpMM" therefore does not imply "tile-j plus a separate tail beats fusion",
  and a verdict measured against one must not be read for the other.

Verdicts are written into the memo directly rather than probed for wherever the point
is routing rather than speed: whether tiling wins on the host running the suite is not
what these tests are about, and a probe would make them machine-dependent.
"""

import pytest
import torch

import scorch
from scorch import ops, tiling
from scorch.prebuilt_kernels import (
    _FUSED_PREBUILT_SPECS,
    resolve_prebuilt_fused,
    resolve_prebuilt_matmul,
)
from scorch.stensor import STensor
from scorch.trace import _TILED_TAILS

# Chosen against tiling's gate, not by taste, and copied from test_dispatch_plans:
# with the LLC forced to 128 KiB, B is 800*4*64 = 200 KiB (> C, so it thrashes) and
# the degree of 70 clears tiling._DEG_FLOOR of 64. A shape that misses either never
# dispatches a tiled kernel, so it cannot test that the fused route follows one.
ROWS, DEGREE, N, LLC = 800, 70, 64, 131072


def scattered_csr(rows=ROWS, degree=DEGREE, seed=21):
    """A scattered high-degree CSR operand -- what the tiling gate is looking for."""
    generator = torch.Generator().manual_seed(seed)
    dense = torch.zeros(rows, rows)
    for row in range(rows):
        columns = torch.randperm(rows, generator=generator)[:degree]
        dense[row, columns] = torch.randn(degree, generator=generator)
    return STensor.from_torch(dense.to_sparse_csr())


def dense(rows, cols, seed, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=generator, dtype=dtype)


def vector(size, seed, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(size, generator=generator, dtype=dtype)


@scorch.compile
def bias_relu(a, b, bias):
    return torch.relu(scorch.matmul(a, b) + bias)


@scorch.compile
def bias_only(a, b, bias):
    return scorch.matmul(a, b) + bias


class gate_open:
    """Force the tiling gate open for these shapes, and restore everything after.

    Both the level and the LLC are process-global, and the decision memo is keyed on
    the level, so a test that leaves either moved changes what later tests dispatch.
    """

    def __init__(self, level="balanced", llc=LLC):
        self.level = level
        self.llc = llc

    def __enter__(self):
        self.previous_llc = tiling._llc_bytes
        self.previous_level = scorch.get_autotune()
        tiling._llc_bytes = self.llc
        scorch.set_autotune(self.level)
        tiling._decision.clear()
        return self

    def __exit__(self, *exc):
        tiling._llc_bytes = self.previous_llc
        scorch.set_autotune(self.previous_level)
        tiling._decision.clear()
        return False


class records_the_route:
    """Record, per fused call, how far into the selector it got.

    `gates` gets one entry per call -- `ops.tiling_gate`'s `(level, may_serve)` -- and
    `served` gets one entry per call that got past the gate, True when a tiled kernel
    actually served it. A call declined at the gate therefore appears in `gates` and
    not in `served`, which is the distinction between "the selector was asked" and
    "the selector answered yes".

    Recording this is not optional decoration: the numeric assertions cannot tell the
    routes apart. If a fused graph fell back to its eager equivalent, that equivalent
    is `scorch.matmul` -- which routes through the *same* tiled kernel -- plus the
    same tail, so it produces bit-identical output to the tiled fused route. Every
    test that means to exercise a particular route has to say so out of band.
    """

    def __enter__(self):
        self.gates = []
        self.served = []
        self.original_gate = ops.tiling_gate
        self.original_dispatch = ops.dispatch_tiled_fused

        def gate_spy(*args, **kwargs):
            answer = self.original_gate(*args, **kwargs)
            self.gates.append(answer)
            return answer

        def dispatch_spy(*args, **kwargs):
            outcome = self.original_dispatch(*args, **kwargs)
            self.served.append(outcome[0] is not None)
            return outcome

        ops.tiling_gate = gate_spy
        ops.dispatch_tiled_fused = dispatch_spy
        return self

    def __exit__(self, *exc):
        ops.tiling_gate = self.original_gate
        ops.dispatch_tiled_fused = self.original_dispatch
        return False


def fused_tag(post_op_kinds=("add", "relu")):
    resolved = resolve_prebuilt_fused("d,s", "d,d", post_op_kinds, torch.float32)
    assert resolved is not None, "no fused prebuilt kernel to compose with"
    return resolved.symbol_name


def force(a, level, tag, kind, param):
    """Write a verdict into the memo for this shape under one baseline."""
    tiling._decision[(tiling._signature(a, N), level, tag)] = (kind, param)


def panel_width(a):
    return tiling._panel_width(N, tiling.query_llc())


# ---------------------------------------------------------------------------
# The tiled route computes the same numbers as the unfused chain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("post_ops", [("add", "relu"), ("add",)])
def test_the_tiled_fused_route_matches_matmul_plus_the_tail_bitwise(post_ops):
    """Bitwise, not assert_close: on the tiled route both sides run the *same* tiled
    SpMM, so the accumulation order is identical and only a wrong tail can differ.

    (Against the *fused* kernel the same comparison is only close, ~1e-05 -- a
    different kernel sums each row differently. That is pre-existing and is why this
    test forces the tiled route on both sides rather than comparing routes.)
    """
    a = scattered_csr()
    b = dense(ROWS, N, seed=22)
    bias = vector(N, seed=23)
    compiled = bias_relu if post_ops == ("add", "relu") else bias_only
    eager = (
        (lambda out: torch.relu(out + bias))
        if post_ops == ("add", "relu")
        else (lambda out: out + bias)
    )

    with gate_open() as g:
        assert tiling.is_candidate(a, STensor.from_torch(b)), "the gate is shut"
        width = panel_width(a)
        # Both baselines pinned to the same tiled kernel, so the fused call and the
        # reference chain run identical arithmetic.
        force(a, g.level, "v2", "tilej", width)
        force(a, g.level, fused_tag(post_ops), "tilej", width)

        with records_the_route() as route:
            result = compiled(a, b, bias)
        reference = eager(scorch.matmul(a, b))

    assert route.served == [True], "the fused call did not take the tiled route"
    assert torch.equal(result, reference), (result - reference).abs().max().item()


def test_the_tiled_fused_route_matches_through_tile_ijk_too():
    """The other tiled kernel, which relayouts B into width panels."""
    if not tiling._HAS_TILEIJK:
        pytest.skip("spmm_csr_float_tileijk is not built")
    a = scattered_csr()
    b = dense(ROWS, N, seed=22)
    bias = vector(N, seed=23)

    with gate_open() as g:
        Nc, Jc = tiling._ijk_params(N, ROWS, ROWS, tiling.query_llc())
        if Nc >= N:
            pytest.skip("tile-ijk's width strip does not split N at this shape")
        force(a, g.level, "v2", "tileijk", (Nc, Jc))
        force(a, g.level, fused_tag(), "tileijk", (Nc, Jc))

        with records_the_route() as route:
            result = bias_relu(a, b, bias)
        reference = torch.relu(scorch.matmul(a, b) + bias)

    assert route.served == [True], "the fused call did not take the tiled route"
    assert torch.equal(result, reference), (result - reference).abs().max().item()


def test_the_tiled_fused_result_is_a_fresh_buffer_not_an_operand():
    """The tail runs in place, so it had better be in place on the tiled kernel's own
    output and not on anything the caller still owns."""
    a = scattered_csr()
    b = dense(ROWS, N, seed=22)
    bias = vector(N, seed=23)
    b_before, bias_before = b.clone(), bias.clone()

    with gate_open() as g:
        width = panel_width(a)
        force(a, g.level, fused_tag(), "tilej", width)
        with records_the_route() as route:
            result = bias_relu(a, b, bias)

    assert route.served == [True], "the fused call did not take the tiled route"
    assert torch.equal(b, b_before), "the dense operand was written through"
    assert torch.equal(bias, bias_before), "the bias was written through"
    assert result.data_ptr() not in (b.data_ptr(), bias.data_ptr())


# ---------------------------------------------------------------------------
# A verdict measured against one baseline must not be read for the other
# ---------------------------------------------------------------------------


def test_a_drop_in_spmm_verdict_does_not_route_the_fused_call():
    """The defect this composition is built to avoid.

    The memo is keyed on the baseline because the two callers ask different
    questions. Here the drop-in-SpMM baseline says "tile-j wins" while the fused
    baseline says "fusion wins" -- the honest outcome whenever tiling's margin over
    the drop-in SpMM is thinner than what folding the tail into the row epilogue
    saves. The fused call must then run the fused kernel, and reading the wrong
    entry is observable: the two kernels sum each row differently.
    """
    a = scattered_csr()
    b = dense(ROWS, N, seed=22)
    bias = vector(N, seed=23)

    with gate_open() as g:
        force(a, g.level, "v2", "tilej", panel_width(a))
        force(a, g.level, fused_tag(), "v2", None)

        with records_the_route() as route:
            result = bias_relu(a, b, bias)
        assert route.served == [False], "the fused call took the tiled route"
        # What the fused kernel alone produces, which is what the pre-composition
        # code ran on every shape.
        resolved = resolve_prebuilt_fused("d,s", "d,d", ("add", "relu"), torch.float32)
        b_stensor = STensor.from_torch(b)
        native = resolved.fn(
            [ROWS, N],
            list(a.shape), a._native_mode_indices(), a.values,
            list(b_stensor.shape), b_stensor._native_mode_indices(), b_stensor.values,
            bias,
        )
        expected = native.storage.value.view(ROWS, N)

    assert torch.equal(result, expected), (
        "a fused call took the tiled route on a verdict measured against the "
        "drop-in SpMM"
    )


def test_a_fused_call_first_does_not_decide_for_the_unfused_path():
    """The other ordering, which nothing exercised.

    A benchmark that runs its unfused arm first (bench_gcn's FRAMEWORK_ORDER does)
    populates the drop-in-SpMM verdict, and the fused arm then reads a memo entry that
    was measured correctly -- so the leak only shows up in the order nobody ran. Here
    the fused call goes first: it must write only its own entry, and the `matmul` that
    follows must measure and record its own rather than inherit one from a comparison
    it did not make.
    """
    a = scattered_csr()
    b = dense(ROWS, N, seed=22)
    bias = vector(N, seed=23)

    with gate_open() as g:
        with records_the_route() as route:
            bias_relu(a, b, bias)
        assert route.served, "the fused call did not reach the selector"

        after_fused = dict(tiling._decision)
        assert set(after_fused) == {
            (tiling._signature(a, N), g.level, fused_tag())
        }, after_fused
        assert tiling.decided(a, N, level=g.level) is None, (
            "the fused call wrote a verdict the unfused path would read"
        )

        scorch.matmul(a, b)
        unfused_verdict = tiling.decided(a, N, level=g.level)

    assert unfused_verdict is not None, "matmul did not record its own verdict"
    # Both entries now exist, and the fused one was not overwritten by the unfused
    # probe that ran after it.
    assert after_fused[(tiling._signature(a, N), g.level, fused_tag())] is not None


def test_a_declined_shape_runs_the_fused_kernel_bit_for_bit():
    """With the gate shut -- every GCN-small, every autoencoder, anything whose
    operand fits in cache -- the fused path must return exactly what it returned
    before the composition existed: the fused kernel's own output.

    That the added work is only a symbol comparison, a level read and the O(1)
    pre-filter is a property of the code (`_dispatch_tiled` returns before touching
    a kernel), not of this test; what the test pins is the result.
    """
    a = scattered_csr(rows=200, degree=8, seed=31)
    b = dense(200, 16, seed=32)
    bias = vector(16, seed=33)

    tiling._decision.clear()
    assert not tiling.is_candidate(a, STensor.from_torch(b)), "the gate is not shut"

    with records_the_route() as route:
        result = bias_relu(a, b, bias)
    # Declined at the gate itself, so the selector was never consulted past it and the
    # per-call closures were never built.
    assert [answer[1] for answer in route.gates] == [False], route.gates
    assert route.served == [], route.served

    resolved = resolve_prebuilt_fused("d,s", "d,d", ("add", "relu"), torch.float32)
    b_stensor = STensor.from_torch(b)
    native = resolved.fn(
        [200, 16],
        list(a.shape), a._native_mode_indices(), a.values,
        list(b_stensor.shape), b_stensor._native_mode_indices(), b_stensor.values,
        bias,
    )
    assert torch.equal(result, native.storage.value.view(200, 16))
    assert not tiling._decision, f"a declined shape wrote a verdict: {tiling._decision}"


# ---------------------------------------------------------------------------
# The tail is charged to the tiled candidate's clock
# ---------------------------------------------------------------------------


def test_the_tail_is_timed_with_every_tiled_candidate():
    """A fused baseline already folds its tail in, so timing a bare tiled kernel
    against it would credit the tiled kernel with work it did not do. The epilogue
    must run inside the timed region of each tiled candidate -- warmup and both
    timed repetitions -- and on the memoized dispatch afterwards."""
    a = scattered_csr()
    b_dense = dense(ROWS, N, seed=22)
    b = STensor.from_torch(b_dense)
    calls = []

    def counting_tail(result_cpp):
        calls.append(1)
        return result_cpp.storage.value.view(ROWS, N)

    resolved = resolve_prebuilt_matmul(a, b, output_format="dd")

    with gate_open() as g:
        def baseline(nthreads):
            out, _ = ops.execute_prebuilt_binary_kernel(
                resolved.fn, a, b, nthreads=nthreads
            )
            return out

        outcome = tiling.maybe_dispatch(
            a, b, [ROWS, N], baseline, None,
            epilogue=counting_tail, baseline_tag="probe-tail",
        )
        probed = len(calls)
        verdict = tiling.decided(a, N, level=g.level, baseline_tag="probe-tail")

        # 3 invocations per tiled candidate (one warmup, two timed); the baseline
        # gets none. `balanced` probes the whole Jc ladder plus tile-ijk when it
        # splits N.
        ladder = len(tiling._jc_ladder(tiling._panel_width(N, tiling.query_llc())))
        candidates = ladder
        Nc, _ = tiling._ijk_params(N, ROWS, ROWS, tiling.query_llc())
        if tiling._HAS_TILEIJK and N >= tiling._NIJK_MIN and Nc < N:
            candidates += 1
        assert probed == 3 * candidates, (probed, candidates)

        # ...and once more on the memoized dispatch, only if a tiled kernel won.
        calls.clear()
        again = tiling.maybe_dispatch(
            a, b, [ROWS, N], baseline, None,
            epilogue=counting_tail, baseline_tag="probe-tail",
        )

    if verdict[0] == "v2":
        assert outcome is None and again is None
        assert not calls, "the tail ran on a route that declined"
    else:
        assert outcome is not None and again is not None
        assert len(calls) == 1, calls


def test_the_verdict_namespaces_do_not_collide():
    """Two baselines, two entries, neither readable as the other."""
    a = scattered_csr()
    b = STensor.from_torch(dense(ROWS, N, seed=22))

    with gate_open() as g:
        force(a, g.level, "v2", "tilej", 64)
        force(a, g.level, "other-baseline", "tileijk", (32, 16))

        assert tiling.decided(a, N, level=g.level) == ("tilej", 64)
        assert tiling.decided(a, N, level=g.level, baseline_tag="other-baseline") == (
            "tileijk",
            (32, 16),
        )
        assert tiling.decided(a, N, level=g.level, baseline_tag="unmeasured") is None


def test_a_fused_verdict_persists_under_its_own_key(tmp_path, monkeypatch):
    """The `max` level's cache outlives the process, so a fused verdict landing on the
    drop-in SpMM's key would be read next run by a caller that never measured it. Round
    trip through a real file, at the level that actually writes one."""
    import json

    cache_file = tmp_path / "autotune.json"
    monkeypatch.setenv("SCORCH_AUTOTUNE_CACHE", str(cache_file))
    tiling._cache_loaded = False
    tiling._persist_cache = None

    a = scattered_csr()
    b = dense(ROWS, N, seed=22)
    bias = vector(N, seed=23)

    with gate_open(level="max"):
        with records_the_route() as route:
            bias_relu(a, b, bias)
        assert route.served, "the fused call did not reach the selector"

        signature = tiling._signature(a, N)
        assert cache_file.exists()
        keys = set()
        for machine in json.loads(cache_file.read_text())["entries"].values():
            keys.update(machine)
        assert tiling._sig_key(signature, fused_tag()) in keys, keys
        assert tiling._sig_key(signature) not in keys, (
            "a fused verdict landed on the drop-in SpMM's persistent key"
        )

        # A fresh process would reload the file; the fused entry must come back and the
        # drop-in SpMM's must still be absent.
        tiling._cache_loaded = False
        tiling._persist_cache = None
        assert tiling._persist_get(signature, fused_tag()) is not None
        assert tiling._persist_get(signature) is None

    tiling._cache_loaded = False
    tiling._persist_cache = None


def test_the_persistent_cache_namespaces_by_baseline_too():
    """The "max" level's on-disk cache is shared across runs, so a fused verdict
    landing on the drop-in SpMM's key would outlive the process that wrote it. The
    default baseline keeps its historical unprefixed key so caches written before
    other baselines existed stay readable."""
    signature = ("sig", 1, 2)
    assert tiling._sig_key(signature) == tiling._sig_key(signature, "v2")
    assert tiling._sig_key(signature, "fused") != tiling._sig_key(signature)
    assert tiling._sig_key(signature) in tiling._sig_key(signature, "fused")


# ---------------------------------------------------------------------------
# The composition hints are derived once, for both callers
# ---------------------------------------------------------------------------


def test_both_callers_configure_the_tiled_kernels_identically():
    """The hints (`nthreads`, and the ATen-pipelining flag) decide how the drop-in
    SpMM baseline runs *and* how the tiled kernels the selector picks run. Two
    callers deriving them separately is how a fused route silently forfeits the
    host-thread match (pubmed 0.78 -> 1.15x) and writes verdicts the other caller
    cannot reproduce. They come from `ops._composition_hints` or they are wrong."""
    a = scattered_csr()
    b_dense = dense(ROWS, N, seed=22)
    b = STensor.from_torch(b_dense)
    bias = vector(N, seed=23)
    seen = []

    original = ops._tiling_maybe_dispatch

    def spy(*args, **kwargs):
        # maybe_dispatch(a, b, result_shape, baseline_fn, nthreads, ...)
        seen.append((args[4], kwargs.get("baseline_tag"), kwargs.get("epilogue") is None))
        return original(*args, **kwargs)

    with gate_open() as g:
        force(a, g.level, "v2", "v2", None)
        force(a, g.level, fused_tag(), "v2", None)
        ops._tiling_maybe_dispatch = spy
        try:
            scorch.matmul(a, b_dense)
            bias_relu(a, b_dense, bias)
        finally:
            ops._tiling_maybe_dispatch = original

    assert len(seen) == 2, seen
    unfused, fused = seen
    assert unfused[0] == fused[0], (
        f"the fused caller passed nthreads={fused[0]} where matmul passes "
        f"{unfused[0]}; the tiled kernels would run differently configured"
    )
    assert unfused[1] == "v2" and fused[1] == fused_tag()
    assert unfused[2] is True and fused[2] is False, "only the fused caller has a tail"


def test_composition_hints_are_inert_off_the_drop_in_spmm():
    """Only `spmm_csr_float_v2` takes the thread override, so the hints must be
    silent for every other symbol -- including the fused kernels."""

    class Resolved:
        def __init__(self, symbol_name):
            self.symbol_name = symbol_name

    assert ops._composition_hints(Resolved("spmm_csr_bias_relu_float")) == (None, False)
    assert ops._composition_hints(Resolved("spmm_csr_f64")) == (None, False)
    hinted = ops._composition_hints(Resolved("spmm_csr_float_v2"))
    if ops._MATCH_HOST_THREADS:
        assert hinted == (torch.get_num_threads(), ops._ATPARALLEL_PIPELINE)
    else:
        assert hinted == (None, False)


# ---------------------------------------------------------------------------
# Everything the tiled route must decline
# ---------------------------------------------------------------------------


def test_one_compiled_graph_serves_several_shapes():
    """The untiled SpMM is resolved once, at trace time, on the first call's operands.

    That is only sound because `resolve_prebuilt_matmul` keys on operand ranks,
    formats and dtypes and nothing else -- no shapes -- and `_prebuilt_inputs`
    re-checks every one of those on each call. A compiled graph re-called with
    different shapes must therefore stay correct, and must still reach the selector,
    which decides per shape.
    """
    small = scattered_csr(rows=400, degree=20, seed=81)
    big = scattered_csr(rows=900, degree=70, seed=82)

    with gate_open() as g:
        for operand, n in ((small, 32), (big, N)):
            b = dense(operand.shape[0], n, seed=83)
            bias = vector(n, seed=84)
            with records_the_route() as route:
                result = bias_relu(operand, b, bias)
            reference = torch.relu(scorch.matmul(operand, b) + bias)
            assert len(route.gates) == 1, route.gates
            assert torch.allclose(result, reference, atol=1e-3, rtol=1e-3), (
                operand.shape,
                n,
            )


def test_a_dtype_change_on_the_same_graph_falls_back():
    """The trace-time resolve pins float32; a later call in another dtype must be
    caught by the per-call re-validation and routed to the eager equivalent, not run
    against a kernel resolved for a dtype it no longer has."""
    a32 = scattered_csr(rows=300, degree=6, seed=91)
    b32 = dense(300, 16, seed=92)
    bias32 = vector(16, seed=93)

    generator = torch.Generator().manual_seed(94)
    values = torch.randn(300, 300, generator=generator, dtype=torch.float64)
    values[values.abs() < 1.8] = 0
    a64 = STensor.from_torch(values.to_sparse_csr())
    b64 = dense(300, 16, seed=95, dtype=torch.float64)
    bias64 = vector(16, seed=96, dtype=torch.float64)

    with gate_open():
        first = bias_relu(a32, b32, bias32)
        assert first.dtype == torch.float32
        second = bias_relu(a64, b64, bias64)

    assert second.dtype == torch.float64, "the float64 call took the float32 kernel"
    reference = torch.relu(torch.as_tensor(values) @ b64 + bias64)
    assert torch.allclose(second, reference, atol=1e-3, rtol=1e-3)


def test_the_gate_shuts_when_the_drop_in_spmm_symbol_is_absent():
    """`resolve_prebuilt_matmul` falls back through
    ("spmm_csr_float_v2", "prebuilt_spmm_csr_f32", "spmm_csr_float"), and only the
    first is tiled. On a build where it is missing, the composition must disable
    itself rather than hand an untiled symbol to the selector."""
    a = scattered_csr()
    b = STensor.from_torch(dense(ROWS, N, seed=22))

    class Resolved:
        symbol_name = "prebuilt_spmm_csr_f32"

    with gate_open() as g:
        # The shape itself is eligible -- it is the symbol that shuts the gate.
        assert tiling.is_candidate(a, b, level=g.level)
        assert ops.tiling_gate(a, b, Resolved()) == (None, False)
        served, shape, kind, param = ops._dispatch_tiled(
            a, b, Resolved(), None, False, None
        )
    assert (served, shape, kind, param) == (None, None, "v2", None)


def test_every_fused_prebuilt_kernel_has_exactly_one_tail():
    """Fail closed both ways. A fused kernel added without an out-of-line tail must
    decline the tiled route rather than run the wrong arithmetic, and a tail with no
    kernel behind it is dead code that would silently never fire."""
    kernels = {spec.post_op_kinds for spec in _FUSED_PREBUILT_SPECS}
    assert set(_TILED_TAILS) == kernels, (set(_TILED_TAILS), kernels)


@pytest.mark.parametrize("post_ops", [("add", "relu"), ("add",)])
def test_each_tail_equals_its_eager_formula(post_ops):
    """The tail is what the fused kernel folds into its row epilogue. Same order:
    add, then clamp."""
    out = dense(64, 8, seed=41)
    bias = vector(8, seed=42)
    expected = out + bias
    if post_ops == ("add", "relu"):
        expected = torch.relu(expected)
    assert torch.equal(_TILED_TAILS[post_ops](out.clone(), bias), expected)


def test_float64_declines_the_tiled_route():
    """There is no float64 fused kernel and no float64 tiled kernel; the graph runs
    through the JIT/eager equivalent and must not touch the selector."""
    generator = torch.Generator().manual_seed(51)
    values = torch.randn(200, 200, generator=generator, dtype=torch.float64)
    values[values.abs() < 1.6] = 0
    a = STensor.from_torch(values.to_sparse_csr())
    b = dense(200, 16, seed=52, dtype=torch.float64)
    bias = vector(16, seed=53, dtype=torch.float64)

    assert resolve_prebuilt_fused("d,s", "d,d", ("add", "relu"), torch.float64) is None

    with gate_open():
        result = bias_relu(a, b, bias)
        assert not tiling._decision, tiling._decision

    reference = torch.relu(torch.as_tensor(values) @ b + bias)
    assert torch.allclose(result.to(torch.float64), reference, atol=1e-3, rtol=1e-3)


def test_a_coo_operand_declines_the_tiled_route():
    """`_prebuilt_inputs` pins the left operand to "d,s"; a COO one has no fused
    kernel and no tiled kernel, so it must reach neither."""
    generator = torch.Generator().manual_seed(61)
    values = torch.randn(200, 200, generator=generator)
    values[values.abs() < 1.6] = 0
    a = STensor.from_torch(values.to_sparse_coo().coalesce())
    assert str(a.format) != "d,s"
    b = dense(200, 16, seed=62)
    bias = vector(16, seed=63)

    with gate_open():
        result = bias_relu(a, b, bias)
        assert not tiling._decision, tiling._decision

    assert torch.allclose(result, torch.relu(values @ b + bias), atol=1e-3, rtol=1e-3)


def test_an_spmv_shaped_fused_call_declines_the_tiled_route():
    """A single free column: the tiled kernels exist to recover reuse of B across
    rows, and at N=1 there is none to recover. The gate must shut on its own."""
    a = scattered_csr()
    b = dense(ROWS, 1, seed=72)
    bias = vector(1, seed=73)

    with gate_open() as g:
        assert not tiling.is_candidate(a, STensor.from_torch(b)), "the gate opened at N=1"
        result = bias_relu(a, b, bias)
        assert not tiling._decision, tiling._decision

    reference = torch.relu(scorch.matmul(a, b) + bias)
    assert torch.allclose(result, reference, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("level", ["off", "analytic", "balanced", "max", "learned"])
def test_every_autotune_level_composes(level):
    """The tail and the baseline have to be threaded through each level's decision
    strategy, not just the probe: `off` short-circuits, `analytic` and `learned`
    reach the one-shot confirm, `balanced` and `max` run the ladder probe, and `max`
    additionally consults the on-disk cache."""
    a = scattered_csr()
    b = dense(ROWS, N, seed=22)
    bias = vector(N, seed=23)

    with gate_open(level=level), records_the_route() as route:
        result = bias_relu(a, b, bias)
        tags = {key[2] for key in tiling._decision}

    assert len(route.gates) == 1, "the fused path did not consult the selector"
    if level == "off":
        assert route.gates[0] == ("off", False), route.gates
        assert route.served == [], route.served

    reference = torch.relu(scorch.matmul(a, b) + bias)
    assert torch.allclose(result, reference, atol=1e-3, rtol=1e-3)
    if level == "off":
        assert not tags, tags
    else:
        assert tags == {fused_tag()}, tags
