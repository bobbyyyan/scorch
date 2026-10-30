"""The tiling probe's own noise floor, and the candidate order it measures in.

Two defects in one routine, both about measurement rather than about kernels.

*Order.* The probe used to time each candidate to completion before starting the
next, and the caller's baseline is always the first candidate. So the baseline was
the one arm that never ran on a machine an earlier candidate had warmed, in the
routine whose entire purpose is to guarantee we never lose to the baseline. The
one-shot confirm had the same bug pointing the other way -- it timed the tiled
candidate first and the baseline second, biasing *towards* the baseline -- and
neither bias had ever been measured.

*Floor.* Nothing measured the noise. The probe took a min of two timings per
candidate and then committed the verdict to a memo for the life of the process,
and at the "max" level to a file for the life of the machine. A cell whose true
margin sits inside the run-to-run spread therefore got a permanent answer from a
coin flip. The fix gives the probe an A/A control -- the baseline entered twice,
at both ends of the candidate list, same function and same arguments -- and
requires a tiled candidate to beat the baseline by more than the gap between
those two identical arms. Inside the floor it fails closed to the baseline.

The timing-dependent tests here inject times rather than measuring them. Whether
a tiled kernel wins on the host running the suite is not what these are about,
and measuring would make them machine-dependent and flaky in exactly the regime
they exist to pin down.
"""

import json
import os
import sys
import warnings

import pytest
import torch

import scorch
from scorch import tiling
from scorch.prebuilt_kernels import (
    execute_prebuilt_binary_kernel,
    resolve_prebuilt_matmul,
)
from scorch.stensor import STensor

ROWS, DEGREE, N, LLC = 800, 70, 64, 131072


def scattered_csr(rows=ROWS, degree=DEGREE, seed=21):
    """A scattered high-degree CSR operand -- what the tiling gate is looking for."""
    generator = torch.Generator().manual_seed(seed)
    dense = torch.zeros(rows, rows)
    for row in range(rows):
        columns = torch.randperm(rows, generator=generator)[:degree]
        dense[row, columns] = torch.randn(degree, generator=generator)
    return STensor.from_torch(dense.to_sparse_csr())


class gate_open:
    """Force the tiling gate open for these shapes, and restore everything after."""

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


class injects_times:
    """Replace the interleaved timer with fixed times, and record what it was given.

    `fns` is the candidate list the probe built, so a test can assert its shape --
    how many entries, and which of them call the baseline -- as well as what the
    probe decided from the times.
    """

    def __init__(self, times_for):
        self.times_for = times_for
        self.fns = None
        self.calls = 0

    def __enter__(self):
        self.original = tiling._interleaved_times

        def fake(fns, rounds=2):
            self.fns = list(fns)
            self.calls += 1
            return self.times_for(len(fns))

        tiling._interleaved_times = fake
        return self

    def __exit__(self, *exc):
        tiling._interleaved_times = self.original
        return False


# ---------------------------------------------------------------------------
# _clears_noise: the floor test itself
# ---------------------------------------------------------------------------
def test_a_win_larger_than_the_floor_is_accepted():
    assert tiling._clears_noise(0.70, 1.00, 1.07)


def test_a_win_inside_the_floor_is_rejected():
    # 5% faster than the baseline, but two timings of the baseline itself differ
    # by 7% -- the win is not distinguishable from the noise that produced it.
    assert not tiling._clears_noise(0.95, 1.00, 1.07)


def test_the_floor_does_not_depend_on_which_arm_was_faster():
    # The two baseline entries are the same function; which one happened to be
    # quicker is noise, so the verdict must not turn on it.
    assert tiling._clears_noise(0.70, 1.07, 1.00) == tiling._clears_noise(
        0.70, 1.00, 1.07
    )
    assert tiling._clears_noise(0.95, 1.07, 1.00) == tiling._clears_noise(
        0.95, 1.00, 1.07
    )


def test_a_candidate_slower_than_the_baseline_is_rejected():
    assert not tiling._clears_noise(1.20, 1.00, 1.00)


def test_a_perfect_floor_still_requires_beating_the_baseline():
    assert not tiling._clears_noise(1.00, 1.00, 1.00)
    assert tiling._clears_noise(0.99, 1.00, 1.00)


def test_the_baseline_is_the_faster_of_the_two_identical_arms():
    # Given arms at 1.00 and 1.20, a candidate at 1.05 beats the slower arm but
    # not the faster one, and must lose. Taking the mean or the slower arm as the
    # baseline would ship it.
    assert not tiling._clears_noise(1.05, 1.00, 1.20)


# ---------------------------------------------------------------------------
# _interleaved_times: warm-then-rotate, and the call budget
# ---------------------------------------------------------------------------
def test_every_candidate_is_warmed_before_any_is_timed():
    order = []
    fns = [(lambda i=i: order.append(i)) for i in range(4)]
    tiling._interleaved_times(fns, rounds=2)
    # The first four calls are one per candidate: no candidate is timed until all
    # of them have run once, so none is timed on a machine only it has warmed.
    assert sorted(order[:4]) == [0, 1, 2, 3]


def test_the_call_budget_is_one_warmup_plus_one_call_per_round():
    counts = [0, 0, 0]

    def bump(i):
        counts[i] += 1

    fns = [(lambda i=i: bump(i)) for i in range(3)]
    tiling._interleaved_times(fns, rounds=2)
    # Same budget the sequential version spent: 1 warmup + 2 timed, per candidate.
    assert counts == [3, 3, 3]


def test_no_candidate_holds_the_first_timed_slot_in_every_round():
    order = []
    fns = [(lambda i=i: order.append(i)) for i in range(4)]
    tiling._interleaved_times(fns, rounds=2)
    timed = order[4:]
    firsts = [timed[0], timed[4]]
    assert firsts[0] != firsts[1], "the start of each round must rotate"


class FakeClock:
    """A clock that advances only when a candidate "runs".

    Each candidate carries a per-call cost list, so one can be made slow on its
    first timed call and quick afterwards -- the cold-start shape the rotation
    exists to neutralize. Faking the clock rather than sleeping makes these exact
    and instant; a test that spins on the wall clock to assert something about a
    timing routine is measuring the machine it runs on, which is the thing this
    file is about not doing.
    """

    def __init__(self, costs):
        self.t = 0.0
        self.costs = {i: list(c) for i, c in enumerate(costs)}
        self.calls = []

    def fn(self, i):
        def run():
            self.calls.append(i)
            seq = self.costs[i]
            self.t += seq.pop(0) if len(seq) > 1 else seq[0]
            return ("result", i)

        return run

    def fns(self):
        return [self.fn(i) for i in sorted(self.costs)]


@pytest.fixture
def clocked(monkeypatch):
    def make(costs):
        clock = FakeClock(costs)
        monkeypatch.setattr(tiling.time, "perf_counter", lambda: clock.t)
        return clock

    return make


def test_the_minimum_over_rounds_is_returned(clocked):
    # A candidate slow once and quick once is scored on the quick one, so a single
    # descheduled call cannot condemn it.
    clock = clocked([[9.0, 1.0], [2.0], [2.0]])
    times = tiling._interleaved_times(clock.fns(), rounds=2)
    assert times[0] == pytest.approx(1.0)


def test_a_cold_first_call_does_not_decide_the_winner(clocked):
    """Candidate 0 is really the fastest but is slow the first time it is timed.

    That is the shape the whole change is about: under the scheme that shipped,
    the arm timed first ran on the coldest machine, and if one slow sample settled
    it then the first-timed arm lost on position rather than on merit.
    """
    clock = clocked([[9.0, 1.0], [2.0], [2.0]])
    times = tiling._interleaved_times(clock.fns(), rounds=2)
    assert min(range(3), key=lambda i: times[i]) == 0


def test_the_warmup_call_is_not_charged_to_any_candidate(clocked):
    """The first call to each candidate is a warmup and must not be timed.

    With three candidates each costing 1.0 per call and two rounds, every
    candidate's reported time is one call's worth -- not two, and not three.
    """
    clock = clocked([[1.0], [1.0], [1.0]])
    times = tiling._interleaved_times(clock.fns(), rounds=2)
    assert times == [pytest.approx(1.0)] * 3


# ---------------------------------------------------------------------------
# the ladder probe
# ---------------------------------------------------------------------------
def _probe(times_for, level="balanced"):
    """Run the balanced/max ladder probe with injected times; return (result, memo)."""
    a = scattered_csr()
    b = torch.randn(ROWS, N)
    bst = STensor.from_torch(b)
    calls = {"baseline": 0}

    def baseline_fn(nthreads):
        calls["baseline"] += 1
        return scorch.matmul(a, bst)

    with gate_open(level=level):
        with injects_times(times_for) as injected:
            out = tiling.maybe_dispatch(
                a, bst, [ROWS, N], baseline_fn, None, level=level
            )
        memo = dict(tiling._decision)
    return out, memo, injected, calls


def test_the_baseline_is_a_candidate_at_both_ends_of_the_list():
    # first and last entries are the same function object: the A/A control.
    _out, _memo, injected, _calls = _probe(lambda n: [1.0] * n)
    assert len(injected.fns) >= 3
    assert injected.fns[0] is injected.fns[-1]


def test_the_probe_declines_a_tiled_win_inside_its_own_floor():
    # baseline arms 1.00 and 1.08 (an 8% floor); best tiled 0.95 (a 5% win).
    def times(n):
        return [1.00] + [0.95] * (n - 2) + [1.08]

    out, memo, _injected, _calls = _probe(times)
    assert out is None, "declining must return None so the caller runs its baseline"
    assert set(memo.values()) == {("v2", None)}


def test_the_probe_takes_a_tiled_win_that_clears_its_floor():
    def times(n):
        return [1.00] + [0.50] * (n - 2) + [1.08]

    out, memo, _injected, _calls = _probe(times)
    assert out is not None and out[1] is True
    assert list(memo.values())[0][0] in ("tilej", "tileijk")


def test_a_baseline_arm_can_never_be_the_winner():
    # Both baseline entries fastest by far; the tiled ones lose. The verdict must
    # be v2, and must not name a tiled kernel just because a control was quick.
    def times(n):
        return [0.10] + [5.0] * (n - 2) + [0.10]

    out, memo, _injected, _calls = _probe(times)
    assert out is None
    assert set(memo.values()) == {("v2", None)}


def test_the_winner_is_re_run_for_its_output_not_retained():
    # The interleaved timer returns times only, so the probe must call the winner
    # once more to get a result -- and that result must be a real tensor.
    def times(n):
        return [1.00] + [0.10] * (n - 2) + [1.00]

    out, _memo, _injected, _calls = _probe(times)
    assert out is not None
    result, used_tiled = out
    assert used_tiled is True
    assert result is not None


def test_the_declined_verdict_is_memoized_so_the_floor_is_paid_once():
    a = scattered_csr()
    bst = STensor.from_torch(torch.randn(ROWS, N))

    def baseline_fn(nthreads):
        return scorch.matmul(a, bst)

    def times(n):
        return [1.00] + [0.95] * (n - 2) + [1.08]

    with gate_open(level="balanced"):
        with injects_times(times) as injected:
            first = tiling.maybe_dispatch(a, bst, [ROWS, N], baseline_fn, None)
            second = tiling.maybe_dispatch(a, bst, [ROWS, N], baseline_fn, None)
        assert first is None and second is None
        # One probe, not two: the second call reads the memo.
        assert injected.calls == 1


# ---------------------------------------------------------------------------
# a candidate that raises
# ---------------------------------------------------------------------------
class PlantedFailure(Exception):
    pass


class tilej_raises:
    """Make the tile-j candidate at panel width ``width`` (every width when None)
    raise on its runs from the ``after``-th on (0: from its first). ``runs`` counts
    every run, per width. It works through _tilej_args, which every tile-j run calls:
    the probe's candidates, the winner's run for its output, and a memoized dispatch."""

    def __init__(self, width=None, after=0):
        self.width = width
        self.after = after
        self.runs = {}

    def __enter__(self):
        self.original = tiling._tilej_args

        def fake(a, b, result_shape, jc, nthreads):
            n = self.runs.get(jc, 0)
            self.runs[jc] = n + 1
            if (self.width is None or jc == self.width) and n >= self.after:
                raise PlantedFailure(f"planted at width {jc}")
            return self.original(a, b, result_shape, jc, nthreads)

        tiling._tilej_args = fake
        return self

    def __exit__(self, *exc):
        tiling._tilej_args = self.original
        return False


class runs_each_once:
    """As injects_times, but the fake timer runs every candidate once, as the real
    one's warm-up does, so a candidate that raises is found."""

    def __init__(self, times_for):
        self.times_for = times_for
        self.fns = None

    def __enter__(self):
        self.original = tiling._interleaved_times

        def fake(fns, rounds=2):
            self.fns = list(fns)
            for fn in self.fns:
                fn()
            return self.times_for(len(self.fns))

        tiling._interleaved_times = fake
        return self

    def __exit__(self, *exc):
        tiling._interleaved_times = self.original
        return False


@pytest.fixture
def own_max_file(tmp_path, monkeypatch):
    """Max's file in this test's directory, read afresh. Returns its path."""
    path = tmp_path / "autotune.json"
    monkeypatch.setenv("SCORCH_AUTOTUNE_CACHE", str(path))
    monkeypatch.setattr(tiling, "_cache_loaded", False)
    monkeypatch.setattr(tiling, "_persist_cache", None)
    return path


def _direct_operands():
    """The probe's operands, with a baseline that calls the drop-in SpMM directly.
    A baseline calling scorch.matmul would reach the selector again from inside the
    probe, and with a timer that runs the candidates the memo is still empty there."""
    a = scattered_csr()
    bst = STensor.from_torch(torch.randn(ROWS, N))
    resolved = resolve_prebuilt_matmul(a, bst, output_format="dd")

    def baseline_fn(nthreads):
        out, _ = execute_prebuilt_binary_kernel(resolved.fn, a, bst, nthreads=nthreads)
        return out

    return a, bst, baseline_fn


def _ladder():
    return tiling._jc_ladder(tiling._panel_width(N, LLC))


def _selector_warnings(caught):
    return [w for w in caught if issubclass(w.category, RuntimeWarning)]


@pytest.mark.parametrize("level", ("balanced", "max"))
def test_every_tiled_candidate_raising_gives_v2_once(level, own_max_file):
    a, bst, baseline_fn = _direct_operands()
    with gate_open(level=level), tilej_raises() as raises:
        # The raising candidates are planted fastest, so only leaving them out gives v2.
        with runs_each_once(lambda n: [1.0] + [0.1] * (n - 2) + [1.0]):
            with pytest.warns(RuntimeWarning, match="PlantedFailure") as caught:
                out = tiling.maybe_dispatch(a, bst, [ROWS, N], baseline_fn, None)
        (warning,) = _selector_warnings(caught)
        assert warning.filename == __file__, "the warning names a library frame"
        assert out is None, "the call must be served by the caller's baseline"
        assert set(tiling._decision.values()) == {("v2", None)}
        runs = dict(raises.runs)
        assert set(runs) == set(_ladder()) and set(runs.values()) == {1}
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            again = tiling.maybe_dispatch(a, bst, [ROWS, N], baseline_fn, None)
        assert again is None
        assert raises.runs == runs, "a later call ran a tiled candidate"
        assert tiling._persist_get(tiling._signature(a, N)) is None
    assert not own_max_file.exists(), "a verdict from a raising probe was written"


def test_the_raising_candidate_is_left_out_and_the_next_best_wins():
    a, bst, baseline_fn = _direct_operands()
    ladder = _ladder()
    assert len(ladder) >= 2, "the shape needs two tile-j widths"
    # Candidates in order: base, ladder[0], ladder[1], ..., base'. ladder[0] is
    # planted fastest and raises; ladder[1] clears the floor.
    planted = [1.0, 0.1, 0.5] + [0.9] * (len(ladder) - 2) + [1.0]
    with gate_open(level="balanced"), tilej_raises(width=ladder[0]):
        with runs_each_once(lambda n: planted):
            with pytest.warns(RuntimeWarning, match=f"width {ladder[0]}") as caught:
                out = tiling.maybe_dispatch(a, bst, [ROWS, N], baseline_fn, None)
        assert len(_selector_warnings(caught)) == 1
        assert f"width {ladder[1]}" not in str(_selector_warnings(caught)[0].message)
        assert out is not None and out[1] is True
        assert tiling.decided(a, N, level="balanced") == ("tilej", ladder[1])


def test_a_winner_that_raises_when_it_runs_for_its_output_gives_v2(own_max_file):
    a, bst, baseline_fn = _direct_operands()
    ladder = _ladder()
    # ladder[0] wins, runs once in the timer, and raises on its run for the output.
    planted = [1.0, 0.1] + [0.9] * (len(ladder) - 1) + [1.0]
    with gate_open(level="max"), tilej_raises(width=ladder[0], after=1) as raises:
        with runs_each_once(lambda n: planted):
            with pytest.warns(RuntimeWarning, match="PlantedFailure") as caught:
                out = tiling.maybe_dispatch(a, bst, [ROWS, N], baseline_fn, None)
        assert len(_selector_warnings(caught)) == 1
        assert raises.runs[ladder[0]] == 2
        assert out is None
        assert set(tiling._decision.values()) == {("v2", None)}
    assert not own_max_file.exists(), "a verdict from a raising probe was written"


def test_a_probe_where_nothing_raised_still_writes_max_s_file(own_max_file):
    # The control for the two "not written" checks above: the same probe, nothing
    # raising, does write the file, so their absence of a file is informative.
    a, bst, baseline_fn = _direct_operands()
    with gate_open(level="max"):
        with runs_each_once(lambda n: [1.0] + [0.1] * (n - 2) + [1.0]):
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                out = tiling.maybe_dispatch(a, bst, [ROWS, N], baseline_fn, None)
    assert out is not None and out[1] is True
    assert own_max_file.exists()


@pytest.mark.parametrize("level", ("analytic", "balanced"))
def test_a_raising_baseline_raises_and_memoizes_nothing(level):
    a, bst, _baseline_fn = _direct_operands()

    def baseline_fn(nthreads):
        raise PlantedFailure("planted in the baseline")

    with gate_open(level=level):
        with runs_each_once(lambda n: [1.0] * n):
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                with pytest.raises(PlantedFailure):
                    tiling.maybe_dispatch(a, bst, [ROWS, N], baseline_fn, None)
        assert tiling._decision == {}


@pytest.fixture
def learned_prefers_tilej(tmp_path, monkeypatch):
    """Returns a function that writes a learned model under which tile-j is predicted
    fastest (test_autotune_learned.py's _PREFER_TILEJ) and loads it. Call it inside
    gate_open: the model's machine id hashes the LLC in force. The loaded model is
    dropped afterwards, so later tests read the session's (absent) one."""
    path = tmp_path / "learned_model.json"

    def write():
        fi = list(tiling._FEATURES).index("f_is_tilej")
        tree = dict(
            feature=[fi, -2, -2],
            threshold=[0.5, -2.0, -2.0],
            left=[1, -1, -1],
            right=[2, -1, -1],
            value=[0.0, 1.0, -1.0],
        )
        spec = dict(
            kind="sklearn_gbr",
            version=tiling._LEARNED_VERSION,
            machine_id=tiling._machine_id(),
            feature_names=list(tiling._FEATURES),
            init=0.0,
            learning_rate=1.0,
            trees=[tree],
        )
        path.write_text(json.dumps(spec))
        monkeypatch.setenv("SCORCH_AUTOTUNE_MODEL", str(path))
        tiling._reset_learned_model_cache()
        assert tiling._load_learned_model() is not None

    yield write
    tiling._reset_learned_model_cache()


@pytest.mark.parametrize("level", ("analytic", "learned"))
@pytest.mark.parametrize("after", (0, 1), ids=("in_the_confirm", "after_the_confirm"))
def test_a_confirmed_pick_that_raises_gives_v2_once(
    after, level, learned_prefers_tilej
):
    """analytic's (the default level's) or learned's pick raises while the confirm
    times it (after=0), or on its run for this call's output once it has cleared the
    confirm (after=1). learned runs with a model that picks tile-j, since without one
    it is analytic."""
    a, bst, baseline_fn = _direct_operands()
    with gate_open(level=level), tilej_raises(after=after) as raises:
        if level == "learned":
            learned_prefers_tilej()
        with runs_each_once(lambda n: [1.0, 0.5, 1.0]) as timer:
            with pytest.warns(RuntimeWarning, match="PlantedFailure") as caught:
                out = tiling.maybe_dispatch(a, bst, [ROWS, N], baseline_fn, None)
        assert len(timer.fns) == 3, "not the confirm's [base, pick, base]"
        (warning,) = _selector_warnings(caught)
        assert warning.filename == __file__, "the warning names a library frame"
        assert out is None
        assert set(tiling._decision.values()) == {("v2", None)}
        runs = dict(raises.runs)
        assert sum(runs.values()) == after + 1
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert tiling.maybe_dispatch(a, bst, [ROWS, N], baseline_fn, None) is None
        assert raises.runs == runs, "a later call ran the pick"


def test_through_scorch_matmul_the_warning_names_the_callers_line():
    """Through ops.matmul the selector sits several scorch frames below the caller,
    and the warning still names the caller's line. The call returns v2's product."""
    a, bst, _baseline_fn = _direct_operands()
    # in_place=False: to_torch's default densifies the STensor itself, and a dense `a`
    # takes ops.matmul's dense x dense route, which never reaches the selector.
    want = a.to_torch(in_place=False) @ bst.to_torch(in_place=False)
    assert not a.format.is_dense(), "the operand is dense, so no SpMM reaches the gate"
    with gate_open(level="analytic"), tilej_raises() as raises:
        with runs_each_once(lambda n: [1.0, 0.5, 1.0]):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                line = sys._getframe().f_lineno + 1
                got = scorch.matmul(a, bst)
        # The route first, so a call that never reached the selector says so.
        assert sum(raises.runs.values()) == 1, "the call never ran the analytic pick"
        (warning,) = _selector_warnings(caught)
        assert "PlantedFailure" in str(warning.message)
        assert (warning.filename, warning.lineno) == (__file__, line)
        assert set(tiling._decision.values()) == {("v2", None)}
    got = got if isinstance(got, torch.Tensor) else got.to_torch(in_place=False)
    torch.testing.assert_close(got, want, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# the one-shot confirm (the analytic / learned levels)
# ---------------------------------------------------------------------------
def _confirm(times):
    a = scattered_csr()
    bst = STensor.from_torch(torch.randn(ROWS, N))
    seen = {}

    def baseline_fn(nthreads):
        return scorch.matmul(a, bst)

    original = tiling._interleaved_times

    def fake(fns, rounds=2):
        seen["fns"] = list(fns)
        return times

    tiling._interleaved_times = fake
    try:
        with gate_open(level="analytic"):
            verdict = tiling._confirm_vs_baseline(
                a, bst, [ROWS, N], "tilej", 64, -1, baseline_fn, None
            )
    finally:
        tiling._interleaved_times = original
    return verdict, seen["fns"]


def test_the_confirm_brackets_the_candidate_between_two_baseline_arms():
    _verdict, fns = _confirm([1.0, 1.0, 1.0])
    assert len(fns) == 3
    assert fns[0] is fns[2], "the same baseline function at both ends"
    assert fns[1] is not fns[0], "the tiled candidate in the middle"


def test_the_confirm_declines_inside_its_floor():
    # arms 1.00 and 1.08, candidate 0.95: a 5% win against an 8% floor.
    verdict, _fns = _confirm([1.00, 0.95, 1.08])
    assert verdict == ("v2", None)


def test_the_confirm_accepts_a_win_that_clears_its_floor():
    verdict, _fns = _confirm([1.00, 0.50, 1.08])
    assert verdict == ("tilej", 64)


def test_the_confirm_declines_a_candidate_slower_than_the_baseline():
    verdict, _fns = _confirm([1.00, 1.50, 1.00])
    assert verdict == ("v2", None)


# ---------------------------------------------------------------------------
# the kernel gate and the selector gate must agree about the machine
# ---------------------------------------------------------------------------
def test_the_kernel_and_the_selector_read_the_same_cache_size():
    """Two layers answer the same question about the machine; pin that they agree.

    `tiling.query_llc` decides whether a product is eligible for a tiled kernel at
    all, and `scorch_llc_bytes` is what a harness or a future kernel gate reads.
    They query the same sysctl keys on macOS and the same sysfs entries on Linux,
    honour the same SCORCH_LLC_BYTES override and fall back to the same per-platform
    default -- but nothing enforced it, and neither layer would report a
    disagreement. The fallbacks in fact differed (16 vs 36 MiB on Darwin) until this
    test was written, which is exactly the case where agreement matters most: when
    the query fails, the fallback is all the two have to agree on.
    """
    scorch_ops = pytest.importorskip("scorch_ops")
    if not hasattr(scorch_ops, "scorch_llc_bytes"):
        pytest.skip("extension predates the scorch_llc_bytes binding")
    assert scorch_ops.scorch_llc_bytes() == tiling.query_llc()


def test_both_cache_queries_honour_the_same_override():
    import os
    import subprocess
    import sys

    scorch_ops = pytest.importorskip("scorch_ops")
    if not hasattr(scorch_ops, "scorch_llc_bytes"):
        pytest.skip("extension predates the scorch_llc_bytes binding")
    # The C++ side caches on first call and the Python side caches in a module
    # global, so the override has to be set before either runs -- hence a
    # subprocess rather than monkeypatching the environment in place.
    env = dict(os.environ, SCORCH_LLC_BYTES="1048576")
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            # torch first: scorch_ops links libc10, and on Linux importing it
            # first fails with "libc10.so: cannot open shared object file".
            "import torch;"
            "import scorch_ops, scorch.tiling as t;"
            "print(scorch_ops.scorch_llc_bytes(), t.query_llc())",
        ],
        env=env,
        text=True,
    ).split()
    assert out == ["1048576", "1048576"]


# ---------------------------------------------------------------------------
# the SpMM thread policy's new argument, and what the shipped build must ignore
# ---------------------------------------------------------------------------
def test_passing_nnz_to_the_thread_policy_changes_nothing_by_default(torch_threads):
    """The rule gained an `nnz` argument; at the shipped constants it must be inert.

    `scorch_spmm_nthreads` takes the nonzero count so that the row-axis ceiling can be
    expressed in nonzeros per worker instead of rows per worker -- a 71-row matrix
    holding 212536 nonzeros gets four workers from `rows / 16` on a 24-core host, and
    the raise that was meant to cover that is bounded by `nnz * k`, which at k=2 is one
    grain. But the ceiling is off by default (`SCORCH_SPMM_NNZ_PER_THREAD` is 0), so
    passing the count must not move a single answer.

    Worth pinning rather than reading off: the four-argument form is what every
    existing caller and harness uses, and a silent change there would reattribute every
    thread count in the ledger to a rule that had not shipped.
    """
    scorch_ops = pytest.importorskip("scorch_ops")
    # The overrides go up to 24; with torch at 1 every count is 1 and the two forms cannot differ.
    torch_threads(24)
    shapes = [
        (71, 212536), (73, 904910), (64, 18432), (256, 294912),
        (2708, 13264), (19717, 108365), (8081, 6323), (232965, 11461589),
        (1, 1), (0, 0),
    ]
    for rows, nnz in shapes:
        for k in (1, 2, 3, 7, 8, 16, 64, 512):
            work = nnz * max(k, 16)
            work_true = nnz * k
            for override in (-1, 0, 6, 24):
                short = scorch_ops.scorch_spmm_nthreads(work, rows, override, work_true)
                full = scorch_ops.scorch_spmm_nthreads(
                    work, rows, override, work_true, nnz
                )
                assert short == full, (
                    "rows=%d nnz=%d k=%d override=%d: %d without nnz, %d with it"
                    % (rows, nnz, k, override, short, full)
                )


def test_the_thread_policy_never_exceeds_the_machine_or_the_rows(torch_threads):
    """Two bounds the rule must hold whatever the arguments say.

    A count above `omp_get_num_procs()` oversubscribes the box; a count above `rows`
    asks for workers that row-parallel decomposition cannot feed, and the partition
    would hand those ranges zero rows. Both are cheap to state and neither was pinned,
    which matters now that three separate paths (the base proxy, the raise, and the
    composition adoption) can each be the one that sets the answer.
    """
    scorch_ops = pytest.importorskip("scorch_ops")
    # The override is 64; with torch at 1 every count is 1, under every bound below.
    torch_threads(64)
    # The hardware bound the rule uses is omp_get_num_procs(), which is not bound to
    # Python. os.cpu_count() is the same number on both hosts here (it counts logical
    # processors), and using it as an UPPER bound is safe either way: if the two ever
    # disagreed, the rule's own cap is the smaller of them.
    procs = os.cpu_count()
    for rows, nnz in [(1, 4096), (2, 1 << 20), (5, 1 << 22), (71, 212536),
                      (256, 294912), (100000, 1 << 24)]:
        for k in (1, 8, 64, 512):
            n = scorch_ops.scorch_spmm_nthreads(
                nnz * max(k, 16), rows, 64, nnz * k, nnz
            )
            assert n >= 1
            assert n <= max(rows, 1), (
                "rows=%d nnz=%d k=%d resolved %d workers for %d rows"
                % (rows, nnz, k, n, rows)
            )
            if procs is not None:
                assert n <= procs
