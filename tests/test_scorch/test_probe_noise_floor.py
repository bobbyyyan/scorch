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

import pytest
import torch

import scorch
from scorch import tiling
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


def test_the_minimum_over_rounds_is_returned():
    # A candidate that is slow once and quick once is scored on the quick one, so
    # a single descheduled call cannot condemn it.
    state = {"n": 0}

    def alternating():
        state["n"] += 1
        if state["n"] == 2:
            _spin(0.02)

    def quick():
        pass

    times = tiling._interleaved_times([alternating, quick], rounds=3)
    assert times[0] < 0.02


def _spin(seconds):
    import time

    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


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
    """Both layers gate on the last-level cache; a disagreement is invisible.

    `tiling.query_llc` decides whether a product is eligible for a tiled kernel at
    all, and the kernels' own `scorch_llc_bytes` decides whether the wide path may
    stream its stores. They query the same sysctl keys on macOS and the same sysfs
    entries on Linux, and honour the same SCORCH_LLC_BYTES override, so they should
    return one number -- but nothing enforced it, and neither layer would report a
    disagreement. It would show up only as a product routed to a tiled kernel that
    then declines to stream, or the reverse.
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
            "import scorch_ops, scorch.tiling as t;"
            "print(scorch_ops.scorch_llc_bytes(), t.query_llc())",
        ],
        env=env,
        text=True,
    ).split()
    assert out == ["1048576", "1048576"]
