"""The SpMM physical-core cap: what it counts, and that it can only ever lower a count.

The cap exists because a sparse row loop is bound by memory rather than issue width, so a
second thread on the same physical core adds contention without adding bandwidth. It feeds
``SCORCH_SPMM_NT_CAP`` rather than acting on its own, so it inherits that block's measured
recruit decline; it ships OFF (``SCORCH_SPMM_NT_CAP_PHYS`` defaults to 0).

What these pin down:

* the effective cap is the SMALLER of the caller's pool and the physical core count, so the
  knob can never RAISE a worker count;
* the count read is physical cores in the process's affinity mask, which is NOT
  ``scorch_pcore_count()`` -- the two are exported separately because they are easy to
  conflate and differ by exactly the factor that matters inside a cgroup;
* with the knob off, the forced core count changes nothing.

The forced-value cases need a fresh interpreter because both knobs are read once and cached
-- deliberately, since reading sysfs per SpMM call would cost more than the cap saves.
"""
import os
import subprocess
import sys

import pytest

scorch_ops = pytest.importorskip("scorch_ops")

pytestmark = pytest.mark.skipif(
    not hasattr(scorch_ops, "scorch_phys_cores_avail"),
    reason="scorch_ops predates the physical-core cap",
)

# Two products, because the cap DECLINES itself in one regime and that is deliberate.
#
# The cap feeds SCORCH_SPMM_NT_CAP, which refuses to apply where it would drop the resolved
# count below twice the caller's pool -- because that threshold is what routes the drop-in and
# fused SpMM kernels onto their own team to reach cores the pool excludes (Apple silicon's 12
# E-cores sit outside torch's 6-thread pool), worth a measured 2.18x on the largest
# autoencoder bucket. Inheriting that decline is the whole reason the cap lives inside that
# block rather than in front of it.
#
# So a binding test has to use a product whose resolved count is BELOW 2x the pool, which is
# also the regime the cap exists for: a pool sized from logical CPUs, not a hybrid pool
# reaching past itself.
_MODERATE = "512, 209715, 8"      # rows, nnz, k -- resolves at or under the pool
_HUGE = "100000, 10000000, 16"    # resolves past 2x the pool on a hybrid host

_PROBE = """
import torch, scorch_ops as so
rows, nnz, k = %s
print(so.scorch_phys_cores_avail(),
      so.scorch_spmm_nthreads(nnz * max(k, 16), rows, torch.get_num_threads(),
                              nnz * k, nnz))
"""


def _resolve(phys=None, cap=0, shape=_MODERATE):
    """(reported_core_count, resolved_workers) from a fresh process, knobs preset."""
    env = dict(os.environ)
    env["SCORCH_SPMM_NT_CAP_PHYS"] = str(cap)
    if phys is None:
        env.pop("SCORCH_PHYS_CORES", None)
    else:
        env["SCORCH_PHYS_CORES"] = str(phys)
    out = subprocess.run([sys.executable, "-c", _PROBE % shape], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    reported, resolved = (int(x) for x in out.stdout.split())
    return reported, resolved


def test_the_reported_core_count_is_sane():
    assert scorch_ops.scorch_phys_cores_avail() >= 1


def test_physical_cores_and_pcore_count_are_separate_questions():
    """Both are exported so a harness cannot silently take one for the other."""
    assert scorch_ops.scorch_phys_cores_avail() >= 1
    assert scorch_ops.scorch_pcore_count() >= 1


def test_the_forced_core_count_is_what_the_binary_reports():
    """The positive control's precondition: if this fails, every case below is vacuous."""
    for phys in (2, 5, 9):
        reported, _ = _resolve(phys=phys, cap=1)
        assert reported == phys


@pytest.mark.parametrize("phys", [1, 2, 3, 5, 8, 1024])
def test_the_cap_never_raises_a_worker_count(phys):
    """The whole safety argument: two min()s cannot reclassify anything that is fast today."""
    _, uncapped = _resolve(phys=phys, cap=0)
    _, capped = _resolve(phys=phys, cap=1)
    assert capped <= uncapped


def test_a_small_forced_core_count_actually_binds():
    """A positive control, so the cap cannot pass these tests by being dead code."""
    _, uncapped = _resolve(phys=1, cap=0)
    _, capped = _resolve(phys=1, cap=1)
    if uncapped > 1:
        assert capped < uncapped, "the cap did not bind; it may be compiled out"
    assert capped >= 1, "the cap must never ask for fewer than one worker"


def test_with_the_cap_off_the_core_count_knob_is_inert():
    """A negative control: the input the cap reads must not act when the cap is off."""
    _, wide = _resolve(phys=None, cap=0)
    _, narrow = _resolve(phys=1, cap=0)
    assert wide == narrow


def test_the_cap_is_bounded_by_the_forced_core_count():
    """Where it binds it binds AT the core count, not at some other number."""
    for phys in (2, 3, 5):
        _, uncapped = _resolve(phys=phys, cap=0)
        _, capped = _resolve(phys=phys, cap=1)
        if uncapped > phys:
            assert capped <= phys


def test_the_cap_declines_where_it_would_disable_the_out_of_pool_recruit():
    """The safety property that made folding it into SCORCH_SPMM_NT_CAP the right layer.

    On a product whose resolved count reaches twice the caller's pool, the surrounding cap
    block declines -- lowering the count there does not merely use fewer workers, it routes
    the kernel back onto the pool and gives up the out-of-pool recruit. An earlier version of
    this cap sat in FRONT of that block and silently defeated it.

    Skipped where the host never reaches that regime, since there is then nothing to decline;
    the condition is read from the binary rather than assumed.
    """
    import torch
    pool = torch.get_num_threads()
    _, huge = _resolve(phys=None, cap=0, shape=_HUGE)
    if huge < 2 * pool:
        pytest.skip("this host does not resolve past 2x its pool; no recruit to protect")
    _, capped = _resolve(phys=1, cap=1, shape=_HUGE)
    assert capped == huge, (
        "the cap bound where a recruit was at stake; it must decline there"
    )
