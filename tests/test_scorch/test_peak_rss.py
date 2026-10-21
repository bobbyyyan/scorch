"""The peak-memory harness the bound tests measure with (peak_rss.py).

A measurement in a fresh process sees the negative control's 2 GiB, also when the process
that starts it holds more than that (getrusage's ``ru_maxrss`` would hide it: sapling 83661
and 83752), catches a measured call that allocates the same 2 GiB (the positive control: the
leak the bound tests guard against), refuses a function that never marks where its call
starts, and reads Linux's /proc fields in bytes.
"""

from __future__ import annotations

import pytest
import torch

from tests.test_scorch import peak_rss
from tests.test_scorch.peak_rss import (
    CONTROL_BYTES,
    CONTROL_SEEN_BYTES,
    _linux_resident,
    assert_peak_rise_below,
    control_rise,
    peak_rise,
    resident_bytes,
)


def _never_marks(mark):
    """Run in a fresh process: forgets the mark."""


def _leaks_after_the_mark(mark):
    """Run in a fresh process: the control's allocation, under another name."""
    peak_rss.allocate_control(mark)


def test_linux_status_is_read_in_bytes():
    status = "Name:\tpython\nVmPeak:\t 9 kB\nVmHWM:\t  3145728 kB\nVmRSS:\t   1024 kB\n"
    assert _linux_resident(status) == (3 * 2**30, 2**20)
    with pytest.raises(RuntimeError, match="VmHWM or VmRSS"):
        _linux_resident("Name:\tpython\nVmRSS:\t1024 kB\n")


def test_this_process_reads_its_own_resident_set():
    peak, current = resident_bytes()
    assert peak >= current > 0


def test_the_control_is_seen():
    assert control_rise() >= CONTROL_SEEN_BYTES
    # And not invented: the growth is the allocation, within the interpreter's slack.
    assert control_rise() < CONTROL_BYTES + 2**28


def test_a_large_parent_does_not_hide_the_control():
    # The process that starts the measurement holds more than the control's allocation,
    # written, so its own peak stands above anything the child reaches.  A child's
    # ru_maxrss would start at that peak (Linux keeps it across execve) and hide most of
    # the control (83752, under a parent holding 1.5 GiB: 536,555,520 of its
    # 2,147,483,648 B).
    held = torch.ones((CONTROL_BYTES + CONTROL_BYTES // 2) // 8, dtype=torch.float64)
    assert float(held[-1]) == 1.0
    growth = peak_rise("tests.test_scorch.peak_rss:allocate_control")
    del held
    assert CONTROL_SEEN_BYTES <= growth < CONTROL_BYTES + 2**28


def test_a_leak_after_the_mark_fails_the_bound():
    with pytest.raises(AssertionError):
        assert_peak_rise_below(2**30, f"{__name__}:_leaks_after_the_mark")


def test_a_function_that_never_marks_is_refused():
    with pytest.raises(AssertionError, match="mark"):
        peak_rise(f"{__name__}:_never_marks")
