"""How much a call raises the peak resident set, measured in a process of its own.

A process's peak resident set only rises.  Read before and after a call inside the test
session, it measures nothing once an earlier test in that process has peaked higher: the
call's allocation stays under the old peak and the difference is zero.  So a measured call
runs here in a new interpreter, which imports its test module, prepares its operands, calls
``mark()`` just before the call, and reports how far its resident set grew from there: the
peak at the end less the resident set at the mark.

The peak and the resident set are read from one source, the address space's own
(:func:`resident_bytes`): Linux's ``VmHWM`` and ``VmRSS`` in /proc/self/status, macOS's
``resident_size_max`` and ``resident_size`` from task_info(MACH_TASK_BASIC_INFO).  Not
getrusage's ``ru_maxrss``: Linux keeps it across execve (getrusage(2)), so a child started by
a large process, such as a pytest worker, begins with its parent's peak, and an allocation
below that peak shows no rise.  (Under a parent holding 1.5 GiB a child's 2 GiB allocation
showed a ``ru_maxrss`` rise of 536,555,520 B and a ``VmHWM`` rise of 2,150,060,032 B; sapling
83752, diagnosing 83661's blind control.)

Measuring from the resident set at the mark, not the peak, counts whatever the child
allocated and freed before its mark (the peak standing above the resident set) against the
bound: a measurement can overstate a call's growth by that much, never understate it.

A measurement that could not see an allocation proves nothing, so the same harness is run
on a negative control that allocates and writes CONTROL_BYTES after its mark, and every
bound asserted through :func:`assert_peak_rise_below` holds only if the control's growth is
measured too (:func:`control_rise`).
"""

from __future__ import annotations

import ctypes
import importlib
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
#: The negative control's allocation.
CONTROL_BYTES = 2 * 2**30
#: The least growth the control must show.  The measurement does not understate (see the
#: module docstring), so a control that shows less than this is a measurement that cannot
#: see an allocation.  Every bound asserted here is at most 1 GiB, well below it.
CONTROL_SEEN_BYTES = 3 * CONTROL_BYTES // 4
#: A measured call that has not returned in this long has failed.
TIMEOUT_S = 900.0
_TAG = "PEAK_RSS_RISE "


def _linux_resident(status_text: str) -> Tuple[int, int]:
    """(``VmHWM``, ``VmRSS``) in bytes from the text of /proc/<pid>/status (kB there)."""
    fields = {}
    for line in status_text.splitlines():
        name, _, value = line.partition(":")
        if name in ("VmHWM", "VmRSS"):
            fields[name] = int(value.split()[0]) * 1024
    if len(fields) != 2:
        raise RuntimeError(f"peak_rss: VmHWM or VmRSS missing, found {sorted(fields)}")
    return fields["VmHWM"], fields["VmRSS"]


class _TimeValue(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_int), ("microseconds", ctypes.c_int)]


class _MachTaskBasicInfo(ctypes.Structure):
    """struct mach_task_basic_info (<mach/task_info.h>, under #pragma pack(4))."""

    _pack_ = 4
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time", _TimeValue),
        ("system_time", _TimeValue),
        ("policy", ctypes.c_int),
        ("suspend_count", ctypes.c_int),
    ]


_MACH_TASK_BASIC_INFO = 20  # <mach/task_info.h>


def _darwin_resident() -> Tuple[int, int]:
    """(``resident_size_max``, ``resident_size``) in bytes, from task_info."""
    system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    task = ctypes.c_uint.in_dll(system, "mach_task_self_").value
    info = _MachTaskBasicInfo()
    count = ctypes.c_uint(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint))
    status = system.task_info(
        ctypes.c_uint(task),
        _MACH_TASK_BASIC_INFO,
        ctypes.byref(info),
        ctypes.byref(count),
    )
    if status != 0:
        raise RuntimeError(
            f"peak_rss: task_info(MACH_TASK_BASIC_INFO) returned {status}"
        )
    return int(info.resident_size_max), int(info.resident_size)


def resident_bytes() -> Tuple[int, int]:
    """(peak, current) resident set of this process's address space, in bytes, both from
    one source; an error where the platform gives neither (no bound can be measured)."""
    if sys.platform.startswith("linux"):
        with open("/proc/self/status") as status:
            return _linux_resident(status.read())
    if sys.platform == "darwin":
        return _darwin_resident()
    raise RuntimeError(f"peak_rss: no resident-set reading on {sys.platform}")


def allocate_control(mark: Callable[[], None]) -> None:
    """The negative control: CONTROL_BYTES allocated and written after the mark."""
    import torch

    mark()
    block = torch.ones(CONTROL_BYTES // 8, dtype=torch.float64)
    assert float(block[-1]) == 1.0


def peak_rise(target: str, *args: str) -> int:
    """Run ``module:function`` in a new interpreter as ``function(mark, *args)``; the bytes
    its resident set grew from ``mark()``: the peak at the function's return less the
    resident set at the mark."""
    module, function = target.split(":")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), *(path for path in sys.path if path)]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from tests.test_scorch.peak_rss import _measure\n"
            "_measure(sys.argv[1], sys.argv[2], sys.argv[3:])\n",
            module,
            function,
            *args,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )
    reports = [line for line in completed.stdout.splitlines() if line.startswith(_TAG)]
    if completed.returncode != 0 or len(reports) != 1:
        raise AssertionError(
            f"{target}{tuple(args)} in its own process: rc {completed.returncode}\n"
            f"--- stdout\n{completed.stdout[-4000:]}\n--- stderr\n{completed.stderr[-4000:]}"
        )
    report = json.loads(reports[0][len(_TAG) :])
    return int(report["growth"])


@lru_cache(maxsize=1)
def control_rise() -> int:
    """The negative control's measured rise, once per process."""
    return peak_rise("tests.test_scorch.peak_rss:allocate_control")


def assert_peak_rise_below(limit_bytes: int, target: str, *args: str) -> int:
    """``target`` raises the peak by less than ``limit_bytes``, by a measurement that sees
    the control's allocation; its rise."""
    assert limit_bytes < CONTROL_SEEN_BYTES
    control = control_rise()
    assert control >= CONTROL_SEEN_BYTES, (
        f"the measurement is blind: a {CONTROL_BYTES} B allocation rose the peak by "
        f"{control} B"
    )
    rise = peak_rise(target, *args)
    assert rise < limit_bytes, (target, args, rise, limit_bytes)
    return rise


def _measure(module: str, function: str, args: List[str]) -> None:
    """The child's side: call the function with a mark, and report the rise."""
    marks: List[Tuple[int, int]] = []

    def mark() -> None:
        marks.append(resident_bytes())

    getattr(importlib.import_module(module), function)(mark, *args)
    if len(marks) != 1:
        raise AssertionError(f"{module}.{function} called mark() {len(marks)} times")
    (peak_at_mark, resident_at_mark), (peak, _) = marks[0], resident_bytes()
    report = {
        "growth": peak - resident_at_mark,
        "rise": peak - peak_at_mark,
        "headroom": peak_at_mark - resident_at_mark,
    }
    print(_TAG + json.dumps(report), flush=True)
