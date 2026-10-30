"""Focused LeakSanitizer window around a native ownership handoff.

The harness runs this file only on Linux. Tracking is disabled while CPython,
PyTorch, OpenMP, and the extension initialize, then enabled only for the warmed
native call. This avoids treating process-lifetime framework caches as leaks.

CPython must run with PYTHONMALLOC=malloc. Its default allocator keeps small
objects in arenas it maps itself, which LeakSanitizer does not scan, so a block
referenced only from such an object reads as leaked. On CPython 3.12.15 the
check before the native call reported 220 such blocks, all allocated by CPython
before this file could disable tracking. With every object in malloc'd memory
the references are visible, so the probe refuses to run without the setting.
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys

if os.environ.get("PYTHONMALLOC") != "malloc" or sys.flags.ignore_environment:
    raise SystemExit(
        "leak_probe.py needs PYTHONMALLOC=malloc and no -E or -I: LeakSanitizer "
        "does not scan pymalloc's arenas, so blocks they reference read as leaks"
    )

runtime = ctypes.CDLL(None)
try:
    lsan_disable = runtime.__lsan_disable
    lsan_enable = runtime.__lsan_enable
    lsan_check = runtime.__lsan_do_recoverable_leak_check
except AttributeError as error:
    raise SystemExit(
        "LeakSanitizer API is unavailable in the preloaded runtime"
    ) from error

lsan_disable.argtypes = []
lsan_disable.restype = None
lsan_enable.argtypes = []
lsan_enable.restype = None
lsan_check.argtypes = []
lsan_check.restype = ctypes.c_int

# Imports and first-call binding/framework caches are process-lifetime state and
# deliberately excluded from the focused allocation window.
lsan_disable()

import torch  # noqa: E402

import scorch_ops  # noqa: E402

width = 7
a_rows = torch.tensor([0, 1], dtype=torch.int32)
a_columns = torch.tensor([0, 2], dtype=torch.int32)
a_values = torch.tensor([2.0, 3.0])
b_values_2d = torch.arange(3 * width, dtype=torch.float32).reshape(3, width)

arguments = (
    [2, width],
    [2, 3],
    [[a_rows], [a_columns]],
    a_values,
    [3, width],
    [[], []],
    b_values_2d.reshape(-1),
)


def invoke_native_coo_spmm():
    return scorch_ops.spmm_coo_float(*arguments)


warm_result = invoke_native_coo_spmm()
del warm_result
gc.collect()

if lsan_check() != 0:
    raise SystemExit("LeakSanitizer found a leak before the focused native call")

lsan_enable()
try:
    result = invoke_native_coo_spmm()
finally:
    lsan_disable()

# the [2, width] product itself: the kernel allocates it in that shape (spmm.h)
expected = torch.stack((2 * b_values_2d[0], 3 * b_values_2d[2]))
if not torch.equal(result.storage.value, expected):
    raise SystemExit("native COO SpMM returned an incorrect result")

del result
gc.collect()

if lsan_check() != 0:
    raise SystemExit("LeakSanitizer found an allocation lost by native COO SpMM")

print("focused native LeakSanitizer check passed")
