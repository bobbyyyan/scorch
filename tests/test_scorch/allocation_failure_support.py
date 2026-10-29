"""Run a check that makes an allocation fail on purpose in a child Python process.

Under the sanitizer harness (tests/sanitizers/run_sanitizers.sh) AddressSanitizer reports
any request past its largest supported size as an error and halts, even through
``new (std::nothrow)``.  A check whose allocation must fail (an owner of a capacity no
host can map) therefore runs in a child whose ``ASAN_OPTIONS`` adds
``allocator_may_return_null=1``: there a too-large request returns null, as the C++
allocator does.  The option stays out of every other process, where that report is the
overflowed-size defect the harness exists to catch.  The child otherwise inherits this
process's environment, so under the harness it runs sanitized too; without ASan the
option is never read.
"""

from __future__ import annotations

import os
import subprocess
import sys


def run_with_null_allocator(
    code: str, *argv: str, timeout: float = 900
) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    options = env.get("ASAN_OPTIONS")
    # A later option wins, so this one holds whatever the harness set before it.
    env["ASAN_OPTIONS"] = (
        f"{options}:allocator_may_return_null=1"
        if options
        else "allocator_may_return_null=1"
    )
    return subprocess.run(
        [sys.executable, "-c", code, *argv],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
