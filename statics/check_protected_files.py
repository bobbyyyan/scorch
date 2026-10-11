#!/usr/bin/env python3
"""Check the five protected tracked files, two independent ways.

Five files are off limits to the compiler-IR refactor because a separate CUDA
project owns them: ``.gitignore``, ``pyproject.toml``, ``src/scorch/__init__.py``,
``tests/packaging/smoke_install.py`` and ``tests/test_scorch/test_resources.py``.
Milestone gates on that branch report that the five "hash exactly as recorded".

Until review section 61.2 that claim rested on a digest file stored per evidence
folder, whose digests had been taken from a working tree carrying the CUDA
project's uncommitted edits.  So it verified "the working tree has not changed
since this snapshot" rather than "these tracked files are unmodified", and a
clean checkout differed from all five recorded digests.  Two checks replace it,
and they answer different questions:

**The git-derived check** compares the five tracked blobs at the tree's HEAD
against the same five at a reference commit -- by default the branch point.  It
reads no stored digest, so it cannot drift as this file ages, and it is the check
that actually states the obligation: the refactor has not modified these files.

**The snapshot check** compares the five files ON DISK against
``statics/protected-hashes.txt``.  A clean checkout passes.  The live repository
fails while the CUDA project's edits are in the working tree, and that failure is
the correct report rather than a defect -- it distinguishes "somebody is editing
a protected file right now" from "a protected file changed in git".

Both are reported, always, with which is which.  Neither is inferred from the
other, because conflating them is the whole history of this check.

Usage:
    check_protected_files.py TREE [--reference COMMIT] [--baseline FILE]
    check_protected_files.py TREE --write     # regenerate the snapshot

Takes the tree root as its first argument and hardcodes no path.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from typing import Dict, List, Tuple

#: The five, in the order every baseline file has listed them.
PROTECTED_FILES: Tuple[str, ...] = (
    ".gitignore",
    "pyproject.toml",
    "src/scorch/__init__.py",
    "tests/packaging/smoke_install.py",
    "tests/test_scorch/test_resources.py",
)

#: The commit the git-derived check compares against unless told otherwise: the
#: origin this branch left, so "unmodified" means unmodified by the refactor.
DEFAULT_REFERENCE = "a3b8d1e"


def _git(tree: str, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", tree) + args, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(
            f"git {' '.join(args)} failed in {tree}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def disk_digests(tree: str) -> Dict[str, str]:
    digests = {}
    for path in PROTECTED_FILES:
        full = os.path.join(tree, path)
        with open(full, "rb") as handle:
            digests[path] = hashlib.sha256(handle.read()).hexdigest()
    return digests


def read_baseline(path: str) -> Dict[str, str]:
    recorded = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, name = line.split(None, 1)
            recorded[name.strip()] = digest
    return recorded


def git_derived_check(tree: str, reference: str) -> Tuple[bool, List[str]]:
    """The five tracked blobs at HEAD against the same five at ``reference``."""

    head = _git(tree, "rev-parse", "HEAD")
    lines = [
        f"reference {reference} ({_git(tree, 'rev-parse', reference)[:12]})",
        f"HEAD      {head[:12]}",
    ]
    differing = []
    for path in PROTECTED_FILES:
        at_head = _git(tree, "rev-parse", f"{head}:{path}")
        at_ref = _git(tree, "rev-parse", f"{reference}:{path}")
        same = at_head == at_ref
        lines.append(
            f"  {'same' if same else 'DIFFERS':<8} {path}  "
            f"{at_ref[:10]} -> {at_head[:10]}"
        )
        if not same:
            differing.append(path)
    return not differing, lines


def snapshot_check(tree: str, baseline: str) -> Tuple[bool, List[str]]:
    """The five files on disk against the recorded snapshot."""

    recorded = read_baseline(baseline)
    found = disk_digests(tree)
    lines = [f"baseline {baseline}"]
    differing = []
    for path in PROTECTED_FILES:
        want = recorded.get(path)
        have = found[path]
        if want is None:
            lines.append(f"  MISSING  {path}  not listed in the baseline")
            differing.append(path)
            continue
        same = want == have
        lines.append(f"  {'match' if same else 'DIFFERS':<8} {path}  {have[:16]}")
        if not same:
            differing.append(path)
    return not differing, lines


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tree", help="tree root to check")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--baseline",
        default=None,
        help="baseline file; defaults to TREE/statics/protected-hashes.txt",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the baseline's digest lines from TREE, keeping its header",
    )
    args = parser.parse_args(argv)

    tree = os.path.realpath(args.tree)
    baseline = args.baseline or os.path.join(tree, "statics", "protected-hashes.txt")

    if args.write:
        dirty = _git(tree, "status", "--porcelain", "--", *PROTECTED_FILES)
        if dirty:
            raise SystemExit(
                "refusing to regenerate from a tree with uncommitted edits to a "
                f"protected file -- that is exactly how the old baseline came to "
                f"hold working-tree digests:\n{dirty}"
            )
        found = disk_digests(tree)
        with open(baseline) as handle:
            header = [ln for ln in handle if ln.startswith("#")]
        with open(baseline, "w") as handle:
            handle.writelines(header)
            for path in PROTECTED_FILES:
                handle.write(f"{found[path]}  {path}\n")
        print(f"rewrote {baseline} from {tree} at {_git(tree, 'rev-parse', 'HEAD')}")
        return 0

    git_ok, git_lines = git_derived_check(tree, args.reference)
    snap_ok, snap_lines = snapshot_check(tree, baseline)

    print("=== check 1, git-derived: are the five tracked files unmodified? ===")
    print("\n".join(git_lines))
    print(f"  VERDICT: {'PASS' if git_ok else 'FAIL'}")
    print()
    print("=== check 2, snapshot: do the five on disk match the baseline? ===")
    print("\n".join(snap_lines))
    print(f"  VERDICT: {'PASS' if snap_ok else 'FAIL'}")
    print()
    if git_ok and not snap_ok:
        print(
            "Read this pair as: the tracked files are unmodified, and the working\n"
            "tree carries edits to some of them.  On the live repository that is\n"
            "the expected state while the separate CUDA project is in flight.  It\n"
            "is NOT the state a measuring checkout should ever be in."
        )
    return 0 if (git_ok and snap_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
