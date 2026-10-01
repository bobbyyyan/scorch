"""Enumerate ledger CSVs so a QUARANTINED artefact cannot join a recount silently.

WHY. chain194-mrhv-redwood holds `hv_ex4b_float64_p2.contaminated.csv`: a file someone
deliberately marked bad, attested as line 22 of 33 in that directory's SHA256SUMS, and returned by
a plain `*.csv` glob there as 1 of 21. The marker is a middle component of the filename, so it is
invisible to every glob while the seal says the bytes are evidence. It is the only such file in
the ledger -- a search for contaminated/bad/broken/stale/discard/reject/invalid/wip/tmp/partial
returns exactly one, against a control of 2049 hits for "float32" -- so this is one instance and
not a convention, which is precisely why no reader was written to expect it.

TWO EXPOSURE SHAPES, and each of this tree's readers is open to exactly one of them. Measured
with fnmatch against the real patterns:

  * MARKER BEFORE `.csv`, as in the real file. `*_float32_p*.csv` admits
    `..._float32_p2.contaminated.csv`, because the trailing `*` spans the second extension
    component. an_boardscreen, an_c211, an_timedshare and an_degree all admit a marker in that
    position for the dtype and prefix they target; the one real quarantined file escapes them only
    because it is float64 and their patterns pin float32. That is a coincidence, not a guard.
  * MARKER APPENDED AFTER `.csv`, which is what `mv x.csv x.csv.bad` produces. pfs_eta compiles
    its caller's regex and calls `rx.match`, which anchors at the start only, so a pattern ending
    `\\.csv` with no `$` accepts any trailing junk. Safe against the first shape, open to this one.

IT REFUSES RATHER THAN SKIPPING. A reader that silently drops a file changes its own denominator
with nothing in the output saying so, which is the failure this project has hit most often. The
caller is told which file and why, and has to decide -- and the decision is a genuine trade, not
a defect: dropping the file from the manifest loses the attestation that the bad file is the bad
file, while keeping it means the marker is enforced by every reader instead of by the filesystem.
A quarantine subdirectory is the form that lets a glob skip it without losing the seal. That is a
change to sealed evidence, so it is not this module's to make.
"""
import fnmatch
import glob as _glob
import os
import re

# Substrings that mark an artefact as not-for-analysis. Matched case-insensitively against the
# WHOLE filename, so position does not matter and both exposure shapes are covered by one list.
MARKERS = ("contaminated", "quarantine", "broken", "discard", "reject", "invalid",
           "stale", "partial", "bad", "wip", "tmp")


def marker_in(name):
    """The marker found in this filename, or None. Position-independent, case-insensitive.

    The name is split on the separators a filename uses -- dot, underscore, hyphen -- and a marker
    must be a WHOLE token. Substring matching would flag "badge" for "bad" and "tmpdir" for "tmp",
    and a guard with false positives on ordinary names gets switched off within a week. Token
    equality also covers both exposure shapes with one rule, since a token is a token wherever it
    sits: `x.contaminated.csv` and `x.csv.bad` both flag.
    """
    for tok in re.split(r"[._\-]+", os.path.basename(name).lower()):
        if tok in MARKERS:
            return tok
    return None


def scan(names):
    """(clean, flagged) over an iterable of paths. flagged is [(path, marker)]."""
    clean, flagged = [], []
    for n in names:
        m = marker_in(n)
        (flagged.append((n, m)) if m else clean.append(n))
    return sorted(clean), sorted(flagged)


def glob_clean(d, pattern, label="", allow_excluded=False):
    """Glob `pattern` under `d` and REFUSE if any match carries a quarantine marker.

    Returns the sorted clean matches. Refusing is the point: a recount that silently excluded the
    file would report a denominator nobody could check, and one that included it would average a
    file someone marked bad.
    """
    # RUN-LEVEL EXCLUSION FIRST. A bad file is marked in its name; a bad RUN has no filename to
    # mark, because every file in it is individually well-formed. Checking the directory before
    # the filenames means a reader pointed at a set-aside run stops on the larger fact rather
    # than reporting a clean filename scan over data nobody should be averaging.
    import ledger_excluded_runs as LX
    LX.check_run(d, allow_excluded=allow_excluded, label=label)
    hits = _glob.glob(os.path.join(d, pattern))
    clean, flagged = scan(hits)
    if flagged:
        raise SystemExit(
            "REFUSE: %s%d of %d files matching %r under %s carry a quarantine marker:\n%s\n"
            "  A marked artefact is sealed evidence and admitted by this pattern, so neither\n"
            "  including it nor silently dropping it is this reader's call. Resolve the marker\n"
            "  (a quarantine subdirectory keeps the seal and lets a glob skip it), or pass a\n"
            "  pattern that excludes it on purpose."
            % (label + ": " if label else "", len(flagged), len(hits), pattern, d,
               "\n".join("    %s  [%s]" % (os.path.basename(p), m) for p, m in flagged)))
    return clean


def require_end_anchored(pattern):
    """Refuse a filename regex that is not end-anchored, for callers using `re.match`.

    `re.match` anchors at the start only, so `..._p(?P<pass>[0-9]+)\\.csv` accepts
    `..._p2.csv.contaminated` and every other trailing suffix. The anchor is what makes the
    extension mean "the name ends here".
    """
    if not re.search(r"(\$|\\Z)\s*$", pattern):
        raise SystemExit(
            "REFUSE: the filename pattern %r is not end-anchored, and re.match anchors only at "
            "the start, so it accepts any trailing suffix -- including the quarantine markers a "
            "renamed artefact carries (x.csv.bad). Append $ or \\Z." % pattern)
    return pattern


def admits_a_marker(pattern, example, marker="contaminated"):
    """METAMORPHIC: insert `marker` before the extension of a name this pattern already matches,
    and report whether the pattern still admits it.

    The first version of this function compared a FIXED probe name against the caller's pattern,
    which answers a question about the probe rather than about the pattern -- it reported False
    for `dl_*_float32_p*.csv` and `pbdeg_n*_core*.csv` simply because the probe lacked their
    prefixes, i.e. it reported "not exposed" for patterns that are exposed. Transforming a name
    the pattern is known to match is the only form that cannot be wrong for that reason.

    Refuses when `example` does not match to begin with: a pattern that admits nothing cannot
    demonstrate an exposure, and a False from that state would read as safety.
    """
    if not fnmatch.fnmatch(example, pattern):
        raise ValueError("example %r does not match pattern %r, so the transformed name proves "
                         "nothing about this pattern" % (example, pattern))
    stem, ext = os.path.splitext(example)
    return fnmatch.fnmatch("%s.%s%s" % (stem, marker, ext), pattern)
