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


def linked_ancestors(d, anchor):
    """Components of the path to `d` that are symlinks, strictly below `anchor`.

    THE DIRECTORY FORM OF THE SYMLINK PROBLEM, and why it needs an anchor the file form does not.
    A file reached THROUGH a symlinked directory is not itself a symlink, so the per-hit
    os.path.islink check in glob_clean cannot see it: measured, a glob through a linked directory
    returns its files silently. The obvious rule -- refuse when realpath(d) != abspath(d) -- is
    unusable, because on macOS /var is itself a symlink to /private/var, so it refuses every
    temporary directory including the ones this module's own plants build. The platform's prefix
    links are not evidence defects, and nothing in the path itself distinguishes them from one
    inside a ledger. So the caller must say where its evidence tree starts, and only links at or
    below that point count.

    Refuses when `d` is not under `anchor`, because an empty result from a wrong anchor would read
    as "no linked ancestors".
    """
    a = os.path.realpath(anchor)
    cur = os.path.abspath(d)
    if not (os.path.realpath(cur) + os.sep).startswith(a + os.sep) and os.path.realpath(cur) != a:
        raise ValueError("%r is not under anchor %r, so an empty result would be meaningless"
                         % (d, anchor))
    out = []
    while True:
        if os.path.islink(cur):
            out.append(cur)
        nxt = os.path.dirname(cur)
        if nxt == cur or os.path.abspath(cur) == a or os.path.realpath(cur) == a:
            break
        cur = nxt
    return out


def glob_clean(d, pattern, label="", allow_excluded=False, anchor=None):
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
    # SYMLINKS ARE REFUSED, NOT RESOLVED, and the ledger already takes that position. A marker is
    # a property of a filename, so a link under a clean name pointing at a marked file defeats
    # this guard -- and the link is visible to glob and os.walk while a `find -type f` census
    # omits it, which is why two counts of "the CSVs here" can disagree by exactly the class
    # neither evaluates. Resolving the target is the obvious shape and it is the wrong one: a
    # symlink in an evidence directory is itself the defect, because shasum follows it, so a
    # sealed NAME whose bytes live outside the sealed set verifies clean forever while remaining
    # mutable. seal_c210.sh refuses links for that reason, and measurement agrees it is not a
    # restriction in practice -- every sealed directory in the ledger is symlink-free today, and
    # the only linked CSVs are four pytest fixtures in an unsealed directory.
    # The DIRECTORY form, only when the caller named its evidence root -- see linked_ancestors
    # for why an unanchored version cannot work. Checked before the per-hit test because a linked
    # ancestor makes every hit's path wrong, not just one file's.
    if anchor is not None:
        anc = linked_ancestors(d, anchor)
        # THE ANCHOR GOES IN THE OUTPUT, NOT JUST IN THE CALL. This check's correctness claim is
        # conditional on an argument the caller supplies, and the conditional is not small:
        # anchored at a sealed chain directory there is one component to walk and the measured
        # population of symlinked milestone and chain directories is ZERO, so the check cannot
        # fire on sealed evidence as the tree stands; anchored at the ledger root and allowed to
        # descend there are 4871 symlinked directories. Same code, same tree, three orders apart,
        # selected entirely by the argument. So "0 linked ancestors" is not a statement and
        # "0 below <anchor>" is -- a silently misplaced anchor is the vacuous pass, and the
        # number that shows this one was not is the anchor itself.
        print("  linked-ancestor check: %d found below %s" % (len(anc), anchor))
        if anc:
            raise SystemExit(
                "REFUSE: %s%s is reached through %d symlinked director%s below %s:\n%s\n"
                "  Files under it are not themselves symlinks, so a per-file check cannot see\n"
                "  this, and the bytes are not where the path says they are."
                % (label + ": " if label else "", d, len(anc),
                   "y" if len(anc) == 1 else "ies", anchor,
                   "\n".join("    %s -> %s" % (x, os.path.realpath(x)) for x in anc)))
    links = [h for h in hits if os.path.islink(h)]
    if links:
        raise SystemExit(
            "REFUSE: %s%d of %d files matching %r under %s are SYMLINKS:\n%s\n"
            "  A marker lives in a filename, so a clean link name hides a marked target, and the\n"
            "  bytes are outside this directory and outside its seal. Resolve the link into a\n"
            "  real file, or point the reader at the directory that holds the bytes."
            % (label + ": " if label else "", len(links), len(hits), pattern, d,
               "\n".join("    %s -> %s%s" % (
                   os.path.basename(l), os.path.realpath(l),
                   "  [TARGET CARRIES MARKER: %s]" % marker_in(os.path.realpath(l))
                   if marker_in(os.path.realpath(l)) else "") for l in sorted(links))))
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
