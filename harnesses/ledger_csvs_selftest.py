"""Plants for ledger_csvs. Both exposure shapes, the substring ablation, and the real file.

The substring ablation is the load-bearing one: a marker rule that matched substrings would flag
"badge" and "tmpdir", and a guard with false positives on ordinary filenames gets switched off.
Plants write into temporary directories outside this tree.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ledger_csvs as L                                              # noqa: E402

FAIL, CHECKS, NOTES = [], 0, []
N_PLANTS = 15
REAL = os.path.expanduser(
    "~/.cache/scorch-codex/spmm-beat-mkl/chain194-mrhv-redwood")


def eq(label, got, want):
    global CHECKS
    CHECKS += 1
    if got != want:
        FAIL.append("%s: got %r, expected %r" % (label, got, want))


def _dir(files):
    d = tempfile.mkdtemp()
    for f in files:
        open(os.path.join(d, f), "w").write("x")
    return d


def t_the_real_quarantined_filename_flags_with_its_marker_named():
    eq("real", L.marker_in("hv_ex4b_float64_p2.contaminated.csv"), "contaminated")


def t_an_ordinary_pass_csv_does_not_flag():
    eq("clean", L.marker_in("dl_base_float32_p2.csv"), None)


def t_SUBSTRING_lookalikes_do_NOT_flag():
    """ABLATION. Under a substring rule every one of these flags, and the guard becomes noise."""
    for n in ("badge_float32_p1.csv", "tmpdir_float32_p1.csv", "partially_float32_p1.csv",
              "rejection_p1.csv", "invalidate_p1.csv"):
        eq("lookalike-" + n, L.marker_in(n), None)


def t_a_marker_APPENDED_after_the_extension_flags():
    """The `mv x.csv x.csv.bad` shape, which pfs_eta's unanchored re.match would accept."""
    eq("appended", L.marker_in("dl_base_float32_p2.csv.bad"), "bad")
    eq("appended-hyphen", L.marker_in("dl_base_float32_p2.csv-stale"), "stale")


def t_glob_clean_refuses_and_names_the_file():
    d = _dir(["a_float32_p1.csv", "a_float32_p2.contaminated.csv"])
    try:
        L.glob_clean(d, "*_float32_p*.csv")
        eq("refuse", "returned", "raised")
    except SystemExit as e:
        eq("refuse", "raised", "raised")
        eq("refuse-names-file", "a_float32_p2.contaminated.csv" in str(e), True)
        eq("refuse-names-marker", "[contaminated]" in str(e), True)


def t_glob_clean_returns_the_matches_when_none_are_flagged():
    d = _dir(["a_float32_p1.csv", "a_float32_p2.csv"])
    eq("pass-through", len(L.glob_clean(d, "*_float32_p*.csv")), 2)


def t_zero_matches_is_not_a_refusal():
    """The refusal must be specific to a flagged file. An empty directory is the caller's own
    non-vacuity problem and conflating the two would make this guard unusable."""
    eq("empty", L.glob_clean(_dir([]), "*_float32_p*.csv"), [])


def t_require_end_anchored_refuses_an_unanchored_pattern():
    try:
        L.require_end_anchored(r"dl_(?P<build>[a-z_]+)_float32_p(?P<pass>[0-9]+)\.csv")
        eq("unanchored", "returned", "raised")
    except SystemExit:
        eq("unanchored", "raised", "raised")
    eq("dollar", L.require_end_anchored(r"x\.csv$").endswith("$"), True)
    eq("bigz", L.require_end_anchored(r"x\.csv\Z").endswith(r"\Z"), True)


def t_the_documented_exposure_of_each_READER_pattern_is_checked_by_code():
    """The docstring names four readers as admitting a marker before the extension. Each is
    checked here against a filename that reader really reads, so the claim cannot be a sentence
    that disagrees with nothing."""
    cases = [("an_boardscreen", "*_float32_p*.csv", "hv_c0_float32_p1.csv"),
             ("an_c211", "dl_*_float32_p*.csv", "dl_base_float32_p1.csv"),
             ("an_timedshare", "dl_*_float32_p*.csv", "dl_deep_float32_p6.csv"),
             ("an_degree", "pbdeg_n*_core*.csv", "pbdeg_n4_core8.csv")]
    for lab, pat, ex in cases:
        eq("admits-" + lab, L.admits_a_marker(pat, ex), True)
    # A pattern that pins the digits and the extension is NOT exposed -- so the guard is
    # reporting a property of the pattern, not returning True for everything.
    eq("anchored-not-exposed", L.admits_a_marker("*_float32_p[0-9].csv", "a_float32_p1.csv"), False)
    # And it refuses when the example cannot match, rather than reporting safety.
    try:
        L.admits_a_marker("dl_*.csv", "hv_other.csv")
        eq("nonmatching-example", "returned", "raised")
    except ValueError:
        eq("nonmatching-example", "raised", "raised")


def t_the_REAL_sealed_directory_refuses(and_report=True):
    """Live-evidence plant, read-only. If the ledger is absent this REPORTS rather than passing,
    because a plant that silently does not run is the defect it exists to prevent."""
    if not os.path.isdir(REAL):
        NOTES.append("live plant did NOT run: %s is absent, so the strongest plant is untested"
                     % REAL)
        eq("real-dir-missing-reported", True, True)
        return
    try:
        L.glob_clean(REAL, "*.csv", label="chain194")
        eq("real-refuse", "returned", "raised")
    except SystemExit as e:
        eq("real-refuse", "raised", "raised")
        eq("real-names-it", "contaminated" in str(e), True)


def t_a_symlinked_match_is_REFUSED_not_resolved():
    """Shape five: a clean link name pointing at a marked target. The marker is a property of a
    filename, so the name check reads the link's. Refusing is the ledger's own position -- its
    sealer refuses links because shasum follows them, so a sealed NAME whose bytes live outside
    the sealed set verifies clean forever while staying mutable."""
    d = _dir(["a_float32_p1.csv", "real_float32_p2.contaminated.csv"])
    os.symlink(os.path.join(d, "real_float32_p2.contaminated.csv"),
               os.path.join(d, "a_float32_p2.csv"))
    try:
        L.glob_clean(d, "*_float32_p*.csv")
        eq("symlink", "returned", "raised")
    except SystemExit as e:
        eq("symlink", "raised", "raised")
        eq("symlink-named", "a_float32_p2.csv" in str(e), True)
        # And it reports that the TARGET is marked, which is the information a reader needs.
        eq("symlink-target-marker", "TARGET CARRIES MARKER" in str(e), True)


def t_a_symlink_to_a_CLEAN_target_is_still_refused():
    """ABLATION: the refusal is about the link, not about the target's name. A clean target does
    not make the bytes sealed or immutable."""
    d = _dir(["a_float32_p1.csv", "real_float32_p9.csv"])
    os.symlink(os.path.join(d, "real_float32_p9.csv"), os.path.join(d, "a_float32_p2.csv"))
    try:
        L.glob_clean(d, "*_float32_p*.csv")
        eq("symlink-clean", "returned", "raised")
    except SystemExit as e:
        eq("symlink-clean", "raised", "raised")
        eq("symlink-clean-no-marker", "TARGET CARRIES MARKER" not in str(e), True)


def t_a_linked_ANCESTOR_is_caught_when_the_caller_names_its_root():
    """The directory form. Files under a linked directory are not links, so the per-hit check
    returns them silently -- measured before this guard existed."""
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "real"))
    open(os.path.join(root, "real", "a_float32_p1.csv"), "w").write("x")
    os.symlink(os.path.join(root, "real"), os.path.join(root, "link"))
    d = os.path.join(root, "link")
    eq("ancestor-found", len(L.linked_ancestors(d, root)), 1)
    try:
        L.glob_clean(d, "*_float32_p*.csv", anchor=root)
        eq("ancestor-refuse", "returned", "raised")
    except SystemExit as e:
        eq("ancestor-refuse", "raised", "raised")
        eq("ancestor-names-target", os.path.realpath(d) in str(e), True)


def t_the_PLATFORM_prefix_link_does_not_trip_the_anchored_check():
    """ABLATION on the rule that was rejected: /var is a symlink to /private/var on macOS, so
    realpath != abspath refuses every temp directory. Anchoring at the caller's root must not."""
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "plain"))
    open(os.path.join(root, "plain", "a_float32_p1.csv"), "w").write("x")
    d = os.path.join(root, "plain")
    eq("prefix-not-flagged", L.linked_ancestors(d, root), [])
    eq("prefix-passes", len(L.glob_clean(d, "*_float32_p*.csv", anchor=root)), 1)


def t_a_wrong_anchor_is_refused_rather_than_returning_empty():
    """An empty result from an anchor the directory is not under would read as safety."""
    root = tempfile.mkdtemp()
    try:
        L.linked_ancestors(root, os.path.join(tempfile.mkdtemp(), "elsewhere"))
        eq("wrong-anchor", "returned", "raised")
    except ValueError:
        eq("wrong-anchor", "raised", "raised")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    for fn in tests:
        try:
            fn()
        except Exception as e:                                        # noqa: BLE001
            FAIL.append("%s CRASHED: %s: %s" % (fn.__name__, type(e).__name__, e))
    print("ledger_csvs selftest: %d plants, %d assertions" % (len(tests), CHECKS))
    for n in NOTES:
        print("  NOTE " + n)
    rc = 0
    for f in FAIL:
        rc = 1
        print("  FAIL " + f)
    if len(tests) != N_PLANTS:
        rc = 1
        print("  REFUSE: %d plants collected, the file declares %d" % (len(tests), N_PLANTS))
    print("  " + ("OK" if rc == 0 else "FAILED"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
