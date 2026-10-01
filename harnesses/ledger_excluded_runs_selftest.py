"""Plants for ledger_excluded_runs. The citation is checked against the filesystem, not trusted.

A registry whose entries are prose is only as good as the prose, so one plant resolves each
citation's cited path and requires it to exist. That is the difference between recording an
exclusion and asserting one.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ledger_csvs as LC                                             # noqa: E402
import ledger_excluded_runs as LX                                    # noqa: E402

FAIL, CHECKS, NOTES = [], 0, []
N_PLANTS = 8
LEDGER = os.path.expanduser("~/.cache/scorch-codex/spmm-beat-mkl")


def eq(label, got, want):
    global CHECKS
    CHECKS += 1
    if got != want:
        FAIL.append("%s: got %r, expected %r" % (label, got, want))


def t_the_registry_is_not_empty():
    """An empty registry passes every other plant here while protecting nothing."""
    eq("nonempty", len(LX.EXCLUDED) >= 1, True)


def t_every_entry_has_three_nonempty_fields():
    """Structural: an entry cannot be added with a blank reason or a blank citation, which would
    refuse a reader while telling it nothing about why."""
    for name, val in LX.EXCLUDED.items():
        eq("triple-" + name, len(val), 3)
        for i, part in enumerate(val):
            eq("nonblank-%s-%d" % (name, i), bool(part and part.strip()), True)


def t_an_unregistered_run_returns_None():
    eq("clean", LX.citation("chain210-dlmc-replicate-redwood"), None)
    eq("clean-check", LX.check_run("/x/chain210-dlmc-replicate-redwood"), None)


def t_a_registered_run_refuses_and_prints_its_citation():
    try:
        LX.check_run("/anywhere/chain207-dlmc-redwood", label="plant")
        eq("refuse", "returned", "raised")
    except SystemExit as e:
        msg = str(e)
        eq("refuse", "raised", "raised")
        eq("names-run", "chain207-dlmc-redwood" in msg, True)
        eq("names-where", "PROVENANCE.md:10" in msg, True)
        eq("names-consequence", "set aside" in msg, True)


def t_allow_excluded_returns_the_citation_instead_of_raising():
    c = LX.check_run("/anywhere/chain207-dlmc-redwood", allow_excluded=True)
    eq("allowed-triple", len(c), 3)


def t_keying_is_on_the_basename_through_paths_and_trailing_slashes():
    for form in ("chain207-dlmc-redwood", "/a/b/chain207-dlmc-redwood",
                 "/a/b/chain207-dlmc-redwood/", "./chain207-dlmc-redwood"):
        eq("key-" + form, LX.citation(form) is not None, True)


def t_each_citation_names_a_path_that_EXISTS():
    """The registry's prose cites a file and a line. If that file is not there, the entry is an
    assertion rather than a record. Reports rather than passing when the ledger is absent."""
    if not os.path.isdir(LEDGER):
        NOTES.append("live plant did NOT run: %s absent, so no citation was resolved" % LEDGER)
        eq("ledger-missing-reported", True, True)
        return
    for name, (_why, where, _c) in LX.EXCLUDED.items():
        m = re.search(r"([\w.\-/]+\.md):(\d+)", where)
        eq("citation-parses-" + name, m is not None, True)
        if not m:
            continue
        path, line = os.path.join(LEDGER, m.group(1)), int(m.group(2))
        eq("citation-exists-" + name, os.path.exists(path), True)
        if os.path.exists(path):
            lines = open(path, errors="replace").read().split("\n")
            eq("citation-line-present-" + name, len(lines) >= line, True)
            # And the cited line must actually mention the excluded run.
            key = name.split("-")[0]
            eq("citation-line-mentions-" + name, key in "\n".join(lines[max(0, line - 2):line + 1]),
               True)


def t_glob_clean_refuses_on_an_excluded_run_and_honours_the_override():
    """Cross-module: every wired reader inherits the run check through glob_clean."""
    if not os.path.isdir(os.path.join(LEDGER, "chain207-dlmc-redwood")):
        NOTES.append("live plant did NOT run: chain207 directory absent")
        eq("glob-live-reported", True, True)
        return
    d = os.path.join(LEDGER, "chain207-dlmc-redwood")
    try:
        LC.glob_clean(d, "*.csv")
        eq("glob-refuse", "returned", "raised")
    except SystemExit:
        eq("glob-refuse", "raised", "raised")
    eq("glob-override", len(LC.glob_clean(d, "*.csv", allow_excluded=True)) > 0, True)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    for fn in tests:
        try:
            fn()
        except Exception as e:                                        # noqa: BLE001
            FAIL.append("%s CRASHED: %s: %s" % (fn.__name__, type(e).__name__, e))
    print("ledger_excluded_runs selftest: %d plants, %d assertions" % (len(tests), CHECKS))
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
