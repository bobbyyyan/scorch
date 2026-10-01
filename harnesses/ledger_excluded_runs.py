"""Runs set aside as contaminated, recorded where TOOLING can read it instead of only prose.

WHY THIS FILE EXISTS. chain210's PROVENANCE.md line 10 reads "a within-chain diagnostic identifies
chain207 as the contaminated run". chain207's OWN PROVENANCE.md is 146 lines and contains zero
contamination words (control: the string "chain207" appears in it once, so the file is readable and
the zero is not a read failure), and none of its 22 filenames carries a marker. So the exclusion of
an entire run is recorded only in a SIBLING directory's prose, and a reader globbing chain207's
CSVs finds nothing anywhere in that directory saying the run was set aside. The analysis side does
act on it -- chain207's cells are set aside in the board screen -- which means the exclusion lives
in whoever remembers it.

THERE ARE TWO QUARANTINE CONVENTIONS IN THIS LEDGER AND THEY NEED DIFFERENT MECHANISMS.
A bad FILE is marked in its filename (three such files across two milestones:
chain194's hv_ex4b_float64_p2.contaminated.csv, and phase6-relayout-61b68be's
latency-base.contaminated.json and .log), which ledger_csvs.marker_in() reads. A bad RUN has no
filename to mark -- every file in it is individually well-formed -- so a name rule cannot express
it. Of the four places a marker can sit (filename, directory name, a sidecar in the directory,
prose) only the first two are visible to tooling, and this ledger uses the first and the fourth.

WHY A REGISTRY AND NOT A DIRECTORY-NAME RULE. Applying whole-token marker matching to directory
basenames hits 393 directories across the ledger -- overwhelmingly `tmp`, plus `invalid`, `stale`
and `partial` -- and inside the spmm-beat-mkl milestone it hits fixtures/u_bad and fixtures/u_broken,
which are deliberately-bad NEGATIVE TEST FIXTURES. True positives by the rule, false positives in
intent, and the discriminator (evidence versus fixture) is not in the name. So the run-level
mechanism has to be an explicit list, and it lives in the repo rather than in the ledger because
adding a marker file to a sealed directory changes sealed evidence.

IT FAILS CLOSED WITH AN AUDITABLE OVERRIDE. A reader pointed at an excluded run refuses and prints
the citation. A caller that genuinely wants the data -- the board screen that MEASURES the
contamination has to read it -- passes allow_excluded=True, which puts the decision at the call
site in the diff instead of in a memory.
"""
import os

# directory basename -> (what is wrong, where that is recorded, what downstream did about it)
EXCLUDED = {
    "chain207-dlmc-redwood": (
        "flagged by the board contamination screen: the pass's own wall time predicts its "
        "ours/mkl ratio on 3 of 5 widths, spread 3.87%, against 0.66-0.67% for the runs that "
        "cleared",
        "chain210-dlmc-replicate-redwood/PROVENANCE.md:10 -- 'a within-chain diagnostic "
        "identifies chain207 as the contaminated run'. chain207's own PROVENANCE.md says "
        "nothing; this registry is the only machine-readable record.",
        "its 10 behind-MKL cells are set aside; chain206's 14 and chain210's 22 stand"),
}


def citation(name):
    """-> (reason, where_recorded, consequence) or None. Keyed on the directory's basename."""
    return EXCLUDED.get(os.path.basename(os.path.normpath(name)))


def check_run(d, allow_excluded=False, label=""):
    """Refuse if `d` is a run set aside as contaminated, unless the caller says otherwise.

    Returns the citation tuple when the run is excluded and the caller allowed it, so a reader
    that proceeds can print WHY it is reading contaminated data. Returns None for a clean run.
    """
    c = citation(d)
    if c is None:
        return None
    if allow_excluded:
        return c
    why, where, conseq = c
    raise SystemExit(
        "REFUSE: %s%s is a run set aside as contaminated.\n"
        "  what is wrong: %s\n"
        "  recorded at:   %s\n"
        "  downstream:    %s\n"
        "  Nothing in this directory says so -- that is why this registry exists. If you mean to\n"
        "  read it anyway (the screen that measures the contamination must), pass\n"
        "  allow_excluded=True so the decision is in the diff rather than in a memory."
        % (label + ": " if label else "", os.path.basename(os.path.normpath(d)),
           why, where, conseq))
