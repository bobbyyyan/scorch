"""The ONE declaration of which files the four-build comparison opens. Imported, never copied.

WHY THIS FILE EXISTS. A reviewer measured "zero `.done` sentinels in any sealed directory" and
concluded `pfs_bitcmp.py`'s gate had never gated a filed result, offering two readings: the seal
drops `*.done`, or the producer never wrote one. Both were false. The twelve directories counted
over hold a different producer's CSVs entirely -- `dl_base`/`dl_bsb_`/`dl_deep` pass files -- and
no file matching `<prefix>_<build>.csv` has ever been sealed. The gate had never gated a filed
result because its inputs had never been filed.

The control in that measurement was sound (2104 CSVs present in the same milestone directory, so
the search reached files) and still did not help, because a non-vacuity control shows the search
reached SOMETHING, not that what it reached is the tool's domain. Those are two questions, and the
second one can only be answered from the consumer's own input pattern. So the pattern is declared
here once and the consumer DERIVES ITS PATHS FROM IT -- if the declaration drifts from what the
gate opens, the gate breaks loudly instead of an auditor measuring the wrong set quietly.

The second half, `audit()`, is the check that would have caught it: set-difference the declared
inputs against a directory and against its manifest. It reports three disjoint quantities, because
collapsing them is what made the original reading ambiguous -- present-and-sealed,
present-but-unsealed, and absent are three different failures with three different owners.
"""
import os

BUILDS = ("ship", "shpb", "strm", "nopf")
BASE = BUILDS[0]
PREFIXES = ("bits", "bitsf")


def csv_name(prefix, build):
    return "%s_%s.csv" % (prefix, build)


def sentinel_name(prefix, build):
    """The producer writes this only after closing its CSV, so absence means death at the tail.

    Derived from csv_name rather than re-spelled: parameterising the CSV path and not this one is
    a defect that already happened once, and it made the fused run read the drop-in run's
    sentinel.
    """
    return csv_name(prefix, build) + ".done"


def required(prefix):
    """Every file the comparison opens, in a stable order. Nothing else is its input."""
    if prefix not in PREFIXES:
        raise ValueError("no declared input set for prefix %r (known: %s)"
                         % (prefix, ", ".join(PREFIXES)))
    out = []
    for b in BUILDS:
        out.append(csv_name(prefix, b))
        out.append(sentinel_name(prefix, b))
    return out


def manifest_names(path):
    """Filenames listed in a SHA256SUMS-style manifest, or None if there is no manifest.

    None and the empty set are deliberately distinguished: "no manifest" is an unsealed directory
    and "a manifest naming nothing" is a broken seal, and an auditor that returns the same value
    for both cannot tell the reader which one to go fix.
    """
    if not os.path.exists(path):
        return None
    names = set()
    for ln in open(path, errors="replace"):
        parts = ln.split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            names.add(os.path.basename(parts[1].strip().lstrip("*")))
    return names


def audit(d, prefix, manifest="SHA256SUMS"):
    """(sealed, unsealed, absent, manifest_present) over the declared inputs of `prefix`.

    sealed   -- present on disk AND named by the manifest
    unsealed -- present on disk, NOT named by the manifest (mutable after the fact)
    absent   -- not present at all
    """
    names = manifest_names(os.path.join(d, manifest))
    sealed, unsealed, absent = [], [], []
    for n in required(prefix):
        if not os.path.exists(os.path.join(d, n)):
            absent.append(n)
        elif names is None or n not in names:
            unsealed.append(n)
        else:
            sealed.append(n)
    return sealed, unsealed, absent, names is not None


def describe(d, prefix, manifest="SHA256SUMS"):
    """Lines a caller prints. The COUNT of each class, and the names, because a bare count of
    absences cannot say whether the CSVs or only the sentinels are missing -- which is the exact
    distinction the reviewer's two readings turned on."""
    sealed, unsealed, absent, has_man = audit(d, prefix, manifest)
    tot = len(required(prefix))
    out = ["  declared inputs for prefix %r: %d (%d CSVs + %d sentinels)"
           % (prefix, tot, len(BUILDS), len(BUILDS)),
           "  manifest %s: %s" % (manifest, "present" if has_man else "ABSENT (unsealed directory)"),
           "  sealed %d, present-but-unsealed %d, absent %d" % (len(sealed), len(unsealed), len(absent))]
    for lab, xs in (("UNSEALED", unsealed), ("ABSENT", absent)):
        if xs:
            out.append("    %s: %s" % (lab, " ".join(xs)))
    # THREE READINGS OF AN ABSENCE, and they are not interchangeable. The reviewer who measured
    # this population reported the sentinel as missing; in fact the comparison refuses on the CSV
    # first (the existence check precedes the sentinel check by fifty lines), so on a directory
    # holding neither, the sentinel branch is unreachable and a finding about it cannot fire.
    if len(absent) == tot:
        out.append("    ^ EVERY declared input is absent. This directory is not in this tool's")
        out.append("      domain -- it holds no file the comparison opens, so no count over it is")
        out.append("      a statement about this gate. Check the producer, not the seal.")
    elif absent and len(absent) == len(BUILDS) and all(n.endswith(".done") for n in absent):
        out.append("    ^ every absence is a sentinel and every CSV is present: this is a run by a")
        out.append("      producer predating the sentinel, not a seal that dropped files.")
    elif absent:
        out.append("    ^ a MIXED absence: some CSVs and some sentinels. The comparison refuses on")
        out.append("      the first missing CSV, so the sentinel state of the rest is unexamined.")
    return out
