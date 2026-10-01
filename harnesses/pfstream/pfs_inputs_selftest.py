"""Plants for pfs_inputs. The shape that defeated a real audit is plant 4, reproduced exactly.

Every plant asserts a positive detection or a specific classification; two are ablations. The
plants write into a temporary directory OUTSIDE this tree, because committing a planted directory
into the scanned repo has already made one scan report defects on its own instruments.
"""
import ast
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pfs_inputs as P                                                # noqa: E402

FAIL, CHECKS = [], 0
N_PLANTS = 13


def eq(label, got, want):
    global CHECKS
    CHECKS += 1
    if got != want:
        FAIL.append("%s: got %r, expected %r" % (label, got, want))


def _dir(files, manifest=None):
    """Build a temp directory; `manifest` is a list of (digest, name) to write as SHA256SUMS."""
    d = tempfile.mkdtemp()
    for f in files:
        open(os.path.join(d, f), "w").write("x")
    if manifest is not None:
        with open(os.path.join(d, "SHA256SUMS"), "w") as fh:
            for dig, name in manifest:
                fh.write("%s  %s\n" % (dig, name))
    return d


SHA = "a" * 64
CRC = "1234abcd"


def t_the_declared_set_is_four_csvs_and_four_sentinels():
    r = P.required("bits")
    eq("count", len(r), 8)
    eq("csvs", sum(1 for n in r if n.endswith(".csv")), 4)
    eq("sentinels", sum(1 for n in r if n.endswith(".csv.done")), 4)


def t_an_undeclared_prefix_is_refused():
    try:
        P.required("dl_base")
        eq("undeclared", "returned", "raised")
    except ValueError:
        eq("undeclared", "raised", "raised")


def t_the_sentinel_name_is_DERIVED_from_the_csv_name():
    """Metamorphic: the two cannot drift, which is the defect that made a fused run read the
    drop-in run's sentinel."""
    for pre in P.PREFIXES:
        for b in P.BUILDS:
            eq("derived-%s-%s" % (pre, b),
               P.sentinel_name(pre, b).startswith(P.csv_name(pre, b)), True)


def t_the_real_chain_directory_shape_classifies_as_predating_not_dropped():
    """PLANT 4 -- the shape of every sealed directory in the ledger, and the one an auditor read
    as "the seal dropped the sentinels OR the producer never wrote them". Four CSVs, no sentinels,
    with a manifest naming the CSVs."""
    csvs = [P.csv_name("bits", b) for b in P.BUILDS]
    d = _dir(csvs, manifest=[(SHA, n) for n in csvs])
    sealed, unsealed, absent, has_man = P.audit(d, "bits")
    eq("p4-sealed", len(sealed), 4)
    eq("p4-unsealed", len(unsealed), 0)
    eq("p4-absent", len(absent), 4)
    eq("p4-absent-are-sentinels", all(n.endswith(".done") for n in absent), True)
    eq("p4-manifest", has_man, True)
    eq("p4-verdict", any("predating the sentinel" in ln for ln in P.describe(d, "bits")), True)


def t_a_missing_CSV_withdraws_the_predating_verdict():
    """ABLATION on plant 4: drop one CSV and the predating-producer line must NOT print, because
    then the absences are not all sentinels and the benign reading is no longer available."""
    csvs = [P.csv_name("bits", b) for b in P.BUILDS][:3]
    d = _dir(csvs, manifest=[(SHA, n) for n in csvs])
    eq("ablate-verdict", any("predating the sentinel" in ln for ln in P.describe(d, "bits")), False)
    eq("ablate-absent", len(P.audit(d, "bits")[2]), 5)


def t_present_but_unsealed_is_its_own_class():
    """A file on disk that no manifest names is mutable after the fact -- not the same as absent."""
    files = P.required("bits")
    d = _dir(files, manifest=[(SHA, P.csv_name("bits", "ship"))])
    sealed, unsealed, absent, _ = P.audit(d, "bits")
    eq("u-sealed", len(sealed), 1)
    eq("u-unsealed", len(unsealed), 7)
    eq("u-absent", len(absent), 0)


def t_no_manifest_and_an_empty_manifest_are_distinguished():
    """Returning one value for both cannot tell the reader which thing to fix."""
    files = P.required("bits")
    eq("none", P.manifest_names(os.path.join(_dir(files), "SHA256SUMS")), None)
    eq("empty", P.manifest_names(os.path.join(_dir(files, manifest=[]), "SHA256SUMS")), set())


def t_a_CRC32_manifest_line_is_not_read_as_a_digest():
    """9 of 25 manifests in this project are CRC-32; a 64-char field is the only check that
    separates them, and against a CRC file `shasum -c` prints zero OK and zero FAILED."""
    csvs = [P.csv_name("bits", b) for b in P.BUILDS]
    d = _dir(csvs, manifest=[(CRC, n) for n in csvs])
    eq("crc-names", P.manifest_names(os.path.join(d, "SHA256SUMS")), set())
    eq("crc-unsealed", len(P.audit(d, "bits")[1]), 4)


def t_a_manifest_star_prefix_and_a_path_both_resolve_to_the_basename():
    csvs = [P.csv_name("bits", b) for b in P.BUILDS]
    d = _dir(csvs, manifest=[(SHA, "*" + csvs[0]), (SHA, "./" + csvs[1])])
    eq("star-path", len(P.audit(d, "bits")[0]), 2)


def t_an_absent_directory_reports_every_input_absent_not_zero():
    """A missing directory must not read as a clean audit -- that is the vacuous-pass shape."""
    _, absent, = None, None
    sealed, unsealed, absent, has_man = P.audit(os.path.join(tempfile.mkdtemp(), "nope"), "bits")
    eq("nodir-absent", len(absent), 8)
    eq("nodir-sealed", len(sealed), 0)
    eq("nodir-manifest", has_man, False)


def t_bitcmp_and_pfs_inputs_declare_THE_SAME_producers():
    """Cross-file: the comparison's schema table and this declaration must list the same prefixes,
    or an auditor reading one of them measures a set the other never opens. Parsed from source
    rather than imported, because pfs_bitcmp executes at import."""
    tree = ast.parse(open(os.path.join(HERE, "pfs_bitcmp.py")).read())
    found = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SCHEMAS" for t in n.targets):
            found = {k.value for k in n.value.keys}
    eq("schemas-parsed", found is not None, True)
    eq("schemas-agree", found, set(P.PREFIXES))


def t_a_directory_holding_NONE_of_the_inputs_reports_a_domain_miss():
    """The chain-directory shape. All 8 absent is NOT the predating-producer reading: the
    comparison refuses on the first missing CSV, so the sentinel branch is never reached and a
    sentinel count over such a directory is not about this gate at all."""
    d = _dir(["dl_base_float32_p1.csv", "c211_run.log"], manifest=[(SHA, "dl_base_float32_p1.csv")])
    lines = P.describe(d, "bits")
    eq("domain-absent", len(P.audit(d, "bits")[2]), 8)
    eq("domain-said", any("not in this tool's" in ln for ln in lines), True)
    eq("domain-not-predating", any("predating the sentinel" in ln for ln in lines), False)


def t_a_mixed_absence_is_neither_of_the_other_two_readings():
    """ABLATION separating the three branches: two CSVs and one sentinel present."""
    files = [P.csv_name("bits", "ship"), P.csv_name("bits", "shpb"),
             P.sentinel_name("bits", "ship")]
    lines = P.describe(_dir(files), "bits")
    eq("mixed-said", any("MIXED absence" in ln for ln in lines), True)
    eq("mixed-not-domain", any("not in this tool's" in ln for ln in lines), False)
    eq("mixed-not-predating", any("predating the sentinel" in ln for ln in lines), False)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    for fn in tests:
        try:
            fn()
        except Exception as e:                                        # noqa: BLE001
            FAIL.append("%s CRASHED: %s: %s" % (fn.__name__, type(e).__name__, e))
    print("pfs_inputs selftest: %d plants, %d assertions" % (len(tests), CHECKS))
    rc = 0
    for f in FAIL:
        rc = 1
        print("  FAIL " + f)
    # A FLOOR ON THE PLANT COUNT. Plants defined below the entry point, or after a module-level
    # test list, are never reached and the run prints a green shortfall. Counting definitions and
    # comparing is the only check that catches it whatever the cause.
    if len(tests) != N_PLANTS:
        rc = 1
        print("  REFUSE: %d plants collected, the file declares %d -- a shortfall means some plant"
              " was never reached" % (len(tests), N_PLANTS))
    print("  " + ("OK" if rc == 0 else "FAILED"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
