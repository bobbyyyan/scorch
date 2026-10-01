"""Plants for attest.py. Each is a directory shaped so that a defect prints a PASS.

Every plant here existed first as a demonstration in a peer session's transcript and
nowhere in the repo, which is the gap this file closes: a control that is not committed
cannot be re-run against a later edit. The two that matter most are a matched pair --
a tamper that the tool used to ATTEST, and the SAME tamper without the manifest rewrite,
which it correctly refuses. Without the second, the first proves only that the tool is
broken somehow; with it, the pass is informative.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ATTEST = os.path.join(HERE, "attest.py")
FAIL, CHECKS = [], 0


def sha(b):
    return hashlib.sha256(b).hexdigest()


def mkdir_ledger(files, manifest=None, seal=None, extra=None):
    """files: {name: bytes}. manifest/seal default to correct ones over `files`."""
    d = os.path.realpath(tempfile.mkdtemp(prefix="attest."))
    for n, b in files.items():
        with open(os.path.join(d, n), "wb") as fh:
            fh.write(b)
    if manifest is None:
        manifest = "".join("%s  %s\n" % (sha(b), n) for n, b in sorted(files.items()))
    with open(os.path.join(d, "SHA256SUMS"), "w") as fh:
        fh.write(manifest)
    if seal is None:
        seal = "%s  SHA256SUMS\n" % sha(manifest.encode())
    if seal is not False:
        with open(os.path.join(d, "SHA256SUMS.sha256"), "w") as fh:
            fh.write(seal)
    for n, b in (extra or {}).items():
        with open(os.path.join(d, n), "wb") as fh:
            fh.write(b)
    return d


def run(*args):
    r = subprocess.run([sys.executable, ATTEST] + list(args), capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def expect(label, got, rc_want, *must):
    global CHECKS
    CHECKS += 1
    rc, out = got
    if rc != rc_want:
        FAIL.append("%s: rc=%d, expected %d\n    %s" % (label, rc, rc_want, out.strip()[-500:]))
        return
    for m in must:
        CHECKS += 1
        if m not in out:
            FAIL.append("%s: rc was right but %r is absent\n    %s"
                        % (label, m, out.strip()[-500:]))


# ----------------------------------------------- the pair: a tamper that used to ATTEST
def t_stale_seal_of_manifest_refuses():
    """THE DEFECT. `pick()` returned max-by-digest-line-count, and a seal of a manifest has
    ONE line, so it could never win and was discarded after being labelled a real seal.
    Tamper a data file, then rewrite SHA256SUMS to agree: the directory verified clean and
    printed ATTESTED while `shasum -a 256 -c SHA256SUMS.sha256` reported SHA256SUMS: FAILED.
    """
    good = {"a.csv": b"original\n", "b.csv": b"other\n"}
    d = mkdir_ledger(good)                      # seal is over the ORIGINAL manifest
    with open(os.path.join(d, "a.csv"), "wb") as fh:
        fh.write(b"tampered\n")
    files = {"a.csv": b"tampered\n", "b.csv": b"other\n"}
    with open(os.path.join(d, "SHA256SUMS"), "w") as fh:   # rewritten to agree
        fh.write("".join("%s  %s\n" % (sha(b), n) for n, b in sorted(files.items())))
    expect("stale-seal", run(d, "*.csv"), 5, "STALE", "seal(s) of the manifest are STALE")
    shutil.rmtree(d, ignore_errors=True)


def t_the_same_tamper_without_the_rewrite_still_refuses():
    """NON-VACUITY for the plant above. Identical tamper, manifest left alone -> the ordinary
    MISMATCH refusal at step 2. Without this, 'it refuses the tampered directory' would be
    satisfied by a tool that refuses everything."""
    d = mkdir_ledger({"a.csv": b"original\n", "b.csv": b"other\n"})
    with open(os.path.join(d, "a.csv"), "wb") as fh:
        fh.write(b"tampered\n")
    expect("tamper-no-rewrite", run(d, "*.csv"), 3, "MISMATCH", "a.csv")
    shutil.rmtree(d, ignore_errors=True)


def t_a_current_seal_attests():
    """And the untampered directory must still pass, or the two refusals above are trivial."""
    d = mkdir_ledger({"a.csv": b"original\n", "b.csv": b"other\n"})
    expect("clean-attests", run(d, "*.csv"), 0, "ATTESTED", "MATCH", "<- seals SHA256SUMS")
    shutil.rmtree(d, ignore_errors=True)


def t_a_full_manifest_named_SHA256SUMS_sha256_is_still_CHOSEN():
    """THE NON-REGRESSION for the fix. In this ledger `SHA256SUMS.sha256` is a 1-line seal in
    two directories and a 186-line data manifest in another, so classifying by NAME would
    break the second kind. The discriminator is whether its entries name another candidate.
    Here SHA256SUMS is cksum output (verifies nothing) and the .sha256 file is the real one.
    """
    files = {"a.csv": b"original\n", "b.csv": b"other\n"}
    d = os.path.realpath(tempfile.mkdtemp(prefix="attest."))
    for n, b in files.items():
        with open(os.path.join(d, n), "wb") as fh:
            fh.write(b)
    with open(os.path.join(d, "SHA256SUMS"), "w") as fh:
        fh.write("".join("1234567890 %d %s\n" % (len(b), n) for n, b in sorted(files.items())))
    with open(os.path.join(d, "SHA256SUMS.sha256"), "w") as fh:
        fh.write("".join("%s  %s\n" % (sha(b), n) for n, b in sorted(files.items())))
    expect("full-manifest-dot-sha256", run(d, "*.csv"), 0,
           "chose SHA256SUMS.sha256 BY CONTENT", "ATTESTED", "verifies nothing with shasum")
    shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------------- a run that certifies no file
def t_no_globs_is_not_attested():
    """`attest.py "$d" && publish` used to get ATTESTED rc=0 with zero file-level coverage."""
    d = mkdir_ledger({"a.csv": b"original\n"})
    expect("no-globs", run(d), 6, "NOT ATTESTED", "certifies no file")
    rc, out = run(d)
    global CHECKS
    CHECKS += 1
    if "  ATTESTED" in out.replace("NOT ATTESTED", ""):
        FAIL.append("no-globs: the word ATTESTED still appears on its own")
    expect("seal-only", run("--seal-only", d), 0, "SEAL-VERIFIED (NOT ATTESTED)")
    expect("seal-only-with-globs", run("--seal-only", d, "*.csv"), 1, "two different requests")
    shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------- the two latent holes
def t_dropped_manifest_lines_refuse():
    """A line parsing as neither digest shape was dropped and compared against nothing."""
    files = {"a.csv": b"original\n"}
    mf = "%s  a.csv\nthis line is prose and names nothing\n" % sha(b"original\n")
    d = mkdir_ledger(files, manifest=mf)
    expect("dropped-lines", run(d, "*.csv"), 3, "were DROPPED", "non-blank lines")
    shutil.rmtree(d, ignore_errors=True)


def t_cover_uses_one_predicate():
    """`n_in` required sealed AND contained; `outside` tested only sealed, so a sealed name
    whose bytes live outside the directory counted as inside. Unreachable through main()
    because step 2 refuses first, so it is exercised at the function."""
    sys.path.insert(0, HERE)
    import attest                                                    # noqa: E402
    global CHECKS
    out = os.path.realpath(tempfile.mkdtemp(prefix="attest.out."))
    d = os.path.realpath(tempfile.mkdtemp(prefix="attest."))
    with open(os.path.join(out, "real.csv"), "wb") as fh:
        fh.write(b"outside the seal\n")
    os.symlink(os.path.join(out, "real.csv"), os.path.join(d, "a.csv"))
    CHECKS += 1
    if attest.cover(d, ["*.csv"], {"a.csv"}):
        FAIL.append("cover: a sealed name whose bytes escape the directory counted as inside")
    CHECKS += 1
    with open(os.path.join(d, "b.csv"), "wb") as fh:
        fh.write(b"really here\n")
    if not attest.cover(d, ["b.csv"], {"b.csv"}):
        FAIL.append("cover: a genuinely contained sealed file was reported outside, so the "
                    "check above passes for the wrong reason")
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)


def t_cksum_only_still_refuses():
    """The original guard: a cksum manifest verifies nothing and must not read as a pass."""
    d = os.path.realpath(tempfile.mkdtemp(prefix="attest."))
    with open(os.path.join(d, "a.csv"), "wb") as fh:
        fh.write(b"x\n")
    with open(os.path.join(d, "SHA256SUMS"), "w") as fh:
        fh.write("1234567890 2 a.csv\n")
    expect("cksum-only", run(d, "*.csv"), 2, "no candidate is a data manifest")
    shutil.rmtree(d, ignore_errors=True)


def t_unsealed_manifest_says_so():
    """No seal present is not an error, but it must be stated: the manifest's own bytes are
    then unsealed, and a silent step would read as 'checked, fine'."""
    d = mkdir_ledger({"a.csv": b"original\n"}, seal=False)
    expect("no-seal", run(d, "*.csv"), 0, "none present, so this step bounds nothing",
           "ATTESTED")
    shutil.rmtree(d, ignore_errors=True)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("t_")]


def main():
    for fn in TESTS:
        try:
            fn()
        except Exception as e:                                       # noqa: BLE001
            FAIL.append("%s CRASHED: %s: %s" % (fn.__name__, type(e).__name__, e))
    print("attest selftest: %d plants, %d assertions" % (len(TESTS), CHECKS))
    rc = 0
    for f in FAIL:
        rc = 1
        print("  FAIL " + f)
    if CHECKS < 20:
        rc = 1
        print("  REFUSE: only %d assertions ran, so the pass above is vacuous" % CHECKS)
    print("  " + ("OK" if rc == 0 else "FAILED"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
