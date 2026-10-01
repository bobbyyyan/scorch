#!/usr/bin/env python3
"""Attest that the specific files a claim rests on sit inside a manifest that verifies.

Four mechanical steps, no judgement at any of them:

  1. PICK THE MANIFEST BY CONTENT, never by name. Count 64-hex-digest lines in each
     candidate. A file named SHA256SUMS may be `cksum` output -- CRC-32 plus a byte count --
     and `shasum -a 256 -c` on such a file prints zero OK AND zero FAILED, which is
     indistinguishable from a passing verification. In this ledger the same filename denotes
     different functions in different directories: chain204-k32gate-mkt/SHA256SUMS is real
     SHA-256, harnesses/SHA256SUMS is CRC-32 and the real seal is SHA256SUMS.sha256. A script
     hardcoding either convention is silently wrong in the other, and "wrong" means printing
     a pass.
  2. VERIFY, asserting emitted verdicts == manifest lines == OK count. Any one of those alone
     can pass vacuously.
  3. ASSERT THE CLAIM'S OWN INPUTS are in the verified set. A 151-entry manifest can verify
     completely while the 90 CSVs a number came from sit outside it. Directory attestation is
     not file attestation.
  4. FAIL CLOSED with a structured reason, and report candidates beside matches so a zero
     carries information.

Usage:  attest.py [--require-seal] <dir> <glob> [glob ...]
        attest.py [--require-seal] --seal-only <dir>

        Every run ends with one `VERDICT: <TOKEN>` line from the VERDICTS enumeration below,
        so a caller branches on that rather than on a sentence. A manifest with no seal of its
        own gets a DIFFERENT token from one whose seal matched, because rc=0 and the word
        SEAL-VERIFIED were previously identical for both. `--require-seal` turns the absent
        case into a refusal; it is off by default because 2 of 28 chain directories in this
        ledger carry such a seal and 26 do not.

        globs are the claim's input files, relative to <dir>. ATTESTED requires at least one,
        because a run that names no file certifies no file. `--seal-only` runs steps 1, 2 and
        2B alone and says SEAL-VERIFIED (NOT ATTESTED), so the two verdicts cannot be
        confused by a caller writing `attest.py "$d" && publish`.
"""
import glob as globmod
import hashlib
import os
import re
import sys

HEX64 = re.compile(r"^[0-9a-f]{64}$")

# The verdict as a DECLARED ENUMERATION on its own machine-readable line, because the human
# sentences were not distinguishable by a caller. `SEAL-VERIFIED` was printed both when a
# seal matched and when no seal existed at all -- the same word and the same rc=0 for "the
# manifest's bytes are bound" and "nothing binds the manifest's bytes", which is the shape of
# the no-globs defect reproduced inside the mode added to fix it. One null got its own verdict
# and exit, the other got an informational line beside a pass.
#
# NO TOKEN MAY BE A PREFIX OF ANOTHER. A `grep -c "^  FAIL"` matching a suite's own `FAILED`
# summary miscounted four assertions as five earlier today, so the property is asserted in
# attest_selftest.py rather than left to inspection.
VERDICTS = (
    "ATTESTED_SEALED_MANIFEST",            # inputs inside, and a seal binds the manifest
    "ATTESTED_UNSEALED_MANIFEST",          # inputs inside, but nothing binds the manifest
    "SEAL_AND_MANIFEST_VERIFIED",          # --seal-only, seal present and matching
    "MANIFEST_VERIFIED_NO_SEAL_PRESENT",   # --seal-only, nothing binds the manifest
    "NOT_ATTESTED_NO_INPUTS_NAMED",
    "REFUSED_NO_DATA_MANIFEST",
    "REFUSED_MANIFEST_UNCLEAN",
    "REFUSED_SEAL_STALE",
    "REFUSED_INPUTS_OUTSIDE_SEAL",
    "REFUSED_SEAL_REQUIRED_BUT_ABSENT",
    "REFUSED_BAD_USAGE",
)


def verdict(tok, rc):
    assert tok in VERDICTS, tok
    print("VERDICT: %s" % tok)
    return rc


def manifest_kind(path):
    """(n_sha256_lines, n_cksum_lines, n_total_lines) -- the content test, not the name."""
    nh = nc = nt = 0
    with open(path, errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            nt += 1
            f = line.split()
            if len(f) >= 2 and HEX64.match(f[0]):
                nh += 1
            elif len(f) == 3 and f[0].isdigit():
                nc += 1
    return nh, nc, nt


def entry_names(path):
    """The names a manifest's digest lines refer to."""
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 2 and HEX64.match(f[0]):
                out.append(denorm(" ".join(f[1:])))
    return out


def pick(d):
    """-> (data_manifest, [(seal, the manifest it seals)]). BOTH, never one of the two.

    The original chose `max(real, key=n_sha256_lines)`, which can NEVER choose a seal of a
    manifest: such a seal has ONE line and always loses to the manifest it seals, so it was
    listed as a candidate, labelled a real seal, and then discarded. Demonstrated: tamper a
    data file AND rewrite SHA256SUMS to agree, leaving SHA256SUMS.sha256 stale -> ATTESTED,
    rc=0, while `shasum -a 256 -c SHA256SUMS.sha256` reports SHA256SUMS: FAILED.

    The selection cannot be fixed by name or by digest field width: in this ledger
    `SHA256SUMS.sha256` is a 1-line seal-of-manifest in two directories (chain207-dlmc-
    redwood, chain209-dlmc-dims) and a full data manifest in two others (harnesses,
    chain210). The discriminator is whether its entries NAME another candidate manifest --
    content, and not the field width.

    So this does not choose between them. The data manifest is picked as before from the
    candidates that are not seals, and every seal present is returned for verification too.
    That also dissolves the trade: honouring a seal only when it is the chosen manifest
    would mean refusing directories whose manifest was legitimately regenerated, whereas
    verifying both reports exactly which of the two is stale.
    """
    cands = sorted(f for f in os.listdir(d)
                   if re.search(r"SHA256SUMS|MANIFEST|CHECKSUM", f, re.I)
                   and os.path.isfile(os.path.join(d, f)))
    scored = [(f,) + manifest_kind(os.path.join(d, f)) for f in cands]
    cand_set = set(cands)
    seals = []
    for f, nh, _nc, _nt in scored:
        if not nh:
            continue
        names = entry_names(os.path.join(d, f))
        sealed_mfs = [n for n in names if n in cand_set and n != f]
        if names and len(sealed_mfs) == len(names):
            seals.append((f, sealed_mfs))
    seal_names = set(f for f, _ in seals)
    print("  STEP 1  manifest candidates: %d" % len(scored))
    for f, nh, nc, nt in scored:
        if f in seal_names:
            what = "<- seals %s" % ",".join(dict(seals)[f])
        elif nh:
            what = "<- data manifest"
        else:
            what = "(verifies nothing with shasum)"
        print("            %-24s lines=%-5d sha256=%-5d cksum=%-5d %s"
              % (f, nt, nh, nc, what))
    data = [s for s in scored if s[1] > 0 and s[0] not in seal_names]
    if len(data) > 1:
        # Choosing the larger of two data manifests is a judgement, and this tool's premise
        # is that it makes none. chain207-dlmc-redwood carries SHA256SUMS (20 entries) and
        # SHA256SUMS.producing (14) side by side, and the memo about the real seal sitting
        # beside a weaker one under a different name applies to two REAL ones as well. Said
        # out loud rather than resolved, so a reader knows which set was verified.
        print("            NOTE: %d data manifests are present and the largest is chosen by "
              "entry count. The others are NOT verified by this run: %s"
              % (len(data), ", ".join(f for f, *_ in sorted(data, key=lambda s: -s[1])[1:])))
    if not data:
        print("  REFUSE: no candidate is a data manifest containing a 64-hex digest; this "
              "directory has no verifiable seal over its own files, and shasum -c on what "
              "it does have prints zero OK and zero FAILED")
        return None, seals
    return max(data, key=lambda s: s[1])[0], seals


def verify_seals(d, seals):
    """Each seal-of-manifest checked against the manifest it names. A stale one is a REFUSAL.

    A stale seal means the manifest's own bytes changed after it was sealed, which is
    indistinguishable from the tamper above. If the manifest was regenerated on purpose, the
    remedy is to re-seal it; a directory whose two seals disagree is not attestable on its
    own evidence, and saying so is the whole point of holding both.
    """
    print("  STEP 2B seals of the manifest itself: %d" % len(seals))
    if not seals:
        print("            none present, so this step bounds nothing -- the data manifest's "
              "own bytes are unsealed here")
        return True
    bad = 0
    for seal, targets in seals:
        want = {}
        with open(os.path.join(d, seal), errors="replace") as fh:
            for line in fh:
                f = line.split()
                if len(f) >= 2 and HEX64.match(f[0]):
                    want[denorm(" ".join(f[1:]))] = f[0]
        for t in targets:
            got = file_sha256(os.path.join(d, t))
            state = "MATCH" if got == want.get(t) else "STALE"
            if state == "STALE":
                bad += 1
            print("            %-24s over %-20s %s" % (seal, t, state))
            if state == "STALE":
                print("              sealed %s" % want.get(t))
                print("              actual %s" % got)
    if bad:
        print("  REFUSE: %d seal(s) of the manifest are STALE -- the manifest's bytes changed "
              "after it was sealed, which is what a rewritten manifest looks like. Re-seal "
              "it, or the two seals disagree and neither can attest the other" % bad)
    return bad == 0


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def denorm(raw):
    """Strip the manifest's PREFIX markers only. `lstrip("*./")` is a character-set strip and
    would turn `.hidden_p1.csv` into `hidden_p1.csv`, reporting a legitimately dot-prefixed
    sealed file as ABSENT while naming the mangled form rather than the mangling."""
    return raw.removeprefix("*").removeprefix("./")


def contained(d, name):
    """Is the sealed NAME backed by bytes inside the sealed directory? A sealed name is not
    sealed bytes if the name is a symlink: shasum follows it, so the digest matches the
    target's current contents and verification is clean while the bytes live outside the
    ledger with nothing binding them there."""
    p = os.path.join(d, name)
    real = os.path.realpath(p)
    root = os.path.realpath(d)
    return real == root or real.startswith(root + os.sep)


def verify(d, mf):
    entries = {}
    parsed = 0
    dups = []
    _nh, _nc, nt = manifest_kind(os.path.join(d, mf))
    with open(os.path.join(d, mf), errors="replace") as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 2 and HEX64.match(f[0]):
                parsed += 1
                name = denorm(" ".join(f[1:]))
                if name in entries:
                    dups.append(name)
                entries[name] = f[0]
    # A dict collapses duplicate names, and len(entries) is then the DEDUPLICATED count. If it
    # were compared only against itself, a 258-line manifest with 250 distinct names would
    # report 250/250/250 clean, silently drop 8 entries, and print an invariant sentence that
    # is untrue. Refuse on ambiguity WITHIN a manifest, as between manifests.
    # A line matching neither the 64-hex nor the cksum shape is simply not counted, and
    # comparing `parsed` only against itself cannot see that -- the same shape as the
    # duplicate-name hole below, which IS refused. Zero instances in this ledger today
    # (every sha-manifest has nt == parsed), so this is a guard against a future producer,
    # not a repair.
    if parsed != nt:
        print("  STEP 2  REFUSE: manifest has %d non-blank lines but only %d parse as a "
              "digest entry -- %d line(s) were DROPPED, so the counts below would describe "
              "a subset of the seal under the seal's name" % (nt, parsed, nt - parsed))
        return False, set()
    if parsed != len(entries):
        print("  STEP 2  REFUSE: manifest has %d digest lines but only %d distinct names -- "
              "%d duplicated" % (parsed, len(entries), parsed - len(entries)))
        for n in sorted(set(dups))[:10]:
            print("            DUPLICATE ENTRY: %s" % n)
        return False, set()
    # A byte-string distinctness check treats a NAME as an identifier. The filesystem may
    # not: on a case-insensitive volume `data_p1.csv` and `Data_p1.csv` are one file, and on
    # macOS two names differing only by Unicode normalisation are too. Collapsing is then
    # invisible to any string-level test -- shasum reads the single file twice and reports OK
    # twice, certifying two entries against one file's bytes. Identity is the (device, inode)
    # pair, which covers case collision, normalisation collision and hardlink aliasing
    # uniformly rather than special-casing one of them. This ledger is produced on
    # case-sensitive hosts and read on a case-insensitive one, so the trigger is ordinary use.
    inodes = {}
    aliased = []
    for name in entries:
        q = os.path.join(d, name)
        if not os.path.isfile(q):
            continue
        st = os.stat(q)
        key = (st.st_dev, st.st_ino)
        if key in inodes:
            aliased.append((inodes[key], name))
        else:
            inodes[key] = name
    if aliased:
        print("  STEP 2  REFUSE: %d manifest entries resolve to a file another entry already "
              "names -- distinct names, one set of bytes" % len(aliased))
        for a, b in aliased[:10]:
            print("            SAME FILE: %r and %r%s"
                  % (a, b, "  (differ only in case)" if a.lower() == b.lower() else ""))
        return False, set()
    ok = bad = missing = 0
    escaped = []
    failed = []
    for name, want in entries.items():
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            missing += 1
            failed.append((name, "ABSENT"))
            continue
        if file_sha256(p) != want:
            bad += 1
            failed.append((name, "MISMATCH"))
        elif not contained(d, name):
            escaped.append((name, os.path.realpath(os.path.join(d, name))))
            ok += 1
        else:
            ok += 1
    emitted = ok + bad + missing
    print("  STEP 2  digest lines=%d  distinct=%d  emitted=%d  OK=%d  MISMATCH=%d  ABSENT=%d  "
          "ESCAPED=%d" % (parsed, len(entries), emitted, ok, bad, missing, len(escaped)))
    for n, r in escaped[:10]:
        print("            ESCAPES THE SEAL: %s -> %s" % (n, r))
    clean = (emitted == parsed == len(entries) == ok and bad == 0 and missing == 0
             and not escaped)
    if not clean:
        print("  REFUSE: not a clean pass -- digest lines, distinct names, emitted and OK "
              "must all be equal, with nothing escaping the directory")
        for n, why in failed[:10]:
            print("            %-8s %s" % (why, n))
    else:
        print("            emitted == manifest lines == OK, so the pass is not the 0/0 artefact")
    return clean, set(entries)


def cover(d, patterns, sealed):
    print("  STEP 3  claim inputs asserted against the verified set")
    if not patterns:
        print("            NOT EXERCISED -- no input globs given, so this step proves nothing")
        return True
    total = inside = 0
    outside = []
    for pat in patterns:
        hits = sorted(os.path.relpath(p, d)
                      for p in globmod.glob(os.path.join(d, pat), recursive=True)
                      if os.path.isfile(p))
        n_in = sum(1 for h in hits if h in sealed and contained(d, h))
        total += len(hits)
        inside += n_in
        # `n_in` requires sealed AND contained; `outside` must use the SAME predicate or
        # a sealed-but-escaping hit prints outside=1 and the function still returns True.
        # Masked today by step 2's own escape check, which refuses first.
        outside += [h for h in hits if h not in sealed or not contained(d, h)]
        print("            %-34s candidates=%-5d inside=%-5d outside=%d"
              % (pat, len(hits), n_in, len(hits) - n_in))
        if not hits:
            print("              NOTE: 0 candidates, so this glob's 0-outside carries no "
                  "information")
    if total and not outside:
        print("            all %d claim inputs are inside a set that verifies" % total)
    for h in outside[:10]:
        print("            OUTSIDE THE SEAL: %s" % h)
    return total > 0 and not outside


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    argv = sys.argv[1:]
    # `attest.py <dir>` with no globs used to print ATTESTED and exit 0 after reporting
    # step 3 as NOT EXERCISED, so `attest.py "$d" && publish` got a pass with zero
    # file-level coverage -- which lines 17-18 above call not-attestation. The standalone
    # seal check is still available, but it has to be ASKED for, and it says a different
    # word. A caller who wants the old exit code now says so in the command line.
    seal_only = "--seal-only" in argv
    # An ABSENT seal is a weaker attestation than a matching one, but whether it is a FAILURE
    # is a producer convention, not this tool's call: 2 of 28 chain directories in the ledger
    # carry a seal-of-manifest and 26 do not, so refusing by default would refuse the 26 and
    # every pre-chain207 directory. So the verdict always distinguishes them and the exit code
    # changes only when the caller asks for the stricter convention.
    require_seal = "--require-seal" in argv
    argv = [a for a in argv if a not in ("--seal-only", "--require-seal")]
    if not argv:
        print("REFUSE: no directory given")
        return verdict("REFUSED_BAD_USAGE", 1)
    d = os.path.abspath(argv[0])
    pats = argv[1:]
    print("ATTEST %s" % d)
    if not os.path.isdir(d):
        sys.exit("REFUSE: %s is not a directory" % d)
    mf, seals = pick(d)
    if mf is None:
        return verdict("REFUSED_NO_DATA_MANIFEST", 2)
    print("          chose %s BY CONTENT" % mf)
    clean, sealed = verify(d, mf)
    if not clean:
        return verdict("REFUSED_MANIFEST_UNCLEAN", 3)
    if not verify_seals(d, seals):
        return verdict("REFUSED_SEAL_STALE", 5)
    if require_seal and not seals:
        print("  REFUSE: --require-seal was given and no seal of the manifest is present, so "
              "the manifest's own bytes are bound by nothing")
        return verdict("REFUSED_SEAL_REQUIRED_BUT_ABSENT", 7)
    if seal_only:
        if pats:
            print("REFUSE: --seal-only was given together with %d input glob(s). Those globs "
                  "are the claim's files and asking to skip them while naming them is two "
                  "different requests" % len(pats))
            return verdict("REFUSED_BAD_USAGE", 1)
        if seals:
            print("  steps 1, 2 and 2B only. A seal binds the manifest's bytes, but no claim "
                  "inputs were named, so no file is attested by this run")
            return verdict("SEAL_AND_MANIFEST_VERIFIED", 0)
        print("  steps 1 and 2 only. The manifest verifies, NOTHING BINDS ITS BYTES, and no "
              "claim inputs were named, so no file is attested by this run")
        return verdict("MANIFEST_VERIFIED_NO_SEAL_PRESENT", 0)
    if not pats:
        print("  no input globs were given, so step 3 proved nothing and this run certifies "
              "no file. Pass the claim's inputs, or --seal-only to ask for a seal check and "
              "get a zero exit for it")
        return verdict("NOT_ATTESTED_NO_INPUTS_NAMED", 6)
    if not cover(d, pats, sealed):
        return verdict("REFUSED_INPUTS_OUTSIDE_SEAL", 4)
    if seals:
        return verdict("ATTESTED_SEALED_MANIFEST", 0)
    print("  NOTE: the claim's inputs are inside a manifest that verifies, but NOTHING BINDS "
          "THAT MANIFEST'S BYTES -- no seal of it is present. Pass --require-seal to make "
          "that a refusal.")
    return verdict("ATTESTED_UNSEALED_MANIFEST", 0)


if __name__ == "__main__":
    sys.exit(main())
