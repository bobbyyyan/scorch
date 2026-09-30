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

Usage:  attest.py <dir> [glob ...]
        globs are the claim's input files, relative to <dir>. With none, only steps 1-2 run
        and step 3 is reported as NOT EXERCISED rather than as a pass.
"""
import glob as globmod
import hashlib
import os
import re
import sys

HEX64 = re.compile(r"^[0-9a-f]{64}$")


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


def pick(d):
    cands = sorted(f for f in os.listdir(d)
                   if re.search(r"SHA256SUMS|MANIFEST|CHECKSUM", f, re.I)
                   and os.path.isfile(os.path.join(d, f)))
    scored = [(f,) + manifest_kind(os.path.join(d, f)) for f in cands]
    print("  STEP 1  manifest candidates: %d" % len(scored))
    for f, nh, nc, nt in scored:
        print("            %-24s lines=%-5d sha256=%-5d cksum=%-5d %s"
              % (f, nt, nh, nc, "<- real seal" if nh else "(verifies nothing with shasum)"))
    real = [s for s in scored if s[1] > 0]
    if not real:
        print("  REFUSE: no candidate contains a 64-hex digest; this directory has no "
              "verifiable seal, and shasum -c on what it does have prints zero OK and zero "
              "FAILED")
        return None
    return max(real, key=lambda s: s[1])[0]


def verify(d, mf):
    entries = {}
    with open(os.path.join(d, mf), errors="replace") as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 2 and HEX64.match(f[0]):
                entries[" ".join(f[1:]).lstrip("*./")] = f[0]
    ok = bad = missing = 0
    failed = []
    for name, want in entries.items():
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            missing += 1
            failed.append((name, "ABSENT"))
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        if h.hexdigest() == want:
            ok += 1
        else:
            bad += 1
            failed.append((name, "MISMATCH"))
    emitted = ok + bad + missing
    print("  STEP 2  manifest lines=%d  emitted=%d  OK=%d  MISMATCH=%d  ABSENT=%d"
          % (len(entries), emitted, ok, bad, missing))
    clean = emitted == len(entries) == ok and bad == 0 and missing == 0
    if not clean:
        print("  REFUSE: not a clean pass -- emitted, manifest lines and OK must all be equal")
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
        n_in = sum(1 for h in hits if h in sealed)
        total += len(hits)
        inside += n_in
        outside += [h for h in hits if h not in sealed]
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
    d = os.path.abspath(sys.argv[1])
    pats = sys.argv[2:]
    print("ATTEST %s" % d)
    if not os.path.isdir(d):
        sys.exit("REFUSE: %s is not a directory" % d)
    mf = pick(d)
    if mf is None:
        sys.exit(2)
    print("          chose %s BY CONTENT" % mf)
    clean, sealed = verify(d, mf)
    if not clean:
        sys.exit(3)
    if not cover(d, pats, sealed):
        sys.exit(4)
    print("  ATTESTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
