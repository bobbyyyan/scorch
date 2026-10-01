#!/usr/bin/env python3
"""Self-test for an_fmtscan.py: the planted tree is in the repo, not in the scanned tree.

WHY THE FIXTURES ARE .pyplant AND NOT .py. The first version of this committed them as .py under
harnesses/, which put an unparseable file into the tracked set and made the scan report 20 defects
on its own repository -- a negative control that invalidates the measurement it certifies. The
fixtures therefore carry a non-.py suffix (so the scan's own directory walk cannot see them) and are
materialised into a temp directory here.

A PLANT SHOWS A GUARD IS REACHED; ONLY A SILENCE CORPUS SHOWS IT IS NOT OVER-BROAD. Both halves are
required and both are asserted: 19 defect forms must fire, one cause each, and 13 correct forms must
stay silent. Six of the 19 came from a peer session running the scan on inputs it built, after my own
test list had passed -- an author's test set is written from the same enumeration as the
implementation, so it cannot contain the case the implementation forgot.
"""
import os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.join(HERE, "an_fmtscan.py")
PLANTS = os.path.join(HERE, "fmtscan_plants")
# (fixture, expected UNFORMATTED, expected SYNTAX, expected rc)
CASES = [("must_fire", 18, 0, 1), ("must_fire2", 1, 0, 1),
         ("must_be_silent", 0, 0, 0), ("broken", 0, 1, 1)]


def run(paths):
    r = subprocess.run([sys.executable, SCAN] + paths, capture_output=True, text=True)
    cls = {}
    m = re.search(r"^CLASSES (.*)$", r.stdout, re.M)
    if m and m.group(1) != "(none)":
        cls = {k: int(v) for k, v in (p.split("=") for p in m.group(1).split())}
    return r.returncode, cls, r.stdout


fail = []
tmp = tempfile.mkdtemp(prefix="fmtscan-selftest-")
try:
    staged = {}
    for name, _u, _s, _rc in CASES:
        dst = os.path.join(tmp, name + ".py")
        shutil.copyfile(os.path.join(PLANTS, name + ".pyplant"), dst)
        staged[name] = dst
    for name, want_u, want_s, want_rc in CASES:
        rc, cls, out = run([staged[name]])
        got_u, got_s = cls.get("UNFORMATTED", 0), cls.get("SYNTAX", 0)
        ok = (rc, got_u, got_s) == (want_rc, want_u, want_s)
        print("  %-16s rc=%d UNFORMATTED=%-2d SYNTAX=%d   want rc=%d U=%-2d S=%d   %s"
              % (name, rc, got_u, got_s, want_rc, want_u, want_s, "ok" if ok else "FAIL"))
        if not ok:
            fail.append(name)
            print(out)
    # Every planted defect must be attributable to its own line, so a single over-broad rule
    # cannot pass by firing twice on one construct and never on another.
    rc, cls, out = run([staged["must_fire"], staged["must_fire2"]])
    lines = sorted(int(m.group(1)) for m in re.finditer(r"^FINDING UNFORMATTED\s+\S+:(\d+)", out, re.M))
    n_want = sum(c[1] for c in CASES)
    if len(lines) != len(set(lines)) or len(lines) != n_want:
        fail.append("one-cause-per-line")
        print("  FAIL distinct lines: %d finding(s) on %d distinct line(s), want %d of each"
              % (len(lines), len(set(lines)), n_want))
    else:
        print("  %-16s %d finding(s) on %d distinct line(s)" % ("attribution", len(lines), len(set(lines))))
    # The fixtures must be invisible to the scan's own directory walk, or the control would
    # contaminate the population it certifies. This is the defect the .pyplant suffix fixes.
    rc, cls, out = run([PLANTS])
    if rc != 0 or cls:
        fail.append("fixtures-visible")
        print("  FAIL scanning the fixture dir found %s (rc=%d); the suffix is not excluding them"
              % (cls or "nothing", rc))
    else:
        print("  %-16s scanning %s finds nothing -- fixtures are outside the scanned population"
              % ("isolation", os.path.relpath(PLANTS, os.getcwd())))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if fail:
    sys.exit("REFUSE: %d self-test case(s) failed: %s" % (len(fail), ", ".join(fail)))
print("an_fmtscan self-test PASS: %d defect form(s) fire, %d correct form(s) silent, "
      "fixtures isolated" % (sum(c[1] + c[2] for c in CASES), 13))
