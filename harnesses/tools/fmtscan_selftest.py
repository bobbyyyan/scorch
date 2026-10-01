#!/usr/bin/env python3
"""Self-test for an_fmtscan.py: the planted tree is in the repo, not in the scanned tree.

WHY THE FIXTURES ARE .pyplant AND NOT .py. The first version of this committed them as .py under
harnesses/, which put an unparseable file into the tracked set and made the scan report 20 defects
on its own repository -- a negative control that invalidates the measurement it certifies. The
fixtures therefore carry a non-.py suffix (so the scan's own directory walk cannot see them) and are
materialised into a temp directory here.

AND THE SILENCE CORPUS NEEDS ITS SIZE ASSERTED, which the first version also got wrong. Its
expectation was (rc=0, UNFORMATTED=0, SYNTAX=0), and an EMPTY FILE satisfies that identically -- a
peer truncated must_be_silent.pyplant to zero bytes and this self-test still passed. A plant shows a
guard is reached; an ablation shows it was necessary; and a SILENCE corpus shows nothing at all
unless its content is pinned. So the count of specifier-bearing literals is derived from the fixture
and compared against a declared number, and so is the count of near-miss literals that carry a `%`
the regex must NOT match. Emptying the file now fails twice.

EVERY PRINTED NUMBER HAS ONE BINDING. The first version printed three figures with three different
predicates -- 19 in the docstring, 20 in the PASS line, 19 in the attribution check -- and claimed
13 silence cases in a fixture that held 17. A count in a message and the same count in a condition
must be the same name.
"""
import ast, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.join(HERE, "an_fmtscan.py")
PLANTS = os.path.join(HERE, "fmtscan_plants")
# (fixture, expected UNFORMATTED, expected SYNTAX, expected rc)
CASES = [("must_fire", 18, 0, 1), ("must_fire2", 1, 0, 1), ("must_fire3", 15, 0, 1),
         ("must_be_silent", 0, 0, 0), ("known_miss", 0, 0, 0), ("broken", 0, 1, 1)]
N_FIRE = sum(c[1] for c in CASES)          # defect forms that must be detected
N_SYNTAX = sum(c[2] for c in CASES)        # files that must be reported as unreadable
SILENT_SPEC = 19                           # specifier-bearing literals that must stay silent
# SILENCE HAS TWO CAUSES AND ONE CORPUS CANNOT ASSERT EITHER. must_be_silent.pyplant claims "this
# is correct code", and the only way to assert that is to RUN it and require that no specifier
# survives into its output. known_miss.pyplant claims the opposite -- a specifier DOES reach output
# and the scan cannot see it -- so it is asserted to emit exactly MISS_SPECS of them. Without the
# split, three cases sat in the silence corpus emitting %d while labelled correct: translate() with
# an empty table is a no-op, and a lazy-logging ARGUMENT has its own specifiers substituted through.
#
# TWO MEMBERS THEN LEFT known_miss BECAUSE THEY WERE INCONSISTENCIES, NOT LIMITS. An argument to
# .format() and a value argument to lazy logging are both substituted verbatim, exactly as the
# right operand of % is -- and that one was already flagged. A documented cost has to be a limit of
# the method, not a gap in it, so both moved to must_fire3 and MISS_SPECS fell from 4 to 3. The
# replacement is a case that IS a limit: an editing method with a NAME argument, where the scan
# cannot prove specifiers are untouched and a false positive would be worse than a miss.
MISS_SPECS = 3
# Literals carrying a % the regex must NOT match. Enumerated rather than counted, because a bare
# number here is a value without its predicate: '3% bare prose', '% i and % o', '%b is not a
# specifier', the regex '^%[sd]$', and the two bare '%' that are replace() arguments.
SILENT_NEARMISS = 6
FIRE_FIXTURES = [c[0] for c in CASES if c[1]]


def run(paths):
    r = subprocess.run([sys.executable, SCAN] + paths, capture_output=True, text=True)
    cls = {}
    m = re.search(r"^CLASSES (.*)$", r.stdout, re.M)
    if m and m.group(1) != "(none)":
        cls = {k: int(v) for k, v in (p.split("=") for p in m.group(1).split())}
    return r.returncode, cls, r.stdout


def spec_census(path):
    """(matching, near-miss) string literals, using the scan's OWN regex rather than a copy.

    Importing SPEC instead of restating it matters: a harness that recomputes the thing it checks
    is wrong silently and flatteringly, and a second copy of this regex would drift from the one
    that does the work.
    """
    sys.path.insert(0, HERE)
    from an_fmtscan import SPEC
    t = ast.parse(open(path).read())
    lits = [n.value for n in ast.walk(t)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    return (sum(1 for v in lits if SPEC.search(v)),
            sum(1 for v in lits if "%" in v and not SPEC.search(v)))


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
    # THE SILENCE CORPUS IS PINNED BY CONTENT, NOT BY ITS VERDICT. Without this an empty file
    # passes, because "produced no finding" is what an empty file does best.
    got_spec, got_near = spec_census(os.path.join(PLANTS, "must_be_silent.pyplant"))
    ok = (got_spec, got_near) == (SILENT_SPEC, SILENT_NEARMISS)
    print("  %-16s %d specifier-bearing + %d near-miss literal(s)   want %d + %d   %s"
          % ("silence corpus", got_spec, got_near, SILENT_SPEC, SILENT_NEARMISS,
             "ok" if ok else "FAIL"))
    if not ok:
        fail.append("silence-corpus-size")
    # Execute both corpora. This is the runtime half of the pair turned on the fixtures: a claim
    # about what reaches OUTPUT can only be settled by looking at output.
    sys.path.insert(0, HERE)
    from an_fmtscan import SPEC
    for name, want_specs in (("must_be_silent", 0), ("known_miss", MISS_SPECS)):
        r = subprocess.run([sys.executable, staged[name]], capture_output=True, text=True)
        hits = SPEC.findall(r.stdout + r.stderr)
        ok = r.returncode == 0 and len(hits) == want_specs
        print("  %-16s executes rc=%d, %d specifier(s) survive into its output   want rc=0, %d"
              "   %s" % (name, r.returncode, len(hits), want_specs, "ok" if ok else "FAIL"))
        if not ok:
            fail.append(name + "-runtime")
    # Every planted defect must be attributable to its own line, so a single over-broad rule
    # cannot pass by firing twice on one construct and never on another.
    rc, cls, out = run([staged[n] for n in FIRE_FIXTURES])
    # (file, line, COLUMN) -- a site, not a line. Two specifier literals on one line are
    # legitimate (both branches of a ternary) and with (file, line) alone such a fixture would make
    # this check FAIL: fail-closed, but a constraint on the fixture rather than a defect in the
    # scan. The column removes it, so a two-literals-on-one-line case can now be planted.
    lines = sorted(m.group(1) for m in
                   re.finditer(r"^FINDING UNFORMATTED\s+(\S+:\d+:\d+)", out, re.M))
    if len(lines) != len(set(lines)) or len(lines) != N_FIRE:
        fail.append("one-cause-per-line")
        print("  FAIL attribution: %d finding(s) on %d distinct site(s), want %d of each"
              % (len(lines), len(set(lines)), N_FIRE))
    else:
        print("  %-16s %d finding(s) on %d distinct site(s), incl. two on one line"
              % ("attribution", len(lines), len(set(lines))))
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
print("an_fmtscan self-test PASS: %d defect form(s) fire, %d unreadable file(s) reported, "
      "%d specifier-bearing literal(s) stay silent in code that runs clean, %d documented miss(es) "
      "confirmed real, fixtures isolated" % (N_FIRE, N_SYNTAX, SILENT_SPEC, MISS_SPECS))
