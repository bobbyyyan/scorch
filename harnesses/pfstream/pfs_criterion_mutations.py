"""Which assertion catches which mutation, MEASURED. Not inferred from how a check reads.

WHY THIS FILE EXISTS. pfs_criterion_selftest.py establishes that every `raise` site in the
target is reachable. That says nothing about whether a wrong VALUE is caught, and the gap
produced a concrete misattribution: I dropped the `- 1.0` from `detect_prob`, read the FAIL
list, and credited the catch to the `delta == b` invariance assertion because it is the one
that looks structural. It does not catch it. The broken form `z = ci_z*(delta-b)/h` gives
z = 0 at delta == b for every b and every h, so the probability is still flat -- 0.5 instead
of 0.025, spread 0.000e+00 under BOTH formulas. The assertion tests a spread and the
mutation preserves the spread. What fails is the value loop, which reads as the weaker
check of the two and is the only one doing the work.

A check that holds a property the mutation also holds is not covering that mutation, and
the only way to know which assertion is doing the work is to ablate and read the FAIL list.
So the attribution is derived here and a sole catcher is named, which makes a later
simplification that deletes it fail this file instead of silently uncovering a mutation.

The same read also produced a miscount: `grep -cE "^  FAIL"` reported 5 failing assertions
where there are 4, because `FAIL` is a prefix of the suite's own `  FAILED` summary line.
That predicate is checked against that line below, since it is the harness's own input.
"""
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "pfs_criterion.py"
SUITE = "pfs_criterion_selftest.py"

# (name, anchor, replacement, what the mutation means). The expected label set is NOT
# listed here: it is measured and printed, and the assertions below are about its SHAPE
# (non-empty, and a sole catcher that is named) rather than about a transcribed set, so
# this file does not have to be re-blessed every time a label is renamed.
MUTATIONS = [
    ("drop-detect-prob-offset",
     "    z = ci_z * ((delta - b) / h - 1.0)",
     "    z = ci_z * ((delta - b) / h)",
     "detection measured from the bound rather than a half-width beyond it"),
    ("flip-erfc-sign",
     "    return 0.5 * math.erfc(-z / math.sqrt(2.0))",
     "    return 0.5 * math.erfc(z / math.sqrt(2.0))",
     "upper tail for lower; power and its complement swapped"),
    ("bound-not-subtracted",
     "    z = ci_z * ((delta - b) / h - 1.0)",
     "    z = ci_z * (delta / h - 1.0)",
     "the class the delta==b invariance check is FOR: probability becomes a function of "
     "b and h at delta == b"),
    ("ci-z-to-one",
     "def detect_prob(delta, b, h, ci_z=1.959963984540054):",
     "def detect_prob(delta, b, h, ci_z=1.0):",
     "a 68% interval reported under a 95% name"),
    ("min-detect-to-zero",
     "    MIN_DETECT = 0.80",
     "    MIN_DETECT = 0.0",
     "no band is ever UNDERPOWERED, so adequacy goes unstated"),
    ("accept-self-voiding",
     "        if injection <= floor_bound:",
     "        if False:",
     "an injection at or below the bound is accepted"),
    ("declared-counts-as-run",
     "        if self._neg is None:",
     "        if False:",
     "verdicts issued before the negative control is run"),
    ("within-boundary-strict",
     "        elif -self.b <= lo and hi <= self.b:",
     "        elif -self.b < lo and hi < self.b:",
     "an interval exactly on the bound falls through to INDETERMINATE"),
]

LABEL = re.compile(r"^  FAIL ([A-Za-z0-9_.-]+)")


def labels_of(out):
    """The FAIL labels, plus the suite's non-assertion failure modes as pseudo-labels.

    A mutation that deletes a `raise` makes its site uncovered, one that breaks the tracer
    makes the coverage ratio vacuous, and one that crashes a test function is caught without
    any assertion deciding so. None of those is an assertion, so all three need a label.

    THE CRASH BRANCH USED TO BE UNREACHABLE. The suite reports a crash AS a FAIL line --
    `  FAIL <fn> CRASHED: ...` -- which LABEL matches, so an if/elif chain testing the crash
    shape afterwards never ran. Planting `raise RuntimeError` at the top of detect_prob gave
    7 crash FAIL lines and 0 `<crash>` labels; the returned set held one label per crashing
    test FUNCTION. Two costs, both latent while no registered mutation crashes: the printed
    count of distinct catching assertions inflated by the number of crashing functions, and
    `len(real)` was pushed far from 1 so a crash-caught mutation could never reach the
    sole-catcher report -- the one attribution this file exists to produce. The crash test
    now comes FIRST, because it is a property of the line that the label pattern also
    matches.
    """
    labs = set()
    for ln in out.split("\n"):
        if " CRASHED: " in ln or ln.rstrip().endswith("CRASHED"):
            labs.add("<crash>")
        elif LABEL.match(ln):
            labs.add(LABEL.match(ln).group(1))
        elif ln.startswith("  UNCOVERED raise sites"):
            labs.add("<uncovered-raise-site>")
        elif ln.startswith("  REFUSE:"):
            labs.add("<tracer-vacuous>")
    return labs


def run_staged(mutate=None):
    """Stage both files, optionally mutate the target, run the suite there. -> (rc, labels).

    Staged rather than edited in place so a crash cannot leave the repo's copy perturbed,
    and so the suite's own `HERE`-relative TARGET resolves to the mutated file without the
    suite knowing anything about this harness.
    """
    d = tempfile.mkdtemp(prefix="pfsmut.")
    try:
        for f in (TARGET, SUITE):
            shutil.copy(os.path.join(HERE, f), os.path.join(d, f))
        p = os.path.join(d, TARGET)
        if mutate is not None:
            _name, anchor, repl, _why = mutate
            src = open(p).read()
            if anchor not in src:
                return None, {"<anchor-absent>"}
            out = src.replace(anchor, repl, 1)
            if out == src:
                return None, {"<edit-did-not-land>"}
            open(p, "w").write(out)
        r = subprocess.run([sys.executable, SUITE], cwd=d, capture_output=True, text=True)
        labs = labels_of(r.stdout + r.stderr)
        # A mutation that stops the suite before any FAIL line prints -- a syntax error, an
        # import-time exception -- produced an EMPTY label set with rc=1, and the caller's
        # `rc == 0 or not labs` then called it an uncovered mutation. That is the inverse of
        # the truth: the suite could not even start, which is the loudest catch there is.
        if r.returncode != 0 and not labs:
            labs = {"<suite-aborted>"}
        return r.returncode, labs
    finally:
        shutil.rmtree(d, ignore_errors=True)


def label_collisions():
    """Two assertions sharing a label make the sole-catcher report name a GROUP, not a check.

    `underpowered:` prefixed three distinct assertions, so "deleting this uncovers its
    mutation" named three checks of which one was load-bearing. The emptiness test still
    protects -- deleting the load-bearing one does empty the row -- but the attribution
    degrades silently. Detected here rather than renamed once, because the collision class
    recurs every time an assertion is added.

    -> (collisions, n_literal, n_parameterised). The last two make a zero informative: a
    scan that resolved nothing would also report no collisions.
    """
    t = ast.parse(open(os.path.join(HERE, SUITE)).read())
    seen, param, lit = {}, 0, 0
    for n in ast.walk(t):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "append" and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "FAIL" and n.args):
            continue
        node = n.args[0]
        while isinstance(node, ast.BinOp):           # strip `% (...)`
            node = node.left
        while isinstance(node, ast.JoinedStr) and node.values:
            node = node.values[0]
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        head = node.value.split(":")[0]
        if "%" in head or "{" in head:
            param += 1                               # a helper whose label is an argument
            continue
        lit += 1
        seen.setdefault(head, []).append(n.lineno)
    return {k: v for k, v in seen.items() if len(v) > 1}, lit, param


def main():
    bad = []

    # The harness's own input, three ways. Without these, "the mutation was caught" is
    # unreadable: it could be a label the clean run also emits, or a parser that fires on
    # a summary line, or a staging step that never copied anything.
    rc0, lab0 = run_staged(None)
    print("clean staged copy: rc=%s labels=%s" % (rc0, sorted(lab0) or "none"))
    if rc0 != 0 or lab0:
        bad.append("the UNMUTATED staged copy is not clean, so every 'caught' below is "
                   "unreadable: rc=%s labels=%s" % (rc0, sorted(lab0)))
    if LABEL.match("  FAILED"):
        bad.append("the label predicate matches the suite's own '  FAILED' summary line -- "
                   "that is the prefix match that miscounted 4 assertions as 5")
    if not LABEL.match("  FAIL detect-prob-value: at b+0h got 0.500000, expected 0.0250"):
        bad.append("the label predicate does not match a real FAIL line, so an empty label "
                   "set would mean nothing")

    # The crash branch was unreachable: the suite prints a crash AS a FAIL line, which the
    # label pattern matches, so the set filled with one label per crashing test FUNCTION.
    rc, labs = run_staged(("crash-in-detect-prob",
                           "    if h <= 0:",
                           "    raise RuntimeError('planted')\n    if h <= 0:",
                           "every test that calls detect_prob dies"))
    print("crash plant: rc=%s labels=%s" % (rc, sorted(labs)))
    # The other three labels here are correct and not the regression: a crash inside
    # detect_prob also leaves raise sites unreached, makes the tracer's ratio vacuous, and
    # trips the constructor's self-voiding refusal, which calls detect_prob. What must never
    # come back is a TEST FUNCTION's name standing in for an assertion label.
    fnames = [lab for lab in labs if lab.startswith("t_")]
    if "<crash>" not in labs or fnames:
        bad.append("a crashing target must yield '<crash>' and no test-function name; got "
                   "%s. One label per crashing function inflates the distinct-assertion "
                   "count and pushes len(real) away from 1, so a crash-caught mutation "
                   "could never appear in the sole-catcher report" % sorted(labs))

    # And a target that cannot even be imported produced an EMPTY set with rc=1, which the
    # loop below read as "caught by nothing" -- the inverse of the truth.
    rc, labs = run_staged(("unparseable-target", "import math",
                           "import math\nthis is not python", "the suite cannot start"))
    print("abort plant: rc=%s labels=%s" % (rc, sorted(labs)))
    if labs != {"<suite-aborted>"} or rc == 0:
        bad.append("a target that stops the suite before any FAIL line must give exactly "
                   "{'<suite-aborted>'} with a nonzero rc; got rc=%s %s"
                   % (rc, sorted(labs)))

    coll, n_lit, n_par = label_collisions()
    print("label scan: %d literal labels resolved, %d parameterised (helper) and skipped"
          % (n_lit, n_par))
    if n_lit < 8:
        bad.append("only %d literal labels resolved, so 'no collisions' would be vacuous"
                   % n_lit)
    for head, lines in sorted(coll.items()):
        bad.append("label %r is used by %d assertions (lines %s), so a sole-catcher row "
                   "naming it names a GROUP rather than a check"
                   % (head, len(lines), ", ".join(str(x) for x in lines)))

    rows, catchers = [], {}
    for m in MUTATIONS:
        rc, labs = run_staged(m)
        rows.append((m[0], rc, labs, m[3]))
        if rc == 0 or not labs:
            bad.append("mutation %r was caught by NOTHING (rc=%s). It is a real behaviour "
                       "change with no assertion over it: %s" % (m[0], rc, m[3]))
            continue
        if {"<anchor-absent>", "<edit-did-not-land>"} & labs:
            bad.append("mutation %r did not apply (%s) -- its anchor has drifted, so this "
                       "row reports nothing" % (m[0], sorted(labs)))
            continue
        for lab in labs:
            catchers.setdefault(lab, set()).add(m[0])

    print("\nmutation -> the assertions that fail")
    for name, rc, labs, why in rows:
        print("  %-26s rc=%s  %s" % (name, rc, ", ".join(sorted(labs)) or "NOTHING"))
        print("  %-26s %s" % ("", why))

    # A sole catcher is load-bearing: delete it and the mutation goes uncaught. This is
    # the part a comment cannot carry, because the sole catcher here is the assertion that
    # reads as the less structural of the two.
    sole = {}
    for name, rc, labs, _why in rows:
        real = labs - {"<anchor-absent>", "<edit-did-not-land>"}
        if len(real) == 1:
            sole.setdefault(next(iter(real)), []).append(name)
    print("\nSOLE catchers -- deleting one of these uncovers its mutation outright")
    if sole:
        for lab, names in sorted(sole.items()):
            print("  %-28s %s" % (lab, ", ".join(sorted(names))))
    else:
        print("  none: every mutation is caught by at least two assertions")

    rc = 0
    for b in bad:
        rc = 1
        print("\nFAIL " + b)
    print("\n%d mutations, %d distinct catching assertions, %d sole catchers -- %s"
          % (len(rows), len(catchers), len(sole), "OK" if rc == 0 else "FAILED"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
