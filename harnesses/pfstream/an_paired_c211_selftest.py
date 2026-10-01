"""Plants for an_paired_c211's sign argument and its zero boundary.

WHY THIS FILE EXISTS. The SIGN paragraph was originally a CONSTANT SENTENCE -- "the arm is
SLOWER and this reads FASTER, so the displacement has the wrong sign to be the arm" -- printed
on every pass, including three where the control's displacement had the SAME sign as the arm
and the argument was exactly unavailable. It was found by a human reading the report, which is
the rung we agreed counts as none.

Replacing it with four branches and no test over them is that defect wearing a condition, so
the branches are driven here. `sign_argument` is a pure function of (arm displacements, control
displacement) precisely so this file can exist; nothing is recomputed, the function under test
is imported.

The second plant is the zero boundary. A guard of `abs(disp) < 1e-9` carried the rationale
"zero to the printed precision", but the value is printed `%.2f%%`, so the printed precision
is 0.005. Six decades of displacements printed `0.00%` beside a finite ratio formed by
dividing by them -- 1000x inflated at 0.001, 250x at 0.004. The crash was fixed and the
misleading number the comment was reasoning about was not.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FAIL, CHECKS = [], 0


def _load():
    """Import the module's functions without running its module-level report, which needs a
    directory of CSVs. Compiling the source and executing only the defs keeps the test honest
    about WHICH implementation it is checking -- there is exactly one."""
    import ast
    src = open(os.path.join(HERE, "an_paired_c211.py")).read()
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PRINTED_ZERO")]
    if not any(isinstance(n, ast.FunctionDef) and n.name == "sign_argument" for n in keep):
        sys.exit("REFUSE: sign_argument is not a module-level function in an_paired_c211.py, "
                 "so this file is testing nothing. It was straight-line code once.")
    mod = ast.Module(body=keep, type_ignores=[])
    ns = {}
    exec(compile(mod, "an_paired_c211.py", "exec"), ns)      # noqa: S102
    return ns


NS = _load()
sign_argument, _sign, PZ = NS["sign_argument"], NS["_sign"], NS["PRINTED_ZERO"]


def eq(label, got, want):
    global CHECKS
    CHECKS += 1
    if got != want:
        FAIL.append("%s: got %r, expected %r" % (label, got, want))


def t_all_four_branches_are_reachable():
    """One input per token. A branch with no input is an untested claim."""
    arm_slow = [9.9, 19.8, 6.6, 8.1]
    eq("opposite", sign_argument(arm_slow, -0.18)[0], "OPPOSITE_SIGN_AVAILABLE")
    eq("same", sign_argument(arm_slow, +1.34)[0], "SAME_SIGN_NOT_AVAILABLE")
    eq("ctl-zero", sign_argument(arm_slow, 0.0)[0], "CONTROL_ZERO")
    eq("arm-mixed", sign_argument([9.9, -3.0, 6.6, 8.1], -0.18)[0], "ARM_HAS_NO_SINGLE_SIGN")
    # and the mirror of the first, so OPPOSITE is not satisfied by a constant
    eq("opposite-mirrored", sign_argument([-9.9, -19.8], +0.18)[0], "OPPOSITE_SIGN_AVAILABLE")
    eq("same-mirrored", sign_argument([-9.9, -19.8], -0.18)[0], "SAME_SIGN_NOT_AVAILABLE")


def t_the_words_match_the_token():
    """A token is only useful if the sentence it selects says the same thing."""
    eq("word-slower", sign_argument([9.9, 6.6], -0.2)[1], "SLOWER")
    eq("word-faster", sign_argument([-9.9, -6.6], +0.2)[1], "FASTER")
    eq("word-mixed", sign_argument([9.9, -6.6], +0.2)[1], "MIXED")
    eq("ctl-word-faster", sign_argument([9.9], -0.2)[2], "FASTER")
    eq("ctl-word-slower", sign_argument([9.9], +0.2)[2], "SLOWER")
    eq("ctl-word-zero", sign_argument([9.9], 0.0)[2], "ZERO")


def t_a_width_at_exactly_no_effect_votes_for_neither():
    """THE BOUNDARY THE FIX MISSED FIRST TIME. An arm point ratio of exactly 1.0 -- 0.00 pp --
    used to bin as -1, FASTER, which flips the arm off a single direction and routes the report
    to "no single arm sign" on the strength of a width where nothing happened. The control's
    sign got a zero case on the very next line and the arm's did not."""
    eq("zero-width-does-not-flip", sign_argument([9.9, 0.0, 6.6], -0.2)[0],
       "OPPOSITE_SIGN_AVAILABLE")
    eq("zero-width-keeps-the-word", sign_argument([9.9, 0.0, 6.6], -0.2)[1], "SLOWER")
    # every width at no effect is a real state and must not read as a direction
    eq("all-zero", sign_argument([0.0, 0.0], -0.2)[0], "ARM_HAS_NO_SINGLE_SIGN")
    eq("all-zero-word", sign_argument([0.0, 0.0], -0.2)[1], "NO EFFECT AT ANY WIDTH")


def t_zero_is_the_printed_precision_not_an_epsilon():
    """`%.2f` of a percentage rounds at 0.005, so the guard must be 0.005 and not 1e-9.
    The three values below all PRINT as 0.00% and the old guard sent two of them to the
    number branch, where the ratio is formed by dividing by them."""
    global CHECKS
    CHECKS += 1
    if PZ != 0.005:
        FAIL.append("printed-zero: PRINTED_ZERO is %r; `%%.2f` of a percentage rounds at 0.005"
                    % PZ)
    for v in (0.0, 0.001, 0.004, -0.004):
        eq("prints-as-zero %g" % v, _sign(v), 0)
    # and the first value that does NOT print as zero must still carry a sign, or the guard
    # has swallowed real displacements
    for v in (0.005, 0.006, -0.01, 1.34):
        eq("prints-nonzero %g" % v, _sign(v), 1 if v > 0 else -1)


def t_the_control_zero_branch_wins_over_the_arm_branch():
    """Order matters: with a zero control AND a mixed arm, the report must say which. Pinned
    so a later reordering is a failure rather than a different paragraph."""
    eq("zero-control-beats-mixed-arm", sign_argument([9.9, -3.0], 0.0)[0], "CONTROL_ZERO")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("t_")]


def main():
    for fn in TESTS:
        try:
            fn()
        except Exception as e:                                       # noqa: BLE001
            FAIL.append("%s CRASHED: %s: %s" % (fn.__name__, type(e).__name__, e))
    toks = set()
    for arm in ([9.9, 6.6], [-9.9, -6.6], [9.9, -6.6], [0.0, 0.0]):
        for ctl in (-0.2, 0.0, +0.2):
            toks.add(sign_argument(arm, ctl)[0])
    print("an_paired_c211 selftest: %d plants, %d assertions, %d of 4 sign tokens reached"
          % (len(TESTS), CHECKS, len(toks)))
    rc = 0
    if len(toks) != 4:
        rc = 1
        print("  FAIL only %d tokens are reachable over the grid: %s" % (len(toks), sorted(toks)))
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
