"""Build the per-slice member order for the floor timing run, and refuse an unbalanced one.

WHY ROTATE AT ALL. The floor members are separate processes, so running them as 16 sequential
blocks would measure later members later in wall-clock time. Any drift -- thermal, a co-tenant
arriving, page-cache state -- then appears as a difference between members, which is exactly the
quantity the floor exists to measure. Cutting the corpus into slices and permuting the member
order per slice spreads each member over the available positions.

TWO AXES, NOT ONE. Members m and m+4 share an alignment class, because class = 16(1+m) mod 64 has
period 4 in m. So a schedule can be balanced in two independent directions:

  within-class   members of the SAME class must have equal mean position, or positional drift
                 shows up as disagreement inside a class -- the number that decides whether a
                 contrast is reported as a matched headline or only as a bound.
  between-class  the class MEANS must be equal, or drift shows up as a difference between
                 classes, which is the effect being hunted.

An earlier version of this check computed the within-class range only and printed "BALANCED" for
a schedule whose between-class range was 3.00 of 15 positions. Both are computed here and both
are printed beside the verdict.

THE CONDITION. Equal position sums across members implies both: if every member's positions sum
to the same value then every member has the same mean, so every within-class range and every
between-class difference is zero. That is one scalar check rather than a condition specific to one
family of schedules, and it is what gets asserted.

A CYCLIC ROTATION CANNOT SATISFY IT. With step K the member m sits at (m - K(s-1)) mod M. For
M=16, S=4, K=4 the four positions of member m are m, m-4, m-8, m-12 mod 16, which is (m mod 4)
plus the four multiples of 4 in some order: sum = 4*(m mod 4) + 24. That varies with m mod 4 --
and m mod 4 IS the class -- giving class means 6, 7, 8, 9 and a between-class range of exactly 3.
So no choice of step fixes it; the schedule has to leave the cyclic family.

THE CONSTRUCTION. With M = S*S, write m = S*a + b. Place it at S*((a+s) mod S) + sigma[s][b],
where sigma is a Latin square of order S. The position sum is then
S*(0+1+...+(S-1)) + (column sum of sigma) = S*S*(S-1)/2 + S*(S-1)/2, the same for every member.

A BALANCED SCHEDULE DOES NOT SURVIVE WINDOWING: analysing a subset of the slices can reintroduce
the imbalance. The slice label goes in every output row so the analysis can re-certify balance on
the window it actually uses.
"""
import sys
from math import gcd as _gcd

CLASS_STRIDE = 16           # class = (CLASS_STRIDE * (1 + m)) mod CLASS_MODULUS
CLASS_MODULUS = 64
# The spacing is a CONSEQUENCE of the two constants above, not an independent knob: the smallest
# p with CLASS_STRIDE*p == 0 mod CLASS_MODULUS. Spelling it as a literal 4 left a constant that
# looked enforced while nothing derived from it, so changing it would have had no effect.
CLASS_SPACING = CLASS_MODULUS // _gcd(CLASS_STRIDE, CLASS_MODULUS)


def alignment_class(m):
    return (CLASS_STRIDE * (1 + m)) % CLASS_MODULUS


def class_period(n=64):
    """The observed period of alignment_class in m, which CLASS_SPACING must equal."""
    base = [alignment_class(m) for m in range(n)]
    for p in range(1, n):
        if all(base[m] == base[m + p] for m in range(n - p)):
            return p
    return None


def classes_of(members):
    out = {}
    for m in members:
        out.setdefault(alignment_class(m), []).append(m)
    return out


def latin_square(S):
    """Cyclic Latin square of order S: every row and every column is a permutation of 0..S-1.

    EITHER SIGN SATISFIES THE BALANCE CONDITION. Both have equal column sums, so both give equal
    per-member position sums, so both are balanced within and between class and both give every
    member every position block. They differ only in WHICH members lead a slice, because the
    leader of slice s is the member with a = -s mod S and sigma[s][b] = 0.

    THE SIGN IS INERT AT S=4, WHICH IS THE CONFIGURATION THIS PROJECT RUNS. Measured leader-class
    coverage against a class spacing of 4, with the reachable ceiling min(S, 4):

        S   gcd(S+1, 4)   (b+s)     (b-s)    reachable
        3        4        1 of 4    3 of 4        3
        4        1        4 of 4    4 of 4        4
        5        2        2 of 4    2 of 4        4
        6        1        4 of 4    4 of 4        4

    So the sign is load-bearing only at S=3 of these, and at S=5 NEITHER sign reaches the ceiling.
    No closed form in a comment here would be trustworthy: a first attempt at one claimed the
    (b-s) leader class is s mod 4, which holds only when 4 divides S -- at S=5 the leaders are
    {0,21,17,13,9}, classes {0,1}. The audit computes the coverage and the check compares it to
    the reachable number, so the sign is chosen by measurement rather than by the rule above.
    (b-s) is kept because it is the one that differs from the other at S=3 and ties everywhere
    else in the table, so it is never worse over the range measured.
    """
    return [[(b - s) % S for b in range(S)] for s in range(S)]


def schedule(M, S):
    """Member -> position per slice. Returns (pos, how) or (None, why_it_cannot_be_built)."""
    if S < 1 or M < 1:
        return None, "M=%d and S=%d must both be positive" % (M, S)
    obs = class_period()
    if obs != CLASS_SPACING:
        return None, ("the alignment class repeats every %s members but CLASS_SPACING says %d, so "
                      "the class grouping below would be wrong" % (obs, CLASS_SPACING))
    if S * S == M:
        sigma = latin_square(S)
        pos = {}
        for m in range(M):
            a, b = divmod(m, S)
            pos[m] = [S * ((a + s) % S) + sigma[s][b] for s in range(S)]
        return pos, "Latin-square construction, M = S*S = %d" % M
    # Outside M = S*S the equal-sum construction above does not apply. Report the cyclic
    # schedule's between-class range rather than running it, so the refusal carries its size.
    if M % S:
        return None, "S=%d does not divide M=%d" % (S, M)
    K = M // S
    pos = {m: [((m - K * s) % M) for s in range(S)] for m in range(M)}
    return pos, "cyclic step %d (M = S*S does not hold, so equal sums are not guaranteed)" % K


def audit(pos, M, S):
    """Compute every balance quantity. Returns a dict; no verdict, no printing."""
    T = M // S if S else 0
    mean = {m: sum(v) / float(len(v)) for m, v in pos.items()}
    sums = {m: sum(v) for m, v in pos.items()}
    cls = classes_of(range(M))
    within = {c: max(mean[m] for m in ms) - min(mean[m] for m in ms) for c, ms in cls.items()}
    cmean = {c: sum(mean[m] for m in ms) / float(len(ms)) for c, ms in cls.items()}
    perms = [sorted(pos[m][s] for m in range(M)) == list(range(M)) for s in range(S)]
    blocks = {m: sorted(p // T for p in pos[m]) for m in range(M)} if T else {}
    block_ok = all(v == list(range(S)) for v in blocks.values()) if T else False
    leaders = [m for s in range(S) for m in range(M) if pos[m][s] == 0]
    return {
        "mean": mean, "sums": sums, "classes": cls,
        "within": within, "worst_within": max(within.values()) if within else 0.0,
        "class_mean": cmean,
        "between": (max(cmean.values()) - min(cmean.values())) if cmean else 0.0,
        "every_slice_a_permutation": all(perms), "slice_permutation_flags": perms,
        "distinct_sums": sorted(set(sums.values())),
        "every_member_visits_every_block": block_ok,
        "leaders": leaders,
        "leader_classes": sorted({alignment_class(m) for m in leaders}),
        "n_classes": len(cls),
        "n_slices": S,
    }


FD_cls = alignment_class


def report(M, S, pos, how, a, stream=sys.stdout):
    w = stream.write
    w("  ROTATION DESIGN: %d members, %d slices -- %s\n" % (M, S, how))
    for c in sorted(a["classes"]):
        ms = a["classes"][c]
        w("    class %-3d members %-18s mean positions %-26s within-class range %.2f\n"
          % (c, ms, [round(a["mean"][m], 2) for m in ms], a["within"][c]))
    w("    class means %s\n" % {c: round(v, 2) for c, v in sorted(a["class_mean"].items())})
    w("    WITHIN-class range (worst over classes) %.2f   BETWEEN-class range %.2f\n"
      % (a["worst_within"], a["between"]))
    w("    distinct per-member position sums %s (one value means both ranges are 0)\n"
      % a["distinct_sums"])
    w("    every slice a permutation of 0..%d: %s\n" % (M - 1, a["every_slice_a_permutation"]))
    w("    every member visits every position block: %s\n" % a["every_member_visits_every_block"])
    w("    leaders %s are classes %s -- %d of the %d reachable with %d slices\n"
      % (a["leaders"], [FD_cls(m) for m in a["leaders"]], len(a["leader_classes"]),
         min(a["n_slices"], a["n_classes"]), a["n_slices"]))


def failures(a):
    """The named reasons a schedule is not usable. Empty list means usable."""
    out = []
    if not a["every_slice_a_permutation"]:
        out.append("not every slice is a permutation of the members (flags %s): some member runs "
                   "twice in a slice and another not at all" % a["slice_permutation_flags"])
    if len(a["distinct_sums"]) != 1:
        out.append("per-member position sums are not all equal (%s), so mean position differs "
                   "between members: within-class range %.2f, between-class range %.2f"
                   % (a["distinct_sums"], a["worst_within"], a["between"]))
    if not a["every_member_visits_every_block"]:
        out.append("some member does not visit every position block, so it never runs early and "
                   "never runs late")
    # With S slices there are only S leading slots, so S classes is the ceiling when S is the
    # smaller number. Demanding n_classes there would be unsatisfiable and the check would refuse
    # every schedule rather than the unbalanced ones.
    reachable = min(a["n_slices"], a["n_classes"])
    if len(a["leader_classes"]) != reachable:
        out.append("the leading position is taken by only %d of the %d classes it could reach "
                   "(%d classes, %d slices), so whatever is special about running first -- a cold "
                   "page cache above all -- lands on those classes"
                   % (len(a["leader_classes"]), reachable, a["n_classes"], a["n_slices"]))
    return out


def main(argv):
    if len(argv) < 3:
        return "usage: floor_design.py <members> <slices> [--emit <path>]"
    M, S = int(argv[1]), int(argv[2])
    emit = None
    if "--emit" in argv:
        emit = argv[argv.index("--emit") + 1]
    pos, how = schedule(M, S)
    if pos is None:
        return "REFUSE: no schedule for %d members in %d slices -- %s" % (M, S, how)
    a = audit(pos, M, S)
    report(M, S, pos, how, a)
    bad = failures(a)
    if bad:
        return ("REFUSE: the schedule is unbalanced in %d way(s):\n" % len(bad)
                + "".join("        - %s\n" % b for b in bad)
                + "        An unbalanced schedule puts positional drift into the very numbers\n"
                  "        this run exists to measure, so it is not run.")
    print("    BALANCED on both axes, every slice complete, every member spans the run")
    if emit:
        # One source of truth: the order bash executes is the order audited above.
        with open(emit, "w") as fh:
            for s in range(S):
                order = sorted(range(M), key=lambda m: pos[m][s])
                fh.write("%d %s\n" % (s + 1, " ".join(str(m) for m in order)))
        print("    schedule emitted to %s (%d slices, one line each)" % (emit, S))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
