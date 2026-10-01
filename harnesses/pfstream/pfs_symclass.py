"""Per-symbol code-placement census across arm objects, for choosing a floor's pad shift.

Usage: pfs_symclass.py <pattern> <baseline-label>=<so> <label>=<so> ...

WHAT THIS IS FOR. A floor for a compile-time macro arm must reproduce the placement difference the
arm actually exhibits. That difference is per-symbol, and a `.text` pad is ONE degree of freedom
while each symbol imposes one congruence, so which symbols matter has to be decided before a pad
size can be chosen. This enumerates every matching symbol in every object, derives which of them
execute the kernel, and prints the pad shift each requires.

IT DELIBERATELY DOES NOT REPORT A "BEST SINGLE PAD". That metric -- the shift matching the most
symbols -- was introduced in review and is not a weaker version of the right question; it is a
different and wrong one. Measured on the four PFSTREAM screen objects: nopf's count-optimal shift
is 16, matching 8 of 11 symbols, while BOTH its hot lambdas require shift 0, so the count-optimal
pad matches neither symbol that executes. strm's is 48, matching 9 of 11, while its float64 lambda
requires 32. Maximising matched symbols selects a pad matched to nothing that runs.

THREE WAYS THE ENUMERATION ITSELF MISLEADS, each handled here:

  * INTERSECTING symbol sets hides an inventory change. On these objects the innermost float64
    lambda is present in ship and shpb and absent in both strm and nopf -- either prefetch
    modification gets it fully inlined -- so an intersected table reports a clean 11-row
    comparison and drops the one difference that is not about placement at all. The symmetric
    difference is printed beside the intersection.
  * SELECTING a symbol by name prefix is ambiguous. Several symbols share
    `_Z16spmm_csr_v2_core<T>Lb0E`: the main body, its `.cold` clone and its `.localalias`. Picking
    one with a prefix match over an unordered set is nondeterministic, and doing that produced a
    decomposition that did not match the published table. Symbols are keyed by their full name.
  * ASSUMING which symbol is hot. Derived instead, by counting prefetch and FMA instructions per
    symbol: on these objects all 61 register-block prefetches inside the matching set sit in the
    two OpenMP lambda bodies and none in any wrapper, and the wrappers contain no FMA at all.
"""
import collections
import re
import subprocess
import sys

SYMHDR = re.compile(r"^([0-9a-f]+) <(.+)>:$")


def census(path, pattern):
    """{symbol: (address, n_insn, n_prefetch, n_fma)} for symbols matching pattern, no @plt."""
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", path],
                         capture_output=True, text=True, check=True).stdout
    syms, cur = {}, None
    for ln in out.split("\n"):
        m = SYMHDR.match(ln)
        if m:
            name = m.group(2)
            cur = name if (re.search(pattern, name) and "@plt" not in name) else None
            if cur:
                syms[cur] = [int(m.group(1), 16), 0, 0, 0]
            continue
        if cur is None:
            continue
        m = re.match(r"^\s+[0-9a-f]+:\s+(\S+)", ln)
        if not m:
            continue
        op = m.group(1)
        syms[cur][1] += 1
        if op.startswith("prefetch"):
            syms[cur][2] += 1
        if "fmadd" in op or "fmla" in op:
            syms[cur][3] += 1
    return {k: tuple(v) for k, v in syms.items()}


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: pfs_symclass.py <pattern> <baseline>=<so> <arm>=<so> ...")
    pattern = sys.argv[1]
    objs = []
    for a in sys.argv[2:]:
        if "=" not in a:
            sys.exit("REFUSE: %r is not label=path" % a)
        lab, path = a.split("=", 1)
        objs.append((lab, path))
    base = objs[0][0]
    C = {lab: census(path, pattern) for lab, path in objs}
    for lab, _ in objs:
        if not C[lab]:
            sys.exit("REFUSE: %s has no symbol matching %r, so every number below is vacuous"
                     % (lab, pattern))

    sets = {lab: set(C[lab]) for lab, _ in objs}
    inter = set.intersection(*sets.values())
    union = set.union(*sets.values())
    print("matching symbols per object: %s" % {lab: len(sets[lab]) for lab, _ in objs})
    print("intersection %d, union %d" % (len(inter), len(union)))
    if union - inter:
        print("SYMBOL INVENTORY DIFFERS -- these are CODE differences, not placement, and an")
        print("intersected table would drop them silently:")
        for s in sorted(union - inter):
            where = "".join("Y" if s in sets[lab] else "." for lab, _ in objs)
            print("    %-*s %s" % (len(objs), where, s[:94]))
        print("    (columns: %s)" % " ".join(lab for lab, _ in objs))
    else:
        print("symbol inventory identical across all objects")

    hot = sorted(s for s in inter if C[base][s][2] > 0)
    tot_pf = sum(C[base][s][2] for s in inter)
    print()
    print("WHICH SYMBOLS EXECUTE THE KERNEL -- derived from the instruction mix, not assumed:")
    print("  %-62s %7s %5s %5s" % ("symbol", "insn", "pf", "fma"))
    for s in sorted(inter, key=lambda s: -C[base][s][2])[:8]:
        a, n, pf, fma = C[base][s]
        print("  %-62s %7d %5d %5d" % (s[:62], n, pf, fma))
    print("  prefetch-bearing symbols: %d of %d; they hold %d of %d prefetches in the matching set"
          % (len(hot), len(inter), sum(C[base][s][2] for s in hot), tot_pf))
    if not hot:
        sys.exit("REFUSE: no matching symbol contains a prefetch, so the kernel under test was "
                 "not located and no pad shift can be justified")

    print()
    print("REQUIRED PAD SHIFT, per arm and per executing symbol. A pad adds one constant to every")
    print("later symbol, so one shift can satisfy several symbols only if the arm's inter-symbol")
    print("offsets match the baseline's.")
    for lab, _ in objs[1:]:
        need = {}
        for s in hot:
            need[s] = (C[lab][s][0] - C[base][s][0]) % 64
        print("  %s:" % lab)
        for s in hot:
            print("    shift %2d   baseline class %2d -> %2d   %s"
                  % (need[s], C[base][s][0] % 64, C[lab][s][0] % 64, s[:70]))
        uniq = set(need.values())
        print("    %s" % ("one pad suffices for all %d executing symbols (shift %d)"
                          % (len(hot), uniq.pop()) if len(uniq) == 1 else
                          "NO single pad matches all %d executing symbols; one pad per symbol "
                          "group is NECESSARY" % len(hot)))
    print()
    print("No 'best single pad' count is reported: see this file's docstring for why that metric")
    print("selects a pad matched to symbols that do not run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
