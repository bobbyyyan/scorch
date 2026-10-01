"""Compare two objects that should differ in LAYOUT and not in the kernel's instructions.

WHAT A FLOOR FOR INTER-OBJECT VARIATION HAS TO BE. A byte-identical pair bounds build-path
variation and nothing else, which is why the existing shpb/ship floor cannot certify a
compile-time macro arm. The replacement pair must satisfy three things, and the third is the one
I nearly left out:

  1. the whole object's .text must DIFFER, or the pair is the byte-identical floor again;
  2. the kernel's own instruction sequence must be IDENTICAL, or the pair is measuring a code
     change rather than a layout change;
  3. the kernel's ADDRESS must differ -- ideally its alignment modulo 64 -- or there is no layout
     difference to measure and the floor reads zero by construction, which is failure mode 1
     wearing a different hat.

WHY NOT DIGEST THE BYTES. A symbol's bytes embed relative branch and call offsets, so two
objects with identical instructions give different bytes as soon as anything moves -- the digest
would report a difference that is not an instruction difference. This compares the disassembled
mnemonic and operand text with absolute addresses and objdump's symbol annotations stripped, so a
moved call reads the same and a changed instruction does not. That is weaker than proving
equality of semantics and stronger than a byte digest; see the note in the project memory that
neither .text digests nor instruction counts prove emission neutrality.

HOW TO BUILD A PAIR THAT PASSES, AND WHY THE OBVIOUS WAY DOES NOT. The tempting construction is a
semantically inert SOURCE change -- add an unused-but-emitted function, or reorder two independent
definitions. Measured on the real translation unit with g++ at -O2, that **changes the kernel**:
inserting one external-linkage function into ops.cpp moved the two hot OpenMP lambda bodies from
4359 to 4340 and from 5941 to 6022 instructions, with different register allocation. The
translation unit is one 56 KB file and whole-unit decisions shift, so "inert at the source level"
is not inert at the instruction level. A pair built that way differs in the kernel's code and
cannot be a layout floor.

The construction that works moves the perturbation past the compiler entirely:

  1. compile ops.o ONCE for the arm -- every object of that arm then links the SAME file, so the
     kernel's instructions are identical by construction rather than by inspection;
  2. assemble a pad that occupies .text, e.g. `.globl p / p: .space N, 0x90 / ret`. A pad made
     from a C array lands in .data and shifts .text by nothing at all -- measured: five pad sizes
     from 0 to 4008 bytes left every kernel address byte-identical;
  3. link ops.o behind the pad, once per pad size.

Measured that way on redwood, pads of 0/8/24/48 bytes put the hot lambda at mod-64 alignment
16/32/48/0 -- all four 16-byte classes -- with all 18 kernel symbols instruction-identical and
every .text digest distinct.

WHAT SUCH A FLOOR BOUNDS, STATED SO IT IS NOT OVERSOLD. A compile-time macro arm varies three
things: the loop body's bytes, the loop's length and hence the placement of everything downstream,
and the loop's own address and alignment. This construction reproduces the second and third and
**cannot** reproduce the first -- no inert change can, because a change to the body IS the arm.
So the floor bounds the placement components, and what survives subtracting it is attributable to
the body change. A pair whose inert change sits entirely AFTER the hot loop reproduces only the
second, leaves the loop at its original address, and is a lower bound rather than the floor.
"""
import collections
import re
import subprocess
import sys

# NOT demangled. `objdump -C` puts C++ template names inside the <...> annotation, whose nested
# angle brackets defeat any non-recursive stripper -- the first version of this file reported 0 of
# 12 symbols identical because `<[^>]*>` left a trailing fragment behind. Undemangled annotations
# contain no unmatched '>', so the strip is well defined; symbol names are demangled separately
# for display only.
ANNOT = re.compile(r"\s*<[^>]*>")
# With --no-show-raw-insn, branch and call targets print as BARE hex with no 0x prefix
# (`call   12920`), so a 0x-anchored pattern misses exactly the operands that move when code is
# relocated -- which is every operand this comparison is meant to look through.
HEXNUM = re.compile(r"\b0x[0-9a-f]+\b|\b[0-9a-f]{4,}\b")


def disasm(path):
    """{symbol: (address, [normalised instruction text])}"""
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", path],
                         capture_output=True, text=True, check=True).stdout
    syms, cur = {}, None
    for ln in out.split("\n"):
        m = re.match(r"^([0-9a-f]+) <(.+)>:$", ln)
        if m:
            cur = m.group(2)
            syms[cur] = (int(m.group(1), 16), [])
            continue
        if cur is None:
            continue
        m = re.match(r"^\s*([0-9a-f]+):\s+(.*)$", ln)
        if not m:
            continue
        txt = ANNOT.sub("", m.group(2)).strip()
        txt = txt.split("#")[0].strip()          # objdump's trailing comment also holds addresses
        txt = HEXNUM.sub("ADDR", txt)
        syms[cur][1].append(txt)
    return syms


def text_digest(path):
    subprocess.run(["objcopy", "-O", "binary", "--only-section=.text", path, path + ".text.bin"],
                   check=True)
    return subprocess.run(["sha256sum", path + ".text.bin"],
                          capture_output=True, text=True, check=True).stdout.split()[0]


def main():
    pa, pb, pattern = sys.argv[1], sys.argv[2], sys.argv[3]
    da, db = text_digest(pa), text_digest(pb)
    print("CONDITION 1 -- the whole object's .text must DIFFER")
    print("  a  %s" % da[:32])
    print("  b  %s" % db[:32])
    print("  %s" % ("DIFFER, as required" if da != db else "IDENTICAL -- this is the"
                    " byte-identical floor again, not a new one"))

    A, B = disasm(pa), disasm(pb)
    both = sorted(set(A) & set(B))
    print()
    print("symbols: %d in a, %d in b, %d in both" % (len(A), len(B), len(both)))
    kern = [s for s in both if re.search(pattern, s)]
    if not kern:
        sys.exit("REFUSE: no symbol matches %r, so conditions 2 and 3 are unevaluated and the"
                 " zero above would be vacuous" % pattern)
    print("symbols matching %r: %d" % (pattern, len(kern)))

    same_insn = diff_insn = 0
    same_addr = diff_addr = 0
    align = collections.Counter()
    examples = []
    for s in kern:
        aa, ia = A[s]
        ab, ib = B[s]
        if ia == ib:
            same_insn += 1
        else:
            diff_insn += 1
            if len(examples) < 3:
                d = next((k for k in range(min(len(ia), len(ib))) if ia[k] != ib[k]), None)
                examples.append((s, len(ia), len(ib), d,
                                 (ia[d], ib[d]) if d is not None else None))
        if aa == ab:
            same_addr += 1
        else:
            diff_addr += 1
        align[(aa % 64, ab % 64)] += 1

    print()
    print("CONDITION 2 -- the kernel's instruction sequence must be IDENTICAL")
    print("  identical: %d of %d   differing: %d" % (same_insn, len(kern), diff_insn))
    for s, na, nb, d, pair in examples:
        print("    %s  %d vs %d instructions, first difference at %s" % (s[:70], na, nb, d))
        if pair:
            print("      a: %s" % pair[0])
            print("      b: %s" % pair[1])

    print()
    print("CONDITION 3 -- the kernel's ADDRESS must differ, else there is no layout difference")
    print("  address differs: %d of %d   same address: %d" % (diff_addr, len(kern), same_addr))
    shifted = sum(n for (x, y), n in align.items() if x != y)
    print("  alignment mod 64 differs: %d of %d" % (shifted, len(kern)))
    print("  (an identical-instruction symbol at an identical alignment contributes NOTHING to a"
          " layout floor)")

    # CONTROL ON THE NORMALISER. Looking through relocated operands is exactly how this
    # comparison could erase the difference it exists to find, so check it still sees one: mutate
    # one instruction of a matched symbol and require the comparison to flag it.
    probe = kern[0]
    mutated = list(A[probe][1])
    mutated[len(mutated) // 2] = mutated[len(mutated) // 2] + " ;MUTANT"
    if mutated == B[probe][1]:
        sys.exit("REFUSE: the normaliser cannot distinguish a mutated instruction sequence, so"
                 " 'identical' above carries no information")
    print()
    print("NORMALISER CONTROL: a one-instruction mutation of %s is still detected" % probe[:48])

    print()
    ok = (da != db) and diff_insn == 0 and shifted > 0
    print("VERDICT: %s" % ("all three conditions hold -- this pair is a candidate floor"
                           if ok else "NOT a usable floor pair as constructed"))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
