"""Build a floor family for one arm by RE-RUNNING THAT ARM'S OWN LINK with a .text pad inserted.

Usage: pfs_floor_link.py <tree-root> <out-dir> <pad-bytes...>

WHY THIS DOES NOT WRITE A LINK LINE. The floor family and the arm must share the same link, or the
floor bounds placement variation in an object that is not shaped like the one being measured. The
two hosts this project measures on do not agree on that link: MKT's system Python passes
`-Wl,-O1 -Wl,-Bsymbolic-functions`, redwood's conda Python does not, and without symbolic binding
the kernel's own calls acquire @plt trampolines. A hand-written link line is therefore host-wrong
by default -- mine matched MKT's by accident. So the line is EXTRACTED from the arm's recorded
build.log and re-run verbatim with one object added and the output path changed.

WHY PYTHON AND NOT SHELL. The flags must not pass through an unquoted shell variable. redwood's
interactive shell is zsh, which does not word-split unquoted parameter expansions, so a flag
string arrives as ONE argument. g++ rejects that loudly for compiler flags; for `-L... -lc10 ...`
it accepts a bogus search path and, because a -shared link permits undefined symbols, SUCCEEDS
with zero DT_NEEDED torch entries. The result has rc=0, a plausible size and correct disassembly,
and cannot be loaded. subprocess with an argument list cannot have that failure.

THE PAD. Assembled `.space N, 0x90` inside .text. A pad declared as a C array lands in .data and
shifts .text by nothing -- measured, pads from 0 to 4008 bytes left every kernel address
byte-identical while rc was 0 and the file grew.

THE ALIGNMENT DESIGN. Pads 64 apart keep the hot loop in ONE mod-64 class; pads 16 apart move it
between classes. Conditioning the floor on the arm's own class is the right comparison, because
the arm sits in exactly one class fixed by the compiler rather than drawn from a distribution --
so the useful family is r members at the ARM's class, which costs r objects rather than 4r. The
statistic over them must be n-invariant: a range inflates 1.93x from n=3 to n=12 on sample size
alone, where a c4-corrected standard deviation is unbiased at every n (measured 0.999 at n = 3, 6,
12 and 24). At n=3 that corrected sd still has a p95/p5 of 7.59x, 3.11x at n=6 and 2.07x at n=12,
which is the argument for spending the objects on one class rather than spreading them over four.
"""
import hashlib
import importlib.util
import os
import re
import shlex
import subprocess
import sys

HOT = "_ZZ16spmm_csr_v2_coreIfLb0E"


def find_link_line(root):
    """The arm's own link command, or a refusal. Must be exactly one candidate."""
    log = os.path.join(root, "build.log")
    if not os.path.exists(log):
        sys.exit("REFUSE: %s has no build.log, so the arm's link line is unknown and writing one "
                 "by hand is host-wrong by default" % root)
    cands = [ln.strip() for ln in open(log, errors="replace")
             if " -shared " in ln and re.search(r"\bops\.o\b", ln) and " -o " in ln]
    if len(cands) != 1:
        sys.exit("REFUSE: found %d candidate link lines in %s, not 1; a guess here would silently "
                 "change the linkage the floor is supposed to share with the arm"
                 % (len(cands), log))
    return cands[0]


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: pfs_floor_link.py <tree-root> <out-dir> <pad-bytes...>")
    root, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    pads = [int(x) for x in sys.argv[3:]]
    os.makedirs(out, exist_ok=True)

    line = find_link_line(root)
    argv = shlex.split(line)
    obj_i = next((i for i, a in enumerate(argv) if a.endswith("ops.o")), None)
    out_i = argv.index("-o") + 1 if "-o" in argv else None
    if obj_i is None or out_i is None:
        sys.exit("REFUSE: the extracted link line has no ops.o or no -o; refusing to edit it")
    obj = argv[obj_i] if os.path.isabs(argv[obj_i]) else os.path.join(root, argv[obj_i])
    if not os.path.exists(obj):
        sys.exit("REFUSE: the link line names %s, which does not exist" % obj)

    print("arm tree      %s" % root)
    print("arm object    %s" % obj)
    print("  sha256      %s" % hashlib.sha256(open(obj, "rb").read()).hexdigest())
    print("  THE SAME file is linked into every member, so the kernel's instructions cannot differ")
    print("link line     extracted from the arm's own build.log, re-run verbatim")
    print("  sha256      %s" % hashlib.sha256(line.encode()).hexdigest()[:32])
    print("  linker args %s" % " ".join(a for a in argv if a.startswith("-Wl,")) or "(none)")
    print()

    fail = []
    print("%-7s %-6s %-20s %-6s %-8s %-6s %s" % ("pad", "link", "hot lambda", "mod64",
                                                 "NEEDED", "@plt", "loads"))
    for n in pads:
        s, o = os.path.join(out, "pad_%d.s" % n), os.path.join(out, "pad_%d.o" % n)
        so = os.path.join(out, "arm_pad%d.so" % n)
        with open(s, "w") as fh:
            fh.write("        .section .text\n        .globl scorch_floor_pad_%d\n"
                     "scorch_floor_pad_%d:\n" % (n, n))
            if n > 0:
                fh.write("        .space %d, 0x90\n" % n)
            fh.write("        ret\n")
        if subprocess.run(["gcc", "-c", "-fPIC", s, "-o", o],
                          capture_output=True).returncode != 0:
            fail.append("pad %d would not assemble" % n)
            continue
        cmd = list(argv)
        cmd[out_i] = so
        cmd[obj_i] = obj
        cmd.insert(obj_i, o)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            fail.append("pad %d link rc=%d: %s" % (n, r.returncode, r.stderr.strip()[:120]))
            continue
        nm = subprocess.run(["nm", so], capture_output=True, text=True).stdout
        addr = next((ln.split()[0] for ln in nm.split("\n")
                     if HOT in ln and len(ln.split()) > 2 and ln.split()[1] in "TtWw"), None)
        rd = subprocess.run(["readelf", "-d", so], capture_output=True, text=True).stdout
        need = len([ln for ln in rd.split("\n") if "NEEDED" in ln and "torch" in ln])
        od = subprocess.run(["objdump", "-d", "--no-show-raw-insn", so],
                            capture_output=True, text=True).stdout
        nplt = len([ln for ln in od.split("\n")
                    if re.match(r"^[0-9a-f]+ <.+>:$", ln) and "@plt" in ln
                    and "spmm_csr_v2_core" in ln])
        # Loadability is the only check the broken-flags failure mode does not pass. The module
        # name MUST be scorch_ops: TORCH_EXTENSION_NAME is fixed at compile time, so the init
        # symbol is PyInit_scorch_ops and any other name fails exactly like a broken object --
        # which is why a load test reporting NO for every member is a bug in the test.
        probe = subprocess.run(
            [sys.executable, "-c",
             "import sys, torch, importlib.util\n"
             "spec = importlib.util.spec_from_file_location('scorch_ops', %r)\n"
             "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
             "sys.exit(0 if hasattr(m, 'spmm_csr_float') else 3)\n" % so],
            capture_output=True, text=True)
        loads = probe.returncode == 0
        if not loads:
            fail.append("pad %d does not load: %s" % (n, probe.stderr.strip().split("\n")[-1][:90]))
        if need < 1:
            fail.append("pad %d has no DT_NEEDED torch entry" % n)
        print("%-7d %-6s 0x%-18s %-6s %-8d %-6d %s"
              % (n, "rc=0", addr or "none", (int(addr, 16) % 64) if addr else "?",
                 need, nplt, "yes" if loads else "NO"))

    print()
    if fail:
        print("REFUSE: %d member(s) unusable. An object with rc=0, a plausible size and correct"
              % len(fail))
        print("        disassembly can still have zero DT_NEEDED torch entries and not load.")
        for f in fail[:6]:
            print("    %s" % f)
        return 1
    print("all %d members link through the ARM'S OWN link line, carry torch, and load." % len(pads))
    return 0


if __name__ == "__main__":
    sys.exit(main())
