#!/bin/bash
export HOME=/scratch/bobbyy/fakehome
source /scratch/bobbyy/miniconda3/etc/profile.d/conda.sh
conda activate scorch
cd /scratch/bobbyy/synchk || exit 1
objdump -dC --no-show-raw-insn deep.o > deep.asm 2>/dev/null
echo "deep.asm lines: $(wc -l < deep.asm)"
python3 - <<'PY'
import re, collections
txt = open("/scratch/bobbyy/synchk/deep.asm").read()
# split into symbol bodies
blocks = re.split(r"\n(?=[0-9a-f]{16} <)", txt)
JMP = re.compile(r"^\s*([0-9a-f]+):\s+(j\w+)\s+([0-9a-f]+)")
INS = re.compile(r"^\s*([0-9a-f]+):\s+([a-z0-9.]+)")
def innermost(lines):
    """smallest backward-jump span containing no ret/call/push/pop, as asmcheck does"""
    addrs, mnem = [], []
    for l in lines:
        m = INS.match(l)
        if m: addrs.append(int(m.group(1),16)); mnem.append(m.group(2))
    best = None
    for l in lines:
        m = JMP.match(l)
        if not m: continue
        src, tgt = int(m.group(1),16), int(m.group(3),16)
        if tgt >= src: continue
        span = [(a,x) for a,x in zip(addrs,mnem) if tgt <= a <= src]
        if any(x in ("ret","call","push","pop") or x.startswith("call") for _,x in span):
            continue
        if best is None or len(span) < len(best): best = span
    return best
rows=[]
for b in blocks:
    h = b.split("\n",1)[0]
    m = re.match(r"[0-9a-f]{16} <(.+)>:", h)
    if not m: continue
    name = m.group(1)
    if "row_regblock_deep" not in name: continue
    lo = innermost(b.split("\n"))
    if not lo: continue
    c = collections.Counter(x for _,x in lo)
    fma = sum(v for k,v in c.items() if "fmadd" in k)
    ld  = sum(v for k,v in c.items() if k.startswith("vmov") or k.startswith("vbroadcast") or "gather" in k)
    short = re.sub(r"scorch_spmm_row_regblock_deep<(.*?), (scorch_simd.*)", r"<\1>", name)
    rows.append((short, len(lo), fma, ld))
print()
print("  %-46s %5s %5s %5s" % ("instantiation <T,NVEC,FULL_LAST,UNROLL,PF>", "insn", "FMA", "ld"))
print("  " + "-"*70)
for r in sorted(rows):
    print("  %-46s %5d %5d %5d" % r)
print()
print("FMA count in the innermost loop = vector-FMAs per body.")
print("nonzeros per body = FMA / NVEC  (each nonzero does NVEC FMAs, one per output vector)")
for name, n, fma, ld in sorted(rows):
    mm = re.search(r"<(\w+), (\d+), (\w+), (\d+)", name)
    if mm and fma:
        nv, un = int(mm.group(2)), int(mm.group(4))
        print("  %-46s FMA %3d / NVEC %d = %5.1f nonzeros/body   (source steps %d)"
              % (name, fma, nv, fma/nv, un))
PY
