#!/usr/bin/env python3
"""Plant each of an_c211.py's refusals and require it to fire, attributably.

THIS IS A TEST AND EXITS NONZERO ON FAILURE. An earlier version was a report: it printed a row
labelled "CONTROL (valid)" that was itself refusing, then printed "OK, each guard reached" and
exited 0. A harness whose own control fails while its verdict line says OK is the exact defect it
exists to catch, so the control is now asserted first and everything else is skipped if it fails.

THREE PROPERTIES, each of which the previous version lacked:

  CONTROL PASSES END TO END. The synthetic tree is built from the REAL dims file's keys, with its
  rows and nnz, so the dims join succeeds. The old tree used `key%05d` with rows=512, which cannot
  join real DLMC paths -- so every guard at or after the join was unreachable by construction,
  including both collinearity refusals. Only the `ms` values are synthetic, varied per pass so the
  18 files are not accidentally byte-identical (which would trip the duplicate guard being
  controlled for).

  ATTRIBUTION, NOT MERELY DISTINCTNESS. `len(set(msgs)) == len(msgs)` would pass if two plants
  tripped each other's guard with distinct messages. Each plant declares an expected substring and
  the refusal must contain it.

  COVERAGE IS DERIVED FROM THE ANALYSER'S SOURCE, not from a list written here. A hand list can
  only cover what its author thought of; the refusal sites are extracted mechanically and
  set-differenced against the plants, so an unplanted site is reported rather than forgotten.
"""
import hashlib, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSER = os.path.join(HERE, "an_c211.py")          # not a hardcoded absolute path
DIMS = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/.cache/scorch-codex/verifier-dlmc-dims/dlmc_dims.csv")

HDR = "build,slice,key,k,rows,nnz,deg,arm,ms,batch\n"
OBJ = ("base", "bsb_", "deep")
KS = (2, 4, 8, 16, 32)
ARMS = ("ours", "oursb", "mkl")

if not os.path.exists(DIMS):
    sys.exit("REFUSE: no dims file at %s; the control cannot pass without a real key set" % DIMS)
import csv as _csv
rows = []
with open(DIMS) as h:
    rdr = _csv.reader(l for l in h if not l.startswith("#"))
    ix = {n: i for i, n in enumerate(next(rdr))}
    for r in rdr:
        rows.append((r[ix["key"]], int(r[ix["rows"]]), int(r[ix["cols"]]), int(r[ix["nnz"]])))
if len(rows) < 3820:
    sys.exit("REFUSE: dims file holds %d keys, need 3820" % len(rows))
KEYS = rows[:3820]

def build(d, dims_override=None):
    os.makedirs(d, exist_ok=True)
    for ob in OBJ:
        for p in range(1, 7):
            with open(os.path.join(d, "dl_%s_float32_p%d.csv" % (ob, p)), "w") as h:
                h.write(HDR)
                for i, (key, nrow, _ncol, nnz) in enumerate(KEYS):
                    deg = nnz / nrow if nrow else 0
                    for k in KS:
                        for a in ARMS:
                            h.write("%s,x,%s,%d,%d,%d,%.4f,%s,%.5f,1\n"
                                    % (ob, key, k, nrow, nnz, deg, a,
                                       1.0 + 0.001 * p + 0.000001 * i))
    return d

def run(d, dims=DIMS):
    r = subprocess.run([sys.executable, ANALYSER, d, dims], capture_output=True, text=True)
    txt = r.stdout + r.stderr
    ref = next((l for l in txt.split("\n") if l.startswith("REFUSE")), None)
    return r.returncode, ref, txt

# ---------------------------------------------------------------- the control, asserted first
work = tempfile.mkdtemp(prefix="plant_c211_")
base = build(os.path.join(work, "control"))
rc, ref, txt = run(base)
print("CONTROL (must pass end to end)")
if ref is not None:
    print("  FAIL: the control refuses -- every later guard is unreachable and no plant below")
    print("        means anything. %s" % ref)
    shutil.rmtree(work, ignore_errors=True)
    sys.exit(1)
if rc != 0:
    print("  FAIL: control exited %d with no REFUSE line; output tail:\n%s" % (rc, txt[-400:]))
    shutil.rmtree(work, ignore_errors=True)
    sys.exit(1)
print("  PASS: no refusal, exit 0 -- the join succeeds and the downstream guards are reachable")

# ---------------------------------------------------------------- the plants
def m_missing(d):   os.remove(os.path.join(d, "dl_deep_float32_p4.csv"))
def m_short(d):
    f = os.path.join(d, "dl_base_float32_p1.csv"); L = open(f).readlines()
    open(f, "w").writelines(L[:-1])
def m_extra(d):
    shutil.copyfile(os.path.join(d, "dl_deep_float32_p3.csv"),
                    os.path.join(d, "dl_deep_float32_p9.csv"))
def m_label(d):
    f = os.path.join(d, "dl_deep_float32_p2.csv"); t = open(f).read()
    open(f, "w").write(t.replace("\ndeep,", "\nbase,", 1))
def m_dup(d):
    shutil.copyfile(os.path.join(d, "dl_deep_float32_p2.csv"),
                    os.path.join(d, "dl_deep_float32_p3.csv"))
def m_repeat(d):
    # SINGLE defect: overwrite a row of key[1] with a row of key[0], so one cell repeats and the
    # deleted cell belongs to a DIFFERENT key. The old version did L[5] = L[4], which duplicated
    # one cell and deleted its neighbour -- two defects in one plant.
    f = os.path.join(d, "dl_base_float32_p1.csv"); L = open(f).readlines()
    L[16] = L[1]
    open(f, "w").writelines(L)
def m_badk(d):
    f = os.path.join(d, "dl_bsb__float32_p1.csv"); L = open(f).readlines()
    L[1] = L[1].replace(",2,", ",7,", 1)
    open(f, "w").writelines(L)
def m_arm(d):
    f = os.path.join(d, "dl_deep_float32_p5.csv"); L = open(f).readlines()
    L[3] = L[3].replace(",mkl,", ",mkx,", 1)
    open(f, "w").writelines(L)
def m_blank(d):
    # Reaches the POST-PARSE row check, which the pre-read line count cannot: a blank line keeps
    # the line count at 57301 while csv.DictReader skips it, so parsed rows come out one short.
    f = os.path.join(d, "dl_bsb__float32_p3.csv"); L = open(f).readlines()
    L.insert(900, "\n")
    open(f, "w").writelines(L[:-1])
def m_meta(d):
    f = os.path.join(d, "dl_deep_float32_p6.csv"); L = open(f).readlines()
    parts = L[1].split(","); parts[4] = str(int(parts[4]) + 7)
    L[1] = ",".join(parts)
    open(f, "w").writelines(L)

PLANTS = [
    ("missing object-pass",        m_missing, "missing object-pass", None),
    ("short file (row count)",     m_short,   "data rows, expected", None),
    ("19th undeclared CSV",        m_extra,   "beyond the 18 declared", None),
    ("build label mismatch",       m_label,   "contains build label", None),
    ("duplicate pass",             m_dup,     "byte-identical", None),
    ("repeated cell",              m_repeat,  "repeats cell", None),
    ("k outside KS",               m_badk,    "outside the declared", None),
    ("wrong arm name",             m_arm,     "arm set is", None),
    ("rows disagree across files", m_meta,    "rows/nnz", None),
    ("blank line (parsed rows)",   m_blank,   "parsed", None),
]

# A dims file whose J is an EXACT function of rows and nnz, to reach the collinearity refusal.
coll = os.path.join(work, "dims_collinear.csv")
with open(coll, "w") as h:
    h.write("key,rows,cols,nnz\n")
    for key, nrow, _nc, nnz in KEYS:
        # cols := nnz, so log J IS log nnz and the residual is exactly zero. A first attempt used
        # cols = round(nnz**0.3 * rows**0.7), which is in the span only BEFORE rounding -- the
        # integer round left a residual above the 1e-18 threshold and the guard correctly did not
        # fire. Exact collinearity needs an exact integer identity, not an approximate one.
        h.write("%s,%d,%d,%d\n" % (key, nrow, nnz, nnz))
PLANTS.append(("log J collinear with nnz,rows", lambda d: None, "reads collinear", coll))
# And a dims file missing some keys, to reach the join refusal.
part = os.path.join(work, "dims_partial.csv")
with open(part, "w") as h:
    h.write("key,rows,cols,nnz\n")
    for key, nrow, nc, nnz in KEYS[:-5]:
        h.write("%s,%d,%d,%d\n" % (key, nrow, nc, nnz))
PLANTS.append(("dims join missing keys", lambda d: None, "absent from the dims join", part))

# The usage refusal takes no tree at all, so it is run directly rather than through build().
rc_u = subprocess.run([sys.executable, ANALYSER], capture_output=True, text=True)
usage_ok = rc_u.returncode != 0 and "usage: an_c211.py" in (rc_u.stdout + rc_u.stderr)

print()
print("%-32s %-8s %s" % ("PLANT", "VERDICT", "REFUSAL (must contain the expected substring)"))
print("-" * 118)
fails, msgs = [], {}
for name, fn, expect, dims in PLANTS:
    d = build(os.path.join(work, re.sub(r"\W+", "_", name)))
    fn(d)
    rc, ref, txt = run(d, dims or DIMS)
    if ref is None:
        fails.append("%s: did NOT refuse" % name); v = "FAIL"
    elif expect not in ref:
        fails.append("%s: refused but not attributably (expected %r)" % (name, expect)); v = "WRONG"
    else:
        v = "ok"
    msgs[name] = ref
    print("%-32s %-8s %s" % (name, v, (ref or "(no refusal)")[:74]))

# ---------------------------------------------------------------- coverage, from the source
print("%-32s %-8s %s" % ("no argument (usage)", "ok" if usage_ok else "FAIL",
                         "REFUSES without a tree root, so no ledger path is defaulted"))
if not usage_ok:
    fails.append("no argument: did not refuse")

src = open(ANALYSER).read()
sites = re.findall(r'sys\.exit\(\s*"(REFUSE: [^"]*)"', src)
sites += re.findall(r'sys\.exit\(\s*"(usage: [^"\\]*)', src)
covered, uncovered = [], []
for s in sites:
    if s.startswith("usage:") and usage_ok:
        covered.append(s); continue
    if any(expect in s for _, _, expect, _ in PLANTS if expect):
        covered.append(s)
    else:
        uncovered.append(s)
print()
print("COVERAGE, derived from the analyser's own sys.exit sites rather than a list written here:")
print("  %d refusal sites; %d covered by a plant, %d not" % (len(sites), len(covered), len(uncovered)))
for s in uncovered:
    print("    UNCOVERED  %s" % s[:92])
print("  The remainder are UNREACHABLE rather than untested, and the argument is given so the")
print("  claim can be checked rather than believed:")
print("    * 'log degree did not read collinear' is a code-integrity check -- degree is computed")
print("      as log nnz - log rows, so it is collinear by construction and only a broken analyser")
print("      can trip it. No input reaches it.")
print("    * the distinct-key count and the missing-width check are both DOMINATED by cell")
print("      uniqueness. A key can hold at most 5 widths x 3 arms = 15 unique cells, and")
print("      3820 x 15 = 57300 exactly, so once the row count and uniqueness both hold, every")
print("      key must carry all five widths and all three arms. Fewer keys, or a key missing a")
print("      width, forces a duplicate cell and refuses earlier. Unreachable, not untested.")

print()
if fails:
    print("FAILURES (%d):" % len(fails))
    for f in fails:
        print("  %s" % f)
    shutil.rmtree(work, ignore_errors=True)
    sys.exit(1)
print("ALL %d PLANTS FIRED ATTRIBUTABLY, and the control passed end to end." % len(PLANTS))
shutil.rmtree(work, ignore_errors=True)
