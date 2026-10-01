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
def m_dimsrows(d):
    # The DIMS-DISAGREEMENT guard: the board against the dims file, as distinct from the
    # cross-file META guard (one object-pass against another). This is the only check standing
    # between a wrong dims file and every coefficient downstream of the join, and it was credited
    # as covered by a plant that cannot reach it. Synthesised rather than taken from an attested
    # wrong copy on purpose: a plant's job is to trip a guard reproducibly from this repo, and a
    # generated input makes the test runnable by anyone without fetching an artefact.
    pass   # the mutation is in the dims override, not the tree

def m_keycount(d):
    # The peer's construction, and it refutes my dominance claim. Drop one key's 15 rows and add
    # key X with 10 cells (five widths x {ours, oursb}) plus key Y with 5 cells (five widths x
    # {mkl}). Row count stays 57300, every (key,k,arm) is unique, every width is present, and the
    # arm set is exactly the three -- X supplies ours and oursb, Y supplies mkl. So uniqueness +
    # row count + k-subset + missing-width + arm-set ALL pass at 3822 keys. Row count and
    # uniqueness bound #keys from BELOW (>= 3820); only the distinct-key guard pins it.
    f = os.path.join(d, "dl_base_float32_p1.csv")
    L = open(f).readlines()
    victim = KEYS[7][0]
    kept = [l for l in L[1:] if l.split(",")[2] != victim]
    assert len(L) - 1 - len(kept) == 15, "expected to drop exactly 15 rows"
    nr, nnz = KEYS[7][1], KEYS[7][3]
    add = []
    for k in KS:
        for a in ("ours", "oursb"):
            add.append("base,x,SYNTH_X,%d,%d,%d,1.0,%s,1.0,1\n" % (k, nr, nnz, a))
    for k in KS:
        add.append("base,x,SYNTH_Y,%d,%d,%d,1.0,mkl,1.0,1\n" % (k, nr, nnz))
    assert len(add) == 15, len(add)
    open(f, "w").writelines([L[0]] + kept + add)

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
    ("rows disagree across files", m_meta,    "seen earlier", None),
    ("blank line (parsed rows)",   m_blank,   "parsed", None),
    ("key count 3822 (peer's)",    m_keycount, "distinct keys", None),
]

# A dims file carrying one key twice with CONTRADICTORY values. Needs its own dims override, so
# it is appended after PLANTS is built, like the collinearity and partial-join plants.
_dupdims = os.path.join(work, "dims_dupkey.csv")
with open(_dupdims, "w") as h:
    h.write("key,rows,cols,nnz\n")
    for key, nrow, nc, nnz in KEYS:
        h.write("%s,%d,%d,%d\n" % (key, nrow, nc, nnz))
    k3 = KEYS[3]
    h.write("%s,%d,%d,%d\n" % (k3[0], k3[1], k3[2] + 64, k3[3]))
PLANTS.append(("dims file repeats a key", lambda d: None, "two different rows/cols/nnz", _dupdims))

_wrongdims = os.path.join(work, "dims_wrongrows.csv")
with open(_wrongdims, "w") as h:
    h.write("key,rows,cols,nnz\n")
    for i, (key, nrow, nc, nnz) in enumerate(KEYS):
        h.write("%s,%d,%d,%d\n" % (key, nrow + (1 if i == 11 else 0), nc, nnz))
PLANTS.append(("dims rows disagree w/ board", m_dimsrows, "disagree with the board", _wrongdims))

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
_sites_pre = re.findall(r'sys\.exit\(\s*"(REFUSE: [^"]*)"', src)
_sites_pre += re.findall(r'sys\.exit\(\s*"(usage: [^"\\]*)', src)
# EACH EXPECTATION MUST MATCH EXACTLY ONE SITE. The per-site check below catches a site claimed by
# two plants; this catches the other two failures it cannot see -- an expectation matching SEVERAL
# sites (so one plant is credited for a guard it cannot reach, which is what happened with
# "rows/nnz") and an expectation matching NONE (a stale string left behind when a message is
# reworded, which would silently drop a plant from the coverage count while it still passes).
print()
print("EXPECTATION DISCRIMINATION -- each plant's expected substring must match exactly one of the")
print("analyser's %d exit messages, so coverage is attributable and survives a rewording:" % len(_sites_pre))
_bad_exp = []
for _n, _f, _e, _d in PLANTS:
    if not _e:
        continue
    _hits = [x for x in _sites_pre if _e in x]
    if len(_hits) != 1:
        _bad_exp.append((_n, _e, len(_hits)))
if _bad_exp:
    for _n, _e, _c in _bad_exp:
        print("  FAIL  %-30s %r matches %d sites" % (_n, _e, _c))
        fails.append("expectation %r for %s matches %d sites (need exactly 1)" % (_e, _n, _c))
else:
    print("  PASS  all %d expectations match exactly one site each"
          % sum(1 for _, _, e, _ in PLANTS if e))

sites = re.findall(r'sys\.exit\(\s*"(REFUSE: [^"]*)"', src)
sites += re.findall(r'sys\.exit\(\s*"(usage: [^"\\]*)', src)
covered, uncovered, ambiguous = [], [], []
for site in sites:
    if site.startswith("usage:") and usage_ok:
        covered.append(site); continue
    hits = [n for n, _, expect, _ in PLANTS if expect and expect in site]
    if len(hits) == 1:
        covered.append(site)
    elif len(hits) > 1:
        # A substring shared by two refusal messages attributes one plant to both sites and
        # inflates coverage. This happened: "rows/nnz" appears in the META disagreement message
        # AND the dims-duplicate message, so adding the second guard made coverage read 15/18
        # with nothing planted for it. Report rather than count.
        ambiguous.append((site, hits))
    else:
        uncovered.append(site)
print()
print("COVERAGE, derived from the analyser's own sys.exit sites rather than a list written here:")
print("  %d refusal sites; %d covered by exactly one plant, %d ambiguous, %d not"
      % (len(sites), len(covered), len(ambiguous), len(uncovered)))
for site, hits in ambiguous:
    print("    AMBIGUOUS  %s" % site[:70])
    print("               matched by %d plants (%s) -- substring is not discriminating"
          % (len(hits), ", ".join(hits)))
    fails.append("coverage substring for %s matches %d sites" % (hits, len(hits)))
for s in uncovered:
    print("    UNCOVERED  %s" % s[:92])
print("  The remainder are UNREACHABLE rather than untested, and the argument is given so the")
print("  claim can be checked rather than believed:")
print("    * 'log degree did not read collinear' is a code-integrity check -- degree is computed")
print("      as log nnz - log rows, so it is collinear by construction and only a broken analyser")
print("      can trip it. No input reaches it.")
print("    * the missing-width check is dominated, but ONLY GIVEN the distinct-key guard, and")
print("      the premise has to be stated because an earlier version of this argument got it")
print("      wrong. Row count plus uniqueness bounds the key count from BELOW only: a key holds")
print("      at most 15 unique cells, so 57300 unique cells force #keys >= 3820, NOT == 3820,")
print("      because a key may carry fewer than 15. Once #keys == 3820 is separately asserted,")
print("      15 cells per key is forced and every key must then carry all five widths and all")
print("      three arms -- so missing-width cannot fire. The distinct-key guard is therefore")
print("      LOAD-BEARING for that argument, and it is planted above (3822 keys) rather than")
print("      argued away.")
print()
print("  WHAT THE DISTINCT-KEY GUARD UNIQUELY REJECTS, stated precisely because an earlier")
print("  version of this comment overclaimed it. Running the plant shows the guard fires FIRST;")
print("  it does NOT show the guard is necessary, which is a counterfactual and needs an")
print("  ablation. Ablated, against the real 3820-key dims file, the 3822-key input is STILL")
print("  refused -- by the join-coverage guard, since the two synthetic keys are absent from J.")
print("  Both directions are in fact closed when |J| == 3820:")
print("    from BELOW  a key holds at most 5x3 = 15 unique cells, so 57300 unique cells force")
print("               >= 3820 keys per file, and META is a union over the 18, so always >= 3820")
print("    from ABOVE  more than 3820 board keys means one is missing from a 3820-key J, so the")
print("               join guard refuses")
print("  So against a 3820-key dims file the set this guard uniquely rejects is EMPTY. Its")
print("  independent value is confined to ONE input class: a dims file with MORE than 3820 keys,")
print("  which this analyser deliberately allows (a larger dims file joins correctly and is")
print("  harmless, so asserting |J| == 3820 would reject a valid run). Verified by construction:")
print("  with an oversized J that also contains the two synthetic keys, the ablated analyser")
print("  accepts 3822 keys with no refusal at all, and the guard is what rejects it. The guard")
print("  is therefore what makes that allowance safe -- a narrower and checkable claim than")
print("  'nothing else catches this input'.")

print()
if fails:
    print("FAILURES (%d):" % len(fails))
    for f in fails:
        print("  %s" % f)
    shutil.rmtree(work, ignore_errors=True)
    sys.exit(1)
print("ALL %d PLANTS FIRED ATTRIBUTABLY, and the control passed end to end." % len(PLANTS))
shutil.rmtree(work, ignore_errors=True)
