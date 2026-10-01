import hashlib, os, shutil, subprocess, sys
A = "/Users/bobby/scorch-beat-mkl/harnesses/an_c211.py"
HDR = "build,slice,key,k,rows,nnz,deg,arm,ms,batch\n"
OBJ = ("base", "bsb_", "deep"); KS = (2, 4, 8, 16, 32); ARMS = ("ours", "oursb", "mkl")

def build(d):
    """A tree valid in EVERY respect the analyser checks before the dims join -- including
    per-pass variation in ms, so the 18 files are not accidentally byte-identical."""
    os.makedirs(d, exist_ok=True)
    for ob in OBJ:
        for p in range(1, 7):
            with open(os.path.join(d, "dl_%s_float32_p%d.csv" % (ob, p)), "w") as h:
                h.write(HDR)
                for i in range(3820):
                    for k in KS:
                        for a in ARMS:
                            h.write("%s,x,key%05d,%d,512,%d,%d,%s,%.4f,1\n"
                                    % (ob, i, k, 1000 + i, 2 + i % 5, a, 1.0 + 0.001 * p + 0.0001 * i))

def run(d):
    r = subprocess.run([sys.executable, A, d], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().split("\n")
    return next((l for l in out if l.startswith("REFUSE")), "(no refusal) " + out[-1][:60])

base = "/tmp/p211/base"
shutil.rmtree("/tmp/p211", ignore_errors=True)
build(base)
results = {}
results["CONTROL (valid)"] = run(base)

plants = {}
def plant(name, fn):
    d = "/tmp/p211/" + name.replace(" ", "_")
    shutil.copytree(base, d); fn(d)
    plants[name] = run(d)

def dup(d):
    shutil.copyfile(os.path.join(d, "dl_deep_float32_p2.csv"), os.path.join(d, "dl_deep_float32_p3.csv"))
def repeat_cell(d):
    f = os.path.join(d, "dl_base_float32_p1.csv"); L = open(f).readlines()
    L[5] = L[4]                      # duplicate one cell; row count unchanged
    open(f, "w").writelines(L)
def bad_k(d):
    f = os.path.join(d, "dl_bsb__float32_p1.csv"); L = open(f).readlines()
    L[1] = L[1].replace(",2,512,", ",7,512,", 1)
    open(f, "w").writelines(L)
def bad_arm(d):
    f = os.path.join(d, "dl_deep_float32_p5.csv"); L = open(f).readlines()
    L[3] = L[3].replace(",mkl,", ",mkx,", 1)
    open(f, "w").writelines(L)
def meta_clash(d):
    f = os.path.join(d, "dl_deep_float32_p6.csv"); L = open(f).readlines()
    L[1] = L[1].replace(",512,1000,", ",999,1000,", 1)
    open(f, "w").writelines(L)

plant("duplicate pass", dup)
plant("repeated cell", repeat_cell)
plant("k outside KS", bad_k)
plant("wrong arm name", bad_arm)
plant("rows disagree across files", meta_clash)

print("%-30s %s" % ("PLANT", "FIRST REFUSAL"))
print("-" * 118)
for k, v in [("CONTROL (valid)", results["CONTROL (valid)"])] + list(plants.items()):
    print("%-30s %s" % (k, v[:86]))
print()
msgs = list(plants.values())
dupmsg = [m for m in set(msgs) if msgs.count(m) > 1]
print("MASKING CHECK -- identical output across distinct plants is the signature:")
print("  distinct plants %d, distinct messages %d -> %s"
      % (len(msgs), len(set(msgs)), "OK, each guard reached" if len(set(msgs)) == len(msgs)
         else "MASKED: %d message(s) shared" % len(dupmsg)))
shutil.rmtree("/tmp/p211", ignore_errors=True)
