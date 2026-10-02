#!/bin/bash
#SBATCH -A mkt
#SBATCH -p mkt
#SBATCH -J pfs-dlmc-f32
#SBATCH -t 12:00:00
# -c 32 IS LOAD-BEARING. Without a cpus-per-task request Slurm gives the default allocation and
# torch.get_num_threads() reads 2 on this node, so OMP_NUM_THREADS=32 cannot help -- the cgroup
# caps it. A first submission of this script ran at pool 2 and would have completed normally,
# producing timings on a 2-CPU allocation that look like the 32-CPU screen's and are comparable
# to nothing. The working float64 screen requests -c 32; this matches it.
#SBATCH -c 32
#SBATCH -o /scr/bobbyy/pfs/screen_dlmc.log
#
# The four-arm prefetch screen over DLMC, float32.
#
# WHY float32 HERE AND WHY THESE WIDTHS. DLMC is pruned neural-network weight matrices, where
# float32 is the dtype the workload actually uses. It is also the dtype in which the register
# block kernel reaches k=32 (8 lanes, NVEC 1..4), which is where this project recorded the
# prefetch removal HARMING 115 matrices -- a width float64 cannot reach at all. So this screen
# covers the side of that trade the float64 screen is structurally blind to.
#
# The width set 2,8,12,16,20,24,32 is checked by the driver against the lane count: k=2 sits
# inside the exact kernel's range, which is k=1..3 for BOTH dtypes (the shipped
# SCORCH_NARROWK_EXACT_HI; the `exact_f32_ ? 7 : 3` in the source only clamps), and is a
# NEGATIVE CONTROL where the
# macro cannot change executed code; 8/16/24/32 are full tails at NVEC 1/2/3/4; 12 and 20 are
# masked tails at NVEC 2 and 3. The historical float32 set 2,4,8,16,32 reached neither NVEC 3 nor
# any masked tail, and the driver refuses it by name.
#
# WHY A STRATIFIED CORPUS. DLMC is a factorial design (model x pruning method x sparsity x layer)
# with unequal layer counts per cell (53-97), so a flat sample would load the corpus onto whichever
# cell has the most layers. 392 keys, 8 from each of the 49 populated strata.
set -u

# DIGEST GATE. sbatch copies this script at submit time, so an edit after submission does not
# reach a queued job. A stale stage must therefore be caught at startup rather than discovered
# in the output.
#
# THIS BLOCK USED TO ONLY PRINT. It recorded each producer's digest and refused on ABSENCE
# only, with no expected value anywhere in the file -- so a stale-but-present file passed and
# had its digest written into the log, which is discovery in the output: exactly what the
# comment above claimed it prevented. `pfs_screen_fused.sbatch` had the real gate the whole
# time, a `declare -A WANT` of full digests compared with an explicit refusal, so this was an
# inconsistency between two files rather than a missing idea. A printed digest binds nothing;
# only a comparison does.
#
# The KEY LIST is gated too, not only its generator. Gating dlmc_corpus.py does not gate
# dlmc_keys.txt: a stale list from an earlier parameterisation, or a hand-edited one, reads
# identically with every generator digest unchanged. The generator is deterministic -- evenly
# spaced indices over an nnz-sorted pool, no seed -- so the list is reproducible from it, which
# is exactly why its digest must be compared rather than inferred. One corpus-identity failure
# on the other host already served one key another key's matrix.
#
# THESE VALUES ARE THE CURRENT WORKTREE'S, NOT WHAT JOB 17693884 RAN. That job staged
# arm_screen.sh at 266ca5fc and dlmc_corpus.py at 61f54f46, both since superseded. A future run
# must restage /scr/bobbyy/pfs from the worktree or this gate will refuse -- which is the gate
# working, not a fault. Update a value here only together with the file it names.
# SHELL VERSION GUARD, AND IT IS FAIL-CLOSED BY CONSTRUCTION RATHER THAN DOWNSTREAM.
# `declare -A` needs bash 4+. Under bash 3.2 -- which is what macOS ships, and this harness
# family does get run on the Mac -- the construct does NOT merely produce an empty array. The
# subscript ["mrgate.py"] is parsed as an INDEXED subscript and `mrgate.py` is evaluated
# arithmetically, which prints "syntax error: invalid arithmetic operator" and does not stop the
# script even under `set -u`. Every ${WANT[$f]} lookup then errors, the erroring expansion means
# the `[` test never receives two operands, and the enclosing `if` takes NEITHER branch -- so
# neither the REFUSE nor the OK line runs and control falls past the loop.
#
# Measured on bash 3.2 against deliberately stale files with the count assertion removed: the
# script printed the gate header, one syntax error, then "GATE PASSED", and exited 0. That is
# FAIL-OPEN, and worse than printing nothing, because the log then asserts the gate ran. An
# earlier comment here claimed the opposite -- that comparisons against an empty string would
# refuse -- which is what the behaviour would be if the array were merely empty. It is not.
#
# `bash -n` returns 0 on this whole construct under 3.2, so a syntax check cannot find it.
[ "${BASH_VERSINFO[0]:-0}" -ge 4 ] || {
  echo "REFUSE: this gate needs bash 4+ for 'declare -A'; this is ${BASH_VERSION:-unknown}."
  echo "        Under bash 3.2 the digest loop falls through and reports a pass on stale files."
  exit 1
}
declare -A WANT=(
  ["mrgate.py"]="a55e07bf03e1fb34644072da299f4e34645f3d4c1363f45aaa3b4fd382163fc8"
  ["csrresolve.py"]="1872f707045dc14e149fd53e845fee4b60e206b48132b144ee8217c227bfba78"
  ["arm_screen.sh"]="78f73444e60c62c33316e2bb172289f874f81c430f0e1d4c8c13064b98dd9d1e"
  ["dlmc_corpus.py"]="22afdbe4c3b5b3f3e7a96f1b10254389dbdad2c61ba855bf681b7587721406d5"
  ["dlmc_keys.txt"]="24e386cee4371ce9fddf02d0bb11378596f4345ec8923ef099b89cd129bbde6b"
)
echo "script $(basename "$0") spooled; producer digest gate:"
for f in mrgate.py csrresolve.py arm_screen.sh dlmc_corpus.py dlmc_keys.txt; do
  [ -f "/scr/bobbyy/pfs/$f" ] || { echo "REFUSE: /scr/bobbyy/pfs/$f absent at startup"; exit 1; }
  got=$(shasum -a 256 "/scr/bobbyy/pfs/$f" | cut -d" " -f1)
  if [ "$got" != "${WANT[$f]}" ]; then
    echo "REFUSE: $f sha256 $got, expected ${WANT[$f]}"
    echo "        A digest mismatch here means /scr holds a different producer than this"
    echo "        script was written against. Restage, or update WANT with the file."
    exit 1
  fi
  printf "  %-18s %s  OK\n" "$f" "${got:0:16}"
done
# Print the count that shows the loop was not empty. An unset or misspelled WANT key would make
# every iteration compare against "" and refuse, so a silent pass is not reachable -- but a
# shortened FILE LIST would skip a producer without any refusal, and the count catches that.
# The count assertion below stays, and the reason is the REVERSE of what was first written
# here: the failure it prevents is ACCEPTANCE, not a confusing refusal. With the version guard
# above it is a genuine second barrier rather than the only one.
echo "  gated ${#WANT[@]} producers against stored digests"
[ "${#WANT[@]}" -eq 5 ] || { echo "REFUSE: WANT holds ${#WANT[@]} entries, expected 5"; exit 1; }

export F64_HARNESS=/scr/bobbyy/pfs
export F64_ACTIVATE=/scr/bobbyy/venv/bin/activate
export F64_CACHE=/scr/bobbyy/mklsweep/csrcache/dlmc
export F64_KEYS=/scr/bobbyy/pfs/dlmc_keys.txt
export F64_OUT=/scr/bobbyy/pfs/csvdlmc
export F64_THREADS=32

bash /scr/bobbyy/pfs/arm_screen.sh /scr/bobbyy float32
rc=$?
echo "MKT_DLMC_SCREEN_EXIT $rc $(date +%H:%M:%S)"
exit $rc
