#!/bin/bash
#SBATCH -A mkt
#SBATCH -p mkt
#SBATCH -J pfs-dlmc-f32
#SBATCH -t 12:00:00
#SBATCH -N 1
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
# inside the exact kernel's float range (k=1..7) and is therefore a NEGATIVE CONTROL where the
# macro cannot change executed code; 8/16/24/32 are full tails at NVEC 1/2/3/4; 12 and 20 are
# masked tails at NVEC 2 and 3. The historical float32 set 2,4,8,16,32 reached neither NVEC 3 nor
# any masked tail, and the driver refuses it by name.
#
# WHY A STRATIFIED CORPUS. DLMC is a factorial design (model x pruning method x sparsity x layer)
# with unequal layer counts per cell (53-97), so a flat sample would load the corpus onto whichever
# cell has the most layers. 392 keys, 8 from each of the 49 populated strata.
set -u

# DIGEST PIN. sbatch copies this script at submit time, so an edit after submission does not
# reach a queued job. Pinning the producer digests means a stale stage is caught at startup
# rather than discovered in the output, and the comparison is against what is on disk NOW.
echo "script $(basename "$0") spooled; producer digests at startup:"
for f in mrgate.py csrresolve.py arm_screen.sh dlmc_corpus.py; do
  if [ -f "/scr/bobbyy/pfs/$f" ]; then
    printf "  %-18s %s\n" "$f" "$(shasum -a 256 "/scr/bobbyy/pfs/$f" | cut -c1-16)"
  else
    echo "REFUSE: /scr/bobbyy/pfs/$f absent at startup"; exit 1
  fi
done

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
