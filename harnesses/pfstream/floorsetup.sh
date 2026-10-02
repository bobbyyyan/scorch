#!/bin/bash
# Present each floor member as an importable `scorch_ops`, and pick the floor's cell set.
#
# WHY A DIRECTORY PER MEMBER. TORCH_EXTENSION_NAME is fixed at compile time, so the init symbol is
# PyInit_scorch_ops; a member loaded under any other module name fails exactly like a broken
# object. That mistake once reported a load failure for all twelve working members of an earlier
# family. Each member gets its own directory holding a copy named scorch_ops.so, and PYTHONPATH
# selects which one is imported.
#
# EQUAL-LENGTH PATHS, pNNN zero-padded. Path length cannot change these binaries -- they are
# already linked -- so this is not the reason the build script needs equal lengths. It removes one
# asymmetry between members for free.
set -u

ROOT=${1:-}
[ -n "$ROOT" ] || { echo "usage: floorsetup.sh <root> [per_band]"; exit 1; }
[ -d "$ROOT" ] || { echo "REFUSE: root $ROOT is not a directory"; exit 1; }
PER_BAND=${2:-12}

PFS=${FLOOR_PFS:-$ROOT/pfs}
RUN=${FLOOR_RUN:-$PFS/floorrun}
CENSUS=${FLOOR_CENSUS:-$ROOT/nnz_census.tsv}
KEYS=${FLOOR_KEYS:-$ROOT/floor_keys.txt}
FAMILY=${FLOOR_FAMILY:-$PFS/floor}

[ -d "$FAMILY" ] || { echo "REFUSE: FAMILY=$FAMILY absent"; exit 1; }
[ -f "$CENSUS" ] || { echo "REFUSE: CENSUS=$CENSUS is not a file"; exit 1; }

# Take the pads from the objects on disk rather than a hardcoded list, so a family built with a
# different pad set cannot be silently half-measured against a stale list of sixteen.
PADS=()
for so in "$FAMILY"/arm_pad*.so; do
  [ -f "$so" ] || continue
  b=$(basename "$so"); b=${b#arm_pad}; PADS+=("${b%.so}")
done
NP=${#PADS[@]}
[ "$NP" -gt 0 ] || { echo "REFUSE: no arm_pad*.so under $FAMILY"; exit 1; }
echo "family: $NP members, pads ${PADS[*]}"

rm -rf "$RUN"; mkdir -p "$RUN" || exit 1
for pad in "${PADS[@]}"; do
  d=$(printf "%s/p%03d" "$RUN" "$pad")
  mkdir -p "$d" || exit 1
  cp "$FAMILY/arm_pad${pad}.so" "$d/scorch_ops.so" || exit 1
done
nd=$(ls -d "$RUN"/p*/ 2>/dev/null | wc -l | tr -d ' ')
[ "$nd" -eq "$NP" ] || { echo "REFUSE: built $nd member dirs for $NP objects"; exit 1; }
# Path lengths must match across members, since that is the asymmetry the pNNN padding removes.
nlen=$(for d in "$RUN"/p*/; do printf "%s\n" "${#d}"; done | sort -u | wc -l | tr -d ' ')
[ "$nlen" -eq 1 ] || { echo "REFUSE: member paths have $nlen distinct lengths, not 1"; exit 1; }
echo "member dirs: $nd, each holding scorch_ops.so, all paths the same length"

# STRATIFIED BY NONZERO COUNT, because the A/A floor is an inverted U in matrix size -- p95 near
# 38% at 1e4-1e6 nonzeros against about 6% at both ends. A floor pooled over bands bounds neither
# end, so the band is a reported axis and each band gets the same number of matrices.
# Selection is EVENLY SPACED within the band after sorting by nnz, not random: reproducible
# without carrying a seed, and it spans the band instead of clustering in it.
python - "$CENSUS" "$KEYS" "$PER_BAND" <<'PY'
import sys
census, out, per = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows, skipped = [], 0
for ln in open(census):
    parts = ln.rstrip("\n").split("\t")
    if len(parts) != 3:
        skipped += 1
        continue
    rows.append((parts[0], int(parts[1]), int(parts[2])))
print("  census: %d rows read, %d skipped" % (len(rows), skipped))
if not rows:
    sys.exit("REFUSE: the census yielded no usable rows, so the key list would be empty")

def band(v):
    for lo, lab in ((1e7, ">=1e7"), (1e6, "1e6-1e7"), (1e5, "1e5-1e6"),
                    (1e4, "1e4-1e5"), (1e3, "1e3-1e4")):
        if v >= lo:
            return lab
    return "<1e3"

by = {}
for r in rows:
    by.setdefault(band(r[2]), []).append(r)
picked = []
for lab in sorted(by):
    pool = sorted(by[lab], key=lambda r: r[2])
    take = min(per, len(pool))
    idx = [round(i * (len(pool) - 1) / max(1, take - 1)) for i in range(take)] if take > 1 else [0]
    sel = [pool[i] for i in sorted(set(idx))]
    picked += sel
    print("  %-9s pool %4d  took %2d  nnz %d..%d" % (lab, len(pool), len(sel),
                                                     sel[0][2], sel[-1][2]))
# Print what the bands sum to beside the pick, so a pool that lost rows to a band label cannot
# read as a complete corpus.
print("  pools total %d of %d census rows" % (sum(len(v) for v in by.values()), len(rows)))
with open(out, "w") as fh:
    for name, _n, _nnz in picked:
        fh.write(name + "\n")
print("  keys written: %d -> %s" % (len(picked), out))
PY
