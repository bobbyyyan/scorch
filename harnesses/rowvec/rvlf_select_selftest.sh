#!/bin/bash
# Plant each refusal in rvlf_select.sh's displacement metric and require it to fire.
#
# WHY THIS EXISTS. The invariant `joined <= min(table sizes)` is satisfied BY CONSTRUCTION once
# symtab collapses to one row per name, so it cannot fire on correct input: it is a self-check on
# the fold, not a check on the data. What was validated when it was written were the metric's
# VALUES (0 for the byte-identical pair, 974 and 6 for the others), which is a different claim.
# A guard that has never been seen to refuse is not known to refuse -- the same reason the
# concurrency guard was run against a live job rather than reasoned about.
#
# The planted inputs are not invented: the uncollapsed `name<TAB>addr` table is the form this
# script actually shipped with at ea78f6b, and it is the form that read 56 on a pair whose true
# displacement is 0.
#
# Each plant must make the metric REFUSE. A plant that passes is a finding.
set -u
R=${1:-/scr/bobbyy/rvec}
W=$(mktemp -d /tmp/rvlfst.XXXXXX); trap 'rm -rf "$W"' EXIT
SHIP=$(ls "$R"/ship/src/scorch_ops*.so 2>/dev/null | head -1)
SHPB=$(ls "$R"/shpb/src/scorch_ops*.so 2>/dev/null | head -1)
for v in "SHIP=$SHIP" "SHPB=$SHPB"; do
  [ -n "${v#*=}" ] || { echo "REFUSE: ${v%%=*} object not found under $R"; exit 1; }
done
echo "self-test against $R"
echo "  ship and shpb are the SAME configuration, so their .text is identical and their true"
echo "  displacement is exactly 0. Every plant below perturbs the INSTRUMENT, not the objects."
echo

. "$(dirname "$0")/rvlf_lib.sh"
rvlf_pick_demangler || exit 1
echo "  demangler: $RVLF_DEMANGLER"
echo

pass=0; faildemo=0
check() { # $1 label  $2 expectation: MUST_REFUSE|MUST_PASS  $3 output
  case "$2" in
    MUST_REFUSE) if [[ "$3" == REFUSE* ]]; then echo "  PASS  $1"; echo "        -> $3"; pass=$((pass+1));
                 else echo "  FAIL  $1 -- the guard did NOT fire"; echo "        -> $3"; faildemo=1; fi ;;
    MUST_PASS)   if [[ "$3" == OK* ]]; then echo "  PASS  $1"; echo "        -> $3"; pass=$((pass+1));
                 else echo "  FAIL  $1 -- refused on clean input"; echo "        -> $3"; faildemo=1; fi ;;
  esac
}

# CONTROL FIRST. Without it a refusal on every plant is indistinguishable from a metric that
# refuses unconditionally, which would pass every plant below while measuring nothing.
rvlf_symtab "$SHIP" "$W/ship"; rvlf_symtab "$SHPB" "$W/shpb"
cp "$W/ship.all" "$W/c_ship"; cp "$W/shpb.all" "$W/c_shpb"
check "control: collapsed tables on the identical pair must PASS and read 0" MUST_PASS \
  "$(rvlf_join_check "$W/c_ship" "$W/c_shpb")"
d=$(rvlf_join_check "$W/c_ship" "$W/c_shpb" | sed 's/.*displaced=//')
if [ "$d" = "0" ]; then echo "  PASS  control reads displaced=0, as a byte-identical pair must"; pass=$((pass+1));
else echo "  FAIL  control read displaced=$d on a byte-identical pair"; faildemo=1; fi

# PLANT 1: the ea78f6b uncollapsed form. Duplicate names make join emit k*m rows.
rvlf_symtab_uncollapsed "$SHIP" "$W/u_ship"; rvlf_symtab_uncollapsed "$SHPB" "$W/u_shpb"
echo "  plant 1 input: uncollapsed tables, $(wc -l < "$W/u_ship") rows for $(wc -l < "$W/c_ship") distinct names"
check "plant 1: uncollapsed table (the shipped-at-ea78f6b form) must REFUSE" MUST_REFUSE \
  "$(rvlf_join_check "$W/u_ship" "$W/u_shpb")"

# PLANT 2: an empty tag table, i.e. nm failed on one side.
: > "$W/empty"
check "plant 2: empty tag table must REFUSE, not read displaced=0" MUST_REFUSE \
  "$(rvlf_join_check "$W/c_ship" "$W/empty")"

# PLANT 3: disjoint names, so the join matches nothing without either table being empty.
sed 's/^/ZZZ_/' "$W/c_shpb" | LC_ALL=C sort -t"$RVLF_TAB" -k1,1 > "$W/disjoint"
check "plant 3: non-empty tag table sharing no names must REFUSE" MUST_REFUSE \
  "$(rvlf_join_check "$W/c_ship" "$W/disjoint")"

# PLANT 4: an unresolvable mangled symbol. This is the guard that protects the EXCLUSION rather
# than the join: XPAT matches the string "double", so a name the demangler could not resolve keeps
# its mangled form where double is encoded as `d`, stays in `rest`, and for a _F64 arm moves
# legitimately -- keeping an arm that measures nothing. It cannot fire on real input now that both
# demanglers are chained (0 of 1135 left mangled), which is exactly why it needs planting.
cp "$W/ship.all" "$W/planted.all"; cp "$W/ship.unres" "$W/planted.unres"
printf '_ZN2at8internal15invoke_parallelIZ16spmm_csr_v2_coreIdLb0EEE._omp_fn.9\t00000000deadbeef\n' \
  | tee -a "$W/planted.all" >> "$W/planted.unres"
check "control: the real table must have NO unresolved symbols" MUST_PASS \
  "$(rvlf_unres_check ship "$W/ship")"
check "plant 4: one unresolvable mangled line must REFUSE" MUST_REFUSE \
  "$(rvlf_unres_check planted "$W/planted")"

echo
echo "$pass checks passed"
[ "$faildemo" = "0" ] || { echo "SELF-TEST FAILED: a guard did not fire on its own planted input"; exit 1; }
echo "SELF-TEST OK: every refusal fired on the input it was written for, and the clean pair reads 0"
