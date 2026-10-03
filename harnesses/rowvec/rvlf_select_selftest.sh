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

# The two table forms. `collapsed` is what ships; `uncollapsed` is the ea78f6b form.
uncollapsed() {
  nm -C --defined-only "$1" 2>/dev/null \
    | sed 's/^\([0-9a-f]*\) \([A-Za-z]\) /\1\t\2\t/' \
    | awk -F'\t' 'NF>=3 { print $3 "\t" $1 }' | LC_ALL=C sort -t'	' -k1,1
}
collapsed() {
  uncollapsed "$1" | LC_ALL=C sort -u \
    | awk -F'\t' '{ if ($1==p) a=a","$2; else { if(p!="") print p"\t"a; p=$1; a=$2 } }
                  END { if(p!="") print p"\t"a }' | LC_ALL=C sort -t'	' -k1,1
}
disp() {  # echoes "joined displaced"
  echo "$(LC_ALL=C join -t'	' -j 1 "$1" "$2" | wc -l) \
$(LC_ALL=C join -t'	' -j 1 -o 1.2,2.2 "$1" "$2" | awk -F'\t' '$1 != $2' | wc -l)"
}
# The guard under test, lifted verbatim in behaviour from rvlf_select.sh.
verdict() { # $1 base table  $2 tag table -> prints REFUSE... or OK <j> <d>
  local nb nt mr j d
  nb=$(wc -l < "$1"); nt=$(wc -l < "$2")
  read -r j d <<<"$(disp "$1" "$2")"
  mr=$(( nb < nt ? nb : nt ))
  if [ "$nt" -eq 0 ]; then echo "REFUSE: tag table empty -- tool failure, not a null"; return; fi
  if [ "$j" -gt "$mr" ]; then echo "REFUSE: join returned $j rows against a smaller table of $mr -- a cross product"; return; fi
  if [ "$j" -eq 0 ]; then echo "REFUSE: join matched 0 names -- not a null displacement"; return; fi
  echo "OK joined=$j displaced=$d"
}

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
collapsed "$SHIP" > "$W/c_ship"; collapsed "$SHPB" > "$W/c_shpb"
check "control: collapsed tables on the identical pair must PASS and read 0" MUST_PASS \
  "$(verdict "$W/c_ship" "$W/c_shpb")"
d=$(verdict "$W/c_ship" "$W/c_shpb" | sed 's/.*displaced=//')
if [ "$d" = "0" ]; then echo "  PASS  control reads displaced=0, as a byte-identical pair must"; pass=$((pass+1));
else echo "  FAIL  control read displaced=$d on a byte-identical pair"; faildemo=1; fi

# PLANT 1: the ea78f6b uncollapsed form. Duplicate names make join emit k*m rows.
uncollapsed "$SHIP" > "$W/u_ship"; uncollapsed "$SHPB" > "$W/u_shpb"
echo "  plant 1 input: uncollapsed tables, $(wc -l < "$W/u_ship") rows for $(wc -l < "$W/c_ship") distinct names"
check "plant 1: uncollapsed table (the shipped-at-ea78f6b form) must REFUSE" MUST_REFUSE \
  "$(verdict "$W/u_ship" "$W/u_shpb")"

# PLANT 2: an empty tag table, i.e. nm failed on one side.
: > "$W/empty"
check "plant 2: empty tag table must REFUSE, not read displaced=0" MUST_REFUSE \
  "$(verdict "$W/c_ship" "$W/empty")"

# PLANT 3: disjoint names, so the join matches nothing without either table being empty.
sed 's/^/ZZZ_/' "$W/c_shpb" | LC_ALL=C sort -t'	' -k1,1 > "$W/disjoint"
check "plant 3: non-empty tag table sharing no names must REFUSE" MUST_REFUSE \
  "$(verdict "$W/c_ship" "$W/disjoint")"

echo
echo "$pass checks passed"
[ "$faildemo" = "0" ] || { echo "SELF-TEST FAILED: a guard did not fire on its own planted input"; exit 1; }
echo "SELF-TEST OK: every refusal fired on the input it was written for, and the clean pair reads 0"
