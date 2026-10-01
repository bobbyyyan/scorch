#!/bin/bash
# Build a FAMILY of objects for one arm that differ only in where the kernel lands.
#
# Usage: pfs_floor_link.sh <tree-root> <out-dir> <pad-bytes...>
#
# WHY A FAMILY AND NOT A PAIR. A floor for a compile-time macro arm has to differ from its
# baseline in the same currency as the arm. A byte-identical rebuild bounds build-path variation
# only. A semantically inert SOURCE change does not work either: on this 56 KB translation unit
# at -O2, inserting one never-called external-linkage function into ops.cpp moved the hot OpenMP
# lambda bodies by 19 and 81 instructions and reallocated registers, so the pair differs in the
# kernel's code rather than its placement.
#
# So the perturbation goes past the compiler. ops.o is compiled ONCE and every object in the
# family links that same file, which makes "the kernel's instructions are identical" true by
# construction rather than by inspection. A pad occupying .text shifts everything placed after it.
#
# THREE THINGS THAT LOOK LIKE THEY WORK AND DO NOT:
#   * A pad declared as a C array lands in .data and shifts .text by nothing. Measured: pads from
#     0 to 4008 bytes left every kernel address byte-identical. Use an assembled .space in .text.
#   * Passing the library flags through an unquoted shell variable. The interactive shell on at
#     least one of these hosts is zsh, which does NOT word-split unquoted parameter expansions, so
#     the whole flag string arrives as one argument. g++ rejects it outright for compiler flags --
#     but for `-L... -lc10 ...` it accepts a bogus search path and, because a -shared link permits
#     undefined symbols, SUCCEEDS while linking no libraries at all. The result has rc=0, a
#     plausible size, correct disassembly, and zero DT_NEEDED torch entries, so it cannot be
#     loaded and cannot be timed. This script exists partly so those flags live in an array.
#   * Omitting -Wl,-Bsymbolic-functions, which production's link passes. Without it the kernel's
#     own symbols acquire @plt trampolines that match any symbol pattern and do not move with a
#     .text pad, which diluted a 12-of-12 placement result into 12 of 18.
#
# Every object is therefore checked for loadability, not just for rc=0.
set -u
[ $# -ge 3 ] || { echo "usage: $0 <tree-root> <out-dir> <pad-bytes...>"; exit 2; }
ROOT=$1; OUT=$2; shift 2
PADS=("$@")

OBJ=$(ls "$ROOT"/build/temp.*/src/scorch/csrc/ops.o 2>/dev/null | head -1)
[ -n "$OBJ" ] || { echo "REFUSE: no compiled ops.o under $ROOT/build/temp.*; build the arm first"; exit 1; }
TORCHLIB=$(python -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))') || exit 1
mkdir -p "$OUT"

echo "arm object: $OBJ"
echo "  sha256  $(sha256sum "$OBJ" | cut -d' ' -f1)"
echo "  THE SAME file is linked into every member below, so the kernel's instructions cannot differ"
echo "torch lib: $TORCHLIB"
echo

# Arrays, so no word-splitting assumption is made about any shell.
CFLAGS=(-pthread -shared -fPIC -O2 -fopenmp -Wl,-O1 -Wl,-Bsymbolic-functions)
LIBS=(-L"$TORCHLIB" -lc10 -ltorch -ltorch_cpu -ltorch_python)

fail=0
printf "%-8s %-10s %-20s %-7s %-8s %s\n" "pad" "link" "hot lambda addr" "mod64" "NEEDED" "loads?"
for n in "${PADS[@]}"; do
  s=$OUT/pad_$n.s; o=$OUT/pad_$n.o; so=$OUT/arm_pad$n.so
  {
    echo "        .section .text"
    echo "        .globl scorch_floor_pad_$n"
    echo "scorch_floor_pad_$n:"
    [ "$n" -gt 0 ] && echo "        .space $n, 0x90"
    echo "        ret"
  } > "$s"
  gcc -c -fPIC "$s" -o "$o" 2>/dev/null || { echo "REFUSE: pad $n would not assemble"; fail=1; continue; }
  if ! g++ "${CFLAGS[@]}" "$o" "$OBJ" "${LIBS[@]}" -o "$so" 2>"$OUT/link_$n.err"; then
    echo "REFUSE: pad $n link failed:"; sed 's/^/      /' "$OUT/link_$n.err" | head -4; fail=1; continue
  fi
  need=$(readelf -d "$so" | grep -c "NEEDED.*torch")
  addr=$(nm "$so" | grep " [TtWw] _ZZ16spmm_csr_v2_coreIfLb0E" | head -1 | cut -d' ' -f1)
  # Loadability is the check the broken-flags failure mode passes everything else on.
  # The module name must be scorch_ops. TORCH_EXTENSION_NAME was fixed at COMPILE time, so the
  # init symbol is PyInit_scorch_ops; loading the same file under any other name fails with
  # "dynamic module does not define module export function" and reads exactly like a broken
  # object. Each check runs in its own process, so the repeated name is not a conflict.
  if python -c "
import sys, torch, importlib.util
spec = importlib.util.spec_from_file_location('scorch_ops', '$so')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.exit(0 if hasattr(m, 'spmm_csr_float') else 3)
" 2>"$OUT/load_$n.err"; then loads=yes; else loads="NO($(head -1 "$OUT/load_$n.err" | cut -c1-40))"; fi
  [ "$need" -ge 3 ] && [ "$loads" = yes ] || fail=1
  printf "%-8s %-10s 0x%-18s %-7s %-8s %s\n" "$n" "rc=0" "${addr:-none}" \
         "$([ -n "$addr" ] && echo $(( 0x$addr % 64 )) || echo '?')" "$need" "$loads"
done
echo
if [ "$fail" != 0 ]; then
  echo "REFUSE: at least one member is unusable. An object with rc=0, a plausible size and"
  echo "        correct disassembly can still have zero DT_NEEDED torch entries and be"
  echo "        unloadable; that is the failure this loop exists to catch."
  exit 1
fi
echo "all ${#PADS[@]} members link, carry torch, and load."
