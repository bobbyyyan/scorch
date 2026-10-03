# Shared symbol-table and displacement functions for the layout-floor selection.
#
# WHY THIS FILE EXISTS. The self-test originally built its own symbol tables with `nm -C`, which
# duplicated rvlf_select.sh's logic -- and the two drifted on the very next change: the script
# moved to a chained demangler and the self-test did not, so its CONTROL case refused on clean
# input, reporting 2 unresolved symbols against the script's 0. A harness that restates the thing
# it is checking is wrong silently and flatteringly; here it was wrong loudly, which was luck.
# Both now source this, so a change to the metric cannot leave the self-test testing the old one.
#
# Sourcing this file must have no side effects: it defines functions and nothing else.

# THE DEMANGLER IS CHOSEN, NOT ASSUMED, BECAUSE XPAT MATCHES A STRING. In a mangled name `double`
# is encoded as `d`, so an unresolved float64 symbol stays in `rest`, moves legitimately for a
# _F64 arm, inflates rest_displ, and KEEPS an arm that measures nothing -- the direction the
# exclusion exists to prevent. Measured on this object, 1135 mangled symbols:
#
#   binutils c++filt   leaves 2 mangled: the SpMM OpenMP outlined bodies (it rejects the
#                      lambda-inside-a-template-argument form)
#   llvm-cxxfilt       leaves 6 mangled: pybind11 SFINAE and conversion-operator forms
#   either order, chained                leaves 0
#
# Different coverage, not one a superset of the other, so picking the better single tool would
# still have left a hole -- in the direction XPAT cannot see.
# A literal tab, because `sort -t'\t'` passes TWO characters and sort rejects it with
# "multi-character tab" -- which emptied every table and made four plants pass for the wrong
# reason while the control failed. Caught by the self-test, twice in a row now.
RVLF_TAB=$(printf '\t')

rvlf_pick_demangler() {
  if command -v llvm-cxxfilt >/dev/null 2>&1 && command -v c++filt >/dev/null 2>&1; then
    demangle() { llvm-cxxfilt | c++filt; }
    RVLF_DEMANGLER="llvm-cxxfilt | c++filt (chained: neither alone covers this object)"
  elif command -v c++filt >/dev/null 2>&1; then
    demangle() { c++filt; }
    RVLF_DEMANGLER="c++filt only -- llvm-cxxfilt absent; expect 2 unresolved and a refusal"
  elif command -v llvm-cxxfilt >/dev/null 2>&1; then
    demangle() { llvm-cxxfilt; }
    RVLF_DEMANGLER="llvm-cxxfilt only -- c++filt absent; expect 6 unresolved and a refusal"
  else
    echo "REFUSE: no demangler on PATH, so the exclusion cannot classify anything"; return 1
  fi
  return 0
}

RVLF_KPAT='scorch_spmm_row_(narrow_exact|narrow_rowvec|regblock|gather)<(float|scalar_t)|spmm_csr_v2_core<float'
RVLF_XPAT='double|_f64|F64|<double'

# ONE ROW PER NAME, CARRYING ITS SORTED ADDRESS LIST. A name<TAB>addr table is NOT join-safe:
# `nm` emits 1190 lines for 1093 distinct names here -- 83 names repeat, from weak and local
# aliases, COMDAT folding and pybind holder templates at up to three addresses each -- and `join`
# on a key present k and m times emits k*m rows. That is how the metric read 56 on a pair whose
# `.text` is byte-identical. Collapsing makes `joined <= min(table sizes)` an invariant, and makes
# "the address SET differs" the definition of displaced, which is right when a name legitimately
# has several addresses.
rvlf_symtab() {  # $1 = .so, $2 = output prefix -> .all .rest .kern .unres
  nm --defined-only "$1" 2>/dev/null \
    | sed 's/^\([0-9a-f]*\) \([A-Za-z]\) /\1\t\2\t/' \
    | awk -F'\t' 'NF>=3 { print $1 "\t" $3 }' \
    | demangle \
    | awk -F'\t' 'NF>=2 { print $2 "\t" $1 }' \
    | LC_ALL=C sort -u \
    | awk -F'\t' '{ if ($1==p) a=a","$2; else { if(p!="") print p"\t"a; p=$1; a=$2 } }
                   END { if(p!="") print p"\t"a }' \
    | LC_ALL=C sort -t"$RVLF_TAB" -k1,1 > "$2.all"
  grep -Ev "$RVLF_XPAT" "$2.all" > "$2.rest" || true
  grep -E  "$RVLF_KPAT" "$2.all" > "$2.kern" || true
  grep '^_Z' "$2.all" > "$2.unres" || true
}

# The pre-collapse form, kept ONLY so the self-test can plant it. This is the table rvlf_select.sh
# shipped with at ea78f6b, and the one that read 56 where the truth is 0.
rvlf_symtab_uncollapsed() {  # $1 = .so, $2 = output file
  nm --defined-only "$1" 2>/dev/null \
    | sed 's/^\([0-9a-f]*\) \([A-Za-z]\) /\1\t\2\t/' \
    | awk -F'\t' 'NF>=3 { print $1 "\t" $3 }' \
    | demangle \
    | awk -F'\t' 'NF>=2 { print $2 "\t" $1 }' \
    | LC_ALL=C sort -t"$RVLF_TAB" -k1,1 > "$2"
}

rvlf_disp() {  # $1 base table  $2 tag table -> echoes "joined displaced"
  echo "$(LC_ALL=C join -t"$RVLF_TAB" -j 1 "$1" "$2" | wc -l) \
$(LC_ALL=C join -t"$RVLF_TAB" -j 1 -o 1.2,2.2 "$1" "$2" | awk -F'\t' '$1 != $2' | wc -l)"
}

# The refusals, in one place so the self-test plants the same code the screen runs.
rvlf_unres_check() {  # $1 = label, $2 = prefix
  local n; n=$(wc -l < "$2.unres")
  [ "${n:-0}" -eq 0 ] && { echo "OK no unresolved"; return 0; }
  echo "REFUSE: $1 has $n symbol(s) the demangler could not resolve -- XPAT matches the string"
  echo "        'double', so a mangled float64 name stays in 'rest' and keeps an arm that"
  echo "        measures nothing:"
  cut -c1-110 "$2.unres" | sed 's/^/        /'
  return 1
}

rvlf_join_check() {  # $1 base table  $2 tag table -> prints REFUSE... or OK joined=.. displaced=..
  local nb nt mr j d
  nb=$(wc -l < "$1"); nt=$(wc -l < "$2")
  if [ "$nt" -eq 0 ]; then echo "REFUSE: tag table empty -- tool failure, not a null"; return 1; fi
  read -r j d <<<"$(rvlf_disp "$1" "$2")"
  mr=$(( nb < nt ? nb : nt ))
  if [ "$j" -gt "$mr" ]; then
    echo "REFUSE: join returned $j rows against a smaller table of $mr -- a cross product"; return 1; fi
  if [ "$j" -eq 0 ]; then
    echo "REFUSE: join matched 0 names -- not a null displacement"; return 1; fi
  echo "OK joined=$j displaced=$d"
  return 0
}
