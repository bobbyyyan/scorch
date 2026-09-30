#!/usr/bin/env python3
"""Mechanical span set-difference: intended source vs emitted file.

Replaces a TYPED list of tokens to grep for. A typed list has an enumeration premise --
it can only catch deletions of tokens the author thought to list, so its "all present"
carries no information about a token neither party considered. This extracts the span set
from the INTENDED SOURCE and set-differences it against the OUTPUT, so the predicate is
"every span in the source appears in the output" with no list to be incomplete.

Complete for the class: a span present in the intended text and absent from the emitted
text. Does NOT cover reordering or duplication, which are separate axes (count/index/
content) and need their own checks.

PRESUPPOSES THE QUOTE-THEN-SUBSTITUTE WORKFLOW. The predicate is "every span in the
intended source appears in the output", so the intended source must be the POST-
substitution text. Point this at a heredoc TEMPLATE and every deliberate interpolation
(a `$DIG` meant to be replaced by a digest) reports as MISSING -- a false positive, and
a tool that cries wolf gets switched off. Quoting the heredoc and substituting values
afterwards is what produces a sound input. The tool composes with that rule; it does not
replace it.

Also outside the predicate: damage to a bare $VAR that sits in no span at all. That is
the same blind spot as a span expanding to nothing, which is why 1 of 13 missing spans
in the planted tree emitted no shell diagnostic either.

The DAMAGED set is not predictable from the span list. Backticks pair sequentially across
the whole heredoc, so a pair can straddle prose between two intended spans and which
spans die depends on the parity of everything before them. Historical damage therefore
hits a subset (6 spans at bb7e5b2) while a fresh replay of repaired text exposes every
span in the block (15). Both counts are correct for their own population; name which one
a number came from.

argv[1] = intended source, argv[2] = emitted file.
"""
import re, sys, collections

SPAN = re.compile(r"`([^`\n]+)`")           # inline code spans
EMPH = re.compile(r"\*\*([^*\n]+)\*\*")     # bold spans

def spans(text):
    out = collections.Counter()
    for pat, kind in ((SPAN, "code"), (EMPH, "bold")):
        for m in pat.finditer(text):
            out[(kind, m.group(1))] += 1
    return out

src = open(sys.argv[1]).read()
out = open(sys.argv[2]).read()
a, b = spans(src), spans(out)

missing = [(k, a[k], b.get(k, 0)) for k in a if b.get(k, 0) < a[k]]
extra   = [(k, a.get(k, 0), b[k]) for k in b if a.get(k, 0) < b[k]]

print("intended spans: %d distinct / %d total" % (len(a), sum(a.values())))
print("emitted  spans: %d distinct / %d total" % (len(b), sum(b.values())))
print("MISSING (in source, under-represented in output): %d" % len(missing))
for (kind, s), na, nb in sorted(missing):
    print("   %-5s %-52s source=%d output=%d" % (kind, repr(s)[:52], na, nb))
print("EXTRA (in output, not in source): %d" % len(extra))
for (kind, s), na, nb in sorted(extra):
    print("   %-5s %-52s source=%d output=%d" % (kind, repr(s)[:52], na, nb))
sys.exit(1 if (missing or extra) else 0)
