#!/usr/bin/env python3
"""Static scan: a string literal carrying a conversion specifier whose consumer is not a formatter.

WHY THIS EXISTS ALONGSIDE THE OUTPUT SCAN, AND WHY NEITHER SUBSUMES THE OTHER. A scan over a
harness's own stdout catches a format string that reached output unsubstituted -- but only on lines
that RAN. A defect inside a refusal message cannot appear in stdout until that refusal fires, and
refusal text is precisely the explanatory, written-last, never-exercised population the output scan
was built for. In one analyser 2 of 18 exit sites are unreachable by any input, so a bad format
string in either is permanently invisible to an output scan. The two populations are disjoint: the
output scan covers what ran, this covers what did not.

HOW IT DECIDES. Every `str` constant containing a conversion specifier is traced to its consumer by
walking parent links. The consumer must be one of:
  * a `%` BinOp with the string on the LEFT                (classic formatting)
  * an f-string                                             (already substituted)
  * a `.format(...)` call on the string
  * `re.compile` / `re.search` / `re.match` / `re.findall`  (a regex, where `%` is a literal)
Anything else is reported.

DOCSTRINGS ARE REPORTED, NOT EXCLUDED. They look inert and are not: `tools/attest.py` does
`sys.exit(__doc__)`, so a specifier in a docstring is reachable under that pattern. They are
reported in a separate class so the caller can judge reachability rather than having the judgement
made for them.
"""
import ast, re, sys, os

SPEC = re.compile(r"%[-+#0]*[0-9]*\.?[0-9]*[sdfgeiouxX]|%%")
RE_FUNCS = {"compile", "search", "match", "findall", "finditer", "sub", "split", "fullmatch"}


def scan(path):
    src = open(path).read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [("SYNTAX", 0, str(e))]
    parent = {}
    docstrings = set()
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            parent[ch] = node
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
               and isinstance(b[0].value.value, str):
                docstrings.add(id(b[0].value))
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if not SPEC.search(node.value):
            continue
        if id(node) in docstrings:
            out.append(("DOCSTRING", node.lineno, node.value.strip().split("\n")[0][:70]))
            continue
        p = parent.get(node)
        ok = False
        # Climb implicit concatenation and parens, then test the consumer.
        while isinstance(p, ast.Constant):
            p = parent.get(p)
        if isinstance(p, ast.BinOp) and isinstance(p.op, ast.Mod) and p.left is node:
            ok = True
        elif isinstance(p, ast.JoinedStr):
            ok = True
        elif isinstance(p, ast.Attribute) and p.attr == "format":
            ok = True
        elif isinstance(p, ast.Call):
            f = p.func
            if isinstance(f, ast.Attribute) and f.attr in RE_FUNCS:
                ok = True
            elif isinstance(f, ast.Name) and f.id in RE_FUNCS:
                ok = True
        if not ok:
            # A string that is the LEFT of a Mod higher up (after implicit concatenation) is fine.
            q, hops = node, 0
            while q in parent and hops < 6:
                q = parent[q]; hops += 1
                if isinstance(q, ast.BinOp) and isinstance(q.op, ast.Mod):
                    ok = True; break
                if isinstance(q, (ast.Call, ast.Expr, ast.stmt)):
                    break
        if not ok:
            out.append(("UNFORMATTED", node.lineno, node.value.strip().split("\n")[0][:70]))
    return out


if len(sys.argv) < 2:
    sys.exit("usage: an_fmtscan.py <file-or-dir> [...]")
files = []
for a in sys.argv[1:]:
    if os.path.isdir(a):
        for r, _d, fs in os.walk(a):
            files += [os.path.join(r, f) for f in fs if f.endswith(".py")]
    else:
        files.append(a)
files = sorted(set(files))
bad = 0
for f in files:
    for cls, ln, txt in scan(f):
        print("  %-12s %s:%d  %s" % (cls, os.path.relpath(f), ln, txt))
        if cls == "UNFORMATTED":
            bad += 1
print()
print("scanned %d file(s); %d UNFORMATTED specifier(s)" % (len(files), bad))
if bad:
    sys.exit(1)
