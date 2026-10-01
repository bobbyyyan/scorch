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
A specifier is reported ONLY when it reaches a text SINK -- `print`, `sys.exit`, `.write`, or the
logging/warnings families -- with no formatter between. See the comment on SINKS for why the rule is
inverted from the obvious one.

ARGPARSE IS NOT MERELY A FALSE POSITIVE -- FLAGGING IT INVITES A CRASH. argparse applies
`self._get_help_string(action) % params` to help text, which is why `%(default)s` works there, so
`%%` in a help string is correct AND REQUIRED. Measured, not assumed:

    help="... = 3%%"     -> format_help() renders "= 3%"
    help="... = 3% bare" -> ValueError: unsupported format character 'b' at index 29

So reporting it tells an author to introduce a `--help` that raises -- and nothing in a harness
check ever runs `--help`, so the recommended edit is both wrong and untested. That is the one
failure mode a lint of this kind must not have, which is why this exclusion is by CONSUMER rather
than by a suppression comment an author has to remember to add.

DOCSTRINGS ARE REPORTED, NOT EXCLUDED. They look inert and are not: `tools/attest.py` does
`sys.exit(__doc__)`, so a specifier in a docstring is reachable under that pattern. They are
reported in a separate class so the caller can judge reachability rather than having the judgement
made for them.
"""
import ast, re, sys, os

SPEC = re.compile(r"%[-+#0]*[0-9]*\.?[0-9]*[sdfgeiouxX]|%%")
# THE RULE IS INVERTED FROM THE OBVIOUS ONE, and the first design is why.
#
# Allow-listing "safe consumers" is unbounded, and each miss is a false positive on correct code.
# Three rounds of it: re.* (a regex where % is literal), argparse help= (where %% is REQUIRED), then
# a widened run produced two more -- `strftime("%Y-%m-%dT%H:%M:%S%z")` and this file's own
# `.replace("%%", "")`, where the literal is a search target rather than a template. That list has
# no end, and a lint that reports correct code on its first wider run is a lint that gets switched
# off.
#
# So flag by SINK instead. The defect is "a format string reached OUTPUT unsubstituted", so the
# only consumers that can exhibit it are the ones that emit text. That set is small and closed:
# print, sys.exit, file.write, and the logging/warnings families. Everything else -- regex, argparse,
# strftime, str.replace, dict keys, comparisons -- is out of scope by construction rather than by
# an exclusion someone has to remember to add.
#
# The cost, stated rather than hidden: a specifier that reaches output through a VARIABLE
# (`msg = "x %d"; print(msg)`) is invisible here. That is the half the runtime output scan covers,
# which is why the two scans are kept as a pair.
SINKS = {"print", "exit"}                      # bare calls: print(...), exit(...)
SINK_ATTRS = {"write", "exit"}                 # sys.exit, fh.write
# THE LOGGING FAMILY FORMATS LAZILY, so a specifier there is correct IF the call supplies args and a
# defect if it does not. Measured, because guessing either way would be wrong half the time:
#   logging.warning("with args %d", 7)  -> emits "with args 7"
#   logging.warning("without args %d")  -> emits "without args %d"   <- literal, a defect
# So these sinks are flagged only when the message is the sole positional argument. print() is NOT
# in this set: `print("a %d", 3)` emits "a %d 3" and is a defect regardless.
LAZY_ATTRS = {"warn", "warning", "info", "error", "debug", "critical", "exception", "fatal", "log"}


def _is_sink(call):
    """True when this call emits the string without formatting it."""
    f = call.func
    if isinstance(f, ast.Name):
        return f.id in SINKS
    if isinstance(f, ast.Attribute):
        if f.attr in SINK_ATTRS:
            return True
        if f.attr in LAZY_ATTRS:
            # Lazy formatting: correct when args follow the message, a defect when none do.
            return len(call.args) <= 1 and not call.keywords
    return False


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
        # Walk up to the nearest enclosing Call, stopping if a formatter intervenes.
        q, formatted, sink = node, False, None
        for _ in range(8):
            par = parent.get(q)
            if par is None:
                break
            if isinstance(par, ast.BinOp) and isinstance(par.op, ast.Mod):
                formatted = True; break
            if isinstance(par, ast.JoinedStr):
                formatted = True; break
            if isinstance(par, ast.Attribute) and par.attr == "format":
                formatted = True; break
            if isinstance(par, ast.Call):
                sink = par; break
            q = par
        if formatted or sink is None or not _is_sink(sink):
            continue
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
        if cls in ("UNFORMATTED", "ARGPARSE-BAD"):
            bad += 1
print()
print("scanned %d file(s) under %s; %d defect(s)"
      % (len(files), ", ".join(sys.argv[1:]), bad))
print("  SCOPE IS THE ARGV PATHS, not the repository. A zero here is a statement about those")
print("  paths only -- an earlier run of this scan covered harnesses/ and was reported as 'the")
print("  tree is clean', while the one live argparse instance sat in tools/.")
if bad:
    sys.exit(1)
