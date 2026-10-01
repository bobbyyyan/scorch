#!/usr/bin/env python3
"""Static scan: a string literal carrying a conversion specifier whose consumer is not a formatter.

WHY THIS EXISTS ALONGSIDE THE OUTPUT SCAN, AND WHY NEITHER SUBSUMES THE OTHER. A scan over a
harness's own stdout catches a format string that reached output unsubstituted -- but only on lines
that RAN. A defect inside a refusal message cannot appear in stdout until that refusal fires, and
refusal text is precisely the explanatory, written-last, never-exercised population the output scan
was built for. In one analyser 2 of 18 exit sites are unreachable by any input, so a bad format
string in either is permanently invisible to an output scan. The two populations are disjoint: the
output scan covers what ran, this covers what did not.

HOW IT DECIDES, AND THE ONE PRINCIPLE BEHIND IT. Every `str` constant containing a conversion
specifier is traced to its consumer by walking parent links. A specifier is reported ONLY when it
reaches a text SINK -- `print`, `sys.exit`, `.write`, `raise`, `assert`, or the logging/warnings
families -- with no formatter between. See the comment on SINKS for why the rule is inverted from
the obvious one.

THE TEMPLATE POSITION IS THE WHOLE RULE. A formatter only substitutes the literal that is its
TEMPLATE; a literal sitting in an ARGUMENT position is data, and its specifiers reach output
untouched. Three constructs, one test, all three measured rather than reasoned about:

    "<%s>" % "inner %d"        -> "<inner %d>"   left operand is the template, right is data
    "x %d".upper()             -> "X %D"         receiver is the string, arguments are search terms
    log.warning("%s", "a %d")  -> "a %d"         the message index is the template, rest are values

So the walk passes THROUGH a `%` whose right operand holds the literal, and through a string method
whose receiver holds the literal, and a lazy-logging sink is judged on the literal's argument index.
Getting this wrong in the obvious direction (treat any `%` as a formatter) produces silent misses;
getting it wrong in the other direction produces false positives on `.replace("%%", "")`, where the
literal is a search target. The position test is what separates them, and it is one test, not three
special cases.

ARGPARSE IS NOT MERELY A FALSE POSITIVE -- FLAGGING IT INVITES A CRASH. argparse applies
`self._get_help_string(action) % params` to help text, which is why `%(default)s` works there, so
`%%` in a help string is correct AND REQUIRED. Measured, not assumed:

    help="... = 3%%"     -> format_help() renders "= 3%"
    help="... = 3% bare" -> ValueError: unsupported format character 'b' at index 29
    help="... = 3%s"     -> renders "= 3{'prog': 'p', 'default': 3}"   <- no exception, a dict repr

So reporting the first form tells an author to introduce a `--help` that raises -- and nothing in a
harness check ever runs `--help`, so the recommended edit is both wrong and untested. argparse is
therefore out of scope BY CONSUMER: it is not a text sink, so it never reaches the rule. The third
line is the cost of that decision, stated plainly: the one argparse form that does not announce
itself at runtime is also the one this scan structurally cannot see. No defect class is counted for
it, because a counter that names a class the code cannot emit advertises coverage that is absent.

DOCSTRINGS ARE REPORTED, NOT EXCLUDED. They look inert and are not: `tools/attest.py` does
`sys.exit(__doc__)`, so a specifier in a docstring is reachable under that pattern. They are
reported in a separate class so the caller can judge reachability rather than having the judgement
made for them.

A FILE THAT DID NOT PARSE IS COUNTED AS A DEFECT. The scan has zero information about such a file,
and "0 defects over N files" is a claim about N files only if all N were read. Letting an
unparseable file exit 0 makes a failure to measure indistinguishable from a clean result, which is
the one reading this scan must never support.
"""
import ast, re, sys, os

# The conversion types Python's % operator accepts, all of them: d i o u x X e E f F g G c r s a %.
# `b` is deliberately absent -- "%b" % 1 raises, so it is not a specifier and prose containing "%b"
# is not a defect. The mapping form "%(name)s" is included; it reaches output literally exactly as
# "%d" does. The space flag is excluded because prose contains "% i" and "% o" as ordinary English.
_T = "[diouxXeEfFgGcrsa]"
_F = r"[-+#0]*(?:\*|[0-9]*)(?:\.(?:\*|[0-9]*))?"
SPEC = re.compile(r"%\([^)]*\)" + _F + _T + "|%" + _F + _T + "|%%")
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
# print, sys.exit, file.write, raise, assert, and the logging/warnings families. Everything else --
# regex, argparse, strftime, str.replace, dict keys, comparisons -- is out of scope by construction
# rather than by an exclusion someone has to remember to add.
#
# raise and assert are in the set because the docstring's own motivation -- refusal text, written
# last and never exercised -- describes them as exactly as it describes sys.exit. They carry zero
# instances in this tree today, and the reason is a house convention (CLAUDE.md's fail-closed rule
# prefers a structured sys.exit) rather than anything about the language, so a sink set that omits
# them is one commit away from being wrong.
#
# The cost, stated rather than hidden: a specifier that reaches output through a VARIABLE
# (`msg = "x %d"; print(msg)`) is invisible here. That is the half the runtime output scan covers,
# which is why the two scans are kept as a pair.
SINKS = {"print", "exit"}                      # bare calls: print(...), exit(...)
SINK_ATTRS = {"write", "exit"}                 # sys.exit, fh.write
# THE LOGGING FAMILY FORMATS LAZILY, so a specifier there is correct IF the call supplies args and a
# defect if it does not -- and three separate things about "which args" are easy to get wrong. All
# measured through a StreamHandler, because guessing any of them would be wrong:
#   log.warning("with args %d", 7)        -> "with args 7"      correct
#   log.warning("no args %d")             -> "no args %d"       defect
#   log.warning("no args %d", exc_info=1) -> "no args %d"       defect -- keywords never format
#   log.log(WARNING, "no args %d")        -> "no args %d"       defect -- the LEVEL is args[0]
#   warnings.warn("%d", UserWarning)      -> "%d"               defect -- args[1] is the CATEGORY
# So: the message index is 1 for `log` and 0 for the rest; a keyword argument is never a format
# argument (logging's are exc_info/stack_info/stacklevel/extra), so testing for their absence can
# only turn a defect into a pass; and `warnings.warn` is not lazy at all despite sharing the method
# name `warn` with the logging family, so it is separated by its receiver, not by its arity.
# print() is in neither set: `print("a %d", 3)` emits "a %d 3", a defect regardless of arity, and
# every argument of a print is a template position because print emits all of them.
LAZY_ATTRS = {"warn", "warning", "info", "error", "debug", "critical", "exception", "fatal", "log"}
LEVEL_FIRST = {"log"}                          # log(level, msg, *args): the message is args[1]
EAGER_RECEIVERS = {"warnings"}                 # warnings.warn formats nothing, ever
# String methods the walk passes through when the literal is the RECEIVER. Not an allow-list of safe
# consumers -- the opposite: these are consumers that PRESERVE the specifier, so stopping at them
# would be a miss. `format` and `format_map` are absent because they are formatters.
STR_METHODS = {"upper", "lower", "title", "capitalize", "casefold", "strip", "lstrip", "rstrip",
               "ljust", "rjust", "center", "zfill", "replace", "expandtabs", "swapcase",
               "removeprefix", "removesuffix", "join"}


def _msg_index(attr):
    return 1 if attr in LEVEL_FIRST else 0


def _receiver_name(node):
    """Left-most dotted name of an attribute chain: warnings.warn -> 'warnings'."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_sink(call, literal_arg_index, parent):
    """True when this call emits the string without formatting it."""
    if isinstance(call, ast.Assert):
        return True                                    # reached only when the literal IS .msg
    f = call.func
    if isinstance(f, ast.Name):
        if f.id in SINKS:
            return True
        # raise SomeError("..."): the constructor call sits directly under the Raise.
        return isinstance(parent.get(call), ast.Raise)
    if isinstance(f, ast.Attribute):
        if f.attr in SINK_ATTRS:
            return True
        if f.attr in LAZY_ATTRS:
            if _receiver_name(f.value) in EAGER_RECEIVERS:
                return True                            # warnings.warn: never formats
            mi = _msg_index(f.attr)
            if literal_arg_index != mi:
                return False                           # a value, not the template
            return len(call.args) <= mi + 1            # no args after the message -> defect
        return isinstance(parent.get(call), ast.Raise)
    return False


def scan(path):
    src = open(path).read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [("SYNTAX", getattr(e, "lineno", 0) or 0, str(e))]
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
        # Walk up to the nearest enclosing sink, stopping if a formatter intervenes. The literal
        # must be in the TEMPLATE position of whatever it meets -- see the docstring.
        q, formatted, sink = node, False, None
        for _ in range(10):
            par = parent.get(q)
            if par is None:
                break
            if isinstance(par, ast.BinOp) and isinstance(par.op, ast.Mod):
                if par.left is q:
                    formatted = True
                    break
                q = par                      # right operand: data, its specifiers survive
                continue
            if isinstance(par, ast.JoinedStr):
                formatted = True; break
            if isinstance(par, ast.Attribute) and par.attr in ("format", "format_map"):
                formatted = True; break
            if isinstance(par, ast.Attribute) and par.attr in STR_METHODS and par.value is q:
                gp = parent.get(par)         # skip the literal's own method call, not a sink
                q = gp if isinstance(gp, ast.Call) else par
                continue
            if isinstance(par, ast.Assert):
                if par.msg is q:
                    sink = par
                break                        # the test position never reaches output
            if isinstance(par, ast.Call):
                sink = par; break
            q = par
        if formatted or sink is None:
            continue
        ai = sink.args.index(q) if isinstance(sink, ast.Call) and q in sink.args else -1
        if not _is_sink(sink, ai, parent):
            continue
        out.append(("UNFORMATTED", node.lineno, node.value.strip().split("\n")[0][:70]))
    return out


COUNTED = ("UNFORMATTED", "SYNTAX")            # every class scan() emits except DOCSTRING

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
# A scope that is a file LIST must not print as 166 paths -- it buries the count the reader came
# for. Directories echo as given; explicit files collapse to their count and common prefix.
_dirs = [a for a in sys.argv[1:] if os.path.isdir(a)]
_fs = [a for a in sys.argv[1:] if not os.path.isdir(a)]
SCOPE = ", ".join(_dirs)
if _fs:
    SCOPE += ("; " if _dirs else "") + "%d file(s) listed under %s" % (
        len(_fs), os.path.commonpath([os.path.abspath(f) for f in _fs]) if len(_fs) > 1 else _fs[0])

bad, seen = 0, {}
for f in files:
    for cls, ln, txt in scan(f):
        # FINDING lines are prefixed so a consumer's class census cannot pick up the explanatory
        # footnotes below. An earlier census did exactly that: grepping '^  [A-Z]+' returned
        # SCOPE and SYNTAX from this file's own prose, the same shape as the defects it was built
        # to find. The CLASSES line exists so no consumer has to parse the body at all.
        print("FINDING %-12s %s:%d  %s" % (cls, os.path.relpath(f), ln, txt))
        seen[cls] = seen.get(cls, 0) + 1
        if cls in COUNTED:
            bad += 1
print()
print("scanned %d file(s); scope: %s; %d defect(s) in classes %s"
      % (len(files), SCOPE, bad, "+".join(COUNTED)))
print("CLASSES " + (" ".join("%s=%d" % kv for kv in sorted(seen.items())) or "(none)"))
print("  note: SCOPE IS THE ARGV PATHS, not the repository. A zero here is a statement about those")
print("  note: paths only -- an earlier run of this scan covered harnesses/ and was reported as 'the")
print("  note: tree is clean', while the one live argparse instance sat in tools/.")
print("  note: SYNTAX counts as a defect: a file that did not parse was not cleared, and an exit code")
print("  note: that cannot tell those apart makes a failure to measure read as a pass.")
if bad:
    sys.exit(1)
