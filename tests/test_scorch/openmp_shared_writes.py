"""Unsynchronised writes to a shared scalar inside an OpenMP parallel region.

A generated kernel that increments a position counter declared outside a
``#pragma omp parallel for`` loop has every thread read and write the same
variable: a data race.  The SDDMM kernel with a coordinate mask did exactly that
before patch 0047 (``pSampled1++`` under a parallel loop over ``pMask0``).
:func:`shared_scalar_writes` lists every such write in an emitted C++ source, so
a test asserts that a kernel has none and pins the absence of the race rather
than only the bytes of one kernel.

The reading follows the code generator's layout: one statement per line, a
block's ``{`` on the line of the statement that opens it.  A parallel region is
the statement after ``#pragma omp parallel ...``: the associated ``for`` loop for
``parallel for`` (its header included), otherwise the block or statement that
follows.  A write is ``name++``, ``++name``, ``name--`` or ``--name`` anywhere in
a line, a statement starting ``name = ...`` or ``name op= ...``, or the
initialisation or increment of an inner ``for`` header, for a plain identifier
(``a[i]++``, ``s.x++`` and ``p->x++`` write elements and members, not scalars,
and are not read).  A write is shared when its name is not declared inside the
region, is not the iteration variable of a loop the ``parallel for`` is
associated with (``collapse(n)`` associates n loops), and is not named in a
``private``, ``firstprivate``, ``lastprivate``, ``linear`` or ``reduction``
clause of the pragma.  The statement or block after ``#pragma omp atomic`` and
the blocks of ``critical``, ``single``, ``master`` and ``masked`` are
synchronised and not read.
"""

from __future__ import annotations

import re
from typing import List, NamedTuple, Optional, Set, Tuple


class SharedWrite(NamedTuple):
    """One unsynchronised write: the source's 1-based line, the variable, the line."""

    line: int
    name: str
    text: str


_IDENTIFIER = r"[A-Za-z_]\w*"
_PARALLEL = re.compile(r"^\s*#pragma\s+omp\s+parallel\b(?P<clauses>.*)$")
_PRIVATIZING = re.compile(
    r"\b(?:private|firstprivate|lastprivate|linear|reduction)\s*\((?P<names>[^()]*)\)"
)
_COLLAPSE = re.compile(r"\bcollapse\s*\(\s*(?P<count>\d+)\s*\)")
_ATOMIC = re.compile(r"^\s*#pragma\s+omp\s+atomic\b")
_SYNCHRONISED = re.compile(r"^\s*#pragma\s+omp\s+(?:critical|single|master|masked)\b")
_STRING = re.compile(r'"(?:\\.|[^"\\])*"' + r"|'(?:\\.|[^'\\])*'")
_COMMENT = re.compile(r"//.*$|/\*.*?\*/")
_STEP = re.compile(
    rf"(?<![\w.>\]])(?P<post>{_IDENTIFIER})\s*(?:\+\+|--)"
    rf"|(?:\+\+|--)\s*(?P<pre>{_IDENTIFIER})\b(?!\s*(?:\[|\.|->))"
)
_ASSIGNMENT = re.compile(rf"^\s*(?P<name>{_IDENTIFIER})\s*(?:[-+*/%&|^]|<<|>>)?=(?!=)")
_FOR = re.compile(r"\bfor\s*\((?P<init>[^;]*);(?P<condition>[^;]*);(?P<step>[^)]*)\)")
_KEYWORDS = ("return", "else", "case", "goto", "delete", "throw", "sizeof")


def _declares(text: str, name: str) -> bool:
    """Whether ``text`` declares ``name``: a type (with any ``*``, ``&``, ``const``
    or ``__restrict__``) directly before it, at a statement or ``for`` header start."""

    keywords = "|".join(_KEYWORDS)
    declaration = re.compile(
        rf"(?:^|[;{{]|\bfor\s*\()\s*(?!(?:{keywords})\b)"
        rf"[A-Za-z_][\w:<>,\s*&]*?[\s*&]{re.escape(name)}\s*(?:=|;|\[|\{{|,|\))",
        re.MULTILINE,
    )
    return declaration.search(text) is not None


def _code(line: str) -> str:
    """The line without its string and character literals and its comments."""

    return _COMMENT.sub("", _STRING.sub('""', line))


def _block_end(lines: List[str], start: int) -> int:
    """The index of the last line of the statement starting at ``lines[start]``:
    the line closing its first ``{``, or the first line ending in ``;``."""

    depth = 0
    opened = False
    for index in range(start, len(lines)):
        code = _code(lines[index])
        for character in code:
            if character == "{":
                depth += 1
                opened = True
            elif character == "}":
                depth -= 1
        if opened and depth <= 0:
            return index
        if not opened and code.rstrip().endswith(";"):
            return index
    raise ValueError(f"the statement at line {start + 1} is not closed")


def _next_statement(lines: List[str], after: int) -> Optional[int]:
    for index in range(after + 1, len(lines)):
        if lines[index].strip():
            return index
    return None


def _writes(line: str) -> List[str]:
    code = _code(line)
    names = [
        match.group("post") or match.group("pre") for match in _STEP.finditer(code)
    ]
    assignment = _ASSIGNMENT.match(code)
    if assignment is not None:
        names.append(assignment.group("name"))
    for header in _FOR.finditer(code):
        initialisation = _ASSIGNMENT.match(header.group("init"))
        if initialisation is not None:
            names.append(initialisation.group("name"))
        names.extend(
            match.group(1)
            for match in re.finditer(
                rf"(?:^|,)\s*({_IDENTIFIER})\s*(?:[-+*/%&|^]|<<|>>)?=(?!=)",
                header.group("step"),
            )
        )
    return [name for name in names if name not in _KEYWORDS]


def _iteration_variables(
    lines: List[str], start: int, end: int, count: int
) -> Set[str]:
    names: Set[str] = set()
    for index in range(start, end + 1):
        if len(names) == count:
            break
        header = _FOR.search(lines[index])
        if header is None:
            continue
        variable = re.search(rf"({_IDENTIFIER})\s*=", header.group("init"))
        if variable is not None:
            names.add(variable.group(1))
    return names


def _region_writes(
    lines: List[str], pragma: int, clauses: str
) -> Tuple[int, List[SharedWrite]]:
    first = _next_statement(lines, pragma)
    if first is None:
        return pragma, []
    last = _block_end(lines, first)
    region = "\n".join(_code(line) for line in lines[first : last + 1])
    private: Set[str] = set()
    for clause in _PRIVATIZING.finditer(clauses):
        names = clause.group("names").split(":")[-1]
        private.update(name.strip() for name in names.split(",") if name.strip())
    if re.match(r"\s*for\b", clauses):
        collapse = _COLLAPSE.search(clauses)
        private |= _iteration_variables(
            lines, first, last, int(collapse.group("count")) if collapse else 1
        )
    writes: List[SharedWrite] = []
    index = first
    while index <= last:
        line = lines[index]
        if _ATOMIC.match(line) or _SYNCHRONISED.match(line):
            guarded = _next_statement(lines, index)
            index = (_block_end(lines, guarded) if guarded is not None else index) + 1
            continue
        for name in _writes(line):
            if name in private or _declares(region, name):
                continue
            writes.append(SharedWrite(index + 1, name, line.strip()))
        index += 1
    return last, writes


def shared_scalar_writes(cpp: str) -> List[SharedWrite]:
    """Every unsynchronised write to a scalar declared outside the OpenMP parallel
    region that contains it, in source order; an empty list for a race-free kernel."""

    lines = cpp.split("\n")
    writes: List[SharedWrite] = []
    index = 0
    while index < len(lines):
        pragma = _PARALLEL.match(lines[index])
        if pragma is None:
            index += 1
            continue
        last, found = _region_writes(lines, index, pragma.group("clauses"))
        writes.extend(found)
        index = last + 1
    return writes
