"""Every exception a public entry can raise carries a structured code.

Scorch fails closed: an input it cannot handle is refused with an exception, and the
refusal names its cause with a code (snake_case, such as ``index_position_exceeds_int32``)
so that callers and the suites that count refusals can tell causes apart without
matching message text.  A refusal without a code is unclassifiable to everyone above it.

The check is static (``refusal_code_inventory``): every ``raise`` and ``assert`` in the
package's ``.py`` files is found with ``ast``, the call graph from the public entries is
built and pruned by rapid type analysis, and a site fails the test when it carries no
code and some call path from an in-scope entry reaches it with no handler that catches
its class.  Nothing is compiled, so the check is cheap enough for the default suite.

In scope: ``einsum``, ``matmul``, ``spmv``, ``from_torch`` / ``from_coo`` / ``from_csr``
/ ``from_components`` and the public methods of ``STensor``.  Native code (``scorch_ops``,
generated kernels, torch) is not covered.

The first test runs the analysis on a small synthetic package whose answers are known,
one case per rule, so a change to the analysis that loses a rule fails there rather than
silently passing the gate.
"""

import collections
import os

import pytest

import scorch
from tests.test_scorch.refusal_code_inventory import (
    REQUIRED_ENTRIES,
    inventory,
    scorch_inventory,
    uncoded_escaping,
    uncoded_unmodeled,
)

# A package with one site per rule of the analysis.  ``entry`` and ``entry2`` are its
# public functions, ``T`` its public class.
SYNTHETIC = {
    "__init__.py": "from .ops import entry, entry2\nfrom .t import T\nfrom_x = T.from_x\n"
    "__all__ = ['entry', 'entry2', 'T', 'from_x']\n",
    "exceptions.py": (
        "class E(Exception):\n    pass\n"
        "class Coded(E):\n    code = 'c1'\n"
        "class Unsupported(E, NotImplementedError):\n    code: str = ''\n"
        "class Defected(E):\n    def __init__(self, d):\n        self.defect = d\n"
        "def make_error(error_type, message):\n    return error_type(message)\n"
        "class Diag(E):\n    def __init__(self, message, *, diagnostics=()):\n"
        "        super().__init__(message)\n        self.diagnostics = diagnostics\n"
    ),
    "forms.py": (
        "from dataclasses import FrozenInstanceError\n"
        "from typing import Optional\n"
        "from .exceptions import E\n"
        "class Root(Exception):\n"
        "    code: Optional[str] = None\n"
        "    def __init__(self, *args, code=None):\n"
        "        super().__init__(*args)\n"
        "        if code is not None:\n"
        "            self.code = code\n"
        "class One(Root):\n"
        "    code = 'one_cause'\n"
        "class Annotated(Root):\n"
        "    code: Optional[str] = 'annotated_cause'\n"
        "class RootValueError(Root, ValueError):\n"
        "    pass\n"
        "class RootRuntimeError(Root, RuntimeError):\n"
        "    pass\n"
        "class Clob(Root):\n"
        "    code = 'clob_cause'\n"
        "    def __init__(self, message, code=None):\n"
        "        super().__init__(message)\n"
        "        self.code = code\n"
        "class NoStore(E):\n"
        "    pass\n"
        "class RootFrozen(Root, FrozenInstanceError):\n"
        "    pass\n"
        "class FrozenFirst(FrozenInstanceError, Root):\n"
        "    pass\n"
        "class Partial(Root):\n"
        "    def __init__(self, failure, records):\n"
        "        self.failure = failure\n"
        "        super().__init__(str(failure))\n"
        "VARIANTS = {c.__name__: c for c in (RootValueError, RootRuntimeError)}\n"
        "def defect_code(error):\n"
        "    return getattr(error, 'code', None)\n"
        "def literal(x):\n"
        "    raise Root('literal code', code='literal_code')\n"
        "def missing(x):\n"
        "    raise Root('no code')\n"
        "def variable(x, name):\n"
        "    raise Root('code from a variable', code=name)\n"
        "def single(x):\n"
        "    raise One('class code')\n"
        "def annotated(x):\n"
        "    raise Annotated('annotated class code')\n"
        "def variant(x):\n"
        "    raise RootValueError('variant of a built-in', code='variant_code')\n"
        "def builtin_code(x):\n"
        "    raise ValueError('a built-in given a code', code='builtin_code')\n"
        "def clobbered(x):\n"
        "    raise Clob('class code overwritten')\n"
        "def not_stored(x):\n"
        "    raise NoStore('a class that drops code=', code='dropped')\n"
        "def frozen(x):\n"
        "    raise RootFrozen('an external second base', code='frozen_write')\n"
        "def frozen_first(x):\n"
        "    raise FrozenFirst('an external first base', code='frozen_first')\n"
        "def chained(x):\n"
        "    try:\n"
        "        missing(x)\n"
        "    except Root as exc:\n"
        "        raise RootRuntimeError(\n"
        "            str(exc), code=defect_code(exc) or 'chained_fallback'\n"
        "        ) from exc\n"
        "def step(x):\n"
        "    raise Root('step', code='step_failed')\n"
        "def step_bad(x):\n"
        "    raise E('step without a code')\n"
        "def pipeline_good(x):\n"
        "    try:\n"
        "        step(x)\n"
        "    except Exception as failure:\n"
        "        raise Partial(failure, ()) from failure\n"
        "def pipeline_bad(x):\n"
        "    try:\n"
        "        step_bad(x)\n"
        "    except Exception as failure:\n"
        "        raise Partial(failure, ()) from failure\n"
        "def unwrap_good(x):\n"
        "    try:\n"
        "        pipeline_good(x)\n"
        "    except Partial as failure:\n"
        "        raise failure.failure from None\n"
        "def unwrap_bad(x):\n"
        "    try:\n"
        "        pipeline_bad(x)\n"
        "    except Partial as failure:\n"
        "        raise failure.failure from None\n"
        "def keep_bad(x):\n"
        "    kept = None\n"
        "    try:\n"
        "        step_bad(x)\n"
        "    except E as error:\n"
        "        kept = error\n"
        "    if kept is not None:\n"
        "        raise kept\n"
        "def jit_parent(report):\n"
        "    error_type = VARIANTS.get(report['type'], RootRuntimeError)\n"
        "    raise error_type(\n"
        "        report['message'], code=report.get('code') or 'jit_build_failed'\n"
        "    )\n"
        "def sibling_inner(x):\n"
        "    raise ValueError('escapes through a handler that re-raises')\n"
        "def sibling(x):\n"
        "    try:\n"
        "        sibling_inner(x)\n"
        "    except ValueError:\n"
        "        raise\n"
        "    except Exception:\n"
        "        pass\n"
        "def run(x, name, report):\n"
        "    literal(x)\n"
        "    missing(x)\n"
        "    variable(x, name)\n"
        "    single(x)\n"
        "    annotated(x)\n"
        "    variant(x)\n"
        "    builtin_code(x)\n"
        "    clobbered(x)\n"
        "    not_stored(x)\n"
        "    frozen(x)\n"
        "    frozen_first(x)\n"
        "    chained(x)\n"
        "    unwrap_good(x)\n"
        "    unwrap_bad(x)\n"
        "    keep_bad(x)\n"
        "    jit_parent(report)\n"
        "    sibling(x)\n"
    ),
    # Special methods Python calls without a call expression naming them.
    "implicit.py": (
        "from collections.abc import Mapping\n"
        "from typing import Generic, TypeVar\n"
        "K = TypeVar('K')\n"
        "V = TypeVar('V')\n"
        "class Table(dict):\n"
        "    def __missing__(self, key):\n"
        "        raise LookupError('no entry')\n"
        "TABLE = Table(a=1)\n"
        "class Table2(dict):\n"
        "    def __missing__(self, key):\n"
        "        raise LookupError('caught by the only typed caller')\n"
        "TABLE2 = Table2()\n"
        "class Frozen(Generic[K, V]):\n"
        "    def __init__(self, items):\n"
        "        self.items = items\n"
        "    def __getitem__(self, key):\n"
        "        raise KeyError(key)\n"
        "def make_frozen() -> 'Frozen[str, int]':\n"
        "    return Frozen(())\n"
        "class FM(Mapping):\n"
        "    def __getitem__(self, key):\n"
        "        raise KeyError(key)\n"
        "    def __iter__(self):\n"
        "        return iter(())\n"
        "    def __len__(self):\n"
        "        return 0\n"
        "class Scope:\n"
        "    def __enter__(self):\n"
        "        raise RuntimeError('enter')\n"
        "    def __exit__(self, *exc):\n"
        "        return False\n"
        "class Rows:\n"
        "    def __iter__(self):\n"
        "        raise ValueError('rows')\n"
        "class Hook:\n"
        "    def __call__(self, x):\n"
        "        raise TypeError('hook')\n"
        "HOOK = Hook()\n"
        "class Hook2:\n"
        "    def __call__(self, x):\n"
        "        raise TypeError('held hook')\n"
        "class Holder:\n"
        "    def __init__(self):\n"
        "        self.hook = Hook2()\n"
        "    def run(self, x):\n"
        "        return self.hook(x)\n"
        "class Vec:\n"
        "    def __add__(self, other):\n"
        "        raise ArithmeticError('add')\n"
        "class Rv:\n"
        "    def __radd__(self, other):\n"
        "        raise ArithmeticError('radd')\n"
        "class Flag:\n"
        "    def __len__(self):\n"
        "        raise ValueError('len')\n"
        "class Shown:\n"
        "    def __str__(self):\n"
        "        raise ValueError('str')\n"
        "class Bag:\n"
        "    def __contains__(self, item):\n"
        "        raise TypeError('contains')\n"
        "class Pickled:\n"
        "    def __reduce__(self):\n"
        "        raise RuntimeError('pickle hook')\n"
        "class Gone:\n"
        "    def __del__(self):\n"
        "        raise RuntimeError('never propagated')\n"
        "def lookup(m: 'Frozen[str, int]', k):\n"
        "    return m[k]\n"
        "def consume(rows):\n"
        "    for r in rows:\n"
        "        pass\n"
        "def guarded(k):\n"
        "    try:\n"
        "        return TABLE2[k]\n"
        "    except LookupError:\n"
        "        return None\n"
        "def run(x, k):\n"
        "    TABLE[k]\n"
        "    guarded(k)\n"
        "    lookup(make_frozen(), k)\n"
        "    fm = FM()\n"
        "    k in fm\n"
        "    with Scope():\n"
        "        pass\n"
        "    consume(Rows())\n"
        "    HOOK(x)\n"
        "    Holder().run(x)\n"
        "    Vec() + 1\n"
        "    1 + Rv()\n"
        "    if Flag():\n"
        "        pass\n"
        "    f'{Shown()}'\n"
        "    x in Bag()\n"
        "    from .lazy import LATE\n"
        "    len(LATE)\n"
    ),
    # A module-level instance counts as constructed when its module is imported: by a
    # function that runs, here implicit.run; never, here.
    "lazy.py": (
        "class Late:\n"
        "    def __len__(self):\n"
        "        raise ValueError('imported by a function')\n"
        "LATE = Late()\n"
    ),
    "unimported.py": (
        "class Lone:\n"
        "    def __getitem__(self, key):\n"
        "        raise KeyError('a module nothing imports')\n"
        "LONE = Lone()\n"
    ),
    "ops.py": (
        "from .exceptions import E, Coded, Unsupported\n"
        "from . import forms, helpers, implicit\n"
        "from .t import T\n"
        "def entry(x):\n"
        "    if x < 0:\n"
        "        raise ValueError('negative')\n"
        "    forms.run(x, x, x)\n"
        "    implicit.run(x, x)\n"
        "    helpers.lower(x)\n"
        "    try:\n"
        "        helpers.risky(x)\n"
        "    except NotImplementedError:\n"
        "        pass\n"
        "    try:\n"
        "        helpers.risky2(x)\n"
        "    except ValueError as e:\n"
        "        raise e\n"
        "    def inner():\n"
        "        raise LookupError('inner')\n"
        "    inner()\n"
        "    helpers.deep2()\n"
        "    thing = helpers.Thing()\n"
        "    helpers.walk_all(x)\n"
        "    helpers.factory_raise(x)\n"
        "    helpers.keep(x)\n"
        "    helpers.diag_none(x)\n"
        "    helpers.diag_some(x)\n"
        "    helpers.F.apply(x)\n"
        "    return _private(x)\n"
        "def _private(x):\n"
        "    from .helpers import deep\n"
        "    return deep(x)\n"
        "def unreachable():\n"
        "    raise RuntimeError('never')\n"
        "def shadowed():\n"
        "    raise ValueError('reached only through a local of the same name')\n"
        "HANDLERS = {'a': helpers.handle_a}\n"
        "def _refusal(msg):\n"
        "    return Coded(msg)\n"
        "def entry2(t: 'T', key, exc_type):\n"
        "    shadowed = lambda: None\n"
        "    shadowed()\n"
        "    t.check()\n"
        "    HANDLERS[key]()\n"
        "    helpers.Lowerer().lower(t)\n"
        "    b = helpers.Built.make()\n"
        "    b.go()\n"
        "    if key == 'z':\n"
        "        raise _refusal('made coded')\n"
        "    raise exc_type('unknown class')\n"
    ),
    "helpers.py": (
        "import torch\n"
        "from .exceptions import Coded, Unsupported, Defected, make_error, Diag\n"
        "def diag_none(x):\n"
        "    raise Diag('no diagnostics')\n"
        "def diag_some(x):\n"
        "    raise Diag('with diagnostics', diagnostics=(x,))\n"
        "def factory_raise(x):\n"
        "    raise make_error(Coded, 'built from an argument')\n"
        "def keep(x):\n"
        "    kept = None\n"
        "    try:\n"
        "        lower(x)\n"
        "    except Coded as error:\n"
        "        kept = error\n"
        "    if kept is not None:\n"
        "        raise kept\n"
        "class Walker:\n"
        "    def walk(self, x):\n"
        "        return self.leave(x)\n"
        "    def leave(self, x):\n"
        "        return x\n"
        "class UnusedWalker(Walker):\n"
        "    def leave(self, x):\n"
        "        raise ValueError('override never constructed')\n"
        "class TableWalker(Walker):\n"
        "    def leave(self, x):\n"
        "        raise ValueError('constructed through a table')\n"
        "WALKERS = (TableWalker,)\n"
        "def walk_all(x):\n"
        "    if isinstance(x, UnusedWalker):\n"
        "        return\n"
        "    for w in WALKERS:\n"
        "        w().walk(x)\n"
        "class F(torch.autograd.Function):\n"
        "    @staticmethod\n"
        "    def forward(ctx, x):\n"
        "        raise RuntimeError('forward refuses')\n"
        "    @staticmethod\n"
        "    def backward(ctx, g):\n"
        "        return g\n"
        "def lower(x):\n"
        "    if x == 1:\n"
        "        raise Coded('coded')\n"
        "    if x == 2:\n"
        "        raise Defected(object())\n"
        "def risky(x):\n"
        "    raise NotImplementedError('caught')\n"
        "def risky2(x):\n"
        "    raise ValueError('re-raised by name')\n"
        "def deep(x):\n"
        "    err = Unsupported(f'shape {x} not supported')\n"
        "    raise err\n"
        "def deep2():\n"
        "    err = Unsupported('coded on the instance')\n"
        "    err.code = 'set_here'\n"
        "    raise err\n"
        "def handle_a():\n"
        "    raise KeyError('table')\n"
        "class Ghost:\n"
        "    def run(self):\n"
        "        raise KeyError('never constructed')\n"
        "class Built:\n"
        "    @classmethod\n"
        "    def make(cls):\n"
        "        return cls()\n"
        "    def go(self):\n"
        "        raise KeyError('constructed by a classmethod')\n"
        "class Thing:\n"
        "    def run(self):\n"
        "        raise KeyError('by name')\n"
        "class Lowerer:\n"
        "    def lower(self, node):\n"
        "        return getattr(self, 'visit_' + node.kind)(node)\n"
        "    def visit_a(self, node):\n"
        "        raise NotImplementedError('visit_a unsupported')\n"
        "    def visit_b(self, node):\n"
        "        raise self._err()\n"
        "    def _err(self):\n"
        "        return ValueError('made by a method')\n"
        "    def other(self):\n"
        "        raise ValueError('not a visitor')\n"
    ),
    "t.py": (
        "from .exceptions import E\n"
        "class T:\n"
        "    def __add__(self, other):\n"
        "        if not isinstance(other, T):\n"
        "            raise TypeError('bad operand')\n"
        "        n = self.size\n"
        "        return self._combine(other)\n"
        "    def _combine(self, other):\n"
        "        obj = other.payload\n"
        "        obj.run()\n"
        "    @property\n"
        "    def size(self):\n"
        "        raise IndexError('size')\n"
        "    def check(self):\n"
        "        raise ValueError('check')\n"
        "    def _hidden(self):\n"
        "        raise ValueError('private, never called')\n"
        "    @staticmethod\n"
        "    def from_x(v):\n"
        "        assert v is not None, 'v required'\n"
        "        return T()\n"
    ),
}

# (file, function after the package name, class, coded, reach, escapes to an entry)
SYNTHETIC_EXPECTED = [
    ("ops.py", "ops.entry", "ValueError", False, "resolved", True),
    ("ops.py", "ops.entry.<locals>.inner", "LookupError", False, "resolved", True),
    ("ops.py", "ops.unreachable", "RuntimeError", False, "not reached", False),
    ("ops.py", "ops.shadowed", "ValueError", False, "not reached", False),
    ("ops.py", "ops.entry2", "Coded", True, "resolved", True),
    ("ops.py", "ops.entry2", "?exc_type", False, "resolved", True),
    ("helpers.py", "helpers.lower", "Coded", True, "resolved", True),
    ("helpers.py", "helpers.lower", "Defected", True, "resolved", True),
    ("helpers.py", "helpers.risky", "NotImplementedError", False, "resolved", False),
    ("helpers.py", "helpers.risky2", "ValueError", False, "resolved", True),
    ("helpers.py", "helpers.deep", "Unsupported", False, "resolved", True),
    ("helpers.py", "helpers.deep2", "Unsupported", True, "resolved", True),
    ("helpers.py", "helpers.handle_a", "KeyError", False, "resolved", True),
    ("helpers.py", "helpers.Thing.run", "KeyError", False, "method name only", True),
    ("helpers.py", "helpers.Ghost.run", "KeyError", False, "not reached", False),
    ("helpers.py", "helpers.Built.go", "KeyError", False, "method name only", True),
    (
        "helpers.py",
        "helpers.UnusedWalker.leave",
        "ValueError",
        False,
        "not reached",
        False,
    ),
    (
        "helpers.py",
        "helpers.TableWalker.leave",
        "ValueError",
        False,
        "method name only",
        True,
    ),
    ("helpers.py", "helpers.F.forward", "RuntimeError", False, "resolved", True),
    ("helpers.py", "helpers.factory_raise", "Coded", True, "resolved", True),
    ("helpers.py", "helpers.keep", "Coded", True, "resolved", True),
    ("helpers.py", "helpers.diag_none", "Diag", False, "resolved", True),
    ("helpers.py", "helpers.diag_some", "Diag", True, "resolved", True),
    (
        "helpers.py",
        "helpers.Lowerer.visit_a",
        "NotImplementedError",
        False,
        "resolved",
        True,
    ),
    ("helpers.py", "helpers.Lowerer.visit_b", "ValueError", False, "resolved", True),
    ("helpers.py", "helpers.Lowerer.other", "ValueError", False, "not reached", False),
    ("t.py", "t.T.__add__", "TypeError", False, "resolved", True),
    ("t.py", "t.T.size", "IndexError", False, "resolved", True),
    ("t.py", "t.T.check", "ValueError", False, "resolved", True),
    ("t.py", "t.T._hidden", "ValueError", False, "not reached", False),
    ("t.py", "t.T.from_x", "AssertionError", False, "resolved", True),
    # The coding convention, one case per form.
    ("forms.py", "forms.literal", "Root", True, "resolved", True),
    ("forms.py", "forms.missing", "Root", False, "resolved", True),
    ("forms.py", "forms.variable", "Root", False, "resolved", True),
    ("forms.py", "forms.single", "One", True, "resolved", True),
    ("forms.py", "forms.annotated", "Annotated", True, "resolved", True),
    ("forms.py", "forms.variant", "RootValueError", True, "resolved", True),
    ("forms.py", "forms.builtin_code", "ValueError", False, "resolved", True),
    ("forms.py", "forms.clobbered", "Clob", False, "resolved", True),
    ("forms.py", "forms.not_stored", "NoStore", False, "resolved", True),
    ("forms.py", "forms.frozen", "RootFrozen", True, "resolved", True),
    ("forms.py", "forms.frozen_first", "FrozenFirst", False, "resolved", True),
    ("forms.py", "forms.chained", "RootRuntimeError", True, "resolved", True),
    ("forms.py", "forms.step", "Root", True, "resolved", False),
    ("forms.py", "forms.step_bad", "E", False, "resolved", False),
    ("forms.py", "forms.pipeline_good", "Partial", False, "resolved", False),
    ("forms.py", "forms.pipeline_bad", "Partial", False, "resolved", False),
    ("forms.py", "forms.unwrap_good", "failure", True, "resolved", True),
    ("forms.py", "forms.unwrap_bad", "failure", False, "resolved", True),
    ("forms.py", "forms.keep_bad", "E", False, "resolved", True),
    ("forms.py", "forms.jit_parent", "RootValueError", True, "resolved", True),
    ("forms.py", "forms.sibling_inner", "ValueError", False, "resolved", True),
    # Implicit special-method calls, one case per protocol.  FM's and Rows' are
    # reached only by name, through receivers of unknown type (``report[...]`` in
    # forms.jit_parent, ``rows`` in consume): ``k in fm`` runs FM's ``__iter__``, not its
    # ``__getitem__``.
    (
        "implicit.py",
        "implicit.Table.__missing__",
        "LookupError",
        False,
        "resolved",
        True,
    ),
    (
        "implicit.py",
        "implicit.Table2.__missing__",
        "LookupError",
        False,
        "resolved",
        True,
    ),
    ("implicit.py", "implicit.Frozen.__getitem__", "KeyError", False, "resolved", True),
    (
        "implicit.py",
        "implicit.FM.__getitem__",
        "KeyError",
        False,
        "method name only",
        True,
    ),
    (
        "implicit.py",
        "implicit.Scope.__enter__",
        "RuntimeError",
        False,
        "resolved",
        True,
    ),
    (
        "implicit.py",
        "implicit.Rows.__iter__",
        "ValueError",
        False,
        "method name only",
        True,
    ),
    ("implicit.py", "implicit.Hook.__call__", "TypeError", False, "resolved", True),
    ("implicit.py", "implicit.Hook2.__call__", "TypeError", False, "resolved", True),
    ("implicit.py", "implicit.Vec.__add__", "ArithmeticError", False, "resolved", True),
    ("implicit.py", "implicit.Rv.__radd__", "ArithmeticError", False, "resolved", True),
    ("implicit.py", "implicit.Flag.__len__", "ValueError", False, "resolved", True),
    ("implicit.py", "implicit.Shown.__str__", "ValueError", False, "resolved", True),
    ("implicit.py", "implicit.Bag.__contains__", "TypeError", False, "resolved", True),
    (
        "implicit.py",
        "implicit.Pickled.__reduce__",
        "RuntimeError",
        False,
        "not reached",
        False,
    ),
    (
        "implicit.py",
        "implicit.Gone.__del__",
        "RuntimeError",
        False,
        "not reached",
        False,
    ),
    ("lazy.py", "lazy.Late.__len__", "ValueError", False, "resolved", True),
    (
        "unimported.py",
        "unimported.Lone.__getitem__",
        "KeyError",
        False,
        "not reached",
        False,
    ),
]

# How much of the uncoded list a failure prints.
MAX_GROUPS = 40
MAX_LINES_PER_GROUP = 8


def test_inventory_finds_the_known_sites_of_a_synthetic_package(tmp_path):
    """Each rule of the analysis, on a package whose sites, codes and reach are known:
    coded and uncoded classes, a code set on the instance, a class built by a function
    from its argument, a caught exception raised later, ``diagnostics`` passed and not
    passed, handlers that catch and that re-raise, nested functions, a local that
    shadows a module function, dispatch through ``self``, ``getattr``, a module-level
    table and a classmethod constructor, ``autograd.Function.apply``, and methods of
    classes that are never constructed.

    ``forms.py`` has one case per form of the coding convention: a literal ``code=``
    and none, a ``code=`` that is not a literal, a single-cause class code (plain and
    annotated, ``code: Optional[str] = "..."``), a Scorch
    variant of a built-in, a built-in given a code, a class code an ``__init__``
    overwrites, a class that does not store ``code=``, a chained re-raise with a
    literal fallback, a wrapper's caught exception re-raised (all coded, and not), a
    kept exception re-raised, a class looked up in a table as the JIT parent does,
    and a handler that re-raises ahead of a later one in the same ``try``.

    ``implicit.py`` has one case per implicit special-method call: a dict subclass's
    ``__missing__`` behind a module-level instance (and one whose typed lookup is
    caught), ``__getitem__`` on a parameter annotated with a generic class, ``in`` on a
    ``Mapping`` subclass, ``with``, iteration of an untyped value, ``__call__`` on a
    module-level instance and on an instance attribute, ``+`` and its reflection, truth
    through ``__len__``, an f-string through ``__str__``, ``in`` through
    ``__contains__``; a pickle hook and a ``__del__``, which no edge models; and the
    module-level instances of a module a function imports, which count as constructed,
    and of a module nothing imports, which do not."""

    pkg = tmp_path / "synthpkg"
    pkg.mkdir()
    for name, text in SYNTHETIC.items():
        (pkg / name).write_text(text)
    result = inventory(str(pkg), "synthpkg", "T")
    got = sorted(
        (
            s["file"].split("/")[-1],
            s["function"].split("synthpkg.", 1)[1],
            s["class"].rsplit(".", 1)[-1],
            bool(s["coded"]),
            s["reach"],
            bool(s["escapes_to"]),
        )
        for s in result.sites
    )
    want = sorted(SYNTHETIC_EXPECTED)
    failures = [f"unexpected {g}" for g in got if g not in want]
    failures += [f"missing {w}" for w in want if w not in got]

    def site(function, cls):
        for s in result.sites:
            if (s["function"].rsplit(".", 1)[-1], s["class"].rsplit(".", 1)[-1]) == (
                function,
                cls,
            ):
                return s
        return {}

    kinds = {
        ("entry", "ValueError"): "invalid input",
        ("deep", "Unsupported"): "unsupported",
        ("from_x", "AssertionError"): "internal",
        ("visit_a", "NotImplementedError"): "unsupported",
    }
    for (function, cls), kind in kinds.items():
        if site(function, cls).get("kind_provisional") != kind:
            failures.append(f"kind of {function} {cls}: want {kind}")
    roots = {label for _, label in result.roots}
    want_roots = {
        "synthpkg.entry",
        "synthpkg.entry2",
        "synthpkg.from_x",
        "T.__add__",
        "T.size",
        "T.check",
        "T.from_x",
    }
    if roots != want_roots:
        failures.append(f"entries {sorted(roots)}, want {sorted(want_roots)}")
    how = site("visit_b", "ValueError").get("how", "").split("; ")[0]
    if how != "returned by synthpkg.helpers.Lowerer._err":
        failures.append(f"visit_b's class found {how!r}")
    coded = site("deep2", "Unsupported").get("coded")
    if coded != "sets err.code":
        failures.append(f"deep2 coded {coded!r}")
    path = [q.rsplit(".", 1)[-1] for q in site("deep", "Unsupported").get("path") or []]
    if path != ["entry", "_private", "deep"]:
        failures.append(f"path to deep {path}")
    step_bad = (
        SYNTHETIC["forms.py"].splitlines().index("    raise E('step without a code')")
    )
    uncoded = site("unwrap_bad", "failure").get("reraise_uncoded")
    if uncoded != [f"synthpkg/forms.py:{step_bad + 1}"]:
        failures.append(f"unwrap_bad re-raises the uncoded sites {uncoded}")
    for function, cls, why in (
        ("variable", "Root", "code= is not a literal"),
        ("not_stored", "NoStore", "NoStore does not store code="),
        ("clobbered", "Clob", "Clob.__init__ code not passed"),
    ):
        if why not in site(function, cls).get("how", ""):
            failures.append(f"{function}: {why!r} not in how")
    by_function = {s["function"].split("synthpkg.", 1)[1]: s for s in result.sites}
    # The handler in guarded catches the typed subscript of TABLE2; only the untyped
    # ``report[...]`` reaches Table2.__missing__ uncaught.
    escape = by_function.get("implicit.Table2.__missing__", {}).get("escape_reach")
    if escape != "method name only":
        failures.append(f"Table2.__missing__ escapes through {escape!r}")
    path = [
        q.rsplit(".", 2)[-2] + "." + q.rsplit(".", 1)[-1]
        for q in by_function.get("implicit.Table.__missing__", {}).get("path") or []
    ]
    if path != ["ops.entry", "implicit.run", "Table.__missing__"]:
        failures.append(f"path to Table.__missing__ {path}")
    unmodeled = sorted(
        s["function"].split("synthpkg.", 1)[1] for s in uncoded_unmodeled(result.sites)
    )
    if unmodeled != ["implicit.Pickled.__reduce__"]:
        failures.append(f"uncoded sites in unmodeled special methods {unmodeled}")
    assert not failures, "\n".join(failures)


def describe(uncoded):
    """The uncoded sites grouped by (class, function), largest groups first."""

    groups = collections.defaultdict(list)
    for s in uncoded:
        groups[(s["class"], s["function"])].append(s)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    resolved = sum(1 for s in uncoded if s["escape_reach"] == "resolved")
    out = [
        f"{len(uncoded)} raise or assert sites reach an in-scope entry without a code "
        f"({resolved} through resolved calls, {len(uncoded) - resolved} only through "
        f"a call by method name), in {len(groups)} groups by (class, function).  A "
        'site is coded when it passes a literal `code=` (or `<expr> or "<literal>"`) '
        "to a class that stores it, its class has a class-level `code`, it passes "
        "non-empty `diagnostics=`, or it re-raises only coded sites; see "
        "refusal_code_inventory's docstring for every rule."
    ]
    for (cls, function), ss in ordered[:MAX_GROUPS]:
        first = ss[0]
        lines = ", ".join(str(s["line"]) for s in ss[:MAX_LINES_PER_GROUP])
        if len(ss) > MAX_LINES_PER_GROUP:
            lines += f" and {len(ss) - MAX_LINES_PER_GROUP} more"
        entries = first["escapes_in_scope"]
        shown = ", ".join(entries[:3]) + (", ..." if len(entries) > 3 else "")
        path = " > ".join(q.replace("scorch.", "") for q in (first["path"] or [])[:8])
        out.append(
            f"  {cls.replace('builtins.', '')} in {function} "
            f"({len(ss)} sites, {first['kind_provisional']})"
        )
        out.append(f"      {first['file']}: {lines}")
        out.append(f"      raised with: {first['message'][:160]}")
        out.append(f"      escapes to {len(entries)} in-scope entries: {shown}")
        out.append(f"      a shortest call path from an entry: {path}")
        reraised = sorted({u for s in ss for u in s["reraise_uncoded"]})
        if reraised:
            listed = ", ".join(reraised[:MAX_LINES_PER_GROUP])
            if len(reraised) > MAX_LINES_PER_GROUP:
                listed += f" and {len(reraised) - MAX_LINES_PER_GROUP} more"
            out.append(f"      re-raises {len(reraised)} uncoded sites: {listed}")
    if len(ordered) > MAX_GROUPS:
        rest = sum(len(ss) for _, ss in ordered[MAX_GROUPS:])
        out.append(f"  and {len(ordered) - MAX_GROUPS} more groups ({rest} sites)")
    return "\n".join(out)


def test_every_refusal_reaching_an_entry_has_a_code():
    """No site that can raise to an in-scope entry lacks a code, and no site in a
    special method whose implicit calls the call graph does not model lacks one.

    The inventory must also parse every module and find the entries it covers, so
    that a change in how scorch exports them cannot make this test pass by reaching
    nothing."""

    result = scorch_inventory(os.path.dirname(scorch.__file__))
    assert not result.index.parse_errors, result.index.parse_errors
    found = {label for _, label in result.roots}
    absent = sorted(set(result.missing) | (REQUIRED_ENTRIES - found))
    assert not absent, f"entries the inventory did not find: {absent}"
    uncoded = uncoded_escaping(result.sites)
    if uncoded:
        pytest.fail(describe(uncoded), pytrace=False)
    unmodeled = uncoded_unmodeled(result.sites)
    if unmodeled:
        pytest.fail(
            "uncoded raise or assert sites in special methods Python can call without "
            "an edge the inventory models (see its docstring):\n"
            + "\n".join(
                f"  {s['file']}:{s['line']} {s['class']} in {s['function']}"
                for s in unmodeled
            ),
            pytrace=False,
        )
