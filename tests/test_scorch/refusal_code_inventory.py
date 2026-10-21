"""Static inventory of the exceptions scorch's public entries can raise without a code.

Used by ``test_every_refusal_has_a_code.py``.  Python's ``ast`` over the ``.py`` files of a
package directory: nothing in the package is imported and nothing is compiled.

Functions and classes.  Every module is parsed; every function, method, nested function
and nested class gets a qualified name (``scorch.ops.einsum``,
``scorch.stensor.STensor.__add__``, ``scorch.ops.f.<locals>.g``).  A name resolves in its
function's scope chain (nested definitions, the function's own imports -- the lazy
``from ...ops import x`` form -- and, when the function assigns the name, a local
variable, which resolves to nothing), then the module (definitions, imports, aliases such
as ``from_torch = STensor.from_torch``, module-level dict/list/tuple/set tables), then
builtins.

Types.  A parameter's annotation (``Optional`` / ``Union`` / ``X | None`` / string forms,
and ``C[...]`` for a generic package class C) gives its package classes; a local's type
is the union of its assignments' types (a package class call, a call of a function with a
return annotation, an attribute with a known type) and of the classes an ``except ... as``
clause names; a module-level name's type is the union of its module-level assignments'
types (``X = C(...)``, ``X: C = ...``) unless a function rebinds it with ``global``; an
instance attribute's type comes from ``self.x = ...`` / ``self.x: T`` in any method and
from class-level annotations; a property's type is its getter's return annotation.

Call graph.  An edge per call or reference, in one of two tiers:

* resolved: a package function or method named directly; a package class call (its
  ``__new__``, ``__init__``, ``__post_init__``); ``self.m`` / ``cls.m`` over the class, its
  bases and its subclasses' overrides; ``super().m``; ``x.m`` where x's type is known; a
  property read on self or on a typed object; ``self.f(...)`` where ``self.f`` was assigned
  a package function; ``getattr(self, "visit_" + ...)(...)`` to every method of the class
  with that prefix; a function referenced as a value, alone or inside a module-level
  table; ``C.apply(...)`` on a ``torch.autograd.Function`` subclass, to ``C.forward`` and
  ``C.backward``;
* name only: ``x.m(...)`` with x of unknown type, to every package method named m; a
  property read ``x.p`` with x of unknown type, to every package property named p;
  ``getattr(x, "p" + ...)`` with x untyped, to every package method with that prefix.

Special methods Python calls implicitly get the edges the explicit call ``x.__m__(...)``
would get, in the same two tiers: resolved when x is ``self`` or of a known type (to the
method each such class, and each subclass, runs), name only when x's type is unknown (to
the method every package class runs).  A receiver that is plainly a builtin value (a
literal, a display or comprehension, a comparison, ``not``, an f-string, a call of a
builtin such as ``len`` or ``list``, a local only ever assigned those) gets no edge.  The
operations (``IMPLICIT_PROTOCOLS`` gives the methods each one runs, with Python's
fallbacks):

* ``x[k]``, ``x[k] = v``, ``del x[k]``: ``__getitem__`` (and ``__missing__``, which
  ``dict.__getitem__`` calls on a miss), ``__setitem__``, ``__delitem__``, and the key's
  ``__hash__`` / ``__eq__`` / ``__index__``; ``C[...]`` on a package class C:
  ``C.__class_getitem__``;
* ``k in x``: x's ``__contains__``, else its iteration, else its ``__getitem__``; and k's
  ``__hash__`` / ``__eq__``;
* iteration -- ``for``, comprehensions, ``*x``, unpacking ``a, b = x``, ``yield from x``
  and the builtins that consume an iterable (``list``, ``sorted``, ``zip``, ``"".join``
  and the like): ``__iter__`` and ``__next__`` (also of the classes ``__iter__`` is
  annotated to return), else ``__getitem__``; ``async for``: ``__aiter__`` /
  ``__anext__``;
* ``with x``: ``__enter__`` and ``__exit__``; ``async with``: ``__aenter__`` /
  ``__aexit__``; ``await x``: ``__await__``;
* ``x(...)`` on an instance: ``__call__``;
* operators: ``a + b`` runs ``a.__add__`` and ``b.__radd__`` (each binary operator
  alike), ``a += b`` also ``a.__iadd__``; ``-a`` / ``+a`` / ``~a``: ``__neg__`` /
  ``__pos__`` / ``__invert__``; comparisons run the left operand's method and the right
  operand's reflection (``a < b``: ``a.__lt__``, ``b.__gt__``);
* truth -- ``if`` / ``while`` / ``assert`` / conditional-expression tests, ``not``, the
  operands of ``and`` / ``or`` but the last, comprehension filters and ``bool(x)``:
  ``__bool__``, else ``__len__``;
* text: an f-string field runs ``__format__``, else ``__str__``, else ``__repr__`` (``!r``
  / ``!a``: ``__repr__``; ``!s``: ``__str__``); ``str``, ``repr``, ``format`` and
  ``print`` alike;
* builtins: ``len``, ``hash``, ``abs``, ``int``, ``float``, ``complex``, ``round``,
  ``divmod``, ``pow``, ``iter``, ``next``, ``reversed`` run their special methods;
  ``**x`` runs x's ``keys`` and ``__getitem__``;
* attributes: ``x.a`` runs ``__getattribute__`` and ``__getattr__``, ``x.a = v``
  ``__setattr__``, ``del x.a`` ``__delattr__``, when the class defines them.

A protocol method a class takes from a base outside the package is not seen, so the
fallback its package methods give is followed instead, which can only add edges.

Rapid type analysis (Bacon and Sweeney, OOPSLA 1996) then drops every edge that runs a
method on an instance (``self.m``, typed or untyped ``x.m``, property reads, ``getattr``
calls, implicit special-method calls) whose class and subclasses are never constructed in
code reachable from the entries or in code that runs when a module is imported (module
and class bodies, decorators, default values, and what they call) -- of the modules that
are imported: the package's own, what module-level code imports, transitively, and what a
function the entries or that code reach imports.  ``if TYPE_CHECKING:`` blocks import
nothing.

Raise sites.  ``raise C(...)``, ``raise C``, ``raise name`` with name assigned ``C(...)``
in the function, ``raise f(...)`` / ``raise self.m(...)`` where the function returns
``C(...)`` or calls a class it was passed (``return error_type(message)``), ``raise
cls(...)`` where cls was assigned a lookup in a module-level table of classes
(``TABLE[key]``, ``TABLE.get(key, Default)``), ``raise kept`` where kept was assigned an
exception an ``except C as error`` clause caught, and ``raise w.field`` where w is an
``except ... as`` name.  A bare ``raise`` and ``raise e`` for an ``except ... as e`` name
re-raise and are not sites.  An ``assert`` is a site of ``AssertionError``.  A class
that cannot be resolved is reported as ``"?expression"`` and counts as uncoded.

Coded.  A code must be a snake_case string literal, or ``<expression> or "<literal>"``
(``code=defect_code(exc) or "fallback"``).  A raise is coded when the class or a package
base has a class-level ``code`` literal or a ``defect``, declares ``code`` / ``defect`` as
dataclass fields or defines them as properties; or the ``__init__`` the call runs stores
``self.code`` / ``self.defect`` / ``self.diagnostics`` unconditionally, or passes a
literal ``code=`` to its base's ``__init__``; or that ``__init__`` stores one from a
defaulted parameter (``code=None``, ``diagnostics=()``, followed through
``super().__init__(..., code=code)`` and ``**kwargs``) and the raise passes a literal
code or a non-empty value for it; or the function sets ``.code`` to a literal (or
``.defect``) on the raised name.  A raise that passes ``code=`` to a class that does not
store it, or a ``code=`` that is not a literal, is uncoded; so are builtin exceptions.
A class-level code does not count when the ``__init__`` stores ``self.code = code``
outside an ``if code is not None`` guard, since a raise passing nothing overwrites it.

Re-raises.  ``raise kept`` and ``raise w.field`` re-raise what a handler caught: they are
coded when every site whose exception can reach that handler is coded, and there is at
least one.  For ``raise w.field`` those sites are found through each construction of
w's class that reaches w's handler, from what it passes for the ``__init__`` parameter
stored in ``field`` (an ``except ... as`` name, or another wrapper's field); a
construction or argument that cannot be followed leaves the re-raise uncoded.

Escape.  Within one ``try`` the first handler that names a site's class (or a base, or
is bare) takes the exception; it catches it unless it re-raises, in which case the
exception goes on to the enclosing ``try``.  A site escapes to an entry when it is not
caught in its own function and some call path from the entry reaches the function with
no edge inside a handler that catches the class.  Escapes through resolved edges are
kept apart from escapes that need a name-only edge.

Entries.  The functions in ``scorch/__init__.py``'s ``__all__`` (aliases included),
every method of ``STensor`` that is a dunder or does not start with ``_`` (operators,
conversions, constructors, properties) but ``__repr__`` and ``__str__``, and
``scorch.ops.spmv``, which ``__all__`` does not list.  In scope: ``einsum``, ``matmul``,
``spmv``, ``from_torch`` / ``from_coo`` / ``from_csr`` / ``from_components`` and
``STensor``'s methods.

Provisional kind, by class and message, to help whoever writes the codes: invalid input
(``TypeError``, ``ValueError``, ``IndexError``, ``KeyError``, ``LookupError`` and the
``TensorValidationError`` / ``TensorTypeError`` families, ``CompileSpecError``
included), unsupported (``NotImplementedError`` and subclasses, or a message saying "not
supported", "unsupported", "cannot lower" and the like), internal (``AssertionError``, or
a ``RuntimeError`` whose message names an invariant), else unclassified.

Special methods the call graph does not follow.  Python also calls special methods the
edges above do not model: a key's ``__hash__`` / ``__eq__`` inside a builtin container's
method (``d.get(k)``, ``s.add(k)``), copy and pickle hooks (``__copy__``, ``__reduce__``,
``__getstate__``, ``__setstate__``), descriptors (``__get__``, ``__set__``,
``__set_name__``), ``__init_subclass__``, ``__instancecheck__`` and the like, and an
outside base's method calling back into a package method (``Mapping.get`` running
``__getitem__``).  A raise or assert in a special method that no protocol above runs
(``unmodeled_special``) is reported apart, whatever its reach, so that the gate can
require a code on it.  ``__del__`` is left out: Python does not propagate its
exceptions.

Not covered: exceptions raised by native code (``scorch_ops``, generated kernels, torch)
and by the standard library; exceptions a context manager suppresses; a generator's
raise is attributed to the call that creates the generator.
"""

import ast
import builtins
import collections
import os
import re

BUILTIN_NAMES = set(dir(builtins))
BUILTIN_EXCEPTIONS = {
    name
    for name in BUILTIN_NAMES
    if isinstance(getattr(builtins, name), type)
    and issubclass(getattr(builtins, name), BaseException)
}
UNSUPPORTED_TEXT = re.compile(
    r"not supported|unsupported|not implemented|cannot (lower|schedule|"
    r"compile|fuse|handle|represent|be lowered)|only .{0,60}supported|"
    r"does not support|no (kernel|lowering|route)|is refused|refuses",
    re.I,
)
INTERNAL_TEXT = re.compile(
    r"internal|invariant|should not happen|unreachable|\bbug\b", re.I
)
INVALID_BASES = {
    "TensorValidationError",
    "TensorTypeError",
    "TypeError",
    "ValueError",
    "IndexError",
    "KeyError",
    "LookupError",
    "CompileSpecError",
}
SCOPE_EXPORTED = {
    "einsum",
    "matmul",
    "from_torch",
    "from_coo",
    "from_csr",
    "from_components",
}
# Entries the inventory must find, so that a change in how scorch exports them cannot
# make the gate pass by reaching nothing.
REQUIRED_ENTRIES = {
    "scorch.einsum",
    "scorch.matmul",
    "scorch.ops.spmv",
    "scorch.from_torch",
    "STensor.__add__",
    "STensor.to_dense",
}
RESOLVED, NAME_ONLY = ("resolved",), ("resolved", "name only")
# A module-level assignment of one of these is a table: every name in it is an element.
TABLE_NODES = (
    ast.Dict,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.DictComp,
    ast.ListComp,
    ast.SetComp,
)
# Implicit special-method calls.  A protocol is the list of alternatives Python tries in
# order on the receiver's class: each alternative is a tuple of method names, and the
# first alternative the class (its package bases included) defines one of is the one
# that runs, every name of it the class defines.
BINARY_METHODS = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.MatMult: "matmul",
    ast.Div: "truediv",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Pow: "pow",
    ast.LShift: "lshift",
    ast.RShift: "rshift",
    ast.BitOr: "or",
    ast.BitXor: "xor",
    ast.BitAnd: "and",
}
COMPARE_METHODS = {
    ast.Eq: ("__eq__", "__eq__"),
    ast.NotEq: ("__ne__", "__ne__"),
    ast.Lt: ("__lt__", "__gt__"),
    ast.LtE: ("__le__", "__ge__"),
    ast.Gt: ("__gt__", "__lt__"),
    ast.GtE: ("__ge__", "__le__"),
}
UNARY_METHODS = {
    ast.USub: "__neg__",
    ast.UAdd: "__pos__",
    ast.Invert: "__invert__",
}
IMPLICIT_PROTOCOLS = {
    "getitem": [("__getitem__", "__missing__")],
    "key": [("__hash__", "__eq__", "__index__")],
    "contains": [("__contains__",), ("__iter__", "__next__"), ("__getitem__",)],
    "iter": [("__iter__", "__next__"), ("__getitem__",)],
    "reversed": [("__reversed__",), ("__len__", "__getitem__")],
    "async iter": [("__aiter__", "__anext__")],
    "with": [("__enter__", "__exit__")],
    "async with": [("__aenter__", "__aexit__")],
    "truth": [("__bool__",), ("__len__",)],
    "format": [("__format__",), ("__str__",), ("__repr__",)],
    "str": [("__str__",), ("__repr__",)],
    "int": [("__int__",), ("__index__",), ("__trunc__",)],
    "getattr": [("__getattribute__", "__getattr__")],
    "mapping unpack": [("keys", "__getitem__")],
}
for _op, _name in BINARY_METHODS.items():
    IMPLICIT_PROTOCOLS[f"__{_name}__"] = [(f"__{_name}__",)]
    IMPLICIT_PROTOCOLS[f"__r{_name}__"] = [(f"__r{_name}__",)]
    IMPLICIT_PROTOCOLS[f"__i{_name}__"] = [(f"__i{_name}__",)]
for _name in (
    [m for pair in COMPARE_METHODS.values() for m in pair]
    + list(UNARY_METHODS.values())
    + [
        "__setitem__",
        "__delitem__",
        "__next__",
        "__await__",
        "__call__",
        "__len__",
        "__hash__",
        "__repr__",
        "__abs__",
        "__float__",
        "__complex__",
        "__index__",
        "__round__",
        "__divmod__",
        "__rdivmod__",
        "__setattr__",
        "__delattr__",
    ]
):
    IMPLICIT_PROTOCOLS.setdefault(_name, [(_name,)])
# Builtins whose call runs a protocol on their first argument, or iterates arguments.
BUILTIN_PROTOCOLS = {
    "len": "__len__",
    "hash": "__hash__",
    "bool": "truth",
    "str": "str",
    "repr": "__repr__",
    "ascii": "__repr__",
    "format": "format",
    "abs": "__abs__",
    "int": "int",
    "float": "__float__",
    "complex": "__complex__",
    "round": "__round__",
    "iter": "iter",
    "next": "__next__",
    "reversed": "reversed",
}
ITERATING_BUILTINS = {
    "list": 0,
    "tuple": 0,
    "set": 0,
    "frozenset": 0,
    "dict": 0,
    "sorted": 0,
    "sum": 0,
    "min": 0,
    "max": 0,
    "any": 0,
    "all": 0,
    "enumerate": 0,
    "zip": 0,
    "map": 1,
    "filter": 1,
}  # name -> the first positional argument it iterates (each one from there on)
# Builtins whose result is a builtin value, so that an operation on it runs no package
# method.
BUILTIN_VALUED = {
    "isinstance",
    "issubclass",
    "len",
    "hasattr",
    "callable",
    "id",
    "hash",
    "str",
    "repr",
    "ascii",
    "format",
    "int",
    "float",
    "bool",
    "complex",
    "tuple",
    "list",
    "dict",
    "set",
    "frozenset",
    "sorted",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "reversed",
    "bytes",
    "bytearray",
    "ord",
    "chr",
}
BUILTIN_VALUE_NODES = (
    ast.Constant,
    ast.JoinedStr,
    ast.Compare,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Lambda,
    ast.Slice,
)
# Special methods that a constructor call runs (``callee_targets``) or that an entry is.
CONSTRUCTOR_METHODS = {"__new__", "__init__", "__post_init__"}
# Special methods Python calls whose exceptions it does not propagate.
NOT_PROPAGATED = {"__del__"}

Inventory = collections.namedtuple(
    "Inventory", "index roots missing sites propagations"
)


def own_nodes(fn_node):
    """Every node of a function's body, not entering nested definitions (yielded)."""

    stack = list(fn_node.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(node))


def name_prefix(expr):
    """(constant prefix of a computed attribute name, whether the name is exactly it)."""

    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value, True
    if isinstance(expr, ast.JoinedStr):
        prefix = ""
        for v in expr.values:
            if not isinstance(v, ast.Constant):
                break
            prefix += str(v.value)
        return prefix, False
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return name_prefix(expr.left)[0], False
    return "", False


def is_exception_class(r):
    """A resolved name that is a package class or a builtin exception class."""

    return bool(r) and (
        r[0] == "cls" or (r[0] == "builtin" and r[1][9:] in BUILTIN_EXCEPTIONS)
    )


class Fn:
    def __init__(self, qual, module, node, cls, parent):
        self.qual, self.module, self.node = qual, module, node
        self.cls, self.parent = cls, parent
        self.nested = {}  # name -> ("fn" | "cls", qual)
        self.imports = {}  # alias -> import target
        self.edges = []  # (target qual, tier, handlers, via)
        self.sites = []
        self.unresolved_calls = 0
        self.dynamic_imports = []
        self.implicit_seen = set()  # implicit special-method edges already added
        self.builtin_locals = None  # locals only ever assigned a builtin value
        self.instantiates = set()  # package classes this function constructs
        self.bound_by = collections.defaultdict(list)  # ``except ... as`` name -> ids
        self.types = {}  # local name -> {class qual}
        self.is_static = any(
            isinstance(d, ast.Name) and d.id == "staticmethod"
            for d in node.decorator_list
        )
        self.is_classmethod = any(
            isinstance(d, ast.Name) and d.id == "classmethod"
            for d in node.decorator_list
        )
        args = node.args.posonlyargs + node.args.args
        self.self_name = (
            args[0].arg if cls is not None and args and not self.is_static else None
        )
        declared = set()
        for n in own_nodes(node):
            if isinstance(n, (ast.Global, ast.Nonlocal)):
                declared.update(n.names)
        params = {
            a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        }
        params |= {a.arg for a in (node.args.vararg, node.args.kwarg) if a is not None}
        stored = {
            n.id
            for n in own_nodes(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
        }
        stored |= {
            n.name
            for n in own_nodes(node)
            if isinstance(n, ast.ExceptHandler) and n.name
        }
        self.locals = (params | stored) - declared


class Cls:
    def __init__(self, qual, module, node, scope_fn):
        self.qual, self.module, self.node, self.scope_fn = qual, module, node, scope_fn
        self.name = node.name
        self.methods = {}  # name -> Fn
        self.attrs = {}  # class-level name -> value node
        self.ann = {}  # class-level annotated name -> annotation node
        self.properties = set()
        self.bases = []
        self.external_bases = []  # base expressions outside the package, as written
        self.external_first = False  # an external base is listed before every other
        self.inst_types = collections.defaultdict(set)
        self.inst_callables = collections.defaultdict(set)
        self.is_dataclass = any(
            "dataclass" in ast.unparse(d) for d in node.decorator_list
        )


class ImportTimeCode:
    """The code a module runs when it is imported -- its top-level statements, the
    statements of its class bodies, and the decorators and default values of its
    definitions -- shaped as a function node so that ``Fn`` and ``walk_function`` can
    record its calls and constructions.  Base classes are left out: naming a class as a
    base does not construct it."""

    def __init__(self, stmts):
        self.name = "<module>"
        self.body = list(self.flatten(stmts))
        self.imports = [
            n
            for s in self.body
            for n in self.runtime_nodes(s)
            if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        self.decorator_list = []
        self.returns = None
        self.args = ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )

    @classmethod
    def flatten(cls, stmts):
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                evaluated = list(s.decorator_list)
                if not isinstance(s, ast.ClassDef):
                    evaluated += s.args.defaults
                    evaluated += [d for d in s.args.kw_defaults if d is not None]
                for e in evaluated:
                    yield ast.Expr(value=e)
                if isinstance(s, ast.ClassDef):
                    yield from cls.flatten(s.body)
            else:
                yield s

    @classmethod
    def runtime_nodes(cls, node):
        """The nodes of a statement that run at import: not function bodies, not an
        ``if TYPE_CHECKING:`` body."""

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        test = node.test if isinstance(node, ast.If) else None
        if getattr(test, "id", getattr(test, "attr", None)) == "TYPE_CHECKING":
            for s in node.orelse:
                yield from cls.runtime_nodes(s)
            return
        yield node
        for child in ast.iter_child_nodes(node):
            yield from cls.runtime_nodes(child)


class Module:
    def __init__(self, name, path, tree, is_package):
        self.name, self.path, self.tree = name, path, tree
        self.is_package = is_package
        self.defs, self.imports, self.aliases, self.tables = {}, {}, {}, {}
        self.assigns = []  # module-level (name, value or None, annotation or None)
        self.runtime_imports = set()  # modules its import-time code imports
        self.types = {}  # module-level name -> {class qual} it holds an instance of


class Index:
    def __init__(self, pkg_dir, pkg_name):
        self.pkg_name = pkg_name
        self.modules, self.fns, self.classes = {}, {}, {}
        self.parse_errors = []
        self.return_cache, self.factory_cache, self.mro_cache = {}, {}, {}
        self.rta = {}
        self.handlers = {}  # handler id -> (function qual, caught class quals or None)
        for dirpath, dirnames, files in os.walk(pkg_dir):
            dirnames.sort()
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dirpath, f)
                rel = os.path.relpath(path, os.path.dirname(pkg_dir))[:-3].split(os.sep)
                is_package = rel[-1] == "__init__"
                name = ".".join(rel[:-1] if is_package else rel)
                try:
                    with open(path) as fh:
                        tree = ast.parse(fh.read(), path)
                except SyntaxError as error:
                    self.parse_errors.append(f"{path}: {error}")
                    continue
                self.modules[name] = Module(name, path, tree, is_package)
        for m in list(self.modules.values()):
            for node in m.tree.body:
                self.collect_stmt(m, node, None, None, top=True)
        for c in self.classes.values():
            for base in c.node.bases:
                r = self.resolve_expr(base, c.module, c.scope_fn)
                if r and (
                    r[0] == "cls"
                    or (
                        r[0] == "builtin"
                        and r[1][9:] in BUILTIN_EXCEPTIONS | {"object"}
                    )
                ):
                    c.bases.append(r[1])
                else:
                    c.external_first |= not c.bases and not c.external_bases
                    c.external_bases.append(ast.unparse(base))
        self.subclasses = collections.defaultdict(set)
        for c in self.classes.values():
            for base in self.mro(c.qual)[1:]:
                self.subclasses[base].add(c.qual)
        self.methods_by_name = collections.defaultdict(list)
        self.property_getters = collections.defaultdict(list)
        for fn in self.fns.values():
            if fn.cls is not None:
                self.methods_by_name[fn.node.name].append(fn.qual)
                if fn.node.name in fn.cls.properties:
                    self.property_getters[fn.node.name].append(fn.qual)
        self.implicit_any = {}  # protocol -> the methods it runs over every class
        self.protocol_cache = {}  # (class qual, protocol) -> the methods it runs
        # Types: module-level names, parameters, then locals, then instance attributes,
        # then locals again.
        self.infer_module_types()
        for fn in self.fns.values():
            self.infer_params(fn)
        for fn in self.fns.values():
            self.infer_locals(fn)
        self.infer_instance_attrs()
        for fn in self.fns.values():
            self.infer_locals(fn)
        for fn in list(self.fns.values()):
            walk_function(self, fn)
        # Import-time code: its constructions count for rapid type analysis; it is no
        # entry, and its sites are not the package's refusals.
        self.import_fns = []
        for m in self.modules.values():
            node = ImportTimeCode(m.tree.body)
            code = Fn(f"{m.name}.<module>", m, node, None, None)
            walk_function(self, code)
            self.import_fns.append(code)
            m.runtime_imports = {
                t
                for i in node.imports
                for _, target in self.import_target(m, i)
                for t in self.imported_modules(target)
            }

    def imported_modules(self, target):
        """The package modules an import target makes Python import."""

        name = target[1]
        return {name} if name in self.modules else set()

    def imported_closure(self, names):
        """The modules importing ``names`` runs: each, its parent packages, and what
        their module-level code imports, transitively."""

        out, work = set(), list(names)
        while work:
            name = work.pop()
            parts = name.split(".")
            for i in range(1, len(parts) + 1):
                m = self.modules.get(".".join(parts[:i]))
                if m is not None and m.name not in out:
                    out.add(m.name)
                    work.extend(m.runtime_imports)
        return out

    def function_imports(self, quals):
        """The package modules the import statements in these functions name."""

        return {
            t
            for q in quals
            for target in self.fns[q].imports.values()
            for t in self.imported_modules(target)
        }

    # -- collection ---------------------------------------------------------------------

    def import_target(self, module, node):
        out = []
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    out.append((a.asname, ("module", a.name)))
                else:
                    out.append((a.name.split(".")[0], ("module", a.name.split(".")[0])))
            return out
        if node.level:
            parts = module.name.split(".")
            base = parts if module.is_package else parts[:-1]
            if node.level > 1:
                base = base[: len(base) - (node.level - 1)]
            source = ".".join(base + ([node.module] if node.module else []))
        else:
            source = node.module or ""
        for a in node.names:
            alias = a.asname or a.name
            full = f"{source}.{a.name}"
            out.append(
                (
                    alias,
                    (
                        ("module", full)
                        if full in self.modules
                        else ("symbol", source, a.name)
                    ),
                )
            )
        return out

    def collect_stmt(self, m, node, cls, parent, top=False):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if top:
                for alias, t in self.import_target(m, node):
                    m.imports[alias] = t
            elif cls is not None:
                # An import in a class body: module scope, as an approximation.
                for alias, t in self.import_target(m, node):
                    m.imports.setdefault(alias, t)
            elif parent is not None:
                for alias, t in self.import_target(m, node):
                    parent.imports[alias] = t
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if cls is not None:
                qual, scope = f"{cls.qual}.{node.name}", cls.scope_fn
            elif parent is not None:
                qual, scope = f"{parent.qual}.<locals>.{node.name}", parent
            else:
                qual, scope = f"{m.name}.{node.name}", None
            fn = Fn(qual, m, node, cls, scope)
            self.fns[qual] = fn
            if cls is not None:
                cls.methods[node.name] = fn
                for d in node.decorator_list:
                    text = ast.unparse(d)
                    if text in (
                        "property",
                        "functools.cached_property",
                        "cached_property",
                    ) or text.endswith(".getter"):
                        cls.properties.add(node.name)
            elif parent is not None:
                parent.nested[node.name] = ("fn", qual)
            elif top:
                m.defs[node.name] = ("fn", qual)
            self.collect_body(m, node.body, fn)
        elif isinstance(node, ast.ClassDef):
            if cls is not None:
                qual, scope = f"{cls.qual}.{node.name}", cls.scope_fn
            elif parent is not None:
                qual, scope = f"{parent.qual}.<locals>.{node.name}", parent
            else:
                qual, scope = f"{m.name}.{node.name}", None
            c = Cls(qual, m, node, scope)
            self.classes[qual] = c
            if top:
                m.defs[node.name] = ("cls", qual)
            elif parent is not None:
                parent.nested[node.name] = ("cls", qual)
            for child in node.body:
                if isinstance(child, ast.Assign):
                    for t in child.targets:
                        if isinstance(t, ast.Name):
                            c.attrs[t.id] = child.value
                elif isinstance(child, ast.AnnAssign) and isinstance(
                    child.target, ast.Name
                ):
                    c.ann[child.target.id] = child.annotation
                    if child.value is not None:
                        c.attrs[child.target.id] = child.value
                self.collect_stmt(m, child, c, None)
        elif top and isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            ann = getattr(node, "annotation", None)
            for t in targets:
                if not isinstance(t, ast.Name):
                    continue
                m.assigns.append((t.id, node.value, ann))
                if node.value is None:
                    continue
                if isinstance(node.value, (ast.Name, ast.Attribute)):
                    m.aliases[t.id] = node.value
                elif isinstance(node.value, TABLE_NODES):
                    m.tables[t.id] = [
                        n
                        for n in ast.walk(node.value)
                        if isinstance(n, (ast.Name, ast.Attribute))
                    ]
        elif top and isinstance(node, (ast.If, ast.Try, ast.With)):
            body = (
                list(node.body)
                + list(getattr(node, "orelse", []))
                + list(getattr(node, "finalbody", []))
            )
            body += [s for h in getattr(node, "handlers", []) for s in h.body]
            for child in body:
                self.collect_stmt(m, child, None, None, top=True)

    def collect_body(self, m, body, fn):
        stack = list(body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias, target in self.import_target(m, node):
                    fn.imports[alias] = target
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                self.collect_stmt(m, node, None, fn)
            else:
                stack.extend(ast.iter_child_nodes(node))

    # -- names --------------------------------------------------------------------------

    def lookup(self, module_name, name, depth=0):
        m = self.modules.get(module_name)
        if m is None or depth > 10:
            return None
        if name in m.defs:
            return m.defs[name]
        if name in m.imports:
            return self.follow(m.imports[name], depth + 1)
        if name in m.aliases:
            return self.resolve_expr(m.aliases[name], m, None, depth + 1)
        if name in m.tables:
            return ("table", module_name, name)
        if f"{module_name}.{name}" in self.modules:
            return ("module", f"{module_name}.{name}")
        return None

    def follow(self, target, depth=0):
        if target[0] == "module":
            if target[1] in self.modules:
                return ("module", target[1])
            return ("external", target[1])
        _, module, name = target
        if module in self.modules:
            return self.lookup(module, name, depth)
        return ("external", f"{module}.{name}")

    def resolve_name(self, name, module, fn):
        scope = fn
        while scope is not None:
            if name in scope.nested:
                return scope.nested[name]
            if name in scope.imports:
                return self.follow(scope.imports[name])
            if name in scope.locals:
                return None
            scope = scope.parent
        found = self.lookup(module.name, name)
        if found:
            return found
        if name in BUILTIN_NAMES:
            return ("builtin", f"builtins.{name}")
        return None

    def resolve_expr(self, expr, module, fn, depth=0):
        if depth > 10:
            return None
        if isinstance(expr, ast.Name):
            return self.resolve_name(expr.id, module, fn)
        if isinstance(expr, ast.Attribute):
            base = self.resolve_expr(expr.value, module, fn, depth + 1)
            if base is None:
                return None
            if base[0] == "module":
                return self.lookup(base[1], expr.attr, depth + 1)
            if base[0] == "cls":
                for c in self.mro(base[1]):
                    if c in self.classes:
                        if expr.attr in self.classes[c].methods:
                            return ("fn", self.classes[c].methods[expr.attr].qual)
                        if expr.attr in self.classes[c].attrs:
                            return None
                return None
            if base[0] == "external":
                return ("external", f"{base[1]}.{expr.attr}")
        return None

    def mro(self, qual):
        if qual in self.mro_cache:
            return self.mro_cache[qual]
        out, queue = [], [qual]
        while queue:
            q = queue.pop(0)
            if q in out:
                continue
            out.append(q)
            if q in self.classes:
                queue.extend(self.classes[q].bases)
            elif q.startswith("builtins."):
                t = getattr(builtins, q[9:], None)
                if isinstance(t, type):
                    out.extend(
                        f"builtins.{b.__name__}"
                        for b in t.__mro__[1:]
                        if f"builtins.{b.__name__}" not in out
                    )
        self.mro_cache[qual] = out
        return out

    def method_in(self, cls_qual, name, subclasses=True):
        found = []
        for c in self.mro(cls_qual):
            if c in self.classes and name in self.classes[c].methods:
                found.append(self.classes[c].methods[name].qual)
                break
        if subclasses:
            for sub in sorted(self.subclasses.get(cls_qual, ())):
                if name in self.classes[sub].methods:
                    found.append(self.classes[sub].methods[name].qual)
        return found

    def all_methods(self, cls_qual):
        names = {}
        for c in reversed(self.mro(cls_qual)):
            if c in self.classes:
                names.update({n: f.qual for n, f in self.classes[c].methods.items()})
        out = set(names.values())
        for sub in self.subclasses.get(cls_qual, ()):
            out |= {f.qual for f in self.classes[sub].methods.values()}
        return out

    # -- types --------------------------------------------------------------------------

    def ann_types(self, expr, module, fn, depth=0):
        if expr is None or depth > 6:
            return set()
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            try:
                expr = ast.parse(expr.value, mode="eval").body
            except SyntaxError:
                return set()
            return self.ann_types(expr, module, fn, depth + 1)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
            return self.ann_types(expr.left, module, fn, depth + 1) | self.ann_types(
                expr.right, module, fn, depth + 1
            )
        if isinstance(expr, ast.Subscript):
            head = expr.value
            name = (
                head.attr
                if isinstance(head, ast.Attribute)
                else getattr(head, "id", "")
            )
            if name in ("Optional", "Union"):
                inner = expr.slice
                out = set()
                for e in inner.elts if isinstance(inner, ast.Tuple) else [inner]:
                    out |= self.ann_types(e, module, fn, depth + 1)
                return out
            # ``C[K, V]`` for a generic package class C is an instance of C.
            expr = head
        r = self.resolve_expr(expr, module, fn)
        return {r[1]} if r and r[0] == "cls" else set()

    def return_types(self, qual):
        if qual not in self.return_cache:
            self.return_cache[qual] = set()
            f = self.fns[qual]
            self.return_cache[qual] = self.ann_types(f.node.returns, f.module, f)
        return self.return_cache[qual]

    def attr_types(self, cls_qual, attr):
        for c in self.mro(cls_qual):
            if c not in self.classes:
                continue
            k = self.classes[c]
            if attr in k.methods and attr in k.properties:
                return self.return_types(k.methods[attr].qual)
            if attr in k.inst_types:
                return k.inst_types[attr]
            if attr in k.ann:
                return self.ann_types(k.ann[attr], k.module, k.scope_fn)
        return set()

    def expr_types(self, expr, fn, depth=0):
        if depth > 4:
            return set()
        if isinstance(expr, ast.Name):
            scope = fn
            while scope is not None:
                if expr.id in scope.types:
                    return scope.types[expr.id]
                if expr.id in scope.locals:
                    return set()
                scope = scope.parent
            return self.global_types(expr.id, fn.module, fn)
        if isinstance(expr, ast.Call):
            r = self.resolve_expr(expr.func, fn.module, fn)
            if r and r[0] == "cls":
                return {r[1]}
            out = set()
            for q, tier, _ in self.callee_targets(expr.func, fn, depth + 1) or []:
                if tier == "resolved":
                    out |= self.return_types(q)
            return out
        if isinstance(expr, ast.Attribute):
            if (
                isinstance(expr.value, ast.Name)
                and expr.value.id == fn.self_name
                and fn.cls is not None
            ):
                bases = {fn.cls.qual}
            else:
                namespace = self.resolve_expr(expr.value, fn.module, fn)
                if namespace and namespace[0] == "module":
                    return self.module_types(namespace[1], expr.attr)
                bases = self.expr_types(expr.value, fn, depth + 1)
            out = set()
            for b in bases:
                out |= self.attr_types(b, expr.attr)
            return out
        if isinstance(expr, ast.IfExp):
            return self.expr_types(expr.body, fn, depth + 1) | self.expr_types(
                expr.orelse, fn, depth + 1
            )
        if isinstance(expr, ast.BoolOp):
            out = set()
            for v in expr.values:
                out |= self.expr_types(v, fn, depth + 1)
            return out
        return set()

    def infer_module_types(self):
        """The package classes each module-level name holds an instance of: the union
        over its module-level assignments (``X = C(...)``, ``X = f(...)`` with f
        annotated to return C, ``X: C = ...``).  A name a function rebinds with
        ``global`` is left untyped."""

        rebound = collections.defaultdict(set)
        for fn in self.fns.values():
            declared = {
                name
                for n in own_nodes(fn.node)
                if isinstance(n, ast.Global)
                for name in n.names
            }
            if declared:
                stored = {
                    n.id
                    for n in own_nodes(fn.node)
                    if isinstance(n, ast.Name)
                    and isinstance(n.ctx, (ast.Store, ast.Del))
                }
                rebound[fn.module.name] |= declared & stored
        for m in self.modules.values():
            for name, value, ann in m.assigns:
                t = self.ann_types(ann, m, None) if ann is not None else set()
                if isinstance(value, ast.Call):
                    r = self.resolve_expr(value.func, m, None)
                    if r and r[0] == "cls":
                        t = t | {r[1]}
                    elif r and r[0] == "fn":
                        t = t | self.return_types(r[1])
                if t and name not in rebound[m.name]:
                    m.types.setdefault(name, set()).update(t)

    def module_types(self, module_name, name, depth=0):
        """The classes the module-level ``name`` of a module holds an instance of,
        following imports."""

        m = self.modules.get(module_name)
        if m is None or depth > 10 or name in m.defs:
            return set()
        if name in m.types:
            return m.types[name]
        target = m.imports.get(name)
        if target and target[0] == "symbol" and target[1] in self.modules:
            return self.module_types(target[1], target[2], depth + 1)
        return set()

    def global_types(self, name, module, fn):
        """``module_types`` for a name ``fn`` does not bind itself."""

        scope = fn
        while scope is not None:
            if name in scope.nested or name in scope.locals:
                return set()
            if name in scope.imports:
                target = scope.imports[name]
                if target[0] == "symbol" and target[1] in self.modules:
                    return self.module_types(target[1], target[2])
                return set()
            scope = scope.parent
        return self.module_types(module.name, name)

    def infer_params(self, fn):
        args = fn.node.args
        for a in args.posonlyargs + args.args + args.kwonlyargs:
            if a.arg != fn.self_name and a.annotation is not None:
                t = self.ann_types(a.annotation, fn.module, fn)
                if t:
                    fn.types.setdefault(a.arg, set()).update(t)

    def infer_locals(self, fn):
        for node in own_nodes(fn.node):
            if (
                isinstance(node, ast.ExceptHandler)
                and node.name
                and node.type is not None
            ):
                exprs = (
                    node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
                )
                for e in exprs:
                    t = self.ann_types(e, fn.module, fn)
                    if t:
                        fn.types.setdefault(node.name, set()).update(t)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                t = self.ann_types(node.annotation, fn.module, fn)
                if t:
                    fn.types.setdefault(node.target.id, set()).update(t)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        t = self.expr_types(node.value, fn)
                        if t:
                            fn.types.setdefault(target.id, set()).update(t)

    def infer_instance_attrs(self):
        for c in self.classes.values():
            for name, ann in c.ann.items():
                c.inst_types[name] |= self.ann_types(ann, c.module, c.scope_fn)
            for m in c.methods.values():
                if m.self_name is None:
                    continue
                for node in own_nodes(m.node):
                    if isinstance(node, ast.Assign):
                        pairs = [(t, node.value, None) for t in node.targets]
                    elif isinstance(node, ast.AnnAssign) and node.value is not None:
                        pairs = [(node.target, node.value, node.annotation)]
                    else:
                        continue
                    for target, value, ann in pairs:
                        if not (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == m.self_name
                        ):
                            continue
                        if ann is not None:
                            c.inst_types[target.attr] |= self.ann_types(
                                ann, m.module, m
                            )
                        c.inst_types[target.attr] |= self.expr_types(value, m)
                        if isinstance(value, (ast.Name, ast.Attribute)):
                            r = self.resolve_expr(value, m.module, m)
                            if r and r[0] == "fn":
                                c.inst_callables[target.attr].add(r[1])

    # -- call targets -------------------------------------------------------------------

    def table_classes(self, r):
        _, module_name, name = r
        m = self.modules[module_name]
        found = (self.resolve_expr(e, m, None) for e in m.tables[name])
        return {t[1] for t in found if t and t[0] == "cls"}

    def table_targets(self, r):
        _, module_name, name = r
        m = self.modules[module_name]
        out = []
        for e in m.tables[name]:
            t = self.resolve_expr(e, m, None)
            if t and t[0] == "fn":
                out.append((t[1], "resolved", f"table {name}"))
        return out

    def constructed(self, func, fn):
        """The package classes a call of ``func`` constructs: a class named directly,
        ``cls(...)`` in a classmethod, ``type(self)(...)`` (the method's class and its
        subclasses)."""

        r = self.resolve_expr(func, fn.module, fn)
        if r and r[0] == "cls":
            return {r[1]}
        own = fn.cls is not None and (
            (
                isinstance(func, ast.Name)
                and func.id == fn.self_name
                and fn.is_classmethod
            )
            or (
                isinstance(func, ast.Call)
                and isinstance(func.func, ast.Name)
                and func.func.id == "type"
                and len(func.args) == 1
                and isinstance(func.args[0], ast.Name)
                and func.args[0].id == fn.self_name
            )
        )
        if own:
            return {fn.cls.qual} | self.subclasses.get(fn.cls.qual, set())
        return set()

    def callee_targets(self, func, fn, depth=0):
        """[(qual, tier, via)] for a call of ``func``; [] when it leaves the package;
        None when it cannot be resolved."""

        if fn.cls is not None and not (
            isinstance(func, (ast.Name, ast.Attribute))
            and self.resolve_expr(func, fn.module, fn)
        ):
            own = self.constructed(func, fn)
            if own:
                return [
                    (q, "resolved", "constructor")
                    for c in sorted(own)
                    for name in ("__new__", "__init__", "__post_init__")
                    for q in self.method_in(c, name, subclasses=False)
                ]
        r = self.resolve_expr(func, fn.module, fn)
        if r:
            if r[0] == "fn":
                return [(r[1], "resolved", "call")]
            if r[0] == "cls":
                return [
                    (q, "resolved", "constructor")
                    for name in ("__new__", "__init__", "__post_init__")
                    for q in self.method_in(r[1], name, subclasses=False)
                ]
            if r == ("external", "importlib.import_module") or r == (
                "builtin",
                "builtins.__import__",
            ):
                fn.dynamic_imports.append(r[1])
            return []
        if isinstance(func, ast.Attribute):
            base = func.value
            if (
                isinstance(base, ast.Name)
                and base.id == fn.self_name
                and fn.cls is not None
            ):
                found = [
                    (q, "resolved", "self")
                    for q in self.method_in(fn.cls.qual, func.attr)
                ]
                if not found:
                    for c in self.mro(fn.cls.qual):
                        if (
                            c in self.classes
                            and func.attr in self.classes[c].inst_callables
                        ):
                            found = [
                                (q, "resolved", "attribute holding a function")
                                for q in sorted(
                                    self.classes[c].inst_callables[func.attr]
                                )
                            ]
                            break
                return found or None
            if (
                isinstance(base, ast.Call)
                and isinstance(base.func, ast.Name)
                and base.func.id == "super"
                and fn.cls is not None
            ):
                return [
                    (q, "resolved", "super")
                    for b in fn.cls.bases
                    for q in self.method_in(b, func.attr, subclasses=False)
                ]
            rb = self.resolve_expr(base, fn.module, fn)
            if rb and rb[0] == "cls":
                # `C.apply(...)` on a torch.autograd.Function subclass runs C.forward
                # now and C.backward when the caller differentiates; any other
                # attribute the package class does not define comes from a base
                # outside the package.
                if func.attr == "apply" and any(
                    b.endswith("autograd.Function")
                    for c in self.mro(rb[1])
                    if c in self.classes
                    for b in self.classes[c].external_bases
                ):
                    return [
                        (q, "resolved", "autograd apply")
                        for name in ("forward", "backward")
                        for q in self.method_in(rb[1], name, subclasses=False)
                    ]
                return []
            types = self.expr_types(base, fn, depth + 1)
            if types:
                return [
                    (q, "resolved", "typed receiver")
                    for t in sorted(types)
                    for q in self.method_in(t, func.attr)
                ]
            return [
                (q, "name only", f".{func.attr}")
                for q in self.methods_by_name.get(func.attr, ())
            ]
        if (
            isinstance(func, ast.Call)
            and isinstance(func.func, ast.Name)
            and func.func.id == "getattr"
            and len(func.args) >= 2
            and self.resolve_name("getattr", fn.module, fn)
            == ("builtin", "builtins.getattr")
        ):
            obj, name = func.args[0], func.args[1]
            prefix, exact = name_prefix(name)
            if (
                isinstance(obj, ast.Name)
                and obj.id == fn.self_name
                and fn.cls is not None
            ):
                classes = {fn.cls.qual}
            else:
                classes = self.expr_types(obj, fn, depth + 1)
            if classes:
                cands = {q for c in classes for q in self.all_methods(c)}
                tier = "resolved" if prefix else "name only"
            elif prefix:
                cands = {q for qs in self.methods_by_name.values() for q in qs}
                tier = "name only"
            else:
                return None
            if exact:
                cands = {q for q in cands if q.rsplit(".", 1)[-1] == prefix}
            else:
                cands = {q for q in cands if q.rsplit(".", 1)[-1].startswith(prefix)}
            return [(q, tier, f"getattr {prefix}*") for q in sorted(cands)]
        return None

    def attribute_refs(self, node, fn):
        """Edges of an attribute read that is not the callee of a call."""

        r = self.resolve_expr(node, fn.module, fn)
        if r and r[0] == "fn":
            return [(r[1], "resolved", "reference")]
        if r and r[0] == "table":
            return self.table_targets(r)
        if r:
            return []
        base = node.value
        if (
            isinstance(base, ast.Name)
            and base.id == fn.self_name
            and fn.cls is not None
        ):
            return [
                (q, "resolved", "self attribute")
                for q in self.method_in(fn.cls.qual, node.attr)
            ]
        types = self.expr_types(base, fn)
        if types:
            return [
                (q, "resolved", "typed attribute")
                for t in sorted(types)
                for q in self.method_in(t, node.attr)
            ]
        return [
            (q, "name only", f".{node.attr}")
            for q in self.property_getters.get(node.attr, ())
        ]

    # -- implicit special-method calls -------------------------------------------------

    def protocol_methods(self, cls_qual, protocol):
        """The methods a protocol runs on an instance of exactly this class: those of
        the first alternative the class defines a method of."""

        if (cls_qual, protocol) not in self.protocol_cache:
            found = []
            for alternative in IMPLICIT_PROTOCOLS[protocol]:
                found = [
                    q
                    for name in alternative
                    for q in self.method_in(cls_qual, name, False)
                ]
                if found:
                    break
            self.protocol_cache[(cls_qual, protocol)] = found
        return self.protocol_cache[(cls_qual, protocol)]

    def protocol_any(self, protocol):
        """The methods a protocol runs on an instance of some package class."""

        if protocol not in self.implicit_any:
            self.implicit_any[protocol] = sorted(
                {q for c in self.classes for q in self.protocol_methods(c, protocol)}
            )
        return self.implicit_any[protocol]

    def implicit_targets(self, expr, protocol, fn, typed_only=False):
        """[(qual, tier, via)] for the special methods ``protocol`` runs on the value of
        ``expr``: resolved when ``expr`` is ``self`` or of a known type (for each class
        and each subclass), else name only (for every package class) unless
        ``typed_only``; none for a builtin value or a name that resolves to something
        other than an instance.  ``C[...]`` on a package class runs
        ``C.__class_getitem__``."""

        if not self.protocol_any(protocol) and not (
            protocol == "getitem" and self.methods_by_name.get("__class_getitem__")
        ):
            return []
        if builtin_value(self, expr, fn):
            return []
        via = f"implicit {protocol}"
        if (
            isinstance(expr, ast.Name)
            and expr.id == fn.self_name
            and fn.cls is not None
        ):
            if fn.is_classmethod:
                return []
            classes = {fn.cls.qual}
        else:
            r = self.resolve_expr(expr, fn.module, fn)
            if r:
                if r[0] == "cls" and protocol == "getitem":
                    return [
                        (q, "resolved", "class subscript")
                        for q in self.method_in(r[1], "__class_getitem__", False)
                    ]
                return []
            classes = self.expr_types(expr, fn)
        if not classes:
            if typed_only:
                return []
            return [(q, "name only", via) for q in self.protocol_any(protocol)]
        found = []
        for c in sorted(classes):
            for k in [c] + sorted(self.subclasses.get(c, ())):
                found += self.protocol_methods(k, protocol)
        # The iterator ``__iter__`` is annotated to return runs ``__next__``.
        for q in list(found):
            if q.endswith(".__iter__"):
                for t in sorted(self.return_types(q)):
                    found += self.method_in(t, "__next__")
        return [(q, "resolved", via) for q in dict.fromkeys(found)]

    def factory_classes(self, qual):
        """The exception classes a function returns an instance of."""

        if qual in self.factory_cache:
            return self.factory_cache[qual]
        self.factory_cache[qual] = set()
        f = self.fns[qual]
        assigned = assigned_calls(f)
        out = set()
        for n in own_nodes(f.node):
            if not isinstance(n, ast.Return) or n.value is None:
                continue
            if isinstance(n.value, ast.Call):
                calls = [n.value]
            elif isinstance(n.value, ast.Name):
                calls = assigned.get(n.value.id, [])
            else:
                calls = []
            for c in calls:
                r = self.resolve_expr(c.func, f.module, f)
                if is_exception_class(r):
                    out.add(r[1])
        self.factory_cache[qual] = out
        return out

    def factory_params(self, qual):
        """Parameters a function calls to build what it returns
        (``return error_type(...)``), as (name, position among the arguments a caller
        passes)."""

        f = self.fns[qual]
        names = [a.arg for a in f.node.args.posonlyargs + f.node.args.args]
        if f.cls is not None and not f.is_static and names:
            names = names[1:]  # self or cls is bound at the call
        assigned = assigned_calls(f)
        out = []
        for n in own_nodes(f.node):
            if not isinstance(n, ast.Return) or n.value is None:
                continue
            if isinstance(n.value, ast.Call):
                calls = [n.value]
            elif isinstance(n.value, ast.Name):
                calls = assigned.get(n.value.id, [])
            else:
                calls = []
            for c in calls:
                if isinstance(c.func, ast.Name) and c.func.id in names:
                    out.append((c.func.id, names.index(c.func.id)))
        return out

    # -- codes --------------------------------------------------------------------------

    def coded_class(self, qual):
        """How a class carries a code: (a description when every instance has one,
        else None; the parameters of the ``__init__`` a call runs that store a code
        when the raise passes them, as (name, position, where)).

        A class-level attribute holds on every instance unless that ``__init__`` stores
        ``self.code`` from a defaulted parameter outside an ``if code is not None``
        guard, which overwrites it with the default when the raise passes nothing."""

        definite, alternatives, clobbers = self.init_carriers(qual, 0)
        return definite or self.class_code(qual, clobbers), alternatives

    def class_code(self, qual, clobbers):
        """The class-level attribute, field or property that codes every instance."""

        if not clobbers:
            for c in self.mro(qual):
                if c not in self.classes:
                    continue
                k = self.classes[c]
                value = k.attrs.get("code")
                # ``code: Optional[str] = None`` declares the attribute, without a code.
                if value is not None and not (
                    isinstance(value, ast.Constant) and value.value is None
                ):
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        if not CODE_TEXT.match(value.value):
                            break
                        return f"class code {value.value!r} ({k.name})"
                    return f"class code expression ({k.name})"
                if "defect" in k.attrs:
                    return f"class defect ({k.name})"
                if k.is_dataclass and ({"code", "defect"} & set(k.ann)):
                    field = "code" if "code" in k.ann else "defect"
                    return f"dataclass field {field} ({k.name})"
                for name in ("code", "defect"):
                    if name in k.methods and name in k.properties:
                        return f"property {name} ({k.name})"
        return None

    def running_init(self, qual, start):
        """(class, ``__init__``) a call runs from position ``start`` of ``qual``'s
        method resolution order, or None when a builtin or external base supplies it.
        A base outside the package listed after a package base comes after that
        base's own chain (``class E(ScorchError, FrozenInstanceError)`` runs
        ``ScorchError.__init__``); one listed first may supply the ``__init__``."""

        for c in self.mro(qual)[start:]:
            if c not in self.classes:
                if c.startswith("builtins.") and c != "builtins.object":
                    return None
                continue
            k = self.classes[c]
            if "__init__" in k.methods:
                return k, k.methods["__init__"]
            if k.external_first:
                return None
        return None

    def stored_param(self, qual, attr):
        """(parameter, position) the ``__init__`` a call of ``qual`` runs stores as
        ``self.<attr>``, or None."""

        found = self.running_init(qual, 0)
        if found is None:
            return None
        init = found[1]
        params = init_param_defaults(init)
        for node in ast.walk(init.node):
            if not (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Name)
                and node.value.id in params
            ):
                continue
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == init.self_name
                    and t.attr == attr
                ):
                    return node.value.id, params[node.value.id][0]
        return None

    def init_carriers(self, qual, start, depth=0):
        """For the ``__init__`` a call of ``qual`` runs (from MRO position ``start``):
        (a description when every call stores a code, else None; the parameters that
        store one when passed, as (name, position, where); whether a store from a
        defaulted parameter is unguarded).  Follows ``super().__init__(...)``, passing
        a literal, a parameter or ``**kwargs`` on."""

        found = self.running_init(qual, start)
        if found is None or depth > 8:
            return None, [], False
        k, init = found
        defaults = init_param_defaults(init)
        definite, alternatives, clobbers = None, [], False
        for node, guarded in guarded_nodes(init.node.body, frozenset()):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if not (
                        isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == init.self_name
                        and t.attr in ("code", "defect", "diagnostics")
                    ):
                        continue
                    read = [
                        n.id
                        for n in ast.walk(node.value)
                        if isinstance(n, ast.Name)
                        and n.id in defaults
                        and defaults[n.id][1]
                    ]
                    if isinstance(node.value, ast.Constant):
                        if t.attr != "code" or literal_code(node.value):
                            definite = f"__init__ sets self.{t.attr} ({k.name})"
                        elif not guarded:
                            clobbers = True  # ``self.code = None``
                        continue
                    if not read:
                        definite = f"__init__ sets self.{t.attr} ({k.name})"
                        continue
                    # ``self.code = code`` where the parameter has a default: coded
                    # only when the raise passes it.
                    where = f"{k.name}.__init__ {read[0]}"
                    alternatives.append((read[0], defaults[read[0]][0], where))
                    clobbers |= t.attr == "code" and read[0] not in guarded
            elif isinstance(node, ast.Call) and super_init_call(node, init):
                node = super_init_call(node, init)
                position = self.mro(qual).index(k.qual) + 1
                base = self.init_carriers(qual, position, depth + 1)
                if base[0]:
                    definite = definite or base[0]
                clobbers |= base[2]
                for name, base_position, where in base[1]:
                    passed = argument_value(node, name, base_position)
                    if passed is None:
                        if any(
                            kw.arg is None
                            and isinstance(kw.value, ast.Name)
                            and init.node.args.kwarg is not None
                            and kw.value.id == init.node.args.kwarg.arg
                            for kw in node.keywords
                        ):
                            # ``**kwargs`` passed on: a keyword the raise passes.
                            alternatives.append((name, None, where))
                        continue
                    if name == "code" and literal_code(passed):
                        definite = definite or f"__init__ passes code= ({k.name})"
                        continue
                    read = [
                        n.id
                        for n in ast.walk(passed)
                        if isinstance(n, ast.Name)
                        and n.id in defaults
                        and defaults[n.id][1]
                    ]
                    if read:
                        alternatives.append((read[0], defaults[read[0]][0], where))
                    elif name != "code":
                        definite = definite or f"__init__ passes {name} ({k.name})"
        return definite, alternatives, clobbers


def assigned_calls(f):
    """{local name: [calls assigned to it]} in a function's own body."""

    assigned = collections.defaultdict(list)
    for n in own_nodes(f.node):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    assigned[t.id].append(n.value)
    return assigned


def init_param_defaults(init):
    """{parameter: (position among the arguments a caller passes, or None for
    keyword-only; whether it has a default)} for an ``__init__``."""

    a = init.node.args
    positional = (a.posonlyargs + a.args)[1:]
    out = {}
    first_default = len(positional) - len(a.defaults)
    for i, p in enumerate(positional):
        out[p.arg] = (i, i >= first_default)
    for p, d in zip(a.kwonlyargs, a.kw_defaults):
        out[p.arg] = (None, d is not None)
    return out


CODE_TEXT = re.compile(r"[a-z][a-z0-9_]*\Z")


def literal_code(value):
    """Whether a ``code=`` value names its code where the gate can read it: a
    snake_case string literal, or ``<expression> or "<snake_case literal>"`` (such as
    ``defect_code(exc) or "fallback"``), whose literal holds when the expression is
    empty."""

    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return bool(CODE_TEXT.match(value.value))
    if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):
        return literal_code(value.values[-1])
    return False


def tested_names(test):
    """The names an ``if`` test requires to be set: ``p is not None``, ``p``, and
    either side of ``and``."""

    if isinstance(test, ast.Name):
        return {test.id}
    if (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.IsNot)
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    ):
        return {test.left.id}
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return set().union(*(tested_names(v) for v in test.values))
    return set()


def guarded_nodes(body, guarded):
    """Every node under the statements but nested definitions, with the names an
    enclosing ``if`` requires to be set (``tested_names``)."""

    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(stmt, ast.If):
            yield from ((n, guarded) for n in ast.walk(stmt.test))
            yield from guarded_nodes(stmt.body, guarded | tested_names(stmt.test))
            yield from guarded_nodes(stmt.orelse, guarded)
            continue
        yield stmt, guarded
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.stmt):
                yield from guarded_nodes([child], guarded)
            elif isinstance(child, ast.ExceptHandler):
                yield from guarded_nodes(child.body, guarded)
            else:
                yield from ((n, guarded) for n in ast.walk(child))


def super_init_call(call, init):
    """For ``super().__init__(...)``, or ``Base.__init__(self, ...)`` in ``init``, the
    call with the arguments the base's ``__init__`` receives after ``self``; else
    None."""

    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr == "__init__"):
        return None
    if (
        isinstance(func.value, ast.Call)
        and isinstance(func.value.func, ast.Name)
        and func.value.func.id == "super"
    ):
        return call
    if (
        call.args
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == init.self_name
    ):
        return ast.Call(func=func, args=call.args[1:], keywords=call.keywords)
    return None


def argument_value(call, name, position):
    """The expression a call passes for the parameter ``name``, or None."""

    value = next((k.value for k in call.keywords if k.arg == name), None)
    if (
        value is None
        and position is not None
        and position < len(call.args)
        and not any(isinstance(x, ast.Starred) for x in call.args[: position + 1])
    ):
        value = call.args[position]
    return value


def passes_argument(call, name, position):
    """Whether a call passes a non-empty value for the parameter ``name``: True, False,
    or "unknown" when the call passes ``**kwargs``.  A ``code`` must be a literal
    (``literal_code``)."""

    value = argument_value(call, name, position)
    if value is None:
        return any(k.arg is None for k in call.keywords) and "unknown"
    if name == "code":
        return literal_code(value)
    if isinstance(value, (ast.Tuple, ast.List)) and not value.elts:
        return False
    if isinstance(value, ast.Constant) and value.value in (None, "", ()):
        return False
    return True


# -- walking a function: edges, raise sites, handlers ---------------------------------


def handler_id(fn, handler):
    return f"{fn.qual}:{handler.lineno}:{handler.col_offset}"


def handler_key(index, handler, fn, try_node):
    """(caught class quals, or None for a bare except; whether the handler re-raises;
    the handler's id).  Records the handler and its ``try`` in ``index.handlers``."""

    hid = handler_id(fn, handler)
    quals, reraises = handler_classes(index, handler, fn)
    try_id = f"{fn.qual}:{try_node.lineno}:{try_node.col_offset}"
    index.handlers[hid] = (fn.qual, quals, try_id)
    return (quals, reraises, hid)


def handler_classes(index, handler, fn):
    body = ast.Module(body=handler.body, type_ignores=[])
    reraises = any(
        isinstance(n, ast.Raise)
        and (
            n.exc is None
            or (
                handler.name
                and isinstance(n.exc, ast.Name)
                and n.exc.id == handler.name
            )
        )
        for n in ast.walk(body)
    )
    if handler.type is None:
        return (None, reraises)
    exprs = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    quals = []
    for e in exprs:
        r = index.resolve_expr(e, fn.module, fn)
        quals.append(r[1] if r and r[0] in ("cls", "builtin") else f"?{ast.unparse(e)}")
    return (tuple(quals), reraises)


def builtin_value(index, expr, fn):
    """``expr`` is plainly a builtin value, so an operation on it runs no package
    method: a literal, a display or comprehension, a comparison, ``not``, an f-string, a
    lambda, a call of a builtin in ``BUILTIN_VALUED`` or of a ``str`` literal's method,
    or a local only ever assigned one of those."""

    if isinstance(expr, BUILTIN_VALUE_NODES):
        return True
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return True
    if isinstance(expr, ast.Call):
        f = expr.func
        if isinstance(f, ast.Name) and f.id in BUILTIN_VALUED:
            return index.resolve_name(f.id, fn.module, fn) == (
                "builtin",
                f"builtins.{f.id}",
            )
        return (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Constant)
            and isinstance(f.value.value, (str, bytes))
        )
    if isinstance(expr, ast.Name):
        return expr.id in builtin_locals(index, fn)
    return False


def builtin_locals(index, fn):
    """The locals of ``fn`` that every store assigns a builtin value (``parts = []``),
    and its ``*args`` / ``**kwargs`` parameters (a tuple and a dict) when it does not
    rebind them."""

    if fn.builtin_locals is None:
        fn.builtin_locals = set()  # while computing it, a name refers to no builtin
        stores = collections.Counter()
        values = collections.defaultdict(list)
        for n in own_nodes(fn.node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                stores[n.id] += 1
            elif (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
            ):
                values[n.targets[0].id].append(n.value)
        args = fn.node.args
        params = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
        params |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
        fn.builtin_locals = {
            name
            for name, vs in values.items()
            if name in fn.locals
            and name not in params
            and len(vs) == stores[name]
            and all(builtin_value(index, v, fn) for v in vs)
        }
        fn.builtin_locals |= {
            a.arg
            for a in (args.vararg, args.kwarg)
            if a is not None and not stores[a.arg]
        }
    return fn.builtin_locals


def implicit_operations(index, fn, node):
    """[(receiver expression, protocol)] for the special methods ``node`` makes Python
    call without a call expression naming them."""

    out = []

    def key(k):
        # A tuple key hashes and compares its elements.
        for e in k.elts if isinstance(k, ast.Tuple) else [k]:
            out.append((e, "key"))

    if isinstance(node, ast.Subscript):
        ctx = type(node.ctx)
        protocol = {ast.Store: "__setitem__", ast.Del: "__delitem__"}.get(
            ctx, "getitem"
        )
        out.append((node.value, protocol))
        key(node.slice)
    elif isinstance(node, ast.Attribute):
        ctx = type(node.ctx)
        protocol = {ast.Store: "__setattr__", ast.Del: "__delattr__"}.get(
            ctx, "getattr"
        )
        out.append((node.value, protocol))
    elif isinstance(node, ast.Compare):
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                out += [(right, "contains"), (left, "key")]
            elif type(op) in COMPARE_METHODS:
                mine, reflected = COMPARE_METHODS[type(op)]
                out += [(left, mine), (right, reflected)]
            left = right
    elif isinstance(node, ast.BinOp):
        name = BINARY_METHODS[type(node.op)]
        out += [(node.left, f"__{name}__"), (node.right, f"__r{name}__")]
    elif isinstance(node, ast.AugAssign):
        name, target = BINARY_METHODS[type(node.op)], node.target
        out += [
            (target, f"__i{name}__"),
            (target, f"__{name}__"),
            (node.value, f"__r{name}__"),
        ]
        # ``x[k] += v`` reads x[k] before it stores it; ``x.a += v`` reads x.a.
        if isinstance(target, ast.Subscript):
            out.append((target.value, "getitem"))
        elif isinstance(target, ast.Attribute):
            out.append((target.value, "getattr"))
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            out.append((node.operand, "truth"))
        else:
            out.append((node.operand, UNARY_METHODS[type(node.op)]))
    elif isinstance(node, ast.BoolOp):
        out += [(v, "truth") for v in node.values[:-1]]
    elif isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
        out.append((node.test, "truth"))
    elif isinstance(node, ast.comprehension):
        out.append((node.iter, "async iter" if node.is_async else "iter"))
        out += [(c, "truth") for c in node.ifs]
    elif isinstance(node, ast.For):
        out.append((node.iter, "iter"))
    elif isinstance(node, ast.AsyncFor):
        out.append((node.iter, "async iter"))
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        protocol = "with" if isinstance(node, ast.With) else "async with"
        out += [(item.context_expr, protocol) for item in node.items]
    elif isinstance(node, ast.Starred) and isinstance(node.ctx, ast.Load):
        out.append((node.value, "iter"))
    elif isinstance(node, ast.Assign):
        if any(isinstance(t, (ast.Tuple, ast.List)) for t in node.targets):
            out.append((node.value, "iter"))
    elif isinstance(node, ast.YieldFrom):
        out.append((node.value, "iter"))
    elif isinstance(node, ast.Await):
        out.append((node.value, "__await__"))
    elif isinstance(node, ast.FormattedValue):
        conversion = {ord("s"): "str", ord("r"): "__repr__", ord("a"): "__repr__"}
        out.append((node.value, conversion.get(node.conversion, "format")))
    elif isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if k is None:
                out.append((v, "mapping unpack"))
            else:
                key(k)
    elif isinstance(node, ast.Set):
        for e in node.elts:
            key(e)
    elif isinstance(node, ast.Call):
        f = node.func
        out += [(k.value, "mapping unpack") for k in node.keywords if k.arg is None]
        args = [a for a in node.args if not isinstance(a, ast.Starred)]
        builtin = isinstance(f, ast.Name) and index.resolve_name(
            f.id, fn.module, fn
        ) == ("builtin", f"builtins.{f.id}")
        if builtin and f.id in BUILTIN_PROTOCOLS and args:
            out.append((args[0], BUILTIN_PROTOCOLS[f.id]))
        if builtin and f.id in ITERATING_BUILTINS:
            out += [(a, "iter") for a in args[ITERATING_BUILTINS[f.id] :]]
        if builtin and f.id == "print":
            out += [(a, "str") for a in args]
        if builtin and f.id in ("divmod", "pow") and len(args) >= 2:
            out += [(args[0], f"__{f.id}__"), (args[1], f"__r{f.id}__")]
        if (
            isinstance(f, ast.Attribute)
            and f.attr == "join"
            and isinstance(f.value, ast.Constant)
            and isinstance(f.value.value, str)
            and args
        ):
            out.append((args[0], "iter"))
    return out


def implicit_edges(index, fn, node):
    """The edges of the special methods ``node`` makes Python call implicitly."""

    return [
        edge
        for receiver, protocol in implicit_operations(index, fn, node)
        for edge in index.implicit_targets(receiver, protocol, fn)
    ]


def add_implicit_edges(fn, targets, handlers):
    """Add each edge once per (method, tier, handlers, protocol): a function
    subscripts the same untyped receivers many times."""

    for q, tier, via in targets:
        if (q, tier, handlers, via) not in fn.implicit_seen:
            fn.implicit_seen.add((q, tier, handlers, via))
            fn.edges.append((q, tier, handlers, via))


def walk_function(index, fn):
    """Record one function's call edges and its raise and assert sites."""

    assigned = collections.defaultdict(list)
    instance_coded, caught_names, caught_types = {}, set(), {}
    bound = {}  # ``except ... as`` name -> the id of the handler being visited
    for node in own_nodes(fn.node):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigned[t.id].append(node.value)
                elif (
                    isinstance(t, ast.Attribute)
                    and t.attr in ("code", "defect")
                    and isinstance(t.value, ast.Name)
                    and (t.attr == "defect" or literal_code(node.value))
                ):
                    instance_coded[t.value.id] = f"sets {t.value.id}.{t.attr}"
        elif isinstance(node, ast.ExceptHandler) and node.name:
            caught_names.add(node.name)
            fn.bound_by[node.name].append(handler_id(fn, node))
            if node.type is not None:
                exprs = (
                    node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
                )
                for e in exprs:
                    r = index.resolve_expr(e, fn.module, fn)
                    if r and r[0] in ("cls", "builtin"):
                        caught_types.setdefault(node.name, set()).add(r[1])

    def add(targets, handlers):
        for q, tier, via in targets or []:
            fn.edges.append((q, tier, handlers, via))

    def add_implicit(targets, handlers):
        add_implicit_edges(fn, targets, handlers)

    def visit(node, handlers):
        if node is not fn.node and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            for d in node.decorator_list:
                visit(d, handlers)
            if isinstance(node, ast.ClassDef):
                for b in node.bases:
                    visit(b, handlers)
            else:
                defaults = node.args.defaults + [
                    d for d in node.args.kw_defaults if d is not None
                ]
                for d in defaults:
                    visit(d, handlers)
            return
        add_implicit(implicit_edges(index, fn, node), handlers)
        if isinstance(node, ast.Try) or (
            hasattr(ast, "TryStar") and isinstance(node, ast.TryStar)
        ):
            inner = handlers + tuple(
                handler_key(index, h, fn, node) for h in node.handlers
            )
            for s in node.body:
                visit(s, inner)
            for h in node.handlers:
                outer = bound.get(h.name)
                if h.name:
                    bound[h.name] = handler_id(fn, h)
                for s in h.body:
                    visit(s, handlers)
                if h.name:
                    bound.pop(h.name)
                    if outer is not None:
                        bound[h.name] = outer
            for s in node.orelse + node.finalbody:
                visit(s, handlers)
            return
        if isinstance(node, ast.Raise) and node.exc is not None:
            site = raise_site(
                index,
                fn,
                node,
                handlers,
                assigned,
                caught_names,
                instance_coded,
                caught_types,
                bound,
            )
            if site is not None:
                fn.sites.append(site)
        if isinstance(node, ast.Assert):
            message = node.msg if node.msg else node.test
            fn.sites.append(
                {
                    "kind": "assert",
                    "line": node.lineno,
                    "classes": ["builtins.AssertionError"],
                    "message": ast.unparse(message)[:300],
                    "coded": None,
                    "how": "assert",
                    "handlers": handlers,
                }
            )
        if isinstance(node, ast.Call):
            visit_call(index, fn, node, handlers, visit, add, add_implicit)
            return
        if isinstance(node, ast.AnnAssign):
            for child in (node.target, node.value):
                if child is not None:
                    visit(child, handlers)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            r = index.resolve_name(node.id, fn.module, fn)
            if r and r[0] == "fn":
                add([(r[1], "resolved", "reference")], handlers)
            elif r and r[0] == "table":
                add(index.table_targets(r), handlers)
                fn.instantiates |= index.table_classes(r)
            elif r and r[0] == "cls":
                # A class passed as a value may be constructed elsewhere.
                fn.instantiates.add(r[1])
            return
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            r = index.resolve_expr(node, fn.module, fn)
            if r and r[0] == "cls":
                fn.instantiates.add(r[1])
            add(index.attribute_refs(node, fn), handlers)
            namespace = index.resolve_expr(node.value, fn.module, fn)
            if not (namespace and namespace[0] == "cls"):
                # `C.attr` does not construct C.
                visit(node.value, handlers)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, handlers)

    for stmt in fn.node.body:
        visit(stmt, ())


def visit_call(index, fn, node, handlers, visit, add, add_implicit):
    """A call's edges, then its callee expression and its arguments."""

    fn.instantiates |= index.constructed(node.func, fn)
    targets = index.callee_targets(node.func, fn)
    if targets is None:
        fn.unresolved_calls += 1
    add(targets, handlers)
    func = node.func
    # Calling an instance runs its ``__call__``: a callee that resolves to no function,
    # or an attribute that is no method of its typed receiver (``self.f(...)`` with
    # ``self.f`` holding an instance).
    if targets is None or (not targets and isinstance(func, ast.Attribute)):
        add_implicit(
            index.implicit_targets(
                func, "__call__", fn, typed_only=targets is not None
            ),
            handlers,
        )
    if isinstance(func, ast.Attribute):
        # The attribute read itself; the callee expression is not visited as one.
        add_implicit(index.implicit_targets(func.value, "getattr", fn), handlers)
        namespace = index.resolve_expr(func.value, fn.module, fn)
        if not (namespace and namespace[0] == "cls"):
            # `C.m(...)` does not construct C.
            visit(func.value, handlers)
    elif (
        isinstance(func, ast.Call)
        and isinstance(func.func, ast.Name)
        and func.func.id == "getattr"
    ):
        for a in func.args:
            visit(a, handlers)
    elif not isinstance(func, ast.Name):
        visit(func, handlers)
    # A class named as isinstance's / issubclass's second argument is only tested,
    # not constructed.
    type_tests = (
        ("builtin", "builtins.isinstance"),
        ("builtin", "builtins.issubclass"),
    )
    type_test = (
        isinstance(func, ast.Name)
        and index.resolve_name(func.id, fn.module, fn) in type_tests
    )
    for i, a in enumerate(node.args):
        if not (type_test and i >= 1):
            visit(a, handlers)
    for k in node.keywords:
        visit(k.value, handlers)


def raise_site(
    index,
    fn,
    node,
    handlers,
    assigned,
    caught_names,
    instance_coded,
    caught_types,
    bound,
):
    exc = node.exc
    call, how, coded, reraise = None, "constructed at the raise", None, None
    if isinstance(exc, ast.Call):
        call = exc
    elif isinstance(exc, ast.Name):
        if exc.id in caught_names and exc.id not in assigned:
            return None
        calls = [v for v in assigned.get(exc.id, []) if isinstance(v, ast.Call)]
        if calls:
            call, how = calls[-1], f"assigned to {exc.id}"
        coded = instance_coded.get(exc.id)
    elif (
        isinstance(exc, ast.Attribute)
        and isinstance(exc.value, ast.Name)
        and exc.value.id in caught_names
    ):
        # ``raise failure.failure``: the exception a caught wrapper holds.
        hids = [bound[exc.value.id]] if exc.value.id in bound else []
        reraise = ("field", hids or fn.bound_by[exc.value.id], exc.attr)
        how = f"the exception held in {ast.unparse(exc)}"
    target = call.func if call is not None else exc
    classes, direct = set(), False
    r = index.resolve_expr(target, fn.module, fn)
    if is_exception_class(r):
        classes, direct = {r[1]}, call is not None
    elif isinstance(target, ast.Name) and target.id in assigned:
        # ``cls = TABLE[name]`` or ``TABLE.get(name, Default)``, then ``raise cls(...)``.
        classes = table_lookup_classes(index, fn, assigned[target.id])
        if classes:
            direct = call is not None
            how = f"a class looked up in a table, {target.id}"
    if not classes and not reraise and call is not None:
        for q, tier, _ in index.callee_targets(call.func, fn) or []:
            if tier != "resolved":
                continue
            if index.factory_classes(q):
                classes |= index.factory_classes(q)
                how = f"returned by {q}"
            for name, position in index.factory_params(q):
                if position < len(call.args) and not isinstance(
                    call.args[position], ast.Starred
                ):
                    passed = call.args[position]
                else:
                    passed = next(
                        (k.value for k in call.keywords if k.arg == name), None
                    )
                rp = (
                    index.resolve_expr(passed, fn.module, fn)
                    if passed is not None
                    else None
                )
                if is_exception_class(rp):
                    classes.add(rp[1])
                    how = f"built by {q} from its argument {name}"
    elif not classes and not reraise and isinstance(exc, ast.Name):
        # `raise kept` where kept was assigned an exception an `except ... as` caught:
        # it re-raises what that handler caught.
        for v in assigned.get(exc.id, []):
            if isinstance(v, ast.Name) and v.id in caught_types:
                classes |= caught_types[v.id]
                how = f"a caught exception kept in {exc.id}"
                reraise = ("handler", list(fn.bound_by[v.id]), None)
    classes = sorted(classes) or [f"?{ast.unparse(target)[:80]}"]
    if coded is None and reraise is None:
        coded, note = site_code(index, call if direct else None, classes)
        how += note
    if call is not None and call.args:
        message = ast.unparse(call.args[0])[:300]
    elif call is not None:
        message = "; ".join(f"{k.arg}={ast.unparse(k.value)}" for k in call.keywords)
        message = message[:300]
    else:
        message = ""
    return {
        "kind": "raise",
        "line": node.lineno,
        "classes": classes,
        "message": message,
        "coded": coded,
        "how": how,
        "handlers": handlers,
        "reraise": reraise,
        # For a wrapper constructed at the raise: its arguments, and which handler
        # binds each ``except ... as`` name there.
        "call": call if direct else None,
        "bound": dict(bound),
    }


def site_code(index, call, classes):
    """(how every class of a raise gives it a code, or None; why not, for ``how``).
    ``call`` constructs the class, or is None when the raise does not show the
    arguments the class receives.  A ``code=`` must be a literal and the class must
    store it."""

    found, notes = [], []
    passed_code = call is not None and any(k.arg == "code" for k in call.keywords)
    for c in classes:
        short = c.rsplit(".", 1)[-1]
        if c not in index.classes:
            notes.append(f"{short} is not a package class")
            found.append(None)
            continue
        definite, alternatives = index.coded_class(c)
        code_position = [p for name, p, _ in alternatives if name == "code"]
        if passed_code and not code_position:
            notes.append(f"{short} does not store code=")
            found.append(None)
            continue
        if passed_code and not passes_argument(call, "code", code_position[0]):
            notes.append("code= is not a literal")
            found.append(None)
            continue
        passed = [
            f"passes {name} ({where})"
            for name, position, where in alternatives
            if call is not None and passes_argument(call, name, position) is True
        ]
        if not (passed or definite):
            missing = ", ".join(where for _, _, where in alternatives)
            notes.append(f"{missing} not passed" if missing else f"{short} uncoded")
        found.append(passed[0] if passed else definite)
    coded = "; ".join(found) if found and all(found) else None
    return coded, "".join(f"; {n}" for n in notes)


def table_lookup_classes(index, fn, values):
    """The classes a local holds when every assignment to it is a package class, a
    lookup in a module-level table (``TABLE[key]``, ``TABLE.get(key, Default)``) or
    ``a or b`` of those; else the empty set."""

    out = set()
    for v in values:
        parts = v.values if isinstance(v, ast.BoolOp) else [v]
        for p in parts:
            default = None
            if (
                isinstance(p, ast.Call)
                and isinstance(p.func, ast.Attribute)
                and p.func.attr == "get"
            ):
                table, default = p.func.value, p.args[1] if len(p.args) > 1 else None
            elif isinstance(p, ast.Subscript):
                table = p.value
            else:
                table = p
            r = index.resolve_expr(table, fn.module, fn)
            if r and r[0] == "table" and index.table_classes(r):
                out |= index.table_classes(r)
            elif is_exception_class(r):
                out.add(r[1])
            else:
                return set()
            if default is not None:
                d = index.resolve_expr(default, fn.module, fn)
                if not is_exception_class(d):
                    return set()
                out.add(d[1])
    return out


def catches(index, handlers, classes):
    """The handlers catch every one of the classes: within one ``try`` the first
    handler naming the class takes it, and when that handler re-raises the exception
    goes on to the enclosing ``try``."""

    groups = []
    for h in handlers:
        try_id = index.handlers[h[2]][2]
        if groups and groups[-1][0] == try_id:
            groups[-1][1].append(h)
        else:
            groups.append((try_id, [h]))

    def caught(c):
        for _, group in reversed(groups):
            first = next((h for h in group if names_class(index, h[0], c)), None)
            if first is not None and not first[1]:
                return True
        return False

    return all(caught(c) for c in classes)


def reaches_handler(index, handlers, hid, classes):
    """An exception of one of the classes, raised under ``handlers``, reaches the
    handler ``hid``: hid names the class, no handler of hid's ``try`` listed before
    it names it, and no ``try`` nested in hid's ``try`` body catches it."""

    ids = [h[2] for h in handlers]
    if hid not in ids:
        return False
    i = ids.index(hid)
    _, quals, try_id = index.handlers[hid]
    same_try = [h for h in handlers[:i] if index.handlers[h[2]][2] == try_id]
    nested = [h for h in handlers[i + 1 :] if index.handlers[h[2]][2] != try_id]
    return any(
        names_class(index, quals, c)
        and not any(names_class(index, h[0], c) for h in same_try)
        and not catches(index, nested, (c,))
        for c in classes
    )


def names_class(index, quals, cls_qual):
    """A handler for ``quals`` (None for a bare except) names the class or a base."""

    if quals is None:
        return True
    if cls_qual.startswith("?"):
        return any(q in ("builtins.Exception", "builtins.BaseException") for q in quals)
    mro = set(index.mro(cls_qual))
    return any(q in mro for q in quals)


def provisional_kind(index, classes, message):
    names = set()
    for c in classes:
        if not c.startswith("?"):
            names |= {q.rsplit(".", 1)[-1] for q in index.mro(c)}
    if "AssertionError" in names:
        return "internal"
    if "NotImplementedError" in names or UNSUPPORTED_TEXT.search(message or ""):
        return "unsupported"
    if names & INVALID_BASES:
        return "invalid input"
    if "RuntimeError" in names and INTERNAL_TEXT.search(message or ""):
        return "internal"
    return "unclassified"


# -- entries, reach, escape -----------------------------------------------------------


def roots_of(index, pkg_name, public_class, extra=()):
    init = index.modules[pkg_name]
    exported = []
    for node in ast.walk(init.tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            exported = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    roots, missing = [], []
    for name in exported:
        r = index.lookup(pkg_name, name)
        if r and r[0] == "fn":
            roots.append((r[1], f"{pkg_name}.{name}"))
        elif r and r[0] == "cls" and r[1].rsplit(".", 1)[-1] == public_class:
            for mname, m in sorted(index.classes[r[1]].methods.items()):
                dunder = mname.startswith("__") and mname.endswith("__")
                if (dunder or not mname.startswith("_")) and mname not in (
                    "__repr__",
                    "__str__",
                ):
                    roots.append((m.qual, f"{public_class}.{mname}"))
    for qual, label in extra:
        if qual in index.fns:
            roots.append((qual, label))
        else:
            missing.append(qual)
    return roots, missing


def propagate(index, roots, tiers, removed):
    """{function: bit mask of the entries that reach it}, over edges of the given
    tiers, without the edges in ``removed``."""

    mask = collections.defaultdict(int)
    work = collections.deque()
    for i, (qual, _) in enumerate(roots):
        mask[qual] |= 1 << i
        work.append(qual)
    queued = set(work)
    while work:
        g = work.popleft()
        queued.discard(g)
        mg = mask[g]
        for eid, (target, tier, _, _) in enumerate(index.fns[g].edges):
            if tier not in tiers or (g, eid) in removed:
                continue
            old = mask[target]
            new = old | mg
            if new != old:
                mask[target] = new
                if target not in queued:
                    work.append(target)
                    queued.add(target)
    return mask


def shortest_paths(index, roots, tiers):
    parent, queue = {}, collections.deque()
    for qual, _ in roots:
        if qual not in parent:
            parent[qual] = None
            queue.append(qual)
    while queue:
        g = queue.popleft()
        for target, tier, _, _ in index.fns[g].edges:
            if tier in tiers and target not in parent:
                parent[target] = g
                queue.append(target)

    def path(q):
        if q not in parent:
            return None
        out = []
        while q is not None:
            out.append(q)
            q = parent[q]
        return out[::-1]

    return path


def labels(mask, roots):
    return sorted({roots[i][1] for i in range(len(roots)) if mask >> i & 1})


DISPATCHED = {"self", "typed receiver", "self attribute", "typed attribute"}


def dispatched(via):
    """An edge that runs a method on an instance: ``self.m``, ``x.m`` with x typed or
    untyped, a property read, a ``getattr`` call, or an implicit special-method
    call."""

    return (
        via in DISPATCHED
        or via.startswith(".")
        or via.startswith("getattr")
        or via.startswith("implicit ")
    )


def rapid_type_analysis(index, roots):
    """Keep an edge that runs a method on an instance only when the method's class, or
    a subclass, is constructed in code reachable from the entries (rapid type
    analysis, Bacon and Sweeney 1996): reach and the set of constructed classes grow
    together to a fixed point, then the other such edges are dropped.  A class counts
    as constructed when reachable code calls it (directly, as ``cls(...)`` in a
    classmethod or ``type(self)(...)``) or passes it as a value (alone or in a
    module-level table), but not when it only names it in ``isinstance`` /
    ``issubclass``, in an annotation, or as the namespace of ``C.attr``.  Reachable code
    is the code the entries reach and the import-time code of the modules that are
    imported (``index.import_fns`` and what it reaches): a module-level ``X = C(...)``
    constructs C when its module is imported -- by the package's own import or by an
    import in a function that runs.  An entry that is an instance method seeds its
    class.  Calls of functions, constructors, ``super()`` and ``C.m(...)`` are kept as
    they are."""

    def alive(q, constructed):
        c = index.fns[q].cls
        return (
            c is None
            or c.qual in constructed
            or bool(index.subclasses.get(c.qual, set()) & constructed)
        )

    def keep(edge, constructed):
        return not dispatched(edge[3]) or alive(edge[0], constructed)

    constructed = {
        index.fns[q].cls.qual
        for q, _ in roots
        if index.fns[q].cls is not None
        and index.fns[q].self_name is not None
        and not index.fns[q].is_classmethod
    }

    def reach(start, constructed):
        seen, frontier = set(start), list(start)
        while frontier:
            nxt = []
            for g in frontier:
                for e in index.fns[g].edges:
                    if e[0] not in seen and keep(e, constructed):
                        seen.add(e[0])
                        nxt.append(e[0])
            frontier = nxt
        return seen

    imported = index.imported_closure({index.pkg_name})
    while True:
        seen = reach({q for q, _ in roots}, constructed)
        import_code = [f for f in index.import_fns if f.module.name in imported]
        at_import = reach(
            {e[0] for f in import_code for e in f.edges if keep(e, constructed)},
            constructed,
        )
        grown = set().union(
            *(index.fns[q].instantiates for q in seen | at_import),
            *(f.instantiates for f in import_code),
        )
        more = index.imported_closure(
            imported | index.function_imports(seen | at_import)
        )
        if grown <= constructed and more == imported:
            break
        constructed |= grown
        imported = more
    dropped = collections.Counter()
    for fn in index.fns.values():
        kept = [e for e in fn.edges if keep(e, constructed)]
        for e in fn.edges:
            if not keep(e, constructed):
                dropped[e[1]] += 1
        fn.edges = kept
    index.rta = {
        "constructed classes": len(constructed),
        "edges dropped by tier": dict(dropped),
        "functions reached": len(seen),
        "functions run at import": len(at_import),
        "modules imported": len(imported),
    }
    return constructed


class Reraises:
    """Whether a re-raise of a caught exception is coded: it is when every site whose
    exception can reach the handler that caught it is coded, and there is at least
    one.  ``raise kept`` re-raises what one handler caught; ``raise w.field``
    re-raises the exception that a wrapper caught in ``w`` stores in ``field``, so its
    sites are found through the wrapper's constructions that reach w's handler, from
    the argument each passes for the ``__init__`` parameter stored in ``field``.  A
    construction or argument the analysis cannot follow leaves the re-raise
    uncoded."""

    def __init__(self, index, removed_for):
        self.index, self.removed_for = index, removed_for
        self.producer_cache, self.reach_cache, self.result = {}, {}, {}

    def reach(self, hid, key):
        """Functions an exception of the classes ``key`` can come from and reach the
        handler ``hid`` through calls in its ``try`` body."""

        if (hid, key) in self.reach_cache:
            return self.reach_cache[(hid, key)]
        g = self.index.fns[self.index.handlers[hid][0]]
        removed = self.removed_for(key)
        seen = {
            target
            for target, tier, h, _ in g.edges
            if tier in NAME_ONLY and reaches_handler(self.index, h, hid, key)
        }
        work = list(seen)
        while work:
            q = work.pop()
            for eid, (target, tier, _, _) in enumerate(self.index.fns[q].edges):
                if tier in NAME_ONLY and (q, eid) not in removed and target not in seen:
                    seen.add(target)
                    work.append(target)
        self.reach_cache[(hid, key)] = seen
        return seen

    def producers(self, hid):
        """[(function, site)] for the sites whose exception can reach ``hid``."""

        if hid in self.producer_cache:
            return self.producer_cache[hid]
        index = self.index
        g_qual, quals, _ = index.handlers[hid]
        out = [
            (g_qual, s)
            for s in index.fns[g_qual].sites
            if reaches_handler(index, s["handlers"], hid, tuple(s["classes"]))
        ]
        seen = {id(s) for _, s in out}
        for q in sorted(self.reach_any(hid)):
            for s in index.fns[q].sites:
                key = tuple(s["classes"])
                if (
                    id(s) not in seen
                    and any(names_class(index, quals, c) for c in key)
                    and not catches(index, s["handlers"], key)
                    and q in self.reach(hid, key)
                ):
                    out.append((q, s))
                    seen.add(id(s))
        self.producer_cache[hid] = out
        return out

    def reach_any(self, hid):
        """Every function a call in ``hid``'s ``try`` body reaches, handlers aside: the
        functions ``reach`` can return for any class."""

        g = self.index.fns[self.index.handlers[hid][0]]
        seen = {
            target
            for target, tier, h, _ in g.edges
            if tier in NAME_ONLY and hid in [x[2] for x in h]
        }
        work = list(seen)
        while work:
            q = work.pop()
            for target, tier, _, _ in self.index.fns[q].edges:
                if tier in NAME_ONLY and target not in seen:
                    seen.add(target)
                    work.append(target)
        return seen

    def field_producers(self, hids, attr, visiting):
        """The sites whose exception a wrapper caught in one of ``hids`` holds in
        ``attr``, or None when a construction or argument cannot be followed."""

        index, out = self.index, []
        for hid in hids:
            if (hid, attr) in visiting:
                continue
            visiting.add((hid, attr))
            for _, p in self.producers(hid):
                call = p.get("call")
                if call is None:
                    return None
                for c in p["classes"]:
                    param = index.stored_param(c, attr)
                    if param is None:
                        return None
                    v = argument_value(call, *param)
                    if isinstance(v, ast.Name) and v.id in p["bound"]:
                        out += self.producers(p["bound"][v.id])
                    elif (
                        isinstance(v, ast.Attribute)
                        and isinstance(v.value, ast.Name)
                        and v.value.id in p["bound"]
                    ):
                        inner = self.field_producers(
                            [p["bound"][v.value.id]], v.attr, visiting
                        )
                        if inner is None:
                            return None
                        out += inner
                    else:
                        return None
        return out

    def resolve(self, s, active=()):
        """Set ``coded``, ``how`` and ``reraise_uncoded`` on a re-raise site; return
        whether the site is coded.  A site already being resolved counts as coded: a
        cycle adds no exception of its own."""

        if s.get("reraise") is None:
            return bool(s["coded"])
        if id(s) in self.result:
            return self.result[id(s)]
        if id(s) in active:
            return True
        kind, hids, attr = s["reraise"]
        if kind == "handler":
            found = [p for hid in hids for p in self.producers(hid)]
        else:
            found = self.field_producers(hids, attr, set())
        if found:
            # A site can reach the handler through several wrapper constructions.
            found = list({id(p): (q, p) for q, p in found}.values())
        uncoded = []
        if found is None:
            s["how"] += "; a wrapper construction the analysis cannot follow"
        elif not found:
            s["how"] += "; no site reaches the handler"
        else:
            uncoded = [
                (q, p) for q, p in found if not self.resolve(p, active + (id(s),))
            ]
            if not uncoded:
                s["coded"] = f"re-raises {len(found)} sites, all coded"
            else:
                s["how"] += f"; {len(uncoded)} of the {len(found)} sites it re-raises"
                s["how"] += " are uncoded"
        s["reraise_uncoded"] = [(q, p["line"]) for q, p in uncoded]
        self.result[id(s)] = bool(s["coded"])
        return self.result[id(s)]


def inventory(pkg_dir, pkg_name, public_class, extra=()):
    """Every raise and assert site in the package at ``pkg_dir``, with whether it is
    coded and which entries it can escape to."""

    index = Index(pkg_dir, pkg_name)
    roots, missing = roots_of(index, pkg_name, public_class, extra)
    rapid_type_analysis(index, roots)
    full = {t: propagate(index, roots, t, frozenset()) for t in (RESOLVED, NAME_ONLY)}
    path_of = {t: shortest_paths(index, roots, t) for t in (RESOLVED, NAME_ONLY)}
    handler_edges = [
        (g.qual, eid, h)
        for g in index.fns.values()
        for eid, (_, _, h, _) in enumerate(g.edges)
        if h
    ]
    removed_by_classes, masks = {}, {}

    def removed_for(key):
        if key not in removed_by_classes:
            removed_by_classes[key] = frozenset(
                (q, eid) for q, eid, h in handler_edges if catches(index, h, key)
            )
        return removed_by_classes[key]

    reraises = Reraises(index, removed_for)
    for fn in index.fns.values():
        for s in fn.sites:
            reraises.resolve(s)
    sites = []
    for fn in index.fns.values():
        for s in fn.sites:
            key = tuple(s["classes"])
            removed = removed_for(key)
            for t in (RESOLVED, NAME_ONLY):
                if (removed, t) not in masks:
                    masks[(removed, t)] = propagate(index, roots, t, removed)
            local = catches(index, s["handlers"], key)
            reach_res = full[RESOLVED].get(fn.qual, 0)
            reach_any = full[NAME_ONLY].get(fn.qual, 0)
            esc_res = 0 if local else masks[(removed, RESOLVED)].get(fn.qual, 0)
            esc_any = 0 if local else masks[(removed, NAME_ONLY)].get(fn.qual, 0)
            path = path_of[RESOLVED](fn.qual) or path_of[NAME_ONLY](fn.qual)
            if reach_res:
                reach = "resolved"
            else:
                reach = "method name only" if reach_any else "not reached"
            if esc_res:
                escape_reach = "resolved"
            else:
                escape_reach = "method name only" if esc_any else "none"
            sites.append(
                {
                    "file": os.path.relpath(fn.module.path, os.path.dirname(pkg_dir)),
                    "line": s["line"],
                    "function": fn.qual,
                    "kind": s["kind"],
                    "class": "|".join(key),
                    "coded": s["coded"],
                    "message": s["message"],
                    "how": s["how"],
                    "caught_in_function": local,
                    "reach": reach,
                    "entries": labels(reach_res or reach_any, roots),
                    "escape_reach": escape_reach,
                    "escapes_to": labels(esc_res or esc_any, roots),
                    "path": path,
                    "depth": len(path or [None]) - 1,
                    "kind_provisional": provisional_kind(
                        index, list(key), s["message"]
                    ),
                    "unmodeled_special": unmodeled_special(fn),
                    # For a re-raise: the uncoded sites it re-raises, as file:line.
                    "reraise_uncoded": [
                        os.path.relpath(
                            index.fns[q].module.path, os.path.dirname(pkg_dir)
                        )
                        + f":{line}"
                        for q, line in s.get("reraise_uncoded", [])
                    ],
                }
            )
    return Inventory(index, roots, missing, sites, len(masks))


MODELED_SPECIAL = (
    {name for alts in IMPLICIT_PROTOCOLS.values() for alt in alts for name in alt}
    | CONSTRUCTOR_METHODS
    | {"__class_getitem__"}
)


def unmodeled_special(fn):
    """The function is a special method Python can call that no edge models (a copy or
    pickle hook, a descriptor method, ``__init_subclass__``); ``__del__`` aside."""

    name = fn.node.name
    return (
        fn.cls is not None
        and name.startswith("__")
        and name.endswith("__")
        and name not in MODELED_SPECIAL
        and name not in NOT_PROPAGATED
    )


def in_scope_entries(roots):
    """The entry labels the gate covers: STensor's methods, spmv and SCOPE_EXPORTED."""

    return {
        label
        for _, label in roots
        if label.startswith("STensor.")
        or label == "scorch.ops.spmv"
        or label.split(".", 1)[1] in SCOPE_EXPORTED
    }


def scorch_inventory(pkg_dir):
    """The inventory of the scorch package at ``pkg_dir``; each site also records the
    in-scope entries it escapes to (``escapes_in_scope``)."""

    result = inventory(
        pkg_dir, "scorch", "STensor", [("scorch.ops.spmv", "scorch.ops.spmv")]
    )
    in_scope = in_scope_entries(result.roots)
    for s in result.sites:
        s["escapes_in_scope"] = [g for g in s["escapes_to"] if g in in_scope]
    return result


def uncoded_escaping(sites):
    """The sites that carry no code and escape to an in-scope entry."""

    return [s for s in sites if not s["coded"] and s["escapes_in_scope"]]


def uncoded_unmodeled(sites):
    """The uncoded sites in a special method the call graph does not model, whatever
    their reach: the analysis cannot tell which entries reach them."""

    return [s for s in sites if not s["coded"] and s["unmodeled_special"]]
