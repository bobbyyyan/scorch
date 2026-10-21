"""Float64 twins of the float32 tests in test_loop_invariant_factor_pass.py.

Loop-invariant factor hoisting does not depend on which of FLOAT32 and FLOAT64
the accumulator carries: the same LLIR with the other scalar type must hoist
the same factors, in the same order, under the same names, and differ only in
the scalar type the temporary is declared with.

This module states that as a metamorphic test.  It takes the source of the
float32 tests, replaces every ``DataType.FLOAT32`` by ``DataType.FLOAT64`` and
every expected C++ declaration that starts ``float `` by one that starts
``double ``, and runs the result.  A float64 accumulator is therefore held to
every assertion a float32 one is, including assertions added to that module
later, without a second copy of them to keep in step.

The tests after the marker line below are left out: they are parameterized
over both scalar types already.  The module-overhead latency test is left out
too: it measures the pass manager, not the scalar type.

A failure is reported at the line of the float32 source it was derived from.
"""

from __future__ import annotations

import pathlib
import re

_FLOAT32_SOURCE = pathlib.Path(__file__).with_name("test_loop_invariant_factor_pass.py")
_SCALAR_TYPE_SECTION = "# -- the hoisted factor's scalar type"
_FLOAT32_SCALAR = re.compile(r"\bDataType\.FLOAT32\b")
_FLOAT_DECLARATION = re.compile(r"""(?<=["'])float """)


def _float64_twin_source() -> tuple[str, int, int]:
    text = _FLOAT32_SOURCE.read_text()
    assert text.count(_SCALAR_TYPE_SECTION) == 1, "the section marker moved"
    float32_tests = text[: text.index(_SCALAR_TYPE_SECTION)]
    retyped, scalar_count = _FLOAT32_SCALAR.subn("DataType.FLOAT64", float32_tests)
    twin, declaration_count = _FLOAT_DECLARATION.subn("double ", retyped)
    return twin, scalar_count, declaration_count


_TWIN_SOURCE, _SCALAR_COUNT, _DECLARATION_COUNT = _float64_twin_source()
# Substitutions never add or remove a line, so line numbers are the source's.
exec(compile(_TWIN_SOURCE, str(_FLOAT32_SOURCE), "exec"), globals())

for _name, _value in list(globals().items()):
    if _name.startswith("test_") and "perf_counter_ns" in getattr(
        getattr(_value, "__code__", None), "co_names", ()
    ):
        del globals()[_name]


def test_twin_source_carries_no_float32_scalar_and_no_float_declaration() -> None:
    # A twin that substituted nothing would be the float32 test under another
    # name, so the substitutions are counted and their targets shown absent.
    assert _SCALAR_COUNT > 0
    assert _DECLARATION_COUNT > 0
    assert _FLOAT32_SCALAR.search(_TWIN_SOURCE) is None
    assert _FLOAT_DECLARATION.search(_TWIN_SOURCE) is None
    assert "double _inv_" in _TWIN_SOURCE
    assert "float _inv_" not in _TWIN_SOURCE


def test_every_float32_test_that_pins_a_float_factor_has_a_twin() -> None:
    float32_tests = _FLOAT32_SOURCE.read_text()
    float32_tests = float32_tests[: float32_tests.index(_SCALAR_TYPE_SECTION)]
    pinned = {
        match.group(1)
        for match in re.finditer(
            r"^def (test_\w+)\(.*?(?=^def |\Z)",
            float32_tests,
            flags=re.MULTILINE | re.DOTALL,
        )
        if "float _inv_" in match.group(0)
    }
    assert len(pinned) >= 20
    missing = sorted(name for name in pinned if not callable(globals().get(name)))
    assert missing == []
