"""The name of a merged coordinate (``scorch.compiler.merged_coordinate``).

A loop that merges sparse operands on an index declares each operand's current
coordinate as ``<index>_<tensor>``.  einsum takes the 52 ASCII letters as index letters
and names its operands ``A``, ``B``, ...; with the index ``M`` its fifth operand's
coordinate was ``M_E``, which math.h defines as a number, and the kernel did not build
(``error: expected unqualified-id before numeric constant``).  One capital, an
underscore and capitals is how the C library spells its constants: ``M_PI``, ``F_OK``,
``S_IFMT``.  An index name without a lower-case letter now gets the suffix ``_coord``.

Four things are held here.

* The rule itself.
* Both compilers spell a merged coordinate through the one function, where they reserve
  the name and where they emit it, and no kernel spells one any other way: with the
  function replaced by a marker, no kernel text holds the plain spelling, and putting
  the plain spelling back for the marker gives the kernel's own text.
* The macros of a generated kernel's translation unit (the preprocessor's list for a
  kernel built here): none has the spelling of an identifier a kernel builds from a
  tensor name, for tensor names of every length.  The same list read by the old rule
  finds ``M_E``, ``M_PI`` and ``F_OK``, which shows the list is one that can collide.
  And for each of the 52 index letters as the merged index, no name the emitted kernel
  declares is in that list (nothing built), in both compilers; with the old spelling
  the same check finds ``M_E``.
* Executed: the calls that did not build, their controls, the index letters that are
  also a tensor's name on five operands (``A`` to ``F``, and ``M``), and an upper-case
  and a lower-case spelling of one letter in one call.  float64 against ``torch.einsum``
  to 1e-12, integers exactly; the result format is named; small and larger extents;
  every sparse operand has an empty first row or entry.
"""

import itertools
import re
import shlex
import string
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import scorch
from scorch import ops
from scorch.compiler import iterator
from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir import lower_llir
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.merged_coordinate import merged_coordinate_name
from scorch.compiler.scheduler import Schedule
from scorch.layout import TensorSpec
from tests.test_scorch.kernel_cache_support import empty_kernel_caches
from tests.test_scorch.level_array_support import assert_matches, reference, stored

SMALL = dict(i=5, m=7, M=7)
LARGER = dict(i=37, m=67, M=67)
TENSOR_NAMES = ("A", "B", "E", "Z", "AA", "OK", "PI", "IFMT", "Result")


# --------------------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("index", list(string.ascii_lowercase) + ["i0", "row", "jK"])
def test_an_index_name_with_a_lower_case_letter_keeps_the_spelling(index):
    for tensor in TENSOR_NAMES:
        assert merged_coordinate_name(index, tensor) == f"{index}_{tensor}"


@pytest.mark.parametrize("index", list(string.ascii_uppercase) + ["K1", "IJ", "M_"])
def test_an_index_name_without_a_lower_case_letter_gets_the_suffix(index):
    for tensor in TENSOR_NAMES:
        assert merged_coordinate_name(index, tensor) == f"{index}_{tensor}_coord"


def test_the_names_einsum_can_ask_for_are_all_different():
    """Every index letter with each of einsum's first 80 tensor names."""

    tensors = [
        "".join(letters)
        for length in (1, 2)
        for letters in itertools.product(string.ascii_uppercase, repeat=length)
    ][:80]
    names = {
        merged_coordinate_name(index, tensor)
        for index in string.ascii_letters
        for tensor in tensors
    }
    assert len(names) == 52 * 80
    assert not any(re.fullmatch(r"[A-Z0-9_]+", name) for name in names)


# --------------------------------------------------------------------------------------
# Emitting a kernel through each compiler, nothing built
# --------------------------------------------------------------------------------------


def terms_of(expression):
    return expression.split("->")[0].split(",")


def shipped_text(monkeypatch, expression, formats, result_format, names, extents):
    """The kernel public einsum emits.  einsum names the tensors itself: ``names`` only
    says what it will call them, A, B, ... in order."""

    assert tuple(names) == tuple(string.ascii_uppercase[: len(names)])
    captured = []

    def stop_before_native(prepared):
        captured.append(prepared.request.cpp_sources[-1])
        return object()

    empty_kernel_caches(monkeypatch)
    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", stop_before_native)
    specs = [
        TensorSpec(
            fmt,
            tuple(extents[letter] for letter in term),
            dtype=torch.float64,
            name=name,
        )
        for term, fmt, name in zip(terms_of(expression), formats, names)
    ]
    ops.einsum(expression, *specs, compile_only=True, format=result_format)
    assert len(captured) == 1, f"{len(captured)} kernels prepared"
    return captured[0]


def cin_statement(expression, formats, result_format, names, result_name="Y"):
    """``expression`` as index notation, with the operands named ``names``."""

    assert result_name not in names and len(set(names)) == len(names)
    terms = terms_of(expression)
    output = expression.split("->")[1]
    letters = "".join(dict.fromkeys("".join(terms)))
    variables = {letter: IndexVar(letter) for letter in letters}
    product = None
    for term, fmt, name in zip(terms, formats, names):
        access = TensorVar(name, fmt=fmt, dtype=torch.float64)[
            tuple(variables[letter] for letter in term)
        ]
        product = (
            access if product is None else CINBinaryOp(Operation.MUL, product, access)
        )
    result = TensorVar(result_name, fmt=result_format, dtype=torch.float64)[
        tuple(variables[letter] for letter in output)
    ]
    statement = TensorAssign(result, product, op=Operation.ADD)
    for letter in reversed(letters):
        statement = ForAll(variables[letter], statement)
    return statement


def new_options():
    return replace(
        CompileOptions.from_environment().with_regblock_enabled(False),
        requested_schedule=Schedule(),
    )


def new_text(
    monkeypatch, expression, formats, result_format, names, extents, result_name="Y"
):
    """The kernel the new compiler emits for the same program (it builds nothing)."""

    output = expression.split("->")[1]
    kernel = compile_cin_via_loopir(
        cin_statement(expression, formats, result_format, names, result_name),
        tuple(extents[letter] for letter in output),
        [
            (tuple(extents[letter] for letter in term), torch.float64)
            for term in terms_of(expression)
        ],
        compile_options=new_options(),
    )
    return kernel.cpp_source


EMIT = {"shipped": shipped_text, "new": new_text}

# Programs that merge sparse operands on an index: (expression, operand formats, result
# format).  The index letters are lower case here; UPPER maps them to capitals.
MERGES = [
    ("im,im->im", ("ds", "ds"), "ds"),
    ("im,im,im->im", ("ds", "ds", "ds"), "ds"),
    ("m,m->m", ("s", "s"), "s"),
    ("im,im->i", ("ds", "ds"), "d"),
]
UPPER = str.maketrans("m", "M")


def marker(index_name, tensor_name):
    return f"merged_{index_name}_of_{tensor_name}"


MARKER = re.compile(r"\bmerged_([A-Za-z0-9]+)_of_([A-Za-z0-9]+)\b")


@pytest.mark.parametrize("upper", [False, True], ids=["index m", "index M"])
@pytest.mark.parametrize("program", MERGES, ids=[p[0] for p in MERGES])
@pytest.mark.parametrize("compiler", ["shipped", "new"])
def test_a_kernel_spells_a_merged_coordinate_through_the_one_function(
    monkeypatch, compiler, program, upper
):
    expression, formats, result_format = program
    if upper:
        expression = expression.translate(UPPER)
    names = string.ascii_uppercase[: len(formats)]
    arguments = (expression, formats, result_format, names, SMALL)
    plain = EMIT[compiler](monkeypatch, *arguments)
    monkeypatch.setattr(iterator, "merged_coordinate_name", marker)
    monkeypatch.setattr(lower_llir, "merged_coordinate_name", marker)
    marked = EMIT[compiler](monkeypatch, *arguments)

    index = "M" if upper else "m"
    spelled = {merged_coordinate_name(index, name) for name in names}
    # the program does merge, and on every operand
    assert {f"merged_{index}_of_{name}" for name in names} <= set(
        match.group(0) for match in MARKER.finditer(marked)
    )
    # with the function replaced, no kernel text holds the plain spelling in either form
    for name in names:
        for spelling in (f"{index}_{name}", f"{index}_{name}_coord"):
            assert re.search(rf"\b{spelling}\b", marked) is None, spelling
    # and the function is all that changed
    restored = MARKER.sub(
        lambda match: merged_coordinate_name(match.group(1), match.group(2)), marked
    )
    assert restored == plain
    assert spelled <= set(re.findall(r"\b\w+\b", plain))


@pytest.mark.parametrize("compiler", ["shipped", "new"])
def test_the_fifth_operands_coordinate_of_index_M_is_not_spelled_M_E(
    monkeypatch, compiler
):
    """The shipped compiler through einsum, which names the fifth operand ``E``; the new
    one on two operands named ``A`` and ``E``."""

    if compiler == "shipped":
        text = shipped_text(
            monkeypatch, "M,M,M,M,M->M", ["s"] * 5, "s", "ABCDE", dict(M=40)
        )
    else:
        text = new_text(monkeypatch, "iM,iM->iM", ("ds", "ds"), "ds", "AE", SMALL)
    assert re.search(r"\bint M_E_coord = E\d_crd(_data)?\[", text), text[-3000:]
    assert re.search(r"\bM_E\b", text) is None


def test_the_new_compiler_reserves_the_name_it_emits(monkeypatch):
    """A third operand whose own name is the merged coordinate's is refused, in the
    spelling the kernel would use and in no other."""

    def lower(third_name, expression="iM,iM,iM->iM"):
        return new_text(
            monkeypatch,
            expression,
            ("ds", "ds", "dd"),
            "ds",
            ("A", "B", third_name),
            SMALL,
        )

    for taken in ("M_A_coord", "M_B_coord"):
        with pytest.raises(LoopIRTargetError) as error:
            lower(taken)
        assert error.value.defect.code == "generated_name_collision", taken
    # The spelling a lower-case index has is free under an upper-case one, and the
    # other way round.
    assert "M_A_coord" in lower("M_A")
    with pytest.raises(LoopIRTargetError) as error:
        lower("m_A", "im,im,im->im")
    assert error.value.defect.code == "generated_name_collision"
    assert "m_A" in lower("m_A_coord", "im,im,im->im")
    # Reserved through the one function: with it replaced, the marker is what is taken.
    monkeypatch.setattr(lower_llir, "merged_coordinate_name", marker)
    with pytest.raises(LoopIRTargetError) as error:
        lower("merged_M_of_A")
    assert error.value.defect.code == "generated_name_collision"
    assert "merged_M_of_A" in lower("M_A_coord")


def test_a_tensor_name_ending_in_a_digit_keeps_its_coordinate_apart(monkeypatch):
    """Tensors named ``B1`` and ``M_B``, merged on ``M``.  The coordinate of ``B1`` is
    ``M_B1_coord``; the level-1 coordinate array of ``M_B`` is ``M_B1_crd``.  Had the
    suffix been ``_crd`` the two would be one name."""

    text = new_text(monkeypatch, "iM,iM->iM", ("ds", "ds"), "ds", ("B1", "M_B"), SMALL)
    assert re.search(r"\bint M_B1_coord = B11_crd(_data)?\[", text), text[-3000:]
    assert re.search(r"\bint M_M_B_coord = M_B1_crd(_data)?\[", text), text[-3000:]


# --------------------------------------------------------------------------------------
# The macros of a kernel's translation unit
# --------------------------------------------------------------------------------------

# Every form in which a kernel spells an identifier from a tensor name N (L is a level
# number), counted over the kernel sources both compilers emit for the raw programs, the
# conversions and the sums.
TENSOR_SUFFIXES = (
    "values",
    "shape",
    "mode_indices",
    "val",
    "values_torch",
    "values_data",
    "val_vec",
    "val_tensor",
    "capacity",
    "reduction",
    "nonempty",
)
LEVEL_SUFFIXES = (
    "pos",
    "crd",
    "size",
    "pos_index",
    "pos_torch",
    "pos_data",
    "crd_torch",
    "crd_tensor",
    "crd_vec",
    "crd_data",
)
TENSOR_NAME = r"(?P<{group}>[A-Z][A-Za-z]*)"


def identifier_forms(merged_coordinate):
    """A pattern for a macro that reads as an identifier built from a tensor name, with
    the merged coordinate spelled by ``merged_coordinate`` (a pattern whose groups hold
    the tensor name)."""

    return re.compile(
        TENSOR_NAME.format(group="tensor")
        + r"_(?:"
        + "|".join(TENSOR_SUFFIXES)
        + r")|"
        + TENSOR_NAME.format(group="level")
        + r"\d+_(?:"
        + "|".join(LEVEL_SUFFIXES)
        + r")|p"
        + TENSOR_NAME.format(group="position")
        + r"\d*(?:_end)?|"
        + merged_coordinate
    )


# The merged coordinate of a one-letter index, as einsum's index letters are: the rule
# of merged_coordinate_name, and the spelling both cases had before it.
NEW_RULE = identifier_forms(
    r"[a-z]_"
    + TENSOR_NAME.format(group="lower")
    + r"|[A-Z]_"
    + TENSOR_NAME.format(group="upper")
    + r"_coord"
)
OLD_RULE = identifier_forms(r"[A-Za-z]_" + TENSOR_NAME.format(group="coordinate"))


def test_the_suffix_is_none_a_kernel_puts_on_a_tensor_name():
    """So a merged coordinate is never one of a tensor's own identifiers, whatever the
    tensors are called."""

    name = merged_coordinate_name("M", "B1")
    assert name.startswith("M_B1_")
    suffix = name[len("M_B1_") :]
    assert suffix not in TENSOR_SUFFIXES + LEVEL_SUFFIXES + ("end",)
    assert not any(known.endswith(suffix) for known in TENSOR_SUFFIXES + LEVEL_SUFFIXES)


def tensor_name_spelled_by(macro, rule=NEW_RULE):
    """The tensor name for which ``macro`` is an identifier a kernel builds, or None."""

    match = rule.fullmatch(macro)
    return next(filter(None, match.groups())) if match else None


def test_reading_a_macro_as_an_identifier_built_from_a_tensor_name():
    for macro, name in {
        "i_AB": "AB",
        "M_E_coord": "E",
        "X_IFMT_coord": "IFMT",
        "AB0_crd": "AB",
        "ZZ_values": "ZZ",
        "A12_pos_index": "A",
        "pA1": "A",
        "pAA12_end": "AA",
        "pZ": "Z",
        "Result_values": "Result",
    }.items():
        assert tensor_name_spelled_by(macro) == name, macro
    # Not identifiers a kernel builds: the old spelling with an upper-case index, a
    # suffix no kernel uses (stdio.h has L_tmpnam and P_tmpdir), a bare name, a name
    # and a level, lower case, two index letters, an upper-case suffix.
    for macro in (
        "M_E",
        "M_PI",
        "F_OK",
        "S_IFMT",
        "L_tmpnam",
        "P_tmpdir",
        "AB",
        "B0",
        "EOF",
        "ab_values",
        "ij_A",
        "AB_VALUES",
        "A1_crd_extra",
        "_A",
    ):
        assert tensor_name_spelled_by(macro) is None, macro
    for macro, name in {"M_E": "E", "M_PI": "PI", "F_OK": "OK", "i_A": "A"}.items():
        assert tensor_name_spelled_by(macro, OLD_RULE) == name, macro


def operands(expression, formats, extents, dtype=torch.float64):
    """Dense values and the tensors that store them.  A sparse operand stores about 70%
    of its cells and nothing in its first row or entry."""

    dense, tensors = [], []
    for position, (term, fmt) in enumerate(zip(terms_of(expression), formats)):
        generator = torch.Generator().manual_seed(53 + position)
        shape = tuple(extents[letter] for letter in term)
        if dtype.is_floating_point:
            values = 0.5 + torch.rand(shape, generator=generator, dtype=dtype)
        else:
            values = torch.randint(1, 4, shape, generator=generator, dtype=dtype)
        if set(fmt) != {"d"}:
            values = values * (torch.rand(shape, generator=generator) < 0.7)
            values[0] = 0
        dense.append(values)
        tensors.append(stored(values, fmt, string.ascii_uppercase[position]))
    return dense, tensors


@pytest.fixture(scope="module")
def kernel_macros():
    """The macros defined where a generated kernel is compiled: the preprocessor's list
    (``-dM -E``, which compiles nothing) for the source of a kernel built here, taken
    with the compiler and the flags of that kernel's own build (its ``build.ninja``)."""

    expression, formats = "im,im,im->im", ("ds", "ds", "ds")
    _, tensors = operands(expression, formats, SMALL)
    requests = []
    load = ops._load_validated_prepared_kernel

    def recording(prepared):
        requests.append(prepared.request)
        return load(prepared)

    with pytest.MonkeyPatch.context() as patch:
        empty_kernel_caches(patch)
        patch.setattr(ops, "_load_validated_prepared_kernel", recording)
        scorch.einsum(expression, *tensors, format="ds")
    assert requests, "einsum prepared no kernel for " + expression
    directory = Path(requests[-1].build_directory)
    ninja = (directory / "build.ninja").read_text()
    compiler = re.search(r"^cxx = (.+)$", ninja, re.MULTILINE).group(1)
    flags = re.search(r"^cflags = (.+)$", ninja, re.MULTILINE).group(1)
    listed = subprocess.run(
        [compiler, *shlex.split(flags), "-dM", "-E", str(directory / "main.cpp")],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(re.findall(r"^#define (\w+)", listed.stdout, re.MULTILINE))


def test_no_identifier_built_from_a_tensor_name_is_a_macro(kernel_macros):
    """Every macro of the kernel's translation unit, read as an identifier a kernel
    builds from a tensor name of any length: none is one."""

    spelled = {}
    for macro in sorted(kernel_macros):
        name = tensor_name_spelled_by(macro)
        if name is not None:
            spelled.setdefault(name, []).append(macro)
    assert spelled == {}


def test_the_macro_list_is_one_the_old_spelling_collided_with(kernel_macros):
    """The control for the test above: the list is a real one (thousands of macros), and
    read by the spelling a merged coordinate had it names the tensors that could not be
    merged on an upper-case index: einsum's fifth operand ``E`` among them.

    Only three macros that POSIX puts in math.h and unistd.h are asked for by name, so
    the control holds under any C library; what else a list holds differs by host (under
    gcc and glibc it names more than 20 tensors)."""

    assert len(kernel_macros) > 1000, len(kernel_macros)
    spelled = {}
    for macro in sorted(kernel_macros):
        name = tensor_name_spelled_by(macro, OLD_RULE)
        if name is not None:
            spelled.setdefault(name, []).append(macro)
    assert "M_E" in spelled.get("E", []), sorted(spelled)
    assert "M_PI" in spelled.get("PI", []), sorted(spelled)
    assert "F_OK" in spelled.get("OK", []), sorted(spelled)


DECLARED = re.compile(
    r"\b(?:int|int64_t|double|float|bool|auto|size_t|long)\b\s*[*&]?\s*([A-Za-z_]\w*)"
    r"\s*(?:=|;|\[|\{|:)"
)


def declared_names(text):
    """The names a kernel declares with a basic type: ``int M_E = ...``, ``for (int i =
    0``, ``const int* A0_crd = ...``, ``double _accum;``."""

    return set(DECLARED.findall(text))


def old_spelling(index_name, tensor_name):
    return f"{index_name}_{tensor_name}"


@pytest.mark.parametrize("compiler", ["shipped", "new"])
def test_no_name_a_kernel_declares_is_a_macro(monkeypatch, kernel_macros, compiler):
    """Each of einsum's 52 index letters as the merged index, nothing built: the names
    the emitted kernel declares, against the preprocessor's list.  The lower-case half
    is the spelling every kernel had before the rule (``i_A``); the upper-case half is
    the suffixed one.  The shipped compiler through einsum on five sparse vectors; the
    new one on two ``ds`` operands, the second named ``E`` (an index is never given a
    tensor's name there: the new compiler refuses that).

    The control is the same check with the spelling the coordinate had: it finds
    ``M_E`` under the letter ``M``, and whatever else it finds on this host is an
    upper-case index letter's coordinate in the old spelling, the names the rule
    replaced."""

    def tensors(index):
        if compiler == "shipped":
            return "ABCDE"
        return ("B" if index == "A" else "A", "F" if index == "E" else "E")

    def kernel(index):
        if compiler == "shipped":
            expression = ",".join(index * 5) + "->" + index
            return shipped_text(
                monkeypatch, expression, ["s"] * 5, "s", tensors(index), {index: 40}
            )
        outer = "j" if index == "i" else "i"
        return new_text(
            monkeypatch,
            f"{outer}{index},{outer}{index}->{outer}{index}",
            ("ds", "ds"),
            "ds",
            tensors(index),
            {outer: 5, index: 7},
            result_name="Z" if index == "Y" else "Y",
        )

    for index in string.ascii_letters:
        declared = declared_names(kernel(index))
        # the check reads the index variable and every operand's merged coordinate
        assert index in declared, index
        coordinates = {merged_coordinate_name(index, name) for name in tensors(index)}
        assert coordinates <= declared, index
        if index.isupper():
            replaced = {old_spelling(index, name) for name in tensors(index)}
            assert not replaced & declared, index
        assert not declared & kernel_macros, (index, sorted(declared & kernel_macros))

    monkeypatch.setattr(iterator, "merged_coordinate_name", old_spelling)
    monkeypatch.setattr(lower_llir, "merged_coordinate_name", old_spelling)
    collisions = {
        index: sorted(declared_names(kernel(index)) & kernel_macros)
        for index in string.ascii_letters
    }
    collisions = {index: names for index, names in collisions.items() if names}
    assert "M_E" in collisions.get("M", []), collisions
    for index, names in collisions.items():
        assert index.isupper(), collisions
        replaced = {old_spelling(index, name) for name in tensors(index)}
        assert set(names) <= replaced, collisions


# --------------------------------------------------------------------------------------
# Executed
# --------------------------------------------------------------------------------------

# (expression, operand formats, result format): the first four did not build.
FIVE = [
    ("M,M,M,M,M->M", ("s",) * 5, "s"),
    ("M,M,M,M,M->M", ("s",) * 5, "d"),
    ("iM,iM,iM,iM,iM->iM", ("ds",) * 5, "ds"),
    ("M,M,M,M,M->M", ("o",) * 5, "o"),
    ("m,m,m,m,m->m", ("s",) * 5, "s"),
    ("M,M,M,M->M", ("s",) * 4, "s"),
    ("M,M,M,M,M->M", ("s", "s", "s", "s", "d"), "s"),
    ("M,M,M,M,M,M->M", ("s",) * 6, "s"),
    ("mM,mM->mM", ("ds", "ds"), "ds"),
    ("mM,mM->mM", ("ss", "ss"), "ss"),
]
FIVE_IDS = [f"{program[0]} {','.join(program[1])}" for program in FIVE]


@pytest.mark.parametrize("extents", [SMALL, LARGER], ids=["small", "larger"])
@pytest.mark.parametrize("program", FIVE, ids=FIVE_IDS)
def test_merging_on_an_upper_case_index_is_computed(program, extents):
    expression, formats, result_format = program
    dense, tensors = operands(expression, formats, extents)
    result = scorch.einsum(expression, *tensors, format=result_format)
    assert_matches(result, reference(expression, dense), torch.float64)


@pytest.mark.parametrize("index", "ABCDEFM")
def test_an_index_letter_that_is_also_a_tensors_name_on_five_operands(index):
    """einsum names the five operands ``A`` to ``E`` and the result ``F``, so the index
    letters ``A`` to ``F`` are each also a tensor's name; ``M`` with the fifth operand
    is the pair math.h names.  The other letters are held, with nothing built, by
    test_no_name_a_kernel_declares_is_a_macro."""

    expression = ",".join(index * 5) + "->" + index
    dense, tensors = operands(expression, ("s",) * 5, {index: 41})
    result = scorch.einsum(expression, *tensors, format="s")
    assert_matches(result, reference(expression, dense), torch.float64)


def test_integer_operands_merged_on_an_upper_case_index():
    expression, formats = "iM,iM,iM,iM,iM->iM", ("ds",) * 5
    dense, tensors = operands(expression, formats, LARGER, dtype=torch.int64)
    result = scorch.einsum(expression, *tensors, format="ds")
    assert_matches(result, reference(expression, dense), torch.int64)


@pytest.mark.parametrize("extents", [SMALL, LARGER], ids=["small", "larger"])
def test_the_new_compiler_computes_a_merge_on_an_upper_case_index(extents):
    """Two ``ds`` operands named ``A`` and ``E``, merged on ``M``."""

    expression, formats = "iM,iM->iM", ("ds", "ds")
    dense, tensors = operands(expression, formats, extents)
    result = execute_cin_via_loopir(
        cin_statement(expression, formats, "ds", ("A", "E")),
        tuple(extents[letter] for letter in "iM"),
        *tensors,
        compile_options=new_options(),
    )
    result = result[0] if isinstance(result, tuple) else result
    assert_matches(result, reference(expression, dense), torch.float64)
