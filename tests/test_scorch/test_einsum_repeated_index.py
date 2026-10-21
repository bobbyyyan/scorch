"""``scorch.einsum`` refuses a term that names one index twice, and returns.

``scorch.einsum("ii->i", A)`` never returned.  The loop-order sort
(``utils.topo_sort_characters``) puts an edge from each letter of a term to the next, so
``"ii"`` gave the edge ``i -> i``, and the cycle resolution removed that edge and put it
back on every pass.  With the edge left out the call got as far as the operand relayout
and raised ``TensorLayoutError`` ("mode_order must be a permutation") with no code, as
``"iji->ij"``, whose repeat is not adjacent, already did: no repeated index was computed.

The diagonal an operand's repeated index asks for is not computed yet.  Until it is, the
expression is refused where einsum checks its labels, with ``EinsumRepeatedIndexError``
(code ``einsum_repeated_index``, a ``CompileSpecError``), before a loop order is
selected, an operand moved or a kernel looked up.  A result term that repeats a letter,
which PyTorch rejects too, gets the same code.

Each call below runs under a time limit, so a call that does not return fails its test
instead of stopping the run, and with the loop-order sort and every build replaced by a
function that fails, so the refusal is shown to come before them.
"""

import pickle
import signal
from contextlib import contextmanager

import pytest
import torch

import scorch
from scorch import ops, stensor, utils
from scorch.compiler.compile_options import CompileOptions
from scorch.exceptions import CompileSpecError, EinsumRepeatedIndexError
from scorch.layout import TensorSpec
from tests.test_scorch.level_array_support import assert_matches, reference, stored

# (expression, operand shapes, operand formats, result format, operand, index): the
# spellings of Bobby's second review and one per place a repeat can stand in a term.
REPEATS = [
    ("ii->i", [(3, 3)], ["ds"], "s", 0, "i"),
    ("ii->i", [(3, 3)], ["ds"], "d", 0, "i"),
    ("ii->i", [(3, 3)], ["ss"], "s", 0, "i"),
    ("ii->i", [(3, 3)], ["dd"], "d", 0, "i"),
    ("ii->i", [(3, 3)], ["oo"], "o", 0, "i"),
    ("ii->i", [(41, 41)], ["ds"], "s", 0, "i"),
    ("iij->ij", [(2, 2, 3)], ["dss"], "ds", 0, "i"),
    ("iij->ij", [(2, 2, 3)], ["sss"], "ss", 0, "i"),
    ("iij->ij", [(2, 2, 3)], ["ooo"], "oo", 0, "i"),
    ("iii->i", [(2, 2, 2)], ["sss"], "s", 0, "i"),
    ("jii->ij", [(3, 2, 2)], ["dss"], "ds", 0, "i"),
    ("iji->ij", [(2, 3, 2)], ["sss"], "ss", 0, "i"),
    ("ijj->ij", [(2, 3, 3)], ["dss"], "ds", 0, "j"),
    ("ij,jj->i", [(2, 3), (3, 3)], ["ds", "ds"], "d", 1, "j"),
    ("ij,kk->ij", [(2, 3), (4, 4)], ["ds", "dd"], "ds", 1, "k"),
    ("ijij->ij", [(2, 3, 2, 3)], ["ssss"], "ss", 0, "i"),
]
SECONDS = 5.0


class _DidNotReturn(Exception):
    """The call was still running at its time limit."""


class _Reached(Exception):
    """Raised by the loop-order sort or a build that a refused call reached."""


@contextmanager
def time_limit(seconds):
    def expired(signum, frame):
        raise _DidNotReturn(f"still running after {seconds} s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _reached(*args, **kwargs):
    raise _Reached


def nothing_past_the_label_check(monkeypatch):
    """Replace the loop-order sort and every build by a function that fails, and return
    the list that records each operand move (``STensor.change_mode_order``)."""

    for module, name in [
        (ops, "topo_sort_characters"),
        (utils, "_prepare_jit_build"),
        (utils, "_load_validated_prepared_kernel"),
        (ops, "_prepare_jit_build"),
        (ops, "_load_validated_prepared_kernel"),
        (stensor, "_prepare_jit_build"),
        (stensor, "_load_validated_prepared_kernel"),
    ]:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _reached)
    moves = []
    original = stensor.STensor.change_mode_order

    def recorded(self, *args, **kwargs):
        moves.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(stensor.STensor, "change_mode_order", recorded)
    return moves


def operands_for(shapes, formats, dtype=torch.float64):
    """Dense operands, about half of each zero, and the same stored in ``formats``."""

    dense_operands, built = [], []
    for position, (shape, fmt) in enumerate(zip(shapes, formats)):
        generator = torch.Generator().manual_seed(100 + position)
        dense = torch.rand(shape, generator=generator, dtype=torch.float64)
        dense = (dense * (dense < 0.6)).to(dtype)
        dense_operands.append(dense)
        built.append(stored(dense, fmt, "ABDEFGH"[position]))
    return dense_operands, built


def _id(case):
    expression, shapes, formats, result, _, _ = case
    return (
        f"{expression} {','.join(formats)}->[{result}] {'x'.join(map(str, shapes[0]))}"
    )


@pytest.mark.parametrize("blocking", [False, True], ids=["baseline", "regblock"])
@pytest.mark.parametrize("case", REPEATS, ids=[_id(case) for case in REPEATS])
def test_operand_term_that_repeats_an_index_is_refused(monkeypatch, case, blocking):
    expression, shapes, formats, result_format, operand, index = case
    _, operands = operands_for(shapes, formats)
    moves = nothing_past_the_label_check(monkeypatch)
    options = CompileOptions.from_environment().with_regblock_enabled(blocking)

    with time_limit(SECONDS):
        with pytest.raises(EinsumRepeatedIndexError) as raised:
            scorch.einsum(
                expression, *operands, format=result_format, _compile_options=options
            )

    error = raised.value
    assert error.code == "einsum_repeated_index"
    assert (error.operand, error.index) == (operand, index)
    assert isinstance(error, CompileSpecError) and isinstance(error, ValueError)
    assert f"operand {operand}" in str(error) and repr(index) in str(error)
    assert moves == []
    assert expression not in ops._EXPR_LABELS_CACHE


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
def test_torch_operands_are_refused_the_same_way(monkeypatch, dtype):
    """A dense ``torch.Tensor`` is wrapped before the labels are checked.  The trace
    ``"ii->"`` is refused for its repeated index like the diagonal: its empty result term
    is a rank-0 result, which einsum computes (``test_einsum_scalar_result.py``).

    Every call names a scorch keyword, so its labels are checked as the compiler checks
    them, whichever path computes it; ``ops._einsum_via_compiler`` is the compiler's own
    entry.  The same calls without a scorch keyword are torch's own (next test)."""

    square = torch.arange(16, dtype=dtype).reshape(4, 4)
    nothing_past_the_label_check(monkeypatch)
    with time_limit(SECONDS):
        with pytest.raises(EinsumRepeatedIndexError) as raised:
            scorch.einsum("ii->i", square, format="d")
        with pytest.raises(EinsumRepeatedIndexError):
            scorch.einsum("ij,jj->i", square, square, format="d")
        with pytest.raises(EinsumRepeatedIndexError) as trace:
            ops._einsum_via_compiler("ii->", square)
    assert (raised.value.operand, raised.value.index) == (0, "i")
    assert (trace.value.operand, trace.value.index) == (0, "i")
    # What the refused call asked for, so the reference of the feature is on record.
    assert torch.equal(torch.einsum("ii->i", square), torch.diagonal(square))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
def test_an_all_dense_call_without_a_scorch_keyword_is_torchs(monkeypatch, dtype):
    """``scorch.einsum`` hands a call whose operands and result are all dense, naming no
    scorch keyword, to ``torch.einsum`` unchanged (drop-in note, section 0.1, step 1), so
    the diagonal and the trace that the compiler refuses are computed, by torch, and
    nothing of the compiler runs."""

    square = torch.arange(16, dtype=dtype).reshape(4, 4)
    moves = nothing_past_the_label_check(monkeypatch)
    with time_limit(SECONDS):
        for expression, operands in (
            ("ii->i", (square,)),
            ("ii->", (square,)),
            ("ij,jj->i", (square, square)),
        ):
            result = scorch.einsum(expression, *operands)
            assert type(result) is torch.Tensor
            assert torch.equal(result, torch.einsum(expression, *operands))
    assert moves == []


def test_compile_only_specification_is_refused_the_same_way(monkeypatch):
    nothing_past_the_label_check(monkeypatch)
    spec = TensorSpec("ds", (6, 6), dtype=torch.float64, name="A")
    with time_limit(SECONDS):
        with pytest.raises(EinsumRepeatedIndexError) as raised:
            scorch.einsum("ii->i", spec, format="s", compile_only=True)
    assert raised.value.code == "einsum_repeated_index"


@pytest.mark.parametrize(
    ("expression", "shapes", "formats", "result_format", "index"),
    [
        ("i->ii", [(3,)], ["s"], "ds", "i"),
        ("ik,kj->ii", [(3, 4), (4, 3)], ["ds", "ds"], "ds", "i"),
        ("ik,kj->ijj", [(3, 4), (4, 3)], ["ds", "dd"], "dss", "j"),
    ],
)
def test_result_term_that_repeats_an_index_is_refused(
    monkeypatch, expression, shapes, formats, result_format, index
):
    """PyTorch rejects these too ("output subscript ... appears more than once")."""

    dense, operands = operands_for(shapes, formats)
    with pytest.raises(RuntimeError, match="more than once"):
        torch.einsum(expression, *dense)
    nothing_past_the_label_check(monkeypatch)
    with time_limit(SECONDS):
        with pytest.raises(EinsumRepeatedIndexError) as raised:
            scorch.einsum(expression, *operands, format=result_format)
    assert raised.value.code == "einsum_repeated_index"
    assert (raised.value.operand, raised.value.index) == (None, index)
    assert "result labels must be unique" in str(raised.value)


def test_an_operand_repeat_is_reported_before_a_result_repeat():
    with pytest.raises(EinsumRepeatedIndexError) as raised:
        ops._validated_labels("ii,ij->jj", ["ii", "ij"], "jj")
    assert (raised.value.operand, raised.value.index) == (0, "i")


def test_the_error_survives_pickling():
    with pytest.raises(EinsumRepeatedIndexError) as raised:
        ops._validated_labels("ab,cbc->a", ["ab", "cbc"], "a")
    copy = pickle.loads(pickle.dumps(raised.value))
    assert type(copy) is EinsumRepeatedIndexError
    assert (copy.operand, copy.index, str(copy)) == (1, "c", str(raised.value))
    assert copy.code == "einsum_repeated_index"


@pytest.mark.parametrize("blocking", [False, True], ids=["baseline", "regblock"])
@pytest.mark.parametrize(
    ("expression", "formats", "result_format"),
    [("ij,ji->ij", ["ds", "ds"], "ds"), ("ij,jk->ik", ["ds", "dd"], "dd")],
)
@pytest.mark.parametrize("extent", [5, 37])
def test_terms_with_distinct_letters_are_computed_as_before(
    expression, formats, result_format, extent, blocking
):
    """The check refuses nothing else: letters shared between terms, in either order,
    are an ordinary product."""

    dense, operands = operands_for([(extent, extent), (extent, extent)], formats)
    result = scorch.einsum(
        expression,
        *operands,
        format=result_format,
        _compile_options=CompileOptions.from_environment().with_regblock_enabled(
            blocking
        ),
    )
    assert_matches(result, reference(expression, dense), torch.float64)
