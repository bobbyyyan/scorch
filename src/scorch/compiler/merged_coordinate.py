"""The name of a merged coordinate: one spelling for both compilers.

A loop that merges sparse operands on an index reads each operand's current coordinate
into a variable of its own before it takes their minimum::

    int j_A = A1_crd[pA1];
    int j_B = B1_crd[pB1];
    int j = std::min({j_A, j_B});

Both compilers spell that variable from the index name and the tensor name, and this
module is the one place that says how.

``<index>_<tensor>`` is the spelling, and every kernel emitted for an index name with a
lower-case letter in it has it.

An index name without a lower-case letter gets the suffix ``_coord``. einsum takes the
52 ASCII letters as index letters and names its operands ``A``, ``B``, ..., so the index
``M`` on its fifth operand would otherwise be ``M_E``: one capital, an underscore and
capitals, which is how the C library spells its constants. ``M_E`` and ``M_PI`` are
macros of math.h, ``F_OK`` of unistd.h, ``S_IFMT`` of sys/stat.h, and a kernel that
declares ``int M_E`` does not build. A lower-case letter takes the identifier out of
that spelling. ``_coord`` is none of the suffixes a kernel puts on a tensor name
(``E_values``, ``E0_crd``, ``E0_pos``, ...), so the name is not one of a tensor's own
identifiers either: ``_crd`` would be, for a tensor whose name ends in a digit
(``M_B1_crd`` is the level-1 coordinate array of a tensor named ``M_B``).

tests/test_scorch/test_merged_coordinate_name.py holds the rule to the preprocessor's
macro list for a generated kernel's translation unit, and holds both compilers to this
function: no kernel spells a merged coordinate any other way.
"""


def merged_coordinate_name(index_name: str, tensor_name: str) -> str:
    """The variable for ``tensor_name``'s current coordinate of ``index_name``."""

    name = f"{index_name}_{tensor_name}"
    if any(character.islower() for character in index_name):
        return name
    return f"{name}_coord"
