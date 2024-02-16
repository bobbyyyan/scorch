from typing import List, Optional, Any, Tuple, Callable, Union, Sequence


def assert_equal(actual: Any, expected: str):
    """Asserts `actual` is equal to `expected` while ignoring white space,
    e.g.,
       assert_equal("  a", "\n  a  \t ") # true
       assert_equal("ab", "a")           # false
    """
    if not isinstance(actual, str):
        actual = str(actual)

    def strip(s: Any) -> str:
        s = str(s)
        s = s.replace("\n", "")
        s = s.replace("\t", "")
        s = s.replace(" ", "")
        return s

    sactual = strip(actual)
    sexpected = strip(expected)

    # Prepend a newline for readability.
    if len(actual) > 0 and actual[0] != "\n":
        actual = f"\n{actual}"
    if len(expected) > 0 and expected[0] != "\n":
        expected = f"\n{expected}"
    assert (
        sactual == sexpected
    ), f"(IGNORING WHITESPACE)\nactual:{actual}\nexpected:{expected}\n"
