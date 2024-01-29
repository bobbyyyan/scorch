from typing import List, Optional, Any, Tuple, Callable, Union, Sequence


def assert_equal(actual: Any, expected: str):
    """Asserts `actual` is equal to `expected` while ignoring white space,
    e.g.,
       assert_equal("  a", "\n  a  \t ") # true
       assert_equal("ab", "a")           # false
    """

    def strip(s: Any) -> str:
        s = str(s)
        s = s.strip()
        s = s.replace("\n", "")
        s = s.replace("\t", "")
        s = s.replace(" ", "")
        return

    actual = strip(actual)
    expected = strip(expected)
    assert actual == expected, f"\nactual:{actual}\nexpected:{expected}\n"
