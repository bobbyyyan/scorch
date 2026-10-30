"""The in-process caches a test empties so that its calls choose, lower and build again.

A test asks for them with the ``fresh_kernel_caches`` fixture (tests/conftest.py). A
helper that is handed only ``monkeypatch``, and a test that must empty them again partway
through, call ``empty_kernel_caches``, which is what the fixture runs. Code with no
``monkeypatch`` (a child process that runs several calls) calls ``clear_kernel_caches``.
``KERNEL_CACHES`` is the one list all three read, so a cache added to it is emptied by
every test that asks for fresh caches.

A cache belongs in the list when it outlives a test, its key cannot see a function or a
setting a test replaces, and a hit lets a call skip a step of choosing, scheduling,
lowering, building, checking or tuning its kernel, or of resolving its build options: a
test that replaces one of those functions and is served an earlier test's entry never
runs its replacement. Caches keyed by the generated source (``utils._so_cache``) or by
live operand storages (``result_bound._JOIN_CACHE``), and the memos of immutable values
(layouts, expression labels, result parts, operand copies and transposes, file digests,
preamble text), are not in it.

Most caches are one module-level dict, which ``empty_kernel_caches`` replaces with an
empty dict until the test ends. A cache its module keeps in several variables is emptied
by the function the module provides for that, so that the variables are named in one
place only. That function runs when the caches are emptied and again when the test ends
(``finish_test``, which tests/conftest.py runs after every test), so nothing the test put
in such a cache is kept for later tests either.

Nothing here imports scorch until a function runs, because tests/conftest.py imports this
module and some tests check what importing scorch does.
"""

import importlib
from typing import Any, List, NamedTuple, Optional, Tuple, Union


class OneDict(NamedTuple):
    """A cache held in one module-level dict."""

    module: str
    attribute: str
    #: For a cache kept in a file too, the module-level flag that says the dict has been
    #: read from the file. It is set with the empty dict, so that the next lookup reads
    #: the empty dict and not the file.
    read_flag: Optional[str] = None

    def empty(self, monkeypatch: Any) -> None:
        module = importlib.import_module(self.module)
        monkeypatch.setattr(module, self.attribute, {})
        if self.read_flag is not None:
            monkeypatch.setattr(module, self.read_flag, True)

    def clear(self) -> None:
        module = importlib.import_module(self.module)
        if self.read_flag is None:
            getattr(module, self.attribute).clear()
        else:
            setattr(module, self.attribute, {})
            setattr(module, self.read_flag, True)


class EmptiedByCall(NamedTuple):
    """A cache its module keeps in several variables and empties with one function."""

    module: str
    function: str

    def clear(self) -> None:
        getattr(importlib.import_module(self.module), self.function)()


#: Each cache, with what a hit skips.
KERNEL_CACHES: Tuple[Union[OneDict, EmptiedByCall], ...] = (
    # A hit skips scheduling, lowering, the LLIR passes, codegen and the build.
    OneDict("scorch.ops", "_einsum_dispatch_cache"),
    # Keyed by the scheduled CIN: a hit skips lowering onwards.
    OneDict("scorch.ops", "_kernel_cache"),
    # The result levels each kept kernel checks the total of, read from its source when
    # it is built, under _kernel_cache's key.
    OneDict("scorch.ops", "_kernel_level_totals"),
    # The conversions, spmv, ``+`` and lower_and_exec_cin: a hit skips everything from
    # the CIN on.
    OneDict("scorch.stensor", "_GENERATED_KERNELS"),
    # The result bound of each addition and conversion key: a hit skips computing it.
    OneDict("scorch.stensor", "_RESULT_BOUNDS"),
    # The prebuilt kernel matmul resolved for each pair of formats and dtypes: a hit skips
    # the resolution.
    OneDict("scorch.prebuilt_kernels", "_resolve_matmul_cache"),
    # The cache key of each schedule state validated so far: a hit skips the validation.
    OneDict("scorch.compiler.scheduler", "_SCHEDULE_KEY_MEMO"),
    # The tiling selector's verdict for each shape, autotune level and baseline: a hit
    # skips the choice and its probe.
    OneDict("scorch.tiling", "_decision"),
    # The "max" autotune level's verdicts, kept in a file across processes: a hit skips
    # the probe.
    OneDict("scorch.tiling", "_persist_cache", read_flag="_cache_loaded"),
    # The "learned" autotune level's model, read from its file once per process: a kept
    # model makes the choices of a later test that names another file, or none.
    EmptiedByCall("scorch.tiling", "_reset_learned_model_cache"),
    # The resolved build options, the parsed environment snapshots and their keys: a hit
    # skips resolving the options or parsing the environment.
    EmptiedByCall("scorch.compiler.compile_options", "clear_build_snapshot_cache"),
)

#: Whether empty_kernel_caches ran in the current test, so that finish_test runs the
#: EmptiedByCall functions again when it ends.
_emptied_in_this_test = [False]


def empty_kernel_caches(monkeypatch: Any) -> None:
    """Empty each cache in ``KERNEL_CACHES`` until the test ends.

    A dict is replaced by an empty one and restored when the test ends, so the test's own
    entries are not kept for later tests either; a cache its module empties with a
    function is emptied now and again when the test ends. Called again partway through a
    test, it empties them again.
    """

    for cache in KERNEL_CACHES:
        if isinstance(cache, OneDict):
            cache.empty(monkeypatch)
        else:
            cache.clear()
    _emptied_in_this_test[0] = True


def clear_kernel_caches() -> None:
    """Empty each cache in ``KERNEL_CACHES`` in place, for code without ``monkeypatch``."""

    for cache in KERNEL_CACHES:
        cache.clear()


def finish_test() -> None:
    """Empty again each cache a function empties, if the test that ended emptied them."""

    if _emptied_in_this_test[0]:
        _emptied_in_this_test[0] = False
        for cache in KERNEL_CACHES:
            if isinstance(cache, EmptiedByCall):
                cache.clear()


def kernel_caches() -> List[Tuple[str, Any]]:
    """Each one-dict cache's qualified name and its current dict, for a test that reads
    them. A dict not read from its file yet is listed as an empty dict."""

    caches = []
    for cache in KERNEL_CACHES:
        if isinstance(cache, OneDict):
            value = getattr(importlib.import_module(cache.module), cache.attribute)
            caches.append(
                (f"{cache.module}.{cache.attribute}", {} if value is None else value)
            )
    return caches
