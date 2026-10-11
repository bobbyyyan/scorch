"""Production-neutrality contract of the LoopIR strangler package.

``import scorch``, default compilation, legacy correctness paths, and
release JIT must not load or execute ``scorch.compiler.loopir``; only
dedicated LoopIR tests import it.  These checks fail closed if a production
module grows an import of the package or if plain interpreter startup drags
it in.
"""

import pathlib
import re
import subprocess
import sys

import scorch


def test_plain_scorch_import_does_not_load_loopir():
    code = (
        "import sys\n"
        "import scorch\n"
        "loaded = [name for name in sys.modules "
        "if name.startswith('scorch.compiler.loopir')]\n"
        "assert not loaded, loaded\n"
        "print('CLEAN')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip().endswith("CLEAN")


def test_default_compilation_does_not_load_loopir():
    code = (
        "import sys\n"
        "import torch\n"
        "import scorch\n"
        "from scorch.compiler.cin import "
        "BinaryOp, ForAll, IndexVar, Operation, TensorAssign, TensorVar\n"
        "from scorch.compiler.cin_lowerer import CINLowerer\n"
        "from scorch.compiler.cin_analysis import normalize_cin\n"
        "i, j = IndexVar('i'), IndexVar('j')\n"
        "a, b, c = (TensorVar(n, fmt='dd') for n in 'ABC')\n"
        "assign = TensorAssign("
        "c[i, j], BinaryOp(Operation.ADD, a[i, j], b[i, j]))\n"
        "cin = normalize_cin(ForAll(i, ForAll(j, assign)))\n"
        "for tv, shape in zip(cin.get_rhs_tensor_vars(), [(2, 2), (2, 2)]):\n"
        "    tv.shape = shape\n"
        "    tv.dtype = torch.float32\n"
        "for tv in cin.get_result_tensor_vars():\n"
        "    tv.shape = (2, 2)\n"
        "    tv.dtype = torch.float32\n"
        "CINLowerer()._lower_owned_IndexStmt(cin)\n"
        "loaded = [name for name in sys.modules "
        "if name.startswith('scorch.compiler.loopir')]\n"
        "assert not loaded, loaded\n"
        "print('CLEAN')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip().endswith("CLEAN")


def test_no_production_module_imports_loopir():
    package_root = pathlib.Path(scorch.__file__).resolve().parent
    loopir_root = package_root / "compiler" / "loopir"
    offenders = []
    for path in package_root.rglob("*.py"):
        if loopir_root in path.parents:
            continue
        source = path.read_text()
        if re.search(r"\bloopir\b(?!_spike)", source):
            offenders.append(str(path))
    assert not offenders, offenders


def test_loopir_package_namespace_imports_nothing():
    """Loading the bare package namespace must not pull pipeline modules."""

    code = (
        "import sys\n"
        "import scorch.compiler.loopir\n"
        "loaded = sorted(name for name in sys.modules "
        "if name.startswith('scorch.compiler.loopir'))\n"
        "assert loaded == ['scorch.compiler.loopir'], loaded\n"
        "print('CLEAN')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip().endswith("CLEAN")
