"""Automated dependency and target-neutrality checks for the LoopIR spike.

The spike must stay an isolated experiment: its own import closure may touch
only the Python standard library and the stable identity module, production
modules must never reference it, importing ``scorch`` must not load it, and
its sources must contain no target-syntax escape hatches.
"""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import scorch

SRC_ROOT = Path(scorch.__file__).resolve().parent.parent
SPIKE_DIR = SRC_ROOT / "scorch" / "compiler" / "loopir_spike"
SPIKE_MODULES = ("__init__", "csr", "interp", "nodes", "programs", "verifier")

ALLOWED_STDLIB_ROOTS = {
    "__future__",
    "dataclasses",
    "enum",
    "itertools",
    "math",
    "threading",
    "types",
    "typing",
}

TARGET_SYNTAX_TOKENS = (
    "std::",
    "#include",
    "#pragma",
    "__builtin",
    "openmp",
    "load_inline",
    "cpp_extension",
    "RawStmt",
)


def test_spike_package_contains_exactly_the_expected_modules():
    found = sorted(path.stem for path in SPIKE_DIR.glob("*.py"))
    assert found == sorted(SPIKE_MODULES)


def test_spike_sources_import_only_stdlib_identity_and_siblings():
    for stem in SPIKE_MODULES:
        source = (SPIKE_DIR / f"{stem}.py").read_text()
        tree = ast.parse(source, filename=f"{stem}.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root in ALLOWED_STDLIB_ROOTS, (stem, alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    root = (node.module or "").split(".")[0]
                    assert root in ALLOWED_STDLIB_ROOTS, (stem, node.module)
                elif node.level == 1:
                    assert node.module in set(SPIKE_MODULES) - {"__init__"}, (
                        stem,
                        node.module,
                    )
                else:
                    assert node.level == 2 and node.module == "identity", (
                        stem,
                        node.module,
                    )


def test_spike_import_closure_is_torch_free_and_pipeline_free():
    script = textwrap.dedent(f"""
        import sys
        import types

        src = {str(SRC_ROOT)!r}
        for name, path in (
            ("scorch", src + "/scorch"),
            ("scorch.compiler", src + "/scorch/compiler"),
        ):
            synthetic = types.ModuleType(name)
            synthetic.__path__ = [path]
            sys.modules[name] = synthetic

        import scorch.compiler.loopir_spike.csr
        import scorch.compiler.loopir_spike.interp
        import scorch.compiler.loopir_spike.nodes
        import scorch.compiler.loopir_spike.programs
        import scorch.compiler.loopir_spike.verifier

        banned = sorted(
            name for name in sys.modules if name.split(".")[0] == "torch"
        )
        assert not banned, f"torch reached the closure: {{banned}}"

        allowed = {{
            "scorch",
            "scorch.compiler",
            "scorch.compiler.identity",
            "scorch.compiler.loopir_spike",
            "scorch.compiler.loopir_spike.csr",
            "scorch.compiler.loopir_spike.interp",
            "scorch.compiler.loopir_spike.nodes",
            "scorch.compiler.loopir_spike.programs",
            "scorch.compiler.loopir_spike.verifier",
        }}
        loaded = {{
            name for name in sys.modules if name.split(".")[0] == "scorch"
        }}
        unexpected = sorted(loaded - allowed)
        assert not unexpected, f"unexpected scorch modules: {{unexpected}}"
        print("CLOSURE_OK")
        """)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "CLOSURE_OK" in completed.stdout


def test_importing_scorch_does_not_load_the_spike():
    script = textwrap.dedent("""
        import sys

        import scorch
        import scorch.compiler

        spiked = sorted(
            name for name in sys.modules if "loopir_spike" in name
        )
        assert not spiked, f"production import pulled the spike: {spiked}"
        print("PRODUCTION_OK")
        """)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    assert "PRODUCTION_OK" in completed.stdout


def test_no_production_module_references_the_spike():
    offenders = []
    for path in sorted((SRC_ROOT / "scorch").rglob("*.py")):
        if SPIKE_DIR in path.parents:
            continue
        if "loopir_spike" in path.read_text():
            offenders.append(str(path.relative_to(SRC_ROOT)))
    assert not offenders, offenders


def test_spike_sources_contain_no_target_syntax():
    for stem in SPIKE_MODULES:
        source = (SPIKE_DIR / f"{stem}.py").read_text().lower()
        for token in TARGET_SYNTAX_TOKENS:
            assert token.lower() not in source, (stem, token)
