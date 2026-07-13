# Native sanitizer checks

Run the ownership-focused native tests from a development environment that
already has PyTorch, pytest, Ninja, and the project build dependencies:

```bash
bash tests/sanitizers/run_sanitizers.sh
```

The harness rebuilds the editable `scorch_ops` extension and all JIT modules
through `sanitizer_cxx.sh` with AddressSanitizer and UndefinedBehaviorSanitizer.
It uses a fresh Torch extensions directory and verifies that both the prebuilt
extension and at least one JIT translation unit were instrumented, preventing a
cached unsanitized binary from producing a false-green result.

On Linux, GCC's ASan runtime is located with
`$SCORCH_SANITIZER_CXX -print-file-name=libasan.so` and preloaded before Python
starts. Exit-time leak reporting is disabled because CPython, PyTorch, and
OpenMP keep process-lifetime allocations. Instead, `leak_probe.py` warms those
frameworks with LSan tracking disabled, tracks only a native ownership call,
destroys its result, and runs an explicit leak check. On macOS, install OpenMP
with `brew install libomp`; the harness uses Apple Clang's ASan runtime and
`DYLD_INSERT_LIBRARIES`. Apple's runtime does not reliably provide LSan, so the
focused leak probe is Linux-only. Override the compiler with
`SCORCH_SANITIZER_CXX=/path/to/clang++` if needed.

Run from a clean checkout or remove stale generated extension artifacts first.
The harness rejects an existing `scorch_ops` binary that does not import both
ASan and UBSan symbols.
