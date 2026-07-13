#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON:-python}"

case "$(uname -s)" in
  Darwin)
    real_cxx="${SCORCH_SANITIZER_CXX:-/usr/bin/clang++}"
    ;;
  Linux)
    real_cxx="${SCORCH_SANITIZER_CXX:-c++}"
    ;;
  *)
    echo "ASan/UBSan harness supports Linux and macOS only" >&2
    exit 2
    ;;
esac

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/scorch-sanitizers.XXXXXX")"
cleanup() {
  rm -rf "$work_dir"
}
trap cleanup EXIT

export SCORCH_SANITIZER_CXX="$real_cxx"
export CXX="$repo_root/tests/sanitizers/sanitizer_cxx.sh"
export SCORCH_SANITIZER_LOG="${SCORCH_SANITIZER_LOG:-$work_dir/compiler.log}"
export TORCH_EXTENSIONS_DIR="$work_dir/torch_extensions"
export MAX_JOBS="${MAX_JOBS:-2}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
: >"$SCORCH_SANITIZER_LOG"

# A clean CI checkout always rebuilds the extension here. Locally, the symbol
# checks below intentionally reject a stale, unsanitized extension instead of
# reporting a false-green run.
"$python_bin" -m pip install \
  --editable "$repo_root" \
  --no-build-isolation \
  --no-deps \
  --force-reinstall

case "$(uname -s)" in
  Linux)
    asan_runtime="$("$real_cxx" -print-file-name=libasan.so)"
    if [[ "$asan_runtime" == "libasan.so" || ! -f "$asan_runtime" ]]; then
      echo "Could not locate libasan.so with $real_cxx" >&2
      exit 2
    fi
    export LD_PRELOAD="$asan_runtime${LD_PRELOAD:+:$LD_PRELOAD}"
    linux_asan_options="detect_leaks=1:detect_stack_use_after_return=1:strict_string_checks=1:halt_on_error=1"
    export ASAN_OPTIONS="${ASAN_OPTIONS:+$ASAN_OPTIONS:}$linux_asan_options"
    required_lsan_options="leak_check_at_exit=0"
    export LSAN_OPTIONS="${LSAN_OPTIONS:+$LSAN_OPTIONS:}$required_lsan_options"
    ;;
  Darwin)
    clang_resource_dir="$("$real_cxx" -print-resource-dir)"
    asan_runtime="$clang_resource_dir/lib/darwin/libclang_rt.asan_osx_dynamic.dylib"
    if [[ ! -f "$asan_runtime" ]]; then
      echo "Could not locate the Apple Clang ASan runtime under $clang_resource_dir" >&2
      exit 2
    fi
    export DYLD_INSERT_LIBRARIES="$asan_runtime${DYLD_INSERT_LIBRARIES:+:$DYLD_INSERT_LIBRARIES}"
    macos_asan_options="detect_stack_use_after_return=1:strict_string_checks=1:halt_on_error=1"
    export ASAN_OPTIONS="${ASAN_OPTIONS:+$ASAN_OPTIONS:}$macos_asan_options"
    ;;
esac

required_ubsan_options="halt_on_error=1:print_stacktrace=1"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:+$UBSAN_OPTIONS:}$required_ubsan_options"
export SCORCH_SANITIZER_RUN=1

# Import Torch first so its bundled OpenMP runtime is resident before the
# extension is loaded. This is required by macOS wheels whose libomp install
# name points at PyTorch's build-time path rather than its packaged location.
extension_path="$($python_bin -c 'import torch, scorch_ops; print(scorch_ops.__file__)')"
undefined_symbols="$(nm -u "$extension_path")"
if ! grep -q "asan_" <<<"$undefined_symbols"; then
  echo "scorch_ops is not ASan-instrumented: $extension_path" >&2
  exit 1
fi
if ! grep -q "ubsan_" <<<"$undefined_symbols"; then
  echo "scorch_ops is not UBSan-instrumented: $extension_path" >&2
  exit 1
fi

if [[ "$(uname -s)" == "Linux" ]]; then
  "$python_bin" "$repo_root/tests/sanitizers/leak_probe.py"
fi

"$python_bin" -m pytest \
  -q \
  -p no:cacheprovider \
  "$repo_root/tests/sanitizers/test_native_ownership.py" \
  "$repo_root/tests/test_scorch/test_runtime_ownership_helpers.py"

# The tests must compile at least one load_inline translation unit. This guards
# against accidentally exercising only the prebuilt extension while generated
# kernels remain unsanitized.
if ! grep -q -- "fsanitize=address.*undefined" "$SCORCH_SANITIZER_LOG"; then
  echo "sanitizer compiler shim was not used" >&2
  exit 1
fi
if ! grep -q "main.cpp" "$SCORCH_SANITIZER_LOG"; then
  echo "no sanitizer-instrumented JIT translation unit was compiled" >&2
  exit 1
fi

echo "ASan/UBSan native ownership checks passed"
