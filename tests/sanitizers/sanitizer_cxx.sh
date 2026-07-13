#!/usr/bin/env bash
set -euo pipefail

# Compiler shim used by both setuptools and torch.utils.cpp_extension. Keeping
# the instrumentation here makes direct native builds and generated JIT modules
# use exactly the same sanitizer configuration without adding release-build
# flags to Scorch itself.
real_cxx="${SCORCH_SANITIZER_CXX:-c++}"
sanitizer_flags=(
  -O1
  -g
  -fno-omit-frame-pointer
  -fno-optimize-sibling-calls
  -fno-sanitize-recover=all
  -fsanitize=address,undefined
)

if [[ -n "${SCORCH_SANITIZER_LOG:-}" ]]; then
  {
    printf '%q ' "$real_cxx" "$@" "${sanitizer_flags[@]}"
    printf '\n'
  } >>"$SCORCH_SANITIZER_LOG"
fi

exec "$real_cxx" "$@" "${sanitizer_flags[@]}"
