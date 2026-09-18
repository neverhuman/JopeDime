#!/usr/bin/env bash
# Environment for building the `neural` feature (crates/jope-engine, package jope-parallel:
# tch 0.26.0 + native/runtime.cpp) on the self-hosted xbabe2 lane.
#
# Every value mirrors deploy/receipts/libtorch.json; regenerate both together
# (docs/runbooks/neural-sdk.md). The SDK volume layout is:
#   $JOPE_NEURAL_SDK_ROOT/libtorch               LibTorch 2.13.0+cu126 (unzipped, build-version checked)
#   $JOPE_NEURAL_SDK_ROOT/nvidia/<pkg>/{lib,include}  cudnn, cusparselt, cufile, nccl, nvshmem from the
#                                              pinned nvidia-*-cu12 wheels (LibTorch 2.13 no longer bundles them)
#
# Usage:
#   . deploy/ci/neural-env.sh            # export the variables into the current shell
#   deploy/ci/neural-env.sh >> "$GITHUB_ENV"   # print KEY=VALUE lines for a workflow step
#
# CUDA itself comes from the host (Ubuntu nvidia-cuda-toolkit 12.8: nvcc at /usr/lib/cuda/bin,
# cuda_runtime.h/nccl.h in /usr/include, libcudart.so.12 in /usr/lib/x86_64-linux-gnu).

jope_neural_env_fail() {
  printf 'neural-env: %s\n' "$1" >&2
  return 1
}

jope_neural_env_main() {
  local sdk="${JOPE_NEURAL_SDK_ROOT:-/opt/jope-sdk}"
  local expected_version="2.13.0+cu126"
  local dir actual_version

  for dir in libtorch/lib libtorch/include nvidia/nccl/lib nvidia/nccl/include nvidia/cudnn/lib \
             nvidia/cusparselt/lib nvidia/cufile/lib nvidia/nvshmem/lib; do
    [[ -d "$sdk/$dir" ]] || jope_neural_env_fail "missing $sdk/$dir (provision per docs/runbooks/neural-sdk.md)" || return 1
  done
  actual_version="$(< "$sdk/libtorch/build-version")"
  [[ "$actual_version" == "$expected_version" ]] \
    || jope_neural_env_fail "LibTorch build-version is '$actual_version', receipt expects '$expected_version'" \
    || return 1
  [[ -r /usr/include/cuda_runtime.h ]] || jope_neural_env_fail "missing /usr/include/cuda_runtime.h (nvidia-cuda-dev)" || return 1

  # Read by crates/jope-engine/build.rs and torch-sys 0.26.0 build.rs.
  export LIBTORCH="$sdk/libtorch"
  export LIBTORCH_CXX11_ABI=1
  export CUDA_HOME="${CUDA_HOME:-/usr/lib/cuda}"
  export JOPE_NCCL_INCLUDE="$sdk/nvidia/nccl/include"
  # Order matters: wheel directories precede the host directory so DT_RPATH resolves
  # libnccl.so.2 to the wheel's 2.29.3 (the host's 2.26.2 lacks 10 symbols libtorch_cuda imports).
  export JOPE_CUDA_LIBRARY_PATH="$sdk/nvidia/nccl/lib:$sdk/nvidia/cudnn/lib:$sdk/nvidia/cusparselt/lib:$sdk/nvidia/cufile/lib:$sdk/nvidia/nvshmem/lib:/usr/lib/x86_64-linux-gnu"
  # Verbatim SONAMEs: the wheels ship no unversioned libnccl.so dev symlink, so -lnccl would
  # otherwise resolve to the host library.
  export JOPE_NCCL_SONAME=libnccl.so.2
  export JOPE_CUDART_SONAME=libcudart.so.12
  export CXX="${CXX:-clang++}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  jope_neural_env_main
  for name in LIBTORCH LIBTORCH_CXX11_ABI CUDA_HOME JOPE_NCCL_INCLUDE JOPE_CUDA_LIBRARY_PATH \
              JOPE_NCCL_SONAME JOPE_CUDART_SONAME CXX; do
    printf '%s=%s\n' "$name" "${!name}"
  done
else
  jope_neural_env_main
fi
