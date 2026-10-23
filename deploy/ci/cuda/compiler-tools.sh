#!/usr/bin/env bash
set -euo pipefail
test "$(uname -s)/$(uname -m)" = Linux/x86_64
mode=${1:-compile}
root="${CARGO_TARGET_DIR:-$PWD/target}/cuda-toolkit"
python3 -m pip install --disable-pip-version-check --require-hashes --only-binary=:all: \
  --no-deps --upgrade --target "$root" -r deploy/ci/cuda/tools-requirements.txt
test -x "$root/cmake/data/bin/cmake"
printf '%s\n' "$root/cmake/data/bin" >> "$GITHUB_PATH"
if [ "$mode" = host ]; then exit 0; fi
test "$mode" = compile
python3 -m pip install --disable-pip-version-check --require-hashes --only-binary=:all: \
  --no-deps --upgrade --target "$root" -r deploy/ci/cuda/requirements.txt
# The wheel stores libraries under lib; nvcc's Linux profile searches lib64.
test ! -e "$root/nvidia/cu13/lib64" || test -L "$root/nvidia/cu13/lib64"
if [ ! -L "$root/nvidia/cu13/lib64" ]; then ln -s lib "$root/nvidia/cu13/lib64"; fi
if [ ! -e "$root/nvidia/cu13/lib/libcudart.so" ]; then
  ln -s libcudart.so.13 "$root/nvidia/cu13/lib/libcudart.so"
fi
test -x "$root/nvidia/cu13/bin/nvcc"
"$root/nvidia/cu13/bin/nvcc" --version
printf 'JOPE_NVCC=%s\n' "$root/nvidia/cu13/bin/nvcc" >> "$GITHUB_ENV"
