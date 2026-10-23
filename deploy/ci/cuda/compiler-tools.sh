#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
test "$(uname -s)/$(uname -m)" = Linux/x86_64
mode=${1:-compile}
case "$mode" in host|compile) ;; *) echo 'usage: compiler-tools.sh host|compile' >&2; exit 2;; esac
root="${JOPE_CUDA_TOOLS_DIR:-${CARGO_TARGET_DIR:-$PWD/target}/cuda-toolkit}"
mkdir -p "$root"
root=$(cd "$root" && pwd)
python3 -m pip install --disable-pip-version-check --require-hashes --only-binary=:all: \
  --no-deps --upgrade --target "$root" -r deploy/ci/cuda/tools-requirements.txt
test -x "$root/cmake/data/bin/cmake"
printf 'export PATH=%q:"%s"\n' "$root/cmake/data/bin" "\$PATH" > "$root/env.sh"
if [ "$mode" = host ]; then
  printf 'Build-tool environment: source %q\n' "$root/env.sh"
  exit 0
fi
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
printf 'export JOPE_NVCC=%q\n' "$root/nvidia/cu13/bin/nvcc" >> "$root/env.sh"
printf 'Build-tool environment: source %q\n' "$root/env.sh"
