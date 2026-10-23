#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?usage: qualify.sh host|compile|device}
case "$mode" in host|compile|device) ;; *) echo 'invalid CUDA qualification mode' >&2; exit 2;; esac
export JOPE_CUDA_PARITY="$mode" CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
export CARGO_PROFILE_DEV_CODEGEN_UNITS=1
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2}
if [ "$mode" = host ]; then export JOPE_CUDA_SANITIZER=address,undefined; fi
out="${JOPE_CUDA_EVIDENCE_ROOT:-target/cuda-evidence}/$mode"
mkdir -p "$out" "${CARGO_TARGET_DIR:-$PWD/target}/cuda-tmp"
export TMPDIR="${CARGO_TARGET_DIR:-$PWD/target}/cuda-tmp"
source_sha=$(git rev-parse HEAD)
test "$source_sha" = "${JOPE_SOURCE_SHA:-$source_sha}"
git diff --quiet HEAD
printf '%s\n' "$mode" > "$out/mode.txt"
printf '%s\n' "$source_sha" > "$out/source.txt"
printf 'JOPE_CUDA_SANITIZER=%s\nASAN_OPTIONS=%s\nUBSAN_OPTIONS=%s\n' \
  "${JOPE_CUDA_SANITIZER:-}" "${ASAN_OPTIONS:-}" "${UBSAN_OPTIONS:-}" > "$out/environment.txt"
rustc --version > "$out/toolchains.txt"
g++ --version | head -1 >> "$out/toolchains.txt"
if [ "$mode" != host ]; then "${JOPE_NVCC:-nvcc}" --version >> "$out/toolchains.txt"; fi
if [ "$mode" = device ]; then nvidia-smi -q > "$out/device.txt"; fi
suite() {
  local name=$1
  shift
  # Discover the selected binary's tests before execution. A positive subset or
  # a truncated successful log cannot qualify as the complete selected suite.
  cargo test --locked "$@" -- --list > "$out/$name.list"
  cargo test --locked "$@" -- --test-threads=2 --nocapture 2>&1 | tee "$out/$name.log"
}
suite contracts -p feat-math --no-default-features --test cuda_export
suite census -p feat-math --no-default-features --test cuda_census
suite golden -p feat-math --no-default-features --test golden
suite ordered-bound -p feat-math --no-default-features --lib advanced_v3::emit::
suite bindings -p feat-math --no-default-features --test artifact_target_binding
suite pipeline -p jope-pipeline --test artifact_target_binding
suite cli -p jope-cli --no-default-features --lib kernel::artifact_v3_tests::
python3 deploy/ci/cuda/test_evidence.py
python3 deploy/ci/cuda/evidence.py create "$out" "$mode"
