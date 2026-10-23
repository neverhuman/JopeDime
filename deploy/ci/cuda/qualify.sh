#!/usr/bin/env bash
set -euo pipefail
mode=${1:?usage: qualify.sh host|compile|device}
case "$mode" in host|compile|device) ;; *) echo 'invalid CUDA qualification mode' >&2; exit 2;; esac
export JOPE_CUDA_PARITY="$mode" CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
export CARGO_PROFILE_DEV_CODEGEN_UNITS=1
out="target/cuda-evidence/$mode"
mkdir -p "$out" "${CARGO_TARGET_DIR:-$PWD/target}/cuda-tmp"
export TMPDIR="${CARGO_TARGET_DIR:-$PWD/target}/cuda-tmp"
source_sha=$(git rev-parse HEAD)
test "$source_sha" = "${JOPE_SOURCE_SHA:-$source_sha}"
git diff --quiet HEAD
printf '%s\n' "$mode" > "$out/mode.txt"
printf '%s\n' "$source_sha" > "$out/source.txt"
rustc --version > "$out/toolchains.txt"
g++ --version | head -1 >> "$out/toolchains.txt"
if [ "$mode" != host ]; then "${JOPE_NVCC:-nvcc}" --version >> "$out/toolchains.txt"; fi
if [ "$mode" = device ]; then nvidia-smi -q > "$out/device.txt"; fi
cargo test --locked -p feat-math --no-default-features --test cuda_export -- --test-threads=2 --nocapture 2>&1 | tee "$out/contracts.log"
cargo test --locked -p feat-math --no-default-features --test cuda_census -- --test-threads=2 --nocapture 2>&1 | tee "$out/census.log"
cargo test --locked -p feat-math --no-default-features --test golden -- --test-threads=2 --nocapture 2>&1 | tee "$out/golden.log"
cargo test --locked -p feat-math --no-default-features --lib advanced_v3::emit:: -- --test-threads=2 --nocapture 2>&1 | tee "$out/ordered-bound.log"
cargo test --locked -p feat-math --no-default-features --test artifact_target_binding 2>&1 | tee "$out/bindings.log"
cargo test --locked -p jope-pipeline --test artifact_target_binding 2>&1 | tee "$out/pipeline.log"
cargo test --locked -p jope-cli --no-default-features --lib kernel::artifact_v3_tests:: 2>&1 | tee "$out/cli.log"
python3 deploy/ci/cuda/test_evidence.py
python3 deploy/ci/cuda/evidence.py create "$out" "$mode"
