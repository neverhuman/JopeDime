#!/usr/bin/env bash
# Replay an existing declared CI lane, with the same receipt parser and commands.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $# == 1 ]] || { echo 'usage: ci-local.sh {hygiene|rust-check|rust-test|emitters|inputs|python|e2e-smoke}' >&2; exit 2; }
lane=$1
case "$lane" in
  hygiene) backend=unittest ;;
  rust-check|emitters) backend=rust ;;
  rust-test) backend=nextest ;;
  python) backend=pytest ;;
  inputs|e2e-smoke) backend=none ;;
  *) echo "unknown local CI lane: $lane" >&2; exit 2 ;;
esac
# shellcheck source=ops/ci/lib.sh
source ops/ci/lib.sh
jope_ci_environment
GITHUB_RUN_ID=$(date +%s)
export GITHUB_RUN_ID GITHUB_RUN_ATTEMPT=1 GITHUB_EVENT_NAME=local-proof
JOPE_SOURCE_SHA=$(git rev-parse HEAD)
export JOPE_SOURCE_SHA
export GITHUB_SHA="$JOPE_SOURCE_SHA"
GITHUB_REF=$(git symbolic-ref -q HEAD || git rev-parse HEAD)
export GITHUB_REF
export JOPE_RUST_TEST_PARTITION="${JOPE_RUST_TEST_PARTITION:-slice:1/1}"
export JOPE_PARITY_STRICT=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
output="$PWD/target/ci-local/$GITHUB_RUN_ID/$lane"
args=(--lane "$lane" --output "$output" --backend "$backend")
case "$backend" in pytest|nextest) args+=(--junit "$output/junit.xml") ;; esac
exec python3 deploy/ci/receipt.py "${args[@]}" -- bash ops/ci/fast-lane.sh "$lane"
