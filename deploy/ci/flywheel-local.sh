#!/usr/bin/env bash
# Focused process-local checks. This script never dispatches hosted CI or publishes a status.
# Usage: bash deploy/ci/flywheel-local.sh <storage|feedback|parity|model-store> <full-sha> <new-absolute-output>
set -euo pipefail

usage() {
  echo 'usage: flywheel-local.sh <storage|feedback|parity|model-store> <full-sha> <new-absolute-output>' >&2
  exit 2
}
[[ $# == 3 ]] || usage
suite=$1
expected=$2
output=$3
case "$suite" in
  storage|feedback|parity|model-store) ;;
  *) usage ;;
esac
[[ $expected =~ ^[0-9a-f]{40}$ ]] || usage
[[ $output == /* && ! -e $output && ! -L $output ]] || usage
[[ ${GITHUB_ACTIONS:-false} != true ]] || { echo 'Run this check as a local process, outside GitHub Actions.' >&2; exit 2; }
cd "$(dirname "$0")/../.."
for tool in git cargo python3 jq timeout sha256sum; do
  command -v "$tool" >/dev/null || { echo "missing local prerequisite: $tool" >&2; exit 2; }
done
assert_source() {
  [[ $(git rev-parse HEAD) == "$expected" ]] || { echo 'Source HEAD changed or does not match the requested SHA.' >&2; exit 1; }
  [[ -z $(git status --porcelain --untracked-files=all) ]] || { echo 'Source is dirty; stop the writer and commit before checking.' >&2; exit 1; }
}
assert_source
jobs=${JOPE_LOCAL_JOBS:-2}
limit=${JOPE_LOCAL_TIMEOUT_S:-900}
[[ $jobs =~ ^[1-4]$ && $limit =~ ^[0-9]+$ ]] || usage
(( limit >= 1 && limit <= 1800 )) || usage
# Reuse the lane's existing target; never allocate a new target per check.
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
export CARGO_BUILD_JOBS=$jobs OMP_NUM_THREADS=$jobs OPENBLAS_NUM_THREADS=$jobs MKL_NUM_THREADS=$jobs RAYON_NUM_THREADS=$jobs
export CARGO_NET_OFFLINE=true
# Existing receipt schema uses numeric run IDs. These identify a local run, not a GitHub run.
local_run_id="$(date -u +%Y%m%d%H%M%S)$$"
export GITHUB_EVENT_NAME=local-proof GITHUB_RUN_ID=$local_run_id GITHUB_RUN_ATTEMPT=1
export GITHUB_SHA=$expected JOPE_SOURCE_SHA=$expected GITHUB_REF=local-proof
mkdir -p -- "$(dirname "$output")"
mkdir -- "$output"
# An output inside a nonignored part of the checkout would invalidate the source assertion.
assert_source
printf 'event\tlocal-proof\nsource_sha\t%s\nsuite\t%s\n' "$expected" "$suite" > "$output/identity.tsv"
sha256sum deploy/ci/flywheel-local.sh deploy/ci/receipt.py deploy/ci/results.py Cargo.lock rust-toolchain.toml > "$output/INPUTS.sha256"
deadline=$((SECONDS + limit))
receipt_lane=rust-test
[[ $suite != parity ]] || receipt_lane=emitters

run_tests() {
  local name=$1
  shift
  local remaining=$((deadline - SECONDS))
  (( remaining > 0 )) || { echo 'Local check exhausted its shared wall budget.' >&2; return 1; }
  assert_source
  timeout --signal=TERM --kill-after=10s "${remaining}s" python3 deploy/ci/receipt.py \
    --lane "$receipt_lane" --backend rust --output "$output/$name" -- \
    cargo test --offline --locked -j "$jobs" "$@" -- --test-threads="$jobs"
  # A successful command with no executed tests is never passing coverage.
  jq -e --arg sha "$expected" '
    .event == "local-proof" and .source_sha == $sha and .checkout_sha == $sha
    and .status == "passed" and .exit_status == 0
    and (.counts.executed | type == "number" and . > 0)
    and .counts.failed == 0 and .counts.passed == .counts.executed
    and .counts.skipped == 0
    and .dirty_source == []
  ' "$output/$name/receipt.json" >/dev/null
  assert_source
}

case "$suite" in
  storage)
    run_tests storage -p jope-learning --test retention --test retention_pack \
      --test retention_obligations --test retention_value --test executable_cache --test lake_views
    ;;
  feedback)
    run_tests feedback -p jope-learning --test pools --test selfplay_round_qual \
      --test replay_admission --test replay_sampling_design
    ;;
  parity)
    run_tests parity -p feat-math --test clover_quantile
    ;;
  model-store)
    run_tests model-store -p jope-datasets --lib store::bounded_version_tests
    ;;
esac
assert_source
printf 'Process-local %s checks passed at %s. Receipts: %s\n' "$suite" "$expected" "$output"
printf '%s\n' "$expected" > "$output/PASSED_SOURCE_SHA"
