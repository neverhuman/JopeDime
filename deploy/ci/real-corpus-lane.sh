#!/usr/bin/env bash
# Host lane for the real-corpus and deferred-classification fixtures listed in
# deploy/ci/real-corpus-tests.txt. Every CI family ignores them because their inputs exist
# only in host custody. This lane runs exactly those tests, and only them, with the ADR-0003
# roots pointed at a read-only custody copy of the inputs. It is a manual host command, not a
# CI workflow job; its outcomes are host evidence, never CI verification.
#
# Usage: bash deploy/ci/real-corpus-lane.sh <custody-root>
#   <custody-root> holds MANIFEST.sha256, PROVENANCE.txt, sotarena/, state/ and registry/.
#   Optional deferred binary42 inputs belong under binary42/data and binary42/reports
#   in the same manifest; inherited fixture-root overrides are not used.
set -euo pipefail
cd "$(dirname "$0")/../.."
custody=${1:?custody root required (read-only copy with MANIFEST.sha256 and PROVENANCE.txt)}
shift
[[ $# == 0 ]] || { echo 'extra arguments cannot change the declared test selection' >&2; exit 2; }
custody=$(realpath -e -- "$custody")
for required in MANIFEST.sha256 PROVENANCE.txt sotarena state registry; do
  [[ -e $custody/$required ]] || { echo "unavailable: custody input $custody/$required" >&2; exit 78; }
done
# Read-only custody: a fixture must never be able to rewrite the inputs it is judged against.
writable=$(find "$custody" -perm /222 -print -quit)
[[ -z $writable ]] || { echo "custody copy is writable ($writable); restore it read-only" >&2; exit 1; }
# Reject links/special files and compare the entire closure. Reading paths from an
# untrusted manifest would allow escapes; hashing discovered regular files also
# rejects unlisted inputs and omitted files.
special=$(find "$custody" ! -type d ! -type f -print -quit)
[[ -z $special ]] || { echo "custody has a link or special file ($special)" >&2; exit 1; }
(cd "$custody" && diff -q \
  <(LC_ALL=C sort MANIFEST.sha256) \
  <(find . -type f ! -path './MANIFEST.sha256' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | LC_ALL=C sort)) \
  || { echo 'custody manifest does not cover the exact input closure' >&2; exit 1; }
echo "custody $custody verified: $(wc -l < "$custody/MANIFEST.sha256") files, manifest sha256 $(sha256sum "$custody/MANIFEST.sha256" | cut -d' ' -f1)"
sed 's/^/provenance: /' "$custody/PROVENANCE.txt"

list=deploy/ci/real-corpus-tests.txt
mapfile -t entries < <(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$list")
((${#entries[@]} > 0)) || { echo "no tests listed in $list" >&2; exit 1; }
filter=""
for entry in "${entries[@]}"; do
  read -r binary test rest <<< "$entry"
  [[ -n $binary && -n $test && -z $rest ]] || { echo "malformed $list entry: $entry" >&2; exit 1; }
  filter+="${filter:+ | }(binary_id($binary) & test(=$test))"
done

export SOTARENA_ROOT=$custody/sotarena JOPE_STATE=$custody/state JOPE_REGISTRY=$custody/registry
export SOTARENA_REPORTER_ROOT=$custody/state/arena JOPE_TEST_SOTARENA_ROOT=$custody/state/arena
export JOPE_BINARY42_TEST_DATA_ROOT=$custody/binary42/data JOPE_BINARY42_TEST_REPORT_ROOT=$custody/binary42/reports
export CARGO_INCREMENTAL=${CARGO_INCREMENTAL:-0} CARGO_PROFILE_DEV_DEBUG=${CARGO_PROFILE_DEV_DEBUG:-line-tables-only}
export OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
selection=(-P main --locked -p jope-dime-regressor --run-ignored only -E "$filter")
# Each entry must name exactly one test that is still ignored in CI.
collection=$(cargo nextest list "${selection[@]}" --message-format json)
bash deploy/ci/check-test-routing.sh host <(printf '%s\n' "$collection")
matched=$(jq '[.["rust-suites"][] | .testcases[] | select(.["filter-match"].status == "matches")] | length' <<< "$collection")
((matched == ${#entries[@]})) \
  || { echo "$list names ${#entries[@]} tests but $matched ignored tests match them" >&2; exit 1; }
echo "running ${#entries[@]} ignored real-corpus tests"
cargo nextest run "${selection[@]}" --test-threads "${JOPE_CPU_TEST_THREADS:-2}" --no-fail-fast
