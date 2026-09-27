#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p target/tmp
scratch=$(mktemp -d "$PWD/target/tmp/test-routing.XXXXXX")
trap 'rm -r -- "$scratch"' EXIT
mkdir -p "$scratch/deploy/ci"
cp deploy/ci/check-test-routing.sh "$scratch/deploy/ci/"
printf 'suite contract\n' > "$scratch/deploy/ci/neural-cpu-only-tests.txt"
printf 'suite contract\n' > "$scratch/deploy/ci/real-corpus-tests.txt"
printf '%s\n' '{"rust-suites":{"suite":{"testcases":{"contract":{"ignored":false,"filter-match":{"status":"matches"}}}}}}' > "$scratch/cpu.json"
jq '.["rust-suites"].suite.testcases.contract.ignored=true' "$scratch/cpu.json" > "$scratch/neural.json"
jq 'del(.["rust-suites"].suite.testcases.contract)' "$scratch/cpu.json" > "$scratch/missing.json"
check() { bash "$scratch/deploy/ci/check-test-routing.sh" "$@"; }
refuse() { if check "$@"; then echo "routing accepted invalid fixture: $*" >&2; exit 1; fi; }
jq '.["rust-suites"].suite.testcases.contract["filter-match"]={"status":"mismatch","reason":"expression"}' "$scratch/cpu.json" > "$scratch/filtered.json"
jq 'del(.["rust-suites"].suite.testcases.contract["filter-match"])' "$scratch/cpu.json" > "$scratch/no-filter.json"
check cpu "$scratch/cpu.json"
check neural "$scratch/neural.json"
check host "$scratch/neural.json"
refuse cpu "$scratch/filtered.json"
refuse cpu "$scratch/no-filter.json"
refuse cpu "$scratch/neural.json"
refuse neural "$scratch/cpu.json"
refuse host "$scratch/cpu.json"
refuse cpu "$scratch/missing.json"
refuse neural "$scratch/missing.json"
refuse host "$scratch/missing.json"
printf 'suite contract\nsuite contract\n' > "$scratch/deploy/ci/neural-cpu-only-tests.txt"
refuse cpu "$scratch/cpu.json"
printf 'suite\n' > "$scratch/deploy/ci/neural-cpu-only-tests.txt"
refuse cpu "$scratch/cpu.json"
: > "$scratch/deploy/ci/neural-cpu-only-tests.txt"
refuse cpu "$scratch/cpu.json"
echo 'test routing: 3 authorized controls and 11 refusals passed'
