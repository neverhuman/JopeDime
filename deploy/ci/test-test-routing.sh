#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p target/tmp
scratch=$(mktemp -d "$PWD/target/tmp/test-routing.XXXXXX")
trap 'rm -r -- "$scratch"' EXIT
mkdir -p "$scratch/deploy/ci" "$scratch/crates/x/src" "$scratch/crates/y/src"
cp deploy/ci/check-test-routing.sh "$scratch/deploy/ci/"
# Source attributes the reverse direction reads: one neural-routed contract, one corpus-routed
# fixture, a perf probe with its own reason (routes nowhere), and an unrelated crate reusing the
# contract's function name.
cat > "$scratch/crates/x/src/lib.rs" <<'RS'
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch"
)]
fn contract() {}

#[test]
#[ignore = "real-corpus: private inputs require the explicit host-real-corpus lane"]
fn corpus() {}

#[test]
#[ignore = "perf probe: timing only"]
fn probe() {}
RS
printf '#[test]\nfn contract() {}\n' > "$scratch/crates/y/src/lib.rs"
printf 'suite contract\n' > "$scratch/deploy/ci/neural-cpu-only-tests.txt"
printf 'suite corpus\n' > "$scratch/deploy/ci/real-corpus-tests.txt"
printf '%s\n' '{"rust-suites":{"suite":{"cwd":"/w/crates/x","testcases":{"contract":{"ignored":false,"filter-match":{"status":"matches"}},"corpus":{"ignored":true,"filter-match":{"status":"mismatch","reason":"ignored"}},"probe":{"ignored":true,"filter-match":{"status":"mismatch","reason":"ignored"}}}},"other":{"cwd":"/w/crates/y","testcases":{"contract":{"ignored":false,"filter-match":{"status":"matches"}}}}}}' > "$scratch/cpu.json"
jq '.["rust-suites"].suite.testcases.contract.ignored=true' "$scratch/cpu.json" > "$scratch/neural.json"
jq 'del(.["rust-suites"].suite.testcases.contract)' "$scratch/cpu.json" > "$scratch/missing.json"
jq '.["rust-suites"].suite.testcases.corpus.ignored=false' "$scratch/cpu.json" > "$scratch/corpus-active.json"
jq 'del(.["rust-suites"].suite.testcases.corpus)' "$scratch/cpu.json" > "$scratch/corpus-missing.json"
check() { bash "$scratch/deploy/ci/check-test-routing.sh" "$@"; }
refuse() { if check "$@"; then echo "routing accepted invalid fixture: $*" >&2; exit 1; fi; }
# A reverse refusal must fail for its own reason, not because a forward row broke.
refuse_with() {
  local expected=$1; shift
  if check "$@" 2> "$scratch/stderr"; then echo "routing accepted invalid fixture: $*" >&2; exit 1; fi
  grep -qF "$expected" "$scratch/stderr" || { cat "$scratch/stderr" >&2; echo "refused $* without: $expected" >&2; exit 1; }
  cat "$scratch/stderr" >&2
}
jq '.["rust-suites"].suite.testcases.contract["filter-match"]={"status":"mismatch","reason":"expression"}' "$scratch/cpu.json" > "$scratch/filtered.json"
jq 'del(.["rust-suites"].suite.testcases.contract["filter-match"])' "$scratch/cpu.json" > "$scratch/no-filter.json"
# Authorized controls. The probe and the other crate's `contract` are ignored or unlisted without
# failing: only a routing reason in the declaring crate obliges a row.
check cpu "$scratch/cpu.json"
check neural "$scratch/neural.json"
check host "$scratch/cpu.json"
check host "$scratch/neural.json"
# Forward refusals.
refuse cpu "$scratch/filtered.json"
refuse cpu "$scratch/no-filter.json"
refuse cpu "$scratch/neural.json"
refuse neural "$scratch/cpu.json"
refuse host "$scratch/corpus-active.json"
refuse cpu "$scratch/missing.json"
refuse neural "$scratch/missing.json"
refuse host "$scratch/corpus-missing.json"
# Reverse refusals: a new neural-only contract that nobody listed ...
cat >> "$scratch/crates/x/src/lib.rs" <<'RS'

mod tests {
    #[test]
    #[cfg_attr(feature = "neural", ignore = "cpu-fit-service: added without a row")]
    fn unlisted_contract() {}
}
RS
jq '.["rust-suites"].suite.testcases["tests::unlisted_contract"]={"ignored":false,"filter-match":{"status":"matches"}}' "$scratch/cpu.json" > "$scratch/cpu-new.json"
jq '.["rust-suites"].suite.testcases["tests::unlisted_contract"]={"ignored":true,"filter-match":{"status":"mismatch","reason":"ignored"}}' "$scratch/neural.json" > "$scratch/neural-new.json"
refuse_with 'unlisted: suite tests::unlisted_contract' cpu "$scratch/cpu-new.json"
refuse_with 'unlisted: suite tests::unlisted_contract' neural "$scratch/neural-new.json"
# ... and a new corpus-only fixture that nobody listed.
printf '\n#[test]\n#[ignore = "deferred-classification: added without a row"]\nfn unlisted_corpus() {}\n' >> "$scratch/crates/x/src/lib.rs"
jq '.["rust-suites"].suite.testcases.unlisted_corpus={"ignored":true,"filter-match":{"status":"mismatch","reason":"ignored"}}' "$scratch/cpu-new.json" > "$scratch/cpu-new2.json"
refuse_with 'unlisted: suite unlisted_corpus' host "$scratch/cpu-new2.json"
# Listing both restores the controls.
printf 'suite contract\nsuite tests::unlisted_contract\n' > "$scratch/deploy/ci/neural-cpu-only-tests.txt"
printf 'suite corpus\nsuite unlisted_corpus\n' > "$scratch/deploy/ci/real-corpus-tests.txt"
check cpu "$scratch/cpu-new2.json"
check host "$scratch/cpu-new2.json"
# Stale rows: listed and collected with the right status, but the source no longer routes them.
printf 'suite contract\nsuite tests::unlisted_contract\nother contract\n' > "$scratch/deploy/ci/neural-cpu-only-tests.txt"
refuse_with 'stale: deploy/ci/neural-cpu-only-tests.txt lists other contract' cpu "$scratch/cpu-new2.json"
printf 'suite corpus\nsuite unlisted_corpus\nsuite probe\n' > "$scratch/deploy/ci/real-corpus-tests.txt"
refuse_with 'stale: deploy/ci/real-corpus-tests.txt lists suite probe' host "$scratch/cpu-new2.json"
# Row shape.
printf 'suite contract\nsuite contract\n' > "$scratch/deploy/ci/neural-cpu-only-tests.txt"
refuse cpu "$scratch/cpu.json"
printf 'suite\n' > "$scratch/deploy/ci/neural-cpu-only-tests.txt"
refuse cpu "$scratch/cpu.json"
: > "$scratch/deploy/ci/neural-cpu-only-tests.txt"
refuse cpu "$scratch/cpu.json"
echo 'test routing: 6 authorized controls and 16 refusals passed (forward and reverse)'
