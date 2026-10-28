#!/usr/bin/env bash
# Nonzero Rust contracts in the maintained rust-check lane; no live service or GPU required.
set -euo pipefail
cd "$(dirname "$0")/../.."
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=line-tables-only
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
export RUST_TEST_THREADS=${RUST_TEST_THREADS:-2}
export TMPDIR=${CARGO_TARGET_DIR:-$PWD/target}/narration-tmp
mkdir -p -- "$TMPDIR"
cargo fmt --all -- --check
cargo clippy --locked -p jope-narrator -p jope-pipeline -p jope-observatory -p jope-runtime --all-targets -- -D warnings
# A successful cargo exit alone is insufficient: renamed filters must not turn
# a required contract into a zero-test pass. Preserve each attempt's complete log.
contract() {
  local minimum=$1 label=$2 log result=0
  shift 2
  log=$(mktemp "$TMPDIR/${label}.XXXXXX.log")
  "$@" > "$log" 2>&1 || result=$?
  cat "$log"
  (( result == 0 )) || return "$result"
  python3 - "$minimum" "$label" "$log" <<'PYCOUNT'
import pathlib,re,sys
minimum,label,path=int(sys.argv[1]),sys.argv[2],sys.argv[3]
rows=re.findall(r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out",pathlib.Path(path).read_text())
counts=dict(zip(('passed','failed','ignored','measured','filtered_out'),(sum(int(row[i]) for row in rows) for i in range(5))))
print(f"narration-contract {label}: {counts}; log={path}")
assert counts['passed']>=minimum and counts['failed']==0, f"{label}: expected at least {minimum} executed passing tests"
PYCOUNT
}
contract 20 core cargo test --locked -p jope-narrator
contract 5 pipeline-internal cargo test --locked -p jope-pipeline --lib narration_
contract 4 pipeline-public cargo test --locked -p jope-pipeline --test search_narration
contract 13 transport cargo test --locked -p jope-observatory --test narration
contract 6 prediction cargo test --locked -p jope-runtime --test prediction_narration
