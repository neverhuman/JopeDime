#!/usr/bin/env bash
# deploy/ci/emitters.sh — the emitted-code matrix (plan "Code emission" §4; the rust-emitters lane).
#
# Two genomes, two proofs:
#   * the champion 8944cbdf (histgbm head): the Python/Rust bit-identical pair (the C++17/ES2020
#     emitters do not cover gradient-boosted heads — the exclusion is recorded in targets.txt);
#   * portable-4lang (the champion's transform stack over a ridge head): all four targets, with the
#     C++ compiled by g++ and clang++ under -std=c++17 and -std=c++20, plus the historical
#     g++ C++17 -O2 recipe.
# Every emitted program runs on the committed datasets/test/pmlb-229-pwlinear split; per genome the
# predictions must be bit-identical across programs and equal to the committed expectation, and the
# emitted sources byte-identical (crates/feat-math/tests/fixtures/emitted/<genome>.*.sha256).
# UPDATE=1 rewrites the expectations (review the diff). JOPE_CXX_MARCH adds -march=<value> to
# every C++ build. Nothing is skipped: a missing compiler fails the lane.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATASET=datasets/test/pmlb-229-pwlinear
EXPECT=crates/feat-math/tests/fixtures/emitted
OUT=${EMITTERS_OUT:-target/emitters/$(date -u +%Y%m%dT%H%M%S.%NZ)-$$}
CXX_FLAGS=(-O3 -ffp-contract=off -fno-fast-math)
CXX_O2_FLAGS=(-O2 -ffp-contract=off -fno-fast-math)
if [ -n "${JOPE_CXX_MARCH:-}" ]; then
  CXX_FLAGS+=("-march=${JOPE_CXX_MARCH}")
  CXX_O2_FLAGS+=("-march=${JOPE_CXX_MARCH}")
fi
fail=0

for tool in rustc g++ clang++ node python3 sha256sum timeout prlimit; do
  command -v "$tool" > /dev/null || { echo "missing required tool: $tool"; exit 1; }
done
[ ! -e "$OUT" ] || { echo "refusing to overwrite prior evidence: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
# The historical fixture executes and checks every v1 tag and its emitted source bytes. Keep a
# counted result here as well as in rust-test, so the emitter lane cannot pass with zero cases.
cargo test -p feat-math --locked --test historical_v1 > "$OUT/historical-v1.log" 2>&1
cat "$OUT/historical-v1.log"
python3 - "$OUT/historical-v1.log" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out", text)
if not match or int(match[1]) < 3 or any(int(value) for value in match.groups()[1:]):
    raise SystemExit("historical_v1 golden did not execute all required cases")
PY
# Complete-source streaming is an emission contract. Execute each named selection and
# retain its own receipt; the generalized target below intentionally selects one fixture.
source_contract() { # <label> <expected-passes> <filtered-count-pattern> <cargo arguments...>
  local label=$1 expected=$2 filtered=$3
  shift 3
  cargo test --locked "$@" > "$OUT/$label.log" 2>&1
  cat "$OUT/$label.log"
  if ! grep -qE "test result: ok\\. $expected passed; 0 failed; 0 ignored; 0 measured; $filtered filtered out" "$OUT/$label.log"; then
    echo "$label did not execute its declared source contract" >&2
    exit 1
  fi
}
source_contract legacy-wrapper-pins 1 '[0-9]+' -p feat-math --test capability_manifest \
  legacy_wrapper_body_hashes_are_pinned -- --exact
# Restart assertions share this harness process with subprocess-spawning tests. A forked
# child can briefly retain another test's flock descriptor after its parent drops the store.
# Serialize this target while retaining all 13 cases and the exclusive-owner assertions.
source_contract gp-source-stream 13 0 -p jope-observatory --test gp_source -- --test-threads=1
source_contract gp-source-capture 1 0 -p feat-math --test source_capture
source_contract gp-source-cli 2 0 -p jope-cli --test gp_source_capture
source_contract gp-source-kernel2 1 0 -p jope-pipeline --test gp_source_capture
source_contract gp-source-moves 1 '[0-9]+' -p jope-pipeline --lib \
  kernel2::search::moves::gp_source_move_capture_tests::complete_rewrite_and_linearization_proposals_are_captured -- --exact
source_contract gp-source-engine-grammar 1 '[0-9]+' -p jope-parallel --test grammar \
  grammar_source_capture_retains_decoded_and_projected_programs -- --exact
source_contract gp-source-policy-grammar 1 '[0-9]+' -p jope-policy --test grammar_differential \
  grammar_source_capture_retains_decoded_and_projected_programs -- --exact
source_contract gp-source-generalized 1 '[0-9]+' -p jope-dime-regressor --test generalized_gp \
  generalized_source_capture_retains_parseable_generation_zero_and_successors -- --exact

EMIT=$(cargo run -q --locked -p jope-ci -- cargo-artifact build --package feat-math --example emit)

# The emitted programs take (train.csv, test_features.csv, predictions): derive the features-only
# test file from the committed split (the last column is the target).
ncols=$(head -1 "$DATASET/train.csv" | awk -F, '{print NF}')
cut -d, -f1-"$((ncols - 1))" "$DATASET/test.csv" > "$OUT/test_features.csv"

check() { # <genome> <target> <file>
  local actual; actual=$(sha256sum "$3" | cut -c1-64)
  local pin="$EXPECT/$1.$2.sha256"
  if [ "${UPDATE:-0}" = 1 ]; then echo "$actual" > "$pin"; echo "  wrote $2 $actual"; return; fi
  if [ ! -f "$pin" ]; then echo "  MISSING expectation $pin (run with UPDATE=1)"; fail=1; return; fi
  local expected; expected=$(cat "$pin")
  if [ "$actual" = "$expected" ]; then echo "  ok $2 $actual"; else echo "  DRIFT $2: expected $expected, got $actual"; fail=1; fi
}

matrix() { # <genome-name> <dsl> [--all]
  local name=$1 dsl=$2 strict=${3:-}
  local dir="$OUT/$name"
  echo "== $name: emit"
  # shellcheck disable=SC2086 # $strict is either empty or the literal --all
  "$EMIT" $strict "$dsl" "$dir"
  local targets; targets=$(cat "$dir/targets.txt")
  echo "== $name: compile + run ($targets)"
  local run=() # program names
  run() { local prog=$1; shift; "$@" "$DATASET/train.csv" "$OUT/test_features.csv" "$dir/pred-$prog.txt"; run+=("$prog"); }
  python3 -m py_compile "$dir/model.py"; run python python3 "$dir/model.py"
  rustc -O -o "$dir/model-rs" "$dir/model.rs"; run rs "$dir/model-rs"
  # The C++ and JavaScript artifacts are libraries (C ABI / ES2020 functions); the CSV drivers in
  # deploy/ci/emitters/ are appended to make standalone programs with the same argv contract.
  case " $targets " in *" cpp "*)
    cat "$dir/model.cpp" deploy/ci/emitters/driver.cpp > "$dir/program.cpp"
    for cxx in g++ clang++; do for std in 17 20; do
      "$cxx" "-std=c++$std" "${CXX_FLAGS[@]}" -o "$dir/model-$cxx-$std" "$dir/program.cpp"
      run "$cxx-$std" "$dir/model-$cxx-$std"
    done; done
    # Preserve the existing -O3 matrix and also exercise the historical C++17 recipe.
    g++ -std=c++17 "${CXX_O2_FLAGS[@]}" \
      -o "$dir/model-g++-17-o2" "$dir/program.cpp"
    run g++-17-o2 "$dir/model-g++-17-o2";;
  esac
  case " $targets " in *" js "*)
    cat "$dir/model.js" deploy/ci/emitters/driver.js > "$dir/program.js"
    node --check "$dir/program.js"; run js node "$dir/program.js";;
  esac
  echo "== $name: compare predictions of ${#run[@]} programs (IEEE-754 bit patterns)"
  # Shared with deploy/ci/e2e-smoke.sh, which makes the same bit comparison.
  python3 deploy/ci/emitters/compare-bits.py "$dir" > "$dir/predictions.bits"
  echo "  $(wc -l < "$dir/predictions.bits") rows; programs agree: ${run[*]}"
  echo "== $name: expectations"
  for target in $targets; do
    case $target in py) check "$name" py "$dir/model.py";; rs) check "$name" rs "$dir/model.rs";; cpp) check "$name" cpp "$dir/model.cpp";; js) check "$name" js "$dir/model.js";; esac
  done
  check "$name" targets "$dir/targets.txt"
  check "$name" predictions "$dir/predictions.bits"
}

matrix 8944cbdf crates/feat-math/assets/champions/8944cbdf.dsl
matrix portable-4lang "$EXPECT/portable-4lang.dsl" --all
exit $fail
