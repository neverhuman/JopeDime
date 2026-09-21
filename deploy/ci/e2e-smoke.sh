#!/usr/bin/env bash
# shellcheck disable=SC2317  # predict/compile_and_predict are exported and invoked by name through run (indirect calls look unreachable to shellcheck)
# deploy/ci/e2e-smoke.sh — the `e2e-smoke` lane (main.yml / nightly.yml, self-hosted): one bounded
# pass of the product loop on a committed test dataset, written against the binaries that exist
# in this tree (`jope-parallel`, package jope-parallel; the feat-math `emit` example). Writes
# target/e2e/receipt.json (schema jope.e2e-smoke/v1, inline below) with every stage's command,
# exit code and digests, and exits non-zero unless every stage passed.
#
# Stages, in the plan's order:
#   toolchain   build jope-parallel (CPU) and the feat-math emit example
#   dataset     hash the committed datasets/test/<name> split
#   prepare     a `parallel.prepared.v1` root for the dataset (see below)
#   proposals   step-60000 policy proposals for the prepared cut (neural build + pinned checkpoint)
#   search      jope-parallel search --evaluations 40 --rng-seed 7 [--seeds proposals]
#   emit        emit the best genome to Python/Rust/C++/JavaScript, compile, predict, bit-compare
#
# A stage whose inputs come from a work package that has not landed fails with a precise message
# (status "blocked" in the receipt) instead of being skipped:
#   prepare    needs `jope datasets prepare` (WP7) or JOPE_E2E_PREPARED=<dir with prepared.json>.
#              The engine's own `prepare` subcommand consumes the migration audit
#              (audit.json + transfer.json + meta.json with provenance decisions), which the
#              committed split does not carry, and the `parallel.prepared.v1` hashes are computed
#              by Rust (Preprocessing::fit/transform + csv_bytes) — not reproducible from bash.
#   proposals  needs the step-60000 policy checkpoint from `jope models fetch --pinned` (WP7,
#              JOPE_MODELS_ROOT) and a `--features neural` jope-parallel (`propose` is neural-only).
#              Set JOPE_E2E_PROPOSALS=<seed_population.json> to supply proposals directly.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATASET_NAME=${JOPE_E2E_DATASET:-pmlb-229-pwlinear}
DATASET=datasets/test/$DATASET_NAME
EVALUATIONS=${JOPE_E2E_EVALUATIONS:-40}
SEED=${JOPE_E2E_SEED:-7}
CUT=${JOPE_E2E_CUT:-0}
VOCAB=crates/jope-engine/assets/grammar/featmath.json
export OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JOPE_PARITY_STRICT=1 CARGO_TERM_COLOR=never

for tool in cargo jq sha256sum python3 g++ clang++ node rustc; do
  command -v "$tool" >/dev/null || { echo "e2e-smoke: missing tool $tool" >&2; exit 2; }
done
target=$(cargo metadata --no-deps --locked --format-version 1 | jq -r .target_directory)
OUT=${JOPE_E2E_OUT:-$target/e2e}
rm -rf "$OUT"; mkdir -p "$OUT"
receipt=$OUT/receipt.json
stages='[]'
failed=0

record() { # name status command exit_code seconds detail-json
  stages=$(jq -c --arg n "$1" --arg s "$2" --arg c "$3" --argjson e "$4" --argjson t "$5" --argjson d "$6" \
    '. + [{name:$n,status:$s,command:$c,exit_code:$e,seconds:$t} + $d]' <<<"$stages")
  echo "== $1: $2"
}
run() { # name command [detail-json]  — runs the command, logs, records
  local name=$1 cmd=$2 detail=${3:-'{}'} log="$OUT/$1.log" start end code
  echo "== $name: $cmd"
  start=$(date +%s)
  set +e; bash -o pipefail -c "$cmd" > "$log" 2>&1; code=$?; set -e
  end=$(date +%s)
  detail=$(jq -c --arg log "${log#"$target"/}" --arg sha "$(sha256sum "$log" | cut -c1-64)" '. + {log:$log, log_sha256:$sha}' <<<"$detail")
  if [[ $code -eq 0 ]]; then record "$name" passed "$cmd" "$code" "$((end - start))" "$detail"
  else failed=1; tail -20 "$log"; record "$name" failed "$cmd" "$code" "$((end - start))" "$detail"; fi
  return $code
}
block() { # name command reason
  failed=1
  record "$1" blocked "$2" -1 0 "$(jq -nc --arg r "$3" '{reason:$r}')"
  echo "   $3"
}
finish() {
  local result=passed
  [[ $failed -eq 0 ]] || result=failed
  jq -e 'any(.[]; .status=="blocked")' <<<"$stages" >/dev/null && [[ $result == failed ]] && result=blocked
  jq -n --arg at "$(date -u +%FT%TZ)" --arg sha "$(git rev-parse HEAD)" --arg ds "$DATASET_NAME" \
    --argjson ev "$EVALUATIONS" --argjson seed "$SEED" --argjson cut "$CUT" --argjson stages "$stages" --arg result "$result" \
    '{schema:"jope.e2e-smoke/v1", produced_at:$at, git_sha:$sha, dataset:$ds, evaluations:$ev, rng_seed:$seed, cut:$cut, stages:$stages, result:$result}' > "$receipt"
  echo "e2e-smoke: result=$result receipt=$receipt"
  jq -r '.stages[] | "  \(.name): \(.status)\(if .reason then " — " + .reason else "" end)"' "$receipt"
  [[ $result == passed ]]
}
trap finish EXIT

# ---- toolchain -------------------------------------------------------------------------------
run toolchain "cargo build -q -p jope-parallel --bin jope-parallel --locked && cargo build -q -p feat-math --locked --example emit" || exit 1
ENGINE=$target/debug/jope-parallel
EMIT=$target/debug/examples/emit
[[ -x $ENGINE && -x $EMIT ]] || { echo "e2e-smoke: binaries missing after build" >&2; exit 1; }

# ---- dataset ---------------------------------------------------------------------------------
[[ -f $DATASET/train.csv && -f $DATASET/test.csv ]] || { block dataset "sha256sum $DATASET/*.csv" "committed dataset $DATASET is absent"; exit 1; }
run dataset "sha256sum $DATASET/train.csv $DATASET/test.csv" \
  "$(jq -nc --arg tr "$(sha256sum "$DATASET/train.csv" | cut -c1-64)" --arg te "$(sha256sum "$DATASET/test.csv" | cut -c1-64)" '{train_sha256:$tr,test_sha256:$te}')" || exit 1

# ---- prepare ---------------------------------------------------------------------------------
PREPARED=""
if [[ -n ${JOPE_E2E_PREPARED:-} ]]; then
  [[ -f $JOPE_E2E_PREPARED/prepared.json ]] || { block prepare "JOPE_E2E_PREPARED" "JOPE_E2E_PREPARED=$JOPE_E2E_PREPARED has no prepared.json"; exit 1; }
  PREPARED=$JOPE_E2E_PREPARED
  record prepare passed "JOPE_E2E_PREPARED=$PREPARED (supplied)" 0 0 "$(jq -nc --arg sha "$(sha256sum "$PREPARED/prepared.json" | cut -c1-64)" '{prepared_sha256:$sha}')"
elif cargo metadata --no-deps --locked --format-version 1 | jq -e '.packages[] | select(.name=="jope-datasets")' >/dev/null; then
  PREPARED=$OUT/prepared
  run prepare "cargo run -q -p jope-datasets --locked -- prepare --dataset $DATASET --output $PREPARED" || exit 1
else
  block prepare "cargo run -q -p jope-datasets --locked -- prepare --dataset $DATASET --output $OUT/prepared" \
    "no 'jope-datasets' package in the workspace (WP7: jope datasets prepare) and JOPE_E2E_PREPARED is unset; jope-parallel prepare needs the migration audit (audit.json, transfer.json, meta.json) that the committed split does not carry"
  exit 1
fi

# ---- proposals (step-60000 policy) ----------------------------------------------------------
SEEDS=""
if [[ -n ${JOPE_E2E_PROPOSALS:-} ]]; then
  [[ -f $JOPE_E2E_PROPOSALS ]] || { block proposals "JOPE_E2E_PROPOSALS" "JOPE_E2E_PROPOSALS=$JOPE_E2E_PROPOSALS does not exist"; exit 1; }
  SEEDS=$JOPE_E2E_PROPOSALS
  record proposals passed "JOPE_E2E_PROPOSALS=$SEEDS (supplied)" 0 0 "$(jq -nc --arg sha "$(sha256sum "$SEEDS" | cut -c1-64)" '{seeds_sha256:$sha}')"
else
  MODELS=${JOPE_MODELS_ROOT:-}
  CKPT=""
  [[ -n $MODELS && -f $MODELS/PINNED ]] && CKPT=$(grep -m1 -E 'policy.*step-?60000|step60000' "$MODELS/PINNED" || true)
  if [[ -z $CKPT ]]; then
    block proposals "$ENGINE propose --corpus <corpus> --checkpoint <step-60000> --output $OUT/proposals" \
      "no step-60000 policy checkpoint: JOPE_MODELS_ROOT=${MODELS:-unset} has no PINNED entry for it (WP7: jope models fetch --pinned); set JOPE_E2E_PROPOSALS=<parallel.seed_population.v1 json> to supply proposals"
    exit 1
  fi
  if ! "$ENGINE" propose --help >/dev/null 2>&1; then
    block proposals "$ENGINE propose --checkpoint $MODELS/$CKPT" \
      "'propose' is compiled only with --features neural (jope-parallel main.rs); build with '. deploy/ci/neural-env.sh && cargo build -p jope-parallel --features neural' on the neural-cuda runner"
    exit 1
  fi
  run proposals "$ENGINE --root $OUT propose --corpus $PREPARED --checkpoint $MODELS/$CKPT --output $OUT/proposals --cuts 1 --cut-start $CUT" || exit 1
  SEEDS=$OUT/proposals/seed_population.json
fi

# ---- search ----------------------------------------------------------------------------------
SEARCH=$OUT/search
seed_args=()
[[ -n $SEEDS ]] && seed_args=(--seeds "$SEEDS")
run search "JOPE_CPU_EVALUATOR=$ENGINE $ENGINE --root $OUT search --prepared $PREPARED --cut $CUT --output $SEARCH --evaluations $EVALUATIONS --seconds 600 --qualification --vocabulary $VOCAB --rng-seed $SEED ${seed_args[*]}" || exit 1
[[ -f $SEARCH/observations.json ]] || { echo "search wrote no observations.json" >&2; exit 1; }
best=$(jq -r '[.[] | select(.failure == null)] | min_by(.score // .rmse // 1e300) | .genome' "$SEARCH/observations.json")
[[ -n $best && $best != null ]] || { echo "no successful observation in $SEARCH/observations.json" >&2; exit 1; }
printf '%s\n' "$best" > "$OUT/best.dsl"
echo "   best genome: $best"

# ---- emit → compile → predict → bit-compare (same drivers as deploy/ci/emitters.sh) ----------
EMITTED=$OUT/emitted
ncols=$(head -1 "$DATASET/train.csv" | awk -F, '{print NF}')
cut -d, -f1-"$((ncols - 1))" "$DATASET/test.csv" > "$OUT/test_features.csv"
run emit "$EMIT --all $OUT/best.dsl $EMITTED" || exit 1
targets=$(cat "$EMITTED/targets.txt")
programs=()
predict() { local prog=$1; shift; "$@" "$DATASET/train.csv" "$OUT/test_features.csv" "$EMITTED/pred-$prog.txt"; programs+=("$prog"); }
compile_and_predict() {
  python3 -m py_compile "$EMITTED/model.py"; predict python python3 "$EMITTED/model.py"
  rustc -O -o "$EMITTED/model-rs" "$EMITTED/model.rs"; predict rs "$EMITTED/model-rs"
  case " $targets " in *" cpp "*)
    cat "$EMITTED/model.cpp" deploy/ci/emitters/driver.cpp > "$EMITTED/program.cpp"
    for cxx in g++ clang++; do for std in 17 20; do
      "$cxx" "-std=c++$std" -O3 -ffp-contract=off -fno-fast-math ${JOPE_CXX_MARCH:+-march=$JOPE_CXX_MARCH} -o "$EMITTED/model-$cxx-$std" "$EMITTED/program.cpp"
      predict "$cxx-$std" "$EMITTED/model-$cxx-$std"
    done; done;;
  esac
  case " $targets " in *" js "*)
    cat "$EMITTED/model.js" deploy/ci/emitters/driver.js > "$EMITTED/program.js"
    node --check "$EMITTED/program.js"; predict js node "$EMITTED/program.js";;
  esac
  python3 - "$EMITTED" > "$EMITTED/predictions.bits" <<'PY'
import glob, os, struct, sys
out = sys.argv[1]
bits = {}
for path in sorted(glob.glob(os.path.join(out, "pred-*.txt"))):
    rows = [line.strip() for line in open(path) if line.strip()]
    bits[os.path.basename(path)] = [struct.pack("<d", float(v)).hex() for v in rows]
first = next(iter(bits.values()))
for name, value in bits.items():
    if value != first:
        print(f"MISMATCH {name}", file=sys.stderr); sys.exit(1)
print("\n".join(first))
PY
}
export -f predict compile_and_predict
export EMITTED DATASET OUT targets
run compile-predict-compare "compile_and_predict" \
  "$(jq -nc --arg t "$targets" '{targets:$t}')" || exit 1
jq -c --arg sha "$(sha256sum "$EMITTED/predictions.bits" | cut -c1-64)" --argjson n "$(wc -l < "$EMITTED/predictions.bits")" \
  '.[-1] += {predictions_sha256:$sha, rows:$n}' <<<"$stages" > "$OUT/.stages" && stages=$(cat "$OUT/.stages")
echo "   $(wc -l < "$EMITTED/predictions.bits") predictions, bit-identical across: $targets"
