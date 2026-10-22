#!/usr/bin/env bash
# Real-world objective benchmark. For each dataset, fit three ways on the same training rows and
# score the same held-out test rows with that dataset's objective:
#
#   rmse        --method auto            RMSE selection (the behaviour before objectives existed)
#   selected    --fitness F --calibration none   objective-aware selection only
#   calibrated  --fitness F              selection plus objective-aware output calibration
#
#   bash examples/objectives/real-world/fetch.sh            # once; data stays outside the checkout
#   bash examples/objectives/real-world/benchmark.sh [name ...]
#
# Environment: JOPE_FITNESS_DATA (fetch.sh's directory), JOPEDIME (binary; default: build
# target/release/jopedime), CANDIDATES (comma list), THREADS (0 = all), FOLDS (default 5).
# Writes target/fitness-real-world/results.{json,md}. Needs awk, jq and sha256sum.
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
data=${JOPE_FITNESS_DATA:-$HOME/jope-data/fitness-real-world}
out=${JOPE_FITNESS_OUT:-$root/target/fitness-real-world}
candidates=${CANDIDATES:-ridge,elasticnet,histgbm,extra_trees,gam_boost,ensemble}
threads=${THREADS:-0}
folds=${FOLDS:-5}
recipes="$root/examples/objectives"
if [ -z "${JOPEDIME:-}" ]; then
  export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
  (cd "$root" && cargo build --locked --release -p jope-appliance >&2)
  JOPEDIME="$root/target/release/jopedime"
fi
bin=$JOPEDIME

# name | target | split (chrono or hash) | objective | validation (iid or forward:GAP) | story
datasets=(
  "wind|target|chrono|pinball:quantile=0.1|iid|Firm wind commitment: the daily wind speed exceeded 9 days in 10 (Irish stations, 1961-78)"
  "houses|target|hash|mape:epsilon=1|iid|Valuation: relative error across price levels (California housing, 1990 census)"
  "pm10|target|hash|pinball:quantile=0.9|iid|Air-quality warning: an upper forecast of log PM10 near an Oslo road"
  "no2|target|hash|asymmetric_absolute:under_cost=4,over_cost=1|iid|Exposure alert: missing NO2 is 4x as costly as a false alarm (Oslo road)"
  "sp500|next_return|chrono|sortino:cost=0.001,periods_per_year=12|forward:1|Allocation: next-month S&P 500 exposure scored by after-cost Sortino (Shiller, 1901-2023)"
  "fremtpl|frequency|hash|@$recipes/claims_frequency_exposure.json|iid|Motor insurance pricing: claim frequency, Poisson deviance weighted by exposure (freMTPL2, 1/20 sample)"
)

column() { # csv name -> one value per data row
  awk -F, -v name="$2" -v file="$1" 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == name) c = i
      if (!c) { print "benchmark.sh: no column " name " in " file > "/dev/stderr"; exit 3 }; next }
    { print $c }' "$1"
}
json_array() { awk 'BEGIN { printf "[" } { printf "%s%s", (NR > 1 ? "," : ""), $0 } END { print "]" }'; }

split_rows() { # csv kind train test
  local rows
  rows=$(($(wc -l < "$1") - 1))
  awk -F, -v kind="$2" -v rows="$rows" -v train="$3" -v test="$4" '
    NR == 1 { print > train; print > test; next }
    { i = NR - 1
      held = kind == "chrono" ? (i > int(rows * 0.8)) : ((i * 2654435761) % 4294967296 % 5 == 0)
      print > (held ? test : train) }' "$1"
}

mkdir -p "$out"
: > "$out/results.jsonl"
selected=("$@")
for spec in "${datasets[@]}"; do
  IFS='|' read -r name target split objective validation story <<< "$spec"
  if [ ${#selected[@]} -gt 0 ] && [[ " ${selected[*]} " != *" $name "* ]]; then continue; fi
  csv="$data/csv/$name.csv"
  if [ ! -f "$csv" ]; then
    echo "skip $name: $csv not found (run fetch.sh; freMTPL needs FETCH_FREMTPL=1)" >&2
    continue
  fi
  work="$out/$name"
  rm -rf "$work"
  mkdir -p "$work"
  split_rows "$csv" "$split" "$work/train.csv" "$work/test.csv"
  awk -F, -v name="$target" 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == name) c = i }
    { line = ""; for (i = 1; i <= NF; i++) if (i != c) line = line (line == "" ? "" : ",") $i; print line }' \
    "$work/test.csv" > "$work/input.csv"
  "$bin" fitness validate "$objective" > "$work/objective.json"
  column "$work/test.csv" "$target" | json_array > "$work/targets.json"
  echo '{}' > "$work/columns.json"
  for c in $(jq -r '.columns[]' "$work/objective.json"); do
    column "$work/test.csv" "$c" | json_array > "$work/column.json"
    jq -c --arg c "$c" --slurpfile v "$work/column.json" '. + {($c): $v[0]}' "$work/columns.json" \
      > "$work/columns.next.json"
    mv "$work/columns.next.json" "$work/columns.json"
  done
  validation_args=(--validation iid)
  if [[ $validation == forward:* ]]; then validation_args=(--validation forward --gap "${validation#forward:}"); fi
  for arm in rmse selected calibrated; do
    args=(fit --train "$work/train.csv" --target "$target" --task regression --method auto
      --candidates "$candidates" --folds "$folds" --threads "$threads"
      --output "$work/$arm.jope" --json)
    # Every arm uses the dataset's validation scheme: chronological rows get forward folds.
    args+=("${validation_args[@]}")
    case $arm in
      rmse) ;;
      selected) args+=(--fitness "$objective" --calibration none) ;;
      calibrated) args+=(--fitness "$objective") ;;
    esac
    started=$(date +%s.%N)
    # An arm that cannot be fitted or whose predictions the objective cannot score (for example
    # a negative claim rate under a Poisson deviance) is recorded as undefined with its reason.
    undefined() {
      jq -n --slurpfile spec "$work/objective.json" --arg reason "$1" \
        '{value: null, loss: null, direction: $spec[0].direction, reason: $reason}' > "$work/$arm.score.json"
    }
    if ! "$bin" "${args[@]}" > "$work/$arm.fit.json"; then
      undefined "fit failed: $(jq -r '.error.message // "unknown"' "$work/$arm.fit.json")"
    else
      "$bin" predict --model "$work/$arm.jope" --input "$work/input.csv" \
        --output "$work/$arm.predictions.csv" --json > "$work/$arm.predict.json"
      column "$work/$arm.predictions.csv" prediction | json_array > "$work/$arm.predictions.json"
      jq -n --slurpfile spec "$work/objective.json" --slurpfile y "$work/targets.json" \
        --slurpfile p "$work/$arm.predictions.json" --slurpfile columns "$work/columns.json" \
        '{fitness: $spec[0].specification, targets: $y[0], predictions: $p[0], columns: $columns[0]}' \
        > "$work/$arm.evaluation.json"
      if ! "$bin" fitness evaluate --input "$work/$arm.evaluation.json" > "$work/$arm.score.json" \
        2> "$work/$arm.score.err"; then
        undefined "$(cat "$work/$arm.score.err")"
      fi
    fi
    seconds=$(awk -v a="$started" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')
    jq -c --arg dataset "$name" --arg arm "$arm" --arg story "$story" --arg seconds "$seconds" \
      --slurpfile fit "$work/$arm.fit.json" --slurpfile spec "$work/objective.json" \
      --arg train "$(($(wc -l < "$work/train.csv") - 1))" --arg test "$(($(wc -l < "$work/test.csv") - 1))" \
      '{dataset: $dataset, story: $story, objective: $spec[0].name, direction: .direction,
        arm: $arm, test_value: .value, test_loss: .loss, train_rows: ($train | tonumber),
        test_rows: ($test | tonumber), selected: ($fit[0].selection.selected // null),
        output_transform: ($fit[0].selection.calibration.output_transform // null), seconds: ($seconds | tonumber),
        undefined_reason: .reason}' \
      "$work/$arm.score.json" >> "$out/results.jsonl"
    echo "$name $arm: $(jq -r '"\(.value)"' "$work/$arm.score.json")" >&2
  done
done
jq -s '.' "$out/results.jsonl" > "$out/results.json"
jq -r '
  group_by(.dataset)[] as $rows
  | ($rows | map({(.arm): .}) | add) as $a
  | def show: if . == null then "undefined" else tostring | .[0:8] end;
    "| \($a.rmse.dataset) | \($a.rmse.objective) (\($a.rmse.direction)) | \($a.rmse.train_rows)/\($a.rmse.test_rows) | \($a.rmse.test_value | show) | \($a.selected.test_value | show) | \($a.calibrated.test_value | show) | \((if $a.rmse.test_loss == null or $a.calibrated.test_loss == null or $a.rmse.test_loss == 0 then "n/a" else (100 * ($a.rmse.test_loss - $a.calibrated.test_loss) / ($a.rmse.test_loss | fabs) | round | tostring) + "%" end)) |"
' "$out/results.json" | {
  echo "| dataset | objective | train/test rows | RMSE-selected | objective-selected | + calibration | loss reduction vs RMSE-selected |"
  echo "| --- | --- | ---: | ---: | ---: | ---: | ---: |"
  cat
} > "$out/results.md"
cat "$out/results.md"
