#!/usr/bin/env bash
# Program search ranked by an objective versus by RMSE, on the committed regression datasets.
#
#   bash examples/objectives/real-world/search-benchmark.sh [OBJECTIVE]
#
# For every regression dataset under datasets/test, run `jope pipeline run` twice with the same
# seeds and budget, once ranking candidate programs by held-out RMSE and once by OBJECTIVE
# (default pinball:quantile=0.9), then score each selected program once on the sealed test split
# with OBJECTIVE. Environment: BUDGET (default evaluations=64,seconds=300), CUTS (default 2),
# THREADS (default 1), DATASETS (space-separated names; default every regression dataset) and
# JOPE_BIN (an existing `jope` binary; default builds target/release/jope). Writes target/fitness-search-benchmark/results.{jsonl,md}; the test-score
# registry is private to that directory.
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
objective=${1:-pinball:quantile=0.9}
budget=${BUDGET:-evaluations=64,seconds=300}
cuts=${CUTS:-2}
threads=${THREADS:-1}
out=${JOPE_SEARCH_OUT:-$root/target/fitness-search-benchmark}
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
if [ -n "${JOPE_BIN:-}" ]; then
  jope=$JOPE_BIN
else
  (cd "$root" && cargo build --locked --release --bin jope >&2)
  jope="$root/target/release/jope"
fi
rm -rf "$out"
mkdir -p "$out"
: > "$out/results.jsonl"
for dir in "$root"/datasets/test/*/; do
  name=$(basename "$dir")
  if [ "$(jq -r .task "$dir/metadata.json")" != regression ]; then continue; fi
  if [ -n "${DATASETS:-}" ] && [[ " $DATASETS " != *" $name "* ]]; then continue; fi
  mkdir -p "$out/$name"
  for arm in rmse objective; do
    args=(pipeline run --dataset "$dir" --budget "$budget" --cuts "$cuts" --threads "$threads"
      --policy skip --emit rust --output "$out/$name/$arm" --json)
    if [ "$arm" = objective ]; then args+=(--fitness "$objective"); fi
    "$jope" "${args[@]}" > "$out/$name/$arm.receipt.json" 2> "$out/$name/$arm.log" || {
      echo "$name $arm: run failed (see $out/$name/$arm.log)" >&2; continue; }
    "$jope" pipeline score --receipt "$out/$name/$arm/receipt.json" --split test \
      --registry "$out/registry-$arm" --fitness "$objective" --json > "$out/$name/$arm.score.json"
    jq -c --arg dataset "$name" --arg arm "$arm" \
      '{dataset: $dataset, arm: $arm, program_sha256, test_rmse, objective: .objective.name,
        test_loss: .objective.test_loss, validation_loss: .objective.validation_loss}' \
      "$out/$name/$arm.score.json" >> "$out/results.jsonl"
    echo "$name $arm: $(jq -r '.objective.test_loss' "$out/$name/$arm.score.json")" >&2
  done
done
jq -s -r '
  group_by(.dataset)[] | map({(.arm): .}) | add
  | select(.rmse and .objective)
  | "| \(.rmse.dataset) | \(.rmse.test_loss | tostring | .[0:8]) | \(.objective.test_loss | tostring | .[0:8]) | \(if .rmse.program_sha256 == .objective.program_sha256 then "same" else "different" end) |"
' "$out/results.jsonl" | {
  echo "| dataset | test loss, RMSE-ranked search | test loss, objective-ranked search | selected program |"
  echo "| --- | ---: | ---: | --- |"
  cat
} > "$out/results.md"
cat "$out/results.md"
