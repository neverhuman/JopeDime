#!/usr/bin/env bash
# summarize.sh <run_dir> : per-generation Markdown summary of one run_case.sh output directory.
#
# For every arm/seed run it reads <run>/search/cut-0/observations.json (the evidence that
# `verify-receipt --evidence` re-derives selection from) and reports, per generation:
#   best-so-far objective loss, and the RMSE and resource readings of the best-so-far program;
#   how many candidates were measured and how many were refused as resource_infeasible;
#   whether the best-so-far program was a generation-0 seed (origin "incumbent").
# It also writes generations.csv (one row per arm, seed, generation) for charts. Read-only on
# the evidence; jq and gawk only.
set -euo pipefail
out=${1:?usage: summarize.sh <run_dir>}
spec_name=$(jq -r .name "$out/run.json")
readout=(fit_seconds predict_us_per_row row_latency_p99_us fit_peak_heap_bytes model_bytes program_nodes)
csv=$out/generations.csv
echo "arm,rng_seed,generation,measured,infeasible,best_objective,best_rmse,best_origin,$(IFS=,; echo "${readout[*]}")" > "$csv"
echo "# $spec_name — run $(jq -r .started_utc "$out/run.json")"
echo
echo "Source \`$(jq -r .source_commit "$out/run.json")\`, jope sha256 \`$(jq -r .jope_sha256 "$out/run.json" | cut -c1-16)…\`, budget $(jq -c .budget "$out/run.json"), host $(jq -r .host "$out/run.json")."
echo
echo "| arm | seed | status | champion objective | champion RMSE | fit s | µs/row | p99 µs | heap MB | model KB | nodes | champion origin |"
echo "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
while read -r line; do
  arm=$(jq -r .arm <<<"$line"); seed=$(jq -r .rng_seed <<<"$line"); status=$(jq -r .status <<<"$line")
  dir=$(jq -r '.dir // empty' <<<"$line"); obs=$dir/search/cut-0/observations.json
  if [ -z "$dir" ] || [ ! -f "$obs" ]; then echo "| $arm | $seed | $status | | | | | | | | | |"; continue; fi
  # Per generation, the best-so-far program over measured candidates (lowest objective, then genome).
  jq -r --arg arm "$arm" --arg seed "$seed" --argjson keys "$(printf '%s\n' "${readout[@]}" | jq -R . | jq -s .)" '
    def score: (.objective_loss // .rmse);
    . as $all
    | ($all | map(select(.origin != "remeasure")) | map(.generation) | max) as $last
    | range(0; $last + 1) as $g
    | ($all | map(select(.origin != "remeasure" and .generation <= $g))) as $upto
    | ($upto | map(select(.status == "measured")) | sort_by([score, .genome]) | first) as $best
    | [$arm, $seed, $g,
       ($all | map(select(.origin != "remeasure" and .generation == $g and .status == "measured")) | length),
       ($all | map(select(.origin != "remeasure" and .generation == $g and ((.failure // "") | contains("resource_infeasible")))) | length),
       ($best | if . then score else "" end), ($best | if . then .rmse else "" end), ($best | if . then .origin else "" end)]
      + ($keys | map(. as $k | $best | if . then (.resources[$k] // "") else "" end))
    | @csv' "$obs" | tr -d '"' >> "$csv"
  # The champion: the receipt's selected program, with its recorded readings.
  receipt=$dir/receipt.json
  genome=$(jq -r .search.selected_genome "$receipt")
  jq -r --arg arm "$arm" --arg seed "$seed" --arg status "$status" --arg genome "$genome" \
     --argjson objective "$(jq '.search.best.validation_objective // .search.best.validation_rmse' "$receipt")" '
    map(select(.genome == $genome and .status == "measured")) | first as $c
    | def f(x; d): if x == null then "" else (x * d | round / d | tostring) end;
    "| \($arm) | \($seed) | \($status) | \(f($objective; 10000)) | \(f($c.rmse; 10000)) | \(f($c.resources.fit_seconds; 1000)) | \(f($c.resources.predict_us_per_row; 100)) | \(f($c.resources.row_latency_p99_us; 10)) | \(f(($c.resources.fit_peak_heap_bytes // null) | if . then . / 1048576 else null end; 10)) | \(f(($c.resources.model_bytes // null) | if . then . / 1024 else null end; 10)) | \(f($c.resources.program_nodes; 1)) | \($c.origin) |"' "$obs"
done < "$out/runs.jsonl"
echo
echo "## Best-so-far objective by generation (mean over seeds)"
echo
gawk -F, 'NR > 1 && $6 != "" { k = $1 SUBSEP $3; s[k] += $6; n[k]++; if ($3 > maxg) maxg = $3; arms[$1] = 1 }
  END {
    printf "| generation |"; for (a in arms) printf " %s |", a; printf "\n| --- |"; for (a in arms) printf " --- |"; printf "\n"
    for (g = 0; g <= maxg; g++) { printf "| %d |", g; for (a in arms) { k = a SUBSEP g; if (n[k]) printf " %.5g |", s[k] / n[k]; else printf " |" } printf "\n" }
  }' "$csv"
echo
echo "Per-generation rows for charts: \`$csv\`. Evidence: each run's \`search/cut-0/observations.json\`; verification logs \`<arm>/s<seed>.verify.log\`."
