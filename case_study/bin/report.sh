#!/usr/bin/env bash
# report.sh [stamp] : one Markdown table across every case's latest (or given) main run.
#
# For each case and arm it reads every verified run's receipt (the selected champion) and that
# champion's own observation (its recorded resource readings), then reports means over seeds:
#   accuracy  - the case's accuracy metric for the champion. For constrained objectives the loss
#               is the metric itself; for composite objectives it is RMSE; for the custom
#               serving-cost objective it is the loss minus the per-row serving charge;
#   primary   - the median over seeds of the champions' reading of the primary resource (readout[0]);
#   infeasible- share of candidates the objective refused (resource_infeasible).
# "A vs C" compares the business objective with accuracy-only ranking at the same DL seeds:
# how much cheaper the champion is in the primary resource, and what that costs in accuracy.
# Read-only on the evidence; jq and gawk only.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); top=$(dirname "$here")
out_root=${JOPE_CASE_OUT:-$HOME/jope-data/case_study}
want=${1:-}
echo "| Case | Primary resource | Arm A: accuracy / resource | Arm C: accuracy / resource | A vs C resource | A vs C accuracy | Arm B: accuracy / resource | Refused (A) | Runs A/B/C/D | Mean evaluations A/B/C/D |"
echo "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
for d in "$top"/[0-9][0-9]_*/; do
  d=${d%/}; name=$(basename "$d"); spec=$d/case.json
  if [ -n "$want" ]; then run=$out_root/$name/$want; else
    run=$( { ls -d "$out_root/$name"/2*Z 2>/dev/null || true; } | { grep -v smoke || true; } | while read -r r; do [ -f "$r/SUPERSEDED.txt" ] || ! grep -q '"arm":"A"' "$r/runs.jsonl" 2>/dev/null || echo "$r"; done | tail -1)
  fi
  [ -n "$run" ] && [ -f "$run/runs.jsonl" ] || { echo "| $name | | not run | | | | | | | |"; continue; }
  kind=$(jq -r .objective_kind "$spec"); primary=$(jq -r '.readout[0]' "$spec")
  price=$(jq -r 'if .metric == "custom" then (.expression.right.left.value) else 0 end' "$d/objective.json")
  priced=$(jq -r 'if .metric == "custom" then (.expression.right.right.name) else "" end' "$d/objective.json")
  jq -r 'select(.status == "verified") | [.arm, .dir] | @tsv' "$run/runs.jsonl" | while IFS=$'\t' read -r arm dir; do
    genome=$(jq -r .search.selected_genome "$dir/receipt.json")
    jq -r --arg arm "$arm" --arg genome "$genome" --arg kind "$kind" --arg primary "$primary" \
       --argjson price "$price" --arg priced "$priced" '
      (map(select(.origin != "remeasure")) | length) as $n
      | (map(select((.failure // "") | contains("resource_infeasible"))) | length) as $refused
      | map(select(.genome == $genome and .status == "measured")) | first as $c
      | ($c.resources // {}) as $r
      | (if ($arm == "C" or $arm == "D") then $c.objective_loss
         elif $kind == "constrained" then $c.objective_loss
         elif $kind == "custom" then ($c.objective_loss - $price * ($r[$priced] // 0))
         else $c.rmse end) as $acc
      | [$arm, $acc, ($r[$primary] // "NA"), $refused, $n] | @tsv' "$dir/search/cut-0/observations.json"
  done | gawk -F'\t' -v name="$name" -v primary="$primary" '
    function median(a,   k, i, j, t, v) { k = rn[a]; for (i = 1; i <= k; i++) v[i] = rv[a, i]; for (i = 2; i <= k; i++) { t = v[i]; for (j = i - 1; j >= 1 && v[j] > t; j--) v[j + 1] = v[j]; v[j + 1] = t } return (k % 2) ? v[(k + 1) / 2] : (v[k / 2] + v[k / 2 + 1]) / 2 }
    function ev(a) { return n[a] ? sprintf("%.0f", tot[a] / n[a]) : "–" }
    function fmt(x) { if (x == "") return "–"; a = (x < 0 ? -x : x); if (a >= 1e6) return sprintf("%.3g", x); if (a >= 100) return sprintf("%.0f", x); if (a >= 1) return sprintf("%.3g", x); return sprintf("%.4g", x) }
    { n[$1]++; acc[$1] += $2; if ($3 != "NA") { rn[$1]++; rv[$1, rn[$1]] = $3 + 0 } ref[$1] += $4; tot[$1] += $5 }
    END {
      for (a in n) { ma[a] = acc[a] / n[a]; mr[a] = (rn[a] ? median(a) : "") }
      dres = (n["A"] && n["C"] && mr["C"] > 0 && mr["A"] != "") ? sprintf("%+.0f%%", 100 * (mr["A"] - mr["C"]) / mr["C"]) : "–"
      dacc = (n["A"] && n["C"] && ma["C"] != 0) ? sprintf("%+.1f%%", 100 * (ma["A"] - ma["C"]) / ma["C"]) : "–"
      printf "| %s | `%s` | %s / %s | %s / %s | %s | %s | %s / %s | %s | %d/%d/%d/%d | %s/%s/%s/%s |\n", name, primary,
        (n["A"] ? fmt(ma["A"]) : "–"), (n["A"] ? fmt(mr["A"]) : "–"),
        (n["C"] ? fmt(ma["C"]) : "–"), (n["C"] ? fmt(mr["C"]) : "–"), dres, dacc,
        (n["B"] ? fmt(ma["B"]) : "–"), (n["B"] ? fmt(mr["B"]) : "–"),
        (tot["A"] ? sprintf("%.0f%%", 100 * ref["A"] / tot["A"]) : "–"), n["A"], n["B"], n["C"], n["D"],
        ev("A"), ev("B"), ev("C"), ev("D")
    }'
done
echo
echo "Over verified runs (RNG seeds 7, 11, 13): accuracy is the MEAN of the champions' accuracy metric (lower is better); resource is the MEDIAN of the champions' recorded reading of the primary resource (robust to one slow outlier; per-seed values are in each case's SUMMARY.md) (seconds, µs, bytes or nodes); throughput is higher-is-better. Source: each run's receipt.json and search/cut-0/observations.json."
