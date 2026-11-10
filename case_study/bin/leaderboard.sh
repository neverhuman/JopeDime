#!/usr/bin/env bash
# leaderboard.sh [--extended "STAMP …"] : which case studies are most compelling against the best
# alternative, from compare.sh's tables (validation stage, files only).
#
# For each case:
#   ours        - the best validation mean among our arms A, extended A and E (arm named),
#                 a validation-stage choice that the test look confirms or refutes;
#   alternative - the best alternative under the SAME objective. For latency-limited cases this is
#                 the conservative native-serving proxy (baselines credited with batch µs/row),
#                 otherwise the best fully feasible baseline;
#   margin      - (ours − alternative) / alternative; negative means ours is better.
# Ranking: the worse of temporal_risk and target_leakage_risk (case.json) first (LOW, MEDIUM, HIGH), then
# margin. A case where no alternative is feasible ranks by tier with margin "no feasible
# alternative", always with its comparability caveats. Nothing here is a test-split result.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); top=$(dirname "$here")
extended=""; [ "${1:-}" = "--extended" ] && extended=$2
tables=$("$here/compare.sh" --extended "$extended")
row() { awk -F'|' -v c="$1" -v t="$2" '/^### /{ sec++ } sec == t && $2 ~ c { print; exit }' <<<"$tables"; }
num() {  # the value in parentheses when present ("model (0.123)"), else the first number
  local p; p=$( { grep -oE '\([0-9.eE+-]+\)' <<<"$1" || true; } | head -1 | tr -d '()')
  if [ -n "$p" ]; then echo "$p"; else { grep -oE '[0-9]+(\.[0-9]+)?(e-?[0-9]+)?' <<<"$1" || true; } | head -1; fi
}
echo "| Rank | Case | Leakage risk | Ours (business objective) | Best alternative (same objective) | Margin | Alternatives feasible | Accuracy-only: our C vs best alternative | Caveats |"
echo "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
for d in "$top"/[0-9][0-9]_*/; do
  d=${d%/}; name=$(basename "$d")
  risk=$(jq -r '[.temporal_risk, .target_leakage_risk] | map(select(. != null)) | if index("HIGH") then "HIGH" elif index("MEDIUM") then "MEDIUM" elif length > 0 then "LOW" else "UNKNOWN" end' "$d/case.json")
  caveats=$(jq -r '.comparability // [] | join("; ")' "$d/case.json")
  t1=$(row "$name" 0); t2=$(row "$name" 1); t3=$(row "$name" 2)
  a=$(awk -F'|' '{ print $5 }' <<<"$t1" | xargs); ax=$(awk -F'|' '{ print $6 }' <<<"$t1" | xargs); ae=$(awk -F'|' '{ print $7 }' <<<"$t1" | xargs)
  # Ours = the best validation mean among our arms A, extended A and E (a validation-stage choice;
  # the pre-declared test look confirms or refutes it). The arm is named in the table.
  ours=""; src=""
  for cand in "A:$a" "extended A:$ax" "E:$ae"; do v=$(num "${cand#*:}"); [ -n "$v" ] || continue
    if [ -z "$ours" ] || gawk -v x="$v" -v y="$ours" 'BEGIN { exit !(x < y) }'; then ours=$v; src=${cand%%:*}; fi; done
  if [ -n "$t2" ]; then
    alt=$(awk -F'|' '{ print $4 }' <<<"$t2" | xargs); feas="$(awk -F'|' '{ print $5 }' <<<"$t2" | xargs) (native proxy)"
  else
    alt=$(awk -F'|' '{ print $3 }' <<<"$t1" | xargs); feas=$(awk -F'|' '{ print $4 }' <<<"$t1" | xargs)
  fi
  altv=$(num "$alt")
  margin=$(gawk -v x="$ours" -v b="$altv" -v alt="$alt" 'BEGIN { if (alt ~ /^none/) print "no feasible alternative"; else if (x == "" || b == "" || b == 0) print "pending"; else printf "%+.1f%%\n", 100 * (x - b) / b }')
  acc=$(awk -F'|' '{ print $8 }' <<<"$t3" | xargs)
  # Objectives that price program structure (program_nodes, stage_count) cannot score library models:
  # compare those cases on accuracy alone (the better of our A and extended A vs the best alternative).
  if jq -e '[.. | objects | (.name? // .resource? // empty)] | any(. == "program_nodes" or . == "stage_count")' "$d/objective.json" >/dev/null 2>&1; then
    ours=$(num "$(awk -F'|' '{ print $7 }' <<<"$t3")"); src="A, accuracy"; alt=$(awk -F'|' '{ print $4 }' <<<"$t3" | xargs); altv=$(num "$alt"); feas="n/a (objective prices program structure)"
    xa=$(num "$(awk -F'|' '{ print $9 }' <<<"$t3")")
    if [ -n "$xa" ] && { [ -z "$ours" ] || gawk -v x="$xa" -v y="$ours" 'BEGIN { exit !(x < y) }'; }; then ours=$xa; src="extended A, accuracy"; fi
    margin=$(gawk -v x="$ours" -v b="$altv" 'BEGIN { if (x == "" || b == "" || b == 0) print "pending"; else printf "%+.1f%% (accuracy only)\n", 100 * (x - b) / b }')
  fi
  key=$(case $risk in LOW) echo 1;; MEDIUM) echo 2;; HIGH) echo 3;; *) echo 4;; esac)
  mk=$(gawk -v m="$margin" -v x="$ours" 'BEGIN { if (x == "") print 1000; else if (m ~ /^no feasible/) print -1000; else if (m ~ /^[+-]/) print m + 0; else print 1000 }')
  printf "%s\t%s\t| %s | %s | %s | %s | %s | %s | %s | %s |\n" "$key" "$mk" "$name" "$risk" "${ours:-–} ($src)" "${alt:-–}" "$margin" "${feas:-–}" "${acc:-–}" "$caveats"
done | sort -t$'\t' -k1,1n -k2,2g | gawk -F'\t' '{ printf "| %d %s\n", NR, $3 }'
echo
echo "Validation stage (cut-0 holdout of train.csv); lower objective is better; margin < 0 means ours is better. 'No feasible alternative' means no baseline met every limit on every seed under the measurements listed in the caveats. The single pre-declared test-split look decides the final claims."
