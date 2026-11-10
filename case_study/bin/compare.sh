#!/usr/bin/env bash
# compare.sh [--extended "STAMP [STAMP…]"] : validation-stage (cut-0) comparison, per case, under the case's own
# business objective (objective.json), of
#   the best FULLY feasible external baseline (feasible on all its seeds; lowest mean objective),
#   our arm-A champions (main run), and the final-round arm-A champions of an extended run,
#   our arm-C champions (accuracy-only ranking) re-scored under the business objective.
# Sources are files only: baselines from ~/jope-data/resfit-2071/baselines/<case>/summary.csv
# (scored by the pinned jopedime evaluator), ours from receipt.json + observations.json.
# Re-scoring C uses the objective's own form: a constrained objective is C's accuracy loss when C's
# recorded readings meet every limit (else infeasible); a composite is the weighted sum of its
# terms over C's recorded RMSE and readings; the custom serving-cost objective is C's MAE plus the
# per-row serving charge. Lower is better. jq and gawk only.
set -euo pipefail
# Run dirs recorded inside the xbabe4 container (/w/home/…) are mapped onto this host's evidence tree.
export JOPE_EV=${JOPE_CASE_OUT:-$HOME/jope-data/case_study}
extended=""; [ "${1:-}" = "--extended" ] && extended=$2
here=$(cd "$(dirname "$0")" && pwd); top=$(dirname "$here")
out_root=${JOPE_CASE_OUT:-$HOME/jope-data/case_study}
baselines=${JOPE_CASE_BASELINES:-$HOME/jope-data/resfit-2071/baselines}
echo "| Case | Best feasible baseline (mean objective) | Feasible baselines | Our A (mean objective) | Our A extended | Our E (best-practice seeds) | Our C under business objective | A vs best baseline | A-ext vs best baseline | E vs best baseline |"
echo "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
for d in "$top"/[0-9][0-9]_*/; do
  d=${d%/}; name=$(basename "$d"); obj=$d/objective.json
  run=$( { ls -d "$out_root/$name"/2*Z 2>/dev/null || true; } | { grep -v smoke || true; } | while read -r r; do [ -f "$r/SUPERSEDED.txt" ] || ! grep -q '"arm":"A"' "$r/runs.jsonl" 2>/dev/null || echo "$r"; done | tail -1)
  csv=$baselines/$name/summary.csv
  # Best fully feasible baseline: lowest mean objective_loss among models feasible on every seed.
  best=$( [ -f "$csv" ] && gawk -F, 'NR > 1 { n[$2]++; if ($4 == "true" && $5 != "") { f[$2]++; s[$2] += $5 } }
      END { bm = ""; for (m in n) if (f[m] == n[m] && n[m] > 0) { v = s[m] / f[m]; if (bm == "" || v < bv) { bm = m; bv = v } }
            k = 0; for (m in n) if (f[m] == n[m]) k++; nm = length(n)
            if (bm == "") printf "none\t\t%d/%d\n", k, nm; else printf "%s\t%.6g\t%d/%d\n", bm, bv, k, nm }' "$csv" || printf "pending\t\t\n")
  bmodel=$(cut -f1 <<<"$best"); bval=$(cut -f2 <<<"$best"); bfeas=$(cut -f3 <<<"$best")
  # Our champions' business objective, per run.
  champ() {  # <run dir> <arm>: business objective of the run's champion
    local dir=$1 arm=$2 g; g=$(jq -r .search.selected_genome "$dir/receipt.json")
    jq -r --arg g "$g" --arg arm "$arm" --slurpfile o "$obj" '
      ($o[0]) as $f
      | map(select(.genome == $g and .status == "measured")) | first as $c
      | ($c.resources // {}) as $r
      | if ($arm == "A") then $c.objective_loss
        elif $f.metric == "constrained" then
          (if all($f.limits[]; ($r[.resource] // null) as $v | $v != null
                 and (.max == null or $v <= .max) and (.min == null or $v >= .min)) then $c.objective_loss else "infeasible" end)
        elif $f.metric == "composite" then
          ($f.terms | map(.weight * (if .fitness.metric == "rmse" then $c.rmse else ($r[.fitness.name] // 0) end) / .scale) | add)
        elif $f.metric == "custom" then ($c.objective_loss + $f.expression.right.left.value * ($r[$f.expression.right.right.name] // 0))
        else "unsupported" end' "$dir/search/cut-0/observations.json"
  }
  mean_of() { gawk '{ if ($1 == "infeasible") inf++; else { s += $1; n++ } } END { if (n) printf "%.6g", s / n; else printf "–"; if (inf) printf " (%d infeasible)", inf }'; }
  a="–"; c="–"; ax="–"
  if [ -n "$run" ] && [ -f "$run/runs.jsonl" ]; then
    a=$(jq -r 'select(.status == "verified" and .arm == "A") | (.dir | sub("^.*/jope-data/case_study/"; $ENV.JOPE_EV + "/"))' "$run/runs.jsonl" | while read -r r; do champ "$r" A; done | mean_of)
    c=$(jq -r 'select(.status == "verified" and .arm == "C") | (.dir | sub("^.*/jope-data/case_study/"; $ENV.JOPE_EV + "/"))' "$run/runs.jsonl" | while read -r r; do champ "$r" C; done | mean_of)
  fi
  xdir=""; for x in $extended; do [ -d "$out_root/$name/extended-$x" ] && xdir=$out_root/$name/extended-$x; done
  e="–"
  edir=$( { ls -d "$out_root/$name"/2*Z 2>/dev/null || true; } | { grep -v smoke || true; } | while read -r r; do if grep -q '"arm":"E"' "$r/runs.jsonl" 2>/dev/null; then echo "$r"; fi; done | tail -1)
  if [ -n "$edir" ]; then
    e=$(jq -r 'select(.status == "verified" and .arm == "E") | (.dir | sub("^.*/jope-data/case_study/"; $ENV.JOPE_EV + "/"))' "$edir/runs.jsonl" | while read -r r; do champ "$r" A; done | mean_of)
  fi
  if [ -n "$xdir" ]; then
    ax=$(for s in "$xdir"/A/s*; do
           jq -r 'select(.status == "verified") | (.dir | sub("^.*/jope-data/case_study/"; $ENV.JOPE_EV + "/"))' "$s/rounds.jsonl" | tail -1
         done | while read -r r; do [ -n "$r" ] && champ "$r" A; done | mean_of)
  fi
  rel() { gawk -v x="$1" -v b="$2" 'BEGIN { if (x !~ /^[0-9.eE+-]+$/ || b !~ /^[0-9.eE+-]+$/ || b == 0) print "–"; else printf "%+.1f%% (%s)\n", 100 * (x - b) / b, (x < b ? "ours better" : "baseline better") }'; }
  printf "| %s | %s %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" "$name" "$bmodel" "${bval:+($bval)}" "$bfeas" "$a" "$ax" "$e" "$c" \
    "$(rel "${a%% *}" "$bval")" "$(rel "${ax%% *}" "$bval")" "$(rel "${e%% *}" "$bval")"
done
echo
echo "### Latency-limited cases with the baselines granted native-speed serving"
echo
echo "A Python-served one-row call costs the libraries hundreds of µs. Here every baseline is credited with its amortized batch µs/row as its one-row latency, an optimistic native-serving proxy. Every other limit (model bytes, fit seconds) stays as measured. The contest is then accuracy among programs that meet the budget."
echo
echo "| Case | Limits | Best proxy-feasible baseline (mean accuracy) | Proxy-feasible baselines | Our A (mean accuracy) | A vs best baseline |"
echo "| --- | --- | --- | --- | --- | --- |"
for d in "$top"/[0-9][0-9]_*/; do
  d=${d%/}; name=$(basename "$d"); obj=$d/objective.json; csv=$baselines/$name/summary.csv
  jq -e '.metric == "constrained" and any(.limits[]; .resource | startswith("row_latency"))' "$obj" >/dev/null 2>&1 || continue
  limits=$(jq -r '.limits | map("\(.resource) \(if .max != null then "max" else "min" end) \(.max // .min)") | join(";")' "$obj")
  proxy=$( [ -f "$csv" ] && gawk -F, -v limits="$limits" '
      NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
      { m = $2; n[m]++; ok = 1; L = split(limits, ls, ";")
        for (j = 1; j <= L; j++) { split(ls[j], p, " "); r = p[1]
          v = (r ~ /^row_latency/) ? $col["predict_us_per_row"] : $col[r]
          if (v == "") { ok = 0; continue } if ((p[2] == "max" && v + 0 > p[3] + 0) || (p[2] == "min" && v + 0 < p[3] + 0)) ok = 0 }
        acc = ($col["baseline_loss"] != "" ? $col["baseline_loss"] : $col["rmse"])
        if (ok && acc != "") { f[m]++; s[m] += acc } }
      END { bm = ""; k = 0; for (m in n) if (f[m] == n[m]) { k++; v = s[m] / f[m]; if (bm == "" || v < bv) { bm = m; bv = v } }
            if (bm == "") printf "none\t\t%d/%d\n", k, length(n); else printf "%s\t%.6g\t%d/%d\n", bm, bv, k, length(n) }' "$csv" || printf "pending\t\t\n")
  run=$( { ls -d "$out_root/$name"/2*Z 2>/dev/null || true; } | { grep -v smoke || true; } | while read -r r; do [ -f "$r/SUPERSEDED.txt" ] || ! grep -q '"arm":"A"' "$r/runs.jsonl" 2>/dev/null || echo "$r"; done | tail -1)
  a="–"
  if [ -n "$run" ] && [ -f "$run/runs.jsonl" ]; then
    a=$(jq -r 'select(.status == "verified" and .arm == "A") | (.dir | sub("^.*/jope-data/case_study/"; $ENV.JOPE_EV + "/"))' "$run/runs.jsonl" | while read -r r; do
          g=$(jq -r .search.selected_genome "$r/receipt.json")
          jq -r --arg g "$g" 'map(select(.genome == $g and .status == "measured")) | first | .objective_loss' "$r/search/cut-0/observations.json"
        done | gawk '{ s += $1; n++ } END { if (n) printf "%.6g", s / n; else printf "–" }')
  fi
  bm=$(cut -f1 <<<"$proxy"); bv=$(cut -f2 <<<"$proxy"); bk=$(cut -f3 <<<"$proxy")
  verdict=$(gawk -v x="$a" -v b="$bv" 'BEGIN { if (x !~ /^[0-9.eE+-]+$/ || b !~ /^[0-9.eE+-]+$/ || b == 0) print "–"; else printf "%+.1f%% (%s)\n", 100 * (x - b) / b, (x < b ? "ours better" : "baseline better") }')
  printf "| %s | %s | %s %s | %s | %s | %s |\n" "$name" "${limits//;/; }" "$bm" "${bv:+($bv)}" "$bk" "$a" "$verdict"
done
echo
echo "### Accuracy only (business limits ignored)"
echo
echo "The accuracy ceiling. Each alternative's accuracy metric, ignoring limits, against our accuracy-ranked arm C and our business-ranked arm A. A gap here is the accuracy that the business objective pays for its limits."
echo
echo "| Case | Accuracy metric | Best alternative (mean) | Runner-up alternative | Our C (accuracy-ranked) | Our A (business-ranked) | C vs best alternative | Our A extended (business-ranked) |"
echo "| --- | --- | --- | --- | --- | --- | --- | --- |"
for d in "$top"/[0-9][0-9]_*/; do
  d=${d%/}; name=$(basename "$d"); csv=$baselines/$name/summary.csv; obj=$d/objective.json
  metric=$(jq -r '.accuracy.metric' "$d/case.json")
  alts=$( [ -f "$csv" ] && gawk -F, 'NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
      { acc = ($col["baseline_loss"] != "" ? $col["baseline_loss"] : $col["rmse"]); if (acc != "") { s[$2] += acc; n[$2]++ } }
      END { for (m in n) printf "%.6g\t%s\n", s[m] / n[m], m }' "$csv" | sort -g | head -2 || true)
  b1=$(sed -n 1p <<<"$alts"); b2=$(sed -n 2p <<<"$alts")
  run=$( { ls -d "$out_root/$name"/2*Z 2>/dev/null || true; } | { grep -v smoke || true; } | while read -r r; do [ -f "$r/SUPERSEDED.txt" ] || ! grep -q '"arm":"A"' "$r/runs.jsonl" 2>/dev/null || echo "$r"; done | tail -1)
  ca="–"; aa="–"
  if [ -n "$run" ] && [ -f "$run/runs.jsonl" ]; then
    accs() {  # <arm>: the arm champions' accuracy metric
      jq -r --arg arm "$1" 'select(.status == "verified" and .arm == $arm) | (.dir | sub("^.*/jope-data/case_study/"; $ENV.JOPE_EV + "/"))' "$run/runs.jsonl" | while read -r r; do
        g=$(jq -r .search.selected_genome "$r/receipt.json")
        jq -r --arg g "$g" --arg arm "$1" --slurpfile o "$obj" '($o[0]) as $f
          | map(select(.genome == $g and .status == "measured")) | first as $c
          | if $arm == "C" or $f.metric == "constrained" then $c.objective_loss
            elif $f.metric == "custom" then ($c.objective_loss - $f.expression.right.left.value * ($c.resources[$f.expression.right.right.name] // 0))
            else $c.rmse end' "$r/search/cut-0/observations.json"
      done | gawk '{ s += $1; n++ } END { if (n) printf "%.6g", s / n; else printf "–" }'
    }
    ca=$(accs C); aa=$(accs A)
  fi
  # Extended arm A (final-round champions of the chained rounds), same accuracy definition.
  xa="–"; xdir=""; for x in $extended; do [ -d "$out_root/$name/extended-$x" ] && xdir=$out_root/$name/extended-$x; done
  if [ -n "$xdir" ]; then
    xa=$(for s in "$xdir"/A/s*; do jq -r 'select(.status == "verified") | (.dir | sub("^.*/jope-data/case_study/"; $ENV.JOPE_EV + "/"))' "$s/rounds.jsonl" | tail -1; done | while read -r r; do [ -n "$r" ] || continue
        g=$(jq -r .search.selected_genome "$r/receipt.json")
        jq -r --arg g "$g" --slurpfile o "$obj" '($o[0]) as $f | map(select(.genome == $g and .status == "measured")) | first as $c
          | if $f.metric == "constrained" then $c.objective_loss elif $f.metric == "custom" then ($c.objective_loss - $f.expression.right.left.value * ($c.resources[$f.expression.right.right.name] // 0)) else $c.rmse end' "$r/search/cut-0/observations.json"
      done | gawk '{ s += $1; n++ } END { if (n) printf "%.6g", s / n; else printf "–" }')
  fi
  bv=$(cut -f1 <<<"$b1")
  verdict=$(gawk -v x="$ca" -v b="$bv" 'BEGIN { if (x !~ /^[0-9.eE+-]+$/ || b !~ /^[0-9.eE+-]+$/ || b == 0) print "–"; else printf "%+.1f%% (%s)\n", 100 * (x - b) / b, (x < b ? "ours better" : "alternative better") }')
  printf "| %s | %s | %s | %s | %s | %s | %s | %s |\n" "$name" "$metric" "$(cut -f2 <<<"$b1") ${bv:+($bv)}" \
    "$(cut -f2 <<<"$b2") $( [ -n "$b2" ] && echo "($(cut -f1 <<<"$b2"))")" "$ca" "$aa" "$verdict" "$xa"
done
echo
echo "Validation stage (cut-0 holdout of train.csv), means over RNG seeds 7/11/13; lower objective is better. Baselines: catboost default and strong, lightgbm, xgboost, sklearn HGB (pre-declared configurations, 4 threads). A baseline counts only if it was feasible on every seed. Python-served one-row latency is included in the baselines' readings; see each case's notes."
