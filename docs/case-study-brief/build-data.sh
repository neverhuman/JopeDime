#!/usr/bin/env bash
# build-data.sh : derive every number and chart series of the case-study brief from the evidence.
#
# Inputs (read-only):
#   case_study/<case>/{case.json,objective.json}      the case definitions (this repository)
#   ${JOPE_CASE_OUT:-~/jope-data/case_study}/<case>/… receipts, observations, extended rounds
#   ${JOPE_CASE_BASELINES:-~/jope-data/resfit-2071/baselines}/<case>/summary.csv  (scored by jopedime)
# Outputs: docs/case-study-brief/data/ (CSV series for pgfplots, LaTeX macros and tables, and a
# provenance record hashing every input file read). Nothing is typed by hand: the brief's numbers
# all come from here. bash, jq and gawk only.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd)
cases_dir=$root/case_study; out=$here/data
evidence=${JOPE_CASE_OUT:-$HOME/jope-data/case_study}
export JOPE_EV=$evidence  # run dirs recorded inside the xbabe4 container are mapped onto this tree
baselines=${JOPE_CASE_BASELINES:-$HOME/jope-data/resfit-2071/baselines}
featured=${BRIEF_CASES:-"03 01 02 07"}
mkdir -p "$out" "$out/evo"; : > "$out/inputs.txt"
note() { printf '%s\n' "$1" >> "$out/inputs.txt"; }

latest_with_arm() {  # <case name> <arm>: latest non-superseded main stamp holding verified runs of <arm>
  { ls -d "$evidence/$1"/2*Z 2>/dev/null || true; } | { grep -v smoke || true; } | while read -r r; do
    [ -f "$r/SUPERSEDED.txt" ] && continue
    if grep -q "\"arm\":\"$2\",[^}]*\"status\":\"verified\"" "$r/runs.jsonl" 2>/dev/null; then echo "$r"; fi
  done | tail -1
}
latest_extended() {  # <case name>: the extended stamp with the most verified final rounds
  { ls -d "$evidence/$1"/extended-2* 2>/dev/null || true; } | { grep -v smoke || true; } | while read -r x; do
    n=$(cat "$x"/A/s*/rounds.jsonl 2>/dev/null | grep -c '"status":"verified"' || true); echo "$n $x"
  done | sort -n | tail -1 | cut -d' ' -f2
}
# accuracy and primary-resource reading of one run's champion
champion() {  # <run dir> <arm> <case dir> <primary resource>
  local dir=$1 arm=$2 cdir=$3 primary=$4 g
  g=$(jq -r .search.selected_genome "$dir/receipt.json")
  note "$dir/receipt.json"; note "$dir/search/cut-0/observations.json"
  jq -r --arg g "$g" --arg arm "$arm" --arg p "$primary" --slurpfile o "$cdir/objective.json" '
    ($o[0]) as $f | . as $all
    | map(select(.genome == $g and .status == "measured")) | first as $c
    | ($c.resources // {}) as $r
    | (if $arm == "C" or $f.metric == "constrained" then $c.objective_loss
       elif $f.metric == "custom" then ($c.objective_loss - $f.expression.right.left.value * ($r[$f.expression.right.right.name] // 0))
       else $c.rmse end) as $acc
    | ($all | map(select(.origin != "remeasure")) | length) as $ev
    | [$acc, ($r[$p] // ""), $c.objective_loss, ($r.model_bytes // ""), ($r.fit_seconds // ""), ($r.row_latency_p99_us // ""), $ev, ($g | .[0:60] | gsub(","; ";") | sub(" +$"; ""))] | @csv' \
    "$dir/search/cut-0/observations.json" | tr -d '"'
}

echo "case,arm,seed,accuracy,resource,objective,model_bytes,fit_seconds,p99_us,evaluations,genome" > "$out/champions.csv"
echo "case,model,n,feasible,accuracy,resource,predict_us_per_row,fit_seconds,model_bytes,objective" > "$out/baselines.csv"
echo "case,arm,stamp,host,source_commit,jope_sha256" > "$out/provenance.csv"
: > "$out/macros.tex"
macro() { printf '\\newcommand{\\%s}{%s}\n' "$1" "$2" >> "$out/macros.tex"; }

for num in $featured; do
  cdir=$(ls -d "$cases_dir/${num}"_*); name=$(basename "$cdir"); note "$cdir/case.json"; note "$cdir/objective.json"
  primary=$(jq -r '.readout[0]' "$cdir/case.json")
  for arm in A C E; do
    run=$(latest_with_arm "$name" "$arm"); [ -n "$run" ] || continue
    note "$run/run.json"; jq -r --arg c "$name" --arg a "$arm" --arg s "$(basename "$run")" '[$c, $a, $s, .host, .source_commit, .jope_sha256] | @csv' "$run/run.json" | tr -d '"' >> "$out/provenance.csv"
    jq -r --arg a "$arm" 'select(.status == "verified" and .arm == $a) | "\(.rng_seed) \(.dir | sub("^.*/jope-data/case_study/"; $ENV.JOPE_EV + "/"))"' "$run/runs.jsonl" | while read -r seed dir; do
      echo "$name,$arm,$seed,$(champion "$dir" "$arm" "$cdir" "$primary")" >> "$out/champions.csv"
    done
  done
  x=$(latest_extended "$name")
  if [ -n "$x" ]; then
    for s in "$x"/A/s*; do
      last=$(jq -r 'select(.status == "verified") | (.dir | sub("^.*/jope-data/case_study/"; $ENV.JOPE_EV + "/"))' "$s/rounds.jsonl" | tail -1); [ -n "$last" ] || continue
      seed=${s##*/s}; note "$s/rounds.jsonl"
      note "$s/run.json"; jq -r --arg c "$name" --arg s "$(basename "$x")/s$seed" '[$c, "A-extended", $s, .host, .source_commit, .jope_sha256] | @csv' "$s/run.json" | tr -d '"' >> "$out/provenance.csv"
      echo "$name,A-extended,$seed,$(champion "$last" A "$cdir" "$primary")" >> "$out/champions.csv"
    done
  fi
  csv=$baselines/$name/summary.csv
  if [ -f "$csv" ]; then
    note "$csv"
    gawk -F, -v c="$name" -v p="$primary" 'NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
      { m = $2; n[m]++; if ($4 == "true") f[m]++
        a = ($col["baseline_loss"] != "" ? $col["baseline_loss"] : $col["rmse"]); if (a != "") { sa[m] += a; na[m]++ }
        if ($col[p] != "") { sr[m] += $col[p]; nr[m]++ }
        if ($col["predict_us_per_row"] != "") { su[m] += $col["predict_us_per_row"]; nu[m]++ }
        if ($col["fit_seconds"] != "") { sf[m] += $col["fit_seconds"]; nf[m]++ }
        if ($col["model_bytes"] != "") { sb[m] += $col["model_bytes"]; nb[m]++ }
        if ($col["objective_loss"] != "") { so[m] += $col["objective_loss"]; no[m]++ } }
      END { for (m in n) printf "%s,%s,%d,%d,%s,%s,%s,%s,%s,%s\n", c, m, n[m], f[m] + 0,
              (na[m] ? sa[m] / na[m] : ""), (nr[m] ? sr[m] / nr[m] : ""), (nu[m] ? su[m] / nu[m] : ""),
              (nf[m] ? sf[m] / nf[m] : ""), (nb[m] ? sb[m] / nb[m] : ""), (no[m] ? so[m] / no[m] : "") }' "$csv" >> "$out/baselines.csv"
  fi
  # Per-case macros: limits from objective.json, means per arm, best feasible baseline.
  key=$(echo "$name" | cut -d_ -f1 | sed 's/^0*//'); K=$(echo "$key" | gawk '{ split("One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen", w, " "); print w[$1] }')
  macro "Case${K}Name" "$(jq -r '.dataset.display_name // .dataset.id' "$cdir/case.json" | sed 's/&/\\&/g; s/_/\\_/g')"
  macro "Case${K}Rows" "$(jq -r .dataset.rows "$cdir/case.json")"
  macro "Case${K}Features" "$(jq -r .dataset.features "$cdir/case.json")"
  read -r trows hrows < <(gawk -F, 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next } { print $c["train_rows"] + 0, $c["hold_rows"] + 0; exit }' "$baselines/$name/summary.csv" 2>/dev/null || echo "? ?")
  macro "Case${K}TrainRows" "$trows"; macro "Case${K}HoldRows" "$hrows"  # cut-0 rows used for selection
  macro "Case${K}Limits" "$(jq -r 'if .metric == "constrained" then (.limits | map("\(.resource | gsub("_"; "\\_")) \(if .max != null then "$\\le$ \(.max)" else "$\\ge$ \(.min)" end)") | join(", ")) else "none (priced)" end' "$cdir/objective.json")"
  for arm in A C E A-extended; do
    v=$(gawk -F, -v c="$name" -v a="$arm" '$1 == c && $2 == a && $4 != "" { s += $4; n++ } END { if (n) printf "%.5g", s / n }' "$out/champions.csv")
    r=$(gawk -F, -v c="$name" -v a="$arm" '$1 == c && $2 == a && $5 != "" { v[++n] = $5 + 0 } END { if (!n) exit; asort(v); printf "%.4g", (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2 }' "$out/champions.csv")
    k=$(gawk -F, -v c="$name" -v a="$arm" '$1 == c && $2 == a { n++ } END { print n + 0 }' "$out/champions.csv")
    tag=$(echo "$arm" | tr -d '-')
    macro "Case${K}${tag}Acc" "${v:-n/a}"; macro "Case${K}${tag}Res" "${r:-n/a}"; macro "Case${K}${tag}N" "$k"
  done
done


# Per-case chart series for pgfplots: x = the case's resource axis, y = accuracy (lower is better).
# Axis per case: 01 fit_seconds; 02 fit_cpu_seconds (primary); 03 row_latency_p99_us; 07
# predict_rows_per_second; 09 model_bytes (the structural cost both sides have). Each case also gets
# NN-plots.tex holding only the non-empty series, so every legend entry names the marks it shows.
short='BEGIN { split("xgboost_default XGBoost lightgbm_default LightGBM catboost_default CatBoost catboost_strong CatBoost-1000 sklearn_hgb HistGB autogluon_best AutoGluon ridge_cv Ridge random_forest RF extra_trees ExtraTrees knn10 kNN tabpfn_v2 TabPFN kumo_tabular_small Kumo", w, " "); for (i = 1; i < length(w); i += 2) s[w[i]] = w[i + 1] }'
for num in $featured; do
  name=$(basename "$(ls -d "$cases_dir/${num}"_*)")
  case $num in 09) ours_col=7; base_col=9;; *) ours_col=5; base_col=6;; esac
  structural=$(jq -e '[.. | objects | (.name? // .resource? // empty)] | any(. == "program_nodes" or . == "stage_count")' "$(ls -d "$cases_dir/${num}"_*)/objective.json" >/dev/null 2>&1 && echo 1 || echo 0)  # objective prices program structure: alternatives cannot be scored
  for arm in A A-extended C E; do
    gawk -F, -v c="$name" -v a="$arm" -v xc="$ours_col" 'BEGIN { print "x,y" } $1 == c && $2 == a && $xc != "" && $4 != "" { print $xc "," $4 }' "$out/champions.csv" > "$out/$num-ours-$arm.csv"
  done
  for feas in feasible infeasible; do
    # Labels alternate above and below their mark in x order, so neighbouring labels do not collide.
    gawk -F, -v c="$name" -v xc="$base_col" -v want="$feas" -v st="$structural" "$short"'
      $1 == c && $xc != "" && $5 != "" { f = (st == 1 || $4 == $3) ? "feasible" : "infeasible"; if (f == want) print $xc "," $5 "," (($2 in s) ? s[$2] : $2) }' "$out/baselines.csv" \
      | sort -t, -k1,1g | gawk 'BEGIN { print "x,y,label,anchor" } { print $0 "," ((NR % 2) ? "south west" : "north west") }' > "$out/$num-baselines-$feas.csv"
  done
  {
    plot() {  # <csv> <style> <legend>
      [ "$(wc -l < "$out/$1")" -gt 1 ] || return 0
      printf '\\addplot[%s] table[col sep=comma,x=x,y=y%s]{data/%s};\\addlegendentry{%s}\n' "$2" "$( case $1 in *baselines*) echo ",meta=label";; esac )" "$1" "$3"
    }
    labels='nodes near coords,point meta=explicit symbolic,visualization depends on={value \thisrow{anchor}\as\myanchor},every node near coord/.style={font=\tiny,anchor=\myanchor,inner sep=1pt'
    plot "$num-ours-A.csv" "only marks,mark=*,mark size=2.6pt,color=ours" "Ours: DL seeds (A)"
    plot "$num-ours-A-extended.csv" "only marks,mark=diamond*,mark size=3.4pt,color=oursext" "Ours: DL seeds, extended"
    plot "$num-ours-E.csv" "only marks,mark=square*,mark size=2.4pt,color=ourse" "Ours: best-practice seeds (E)"
    plot "$num-ours-C.csv" "only marks,mark=o,mark size=2.6pt,color=oursc,thick" "Ours: accuracy-only (C)"
    plot "$num-baselines-feasible.csv" "only marks,mark=triangle*,mark size=3pt,color=alt,$labels,color=alt}" "$( [ "$structural" = 1 ] && echo "Alternative (not scorable: no program)" || echo "Alternative, feasible")"
    plot "$num-baselines-infeasible.csv" "only marks,mark=x,mark size=3.2pt,color=bad,thick,$labels,color=bad}" "Alternative, infeasible"
  } > "$out/$num-plots.tex"
done
# Best feasible baseline accuracy per featured case (horizontal reference line).
for num in $featured; do
  name=$(basename "$(ls -d "$cases_dir/${num}"_*)")
  K=$(echo "$num" | sed 's/^0*//' | gawk '{ split("One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen", w, " "); print w[$1] }')
  best=$(gawk -F, -v c="$name" '$1 == c && $4 == $3 && $10 != "" { if (b == "" || $10 + 0 < b + 0) { b = $10; m = $2 } } END { if (b != "") { gsub(/_/, "\\_", m); printf "%s,%s", b, m } }' "$out/baselines.csv")
  macro "Case${K}BestFeasibleObj" "${best%%,*}"; macro "Case${K}BestFeasibleModel" "${best#*,}"
done

# Per-case results tables (LaTeX rows): our arms (mean accuracy, median resource, runs) and every
# alternative (mean accuracy, mean resource, feasible seeds), sorted by accuracy.
pretty() { case $1 in
  catboost_default) echo "CatBoost (default)";; catboost_strong) echo "CatBoost (1000 trees, depth 6)";;
  lightgbm_default) echo "LightGBM (default)";; xgboost_default) echo "XGBoost (default)";;
  sklearn_hgb) echo "scikit-learn HistGB";; autogluon_best) echo "AutoGluon (best\\_quality, 300 s)";;
  tabpfn_v2) echo "TabPFN v2 (Prior Labs, CPU)";; kumo_tabular_small) echo "Kumo Tabular small (NVIDIA, CPU)";;
  ridge_cv) echo "Ridge (CV)";; random_forest) echo "Random forest (300)";; extra_trees) echo "Extra trees (300)";;
  knn10) echo "k-NN (k=10)";; *) echo "$1" | sed 's/_/\\_/g';; esac; }
for num in $featured; do
  name=$(basename "$(ls -d "$cases_dir/${num}"_*)")
  case $num in 09) rc=7; bc=9;; *) rc=5; bc=6;; esac
  structural=$(jq -e '[.. | objects | (.name? // .resource? // empty)] | any(. == "program_nodes" or . == "stage_count")' "$(ls -d "$cases_dir/${num}"_*)/objective.json" >/dev/null 2>&1 && echo 1 || echo 0)  # objective prices program structure: alternatives cannot be scored
  {
    for arm in A A-extended E C; do
      label=$(case $arm in A) echo "Ours: DL seeds (A, 96 evals)";; A-extended) echo "Ours: DL seeds, extended (3 rounds)";;
                            E) echo "Ours: best-practice seeds (E, 96 evals)";; C) echo "Ours: accuracy-only ranking (C)";; esac)
      gawk -F, -v c="$name" -v a="$arm" -v xc="$rc" -v l="$label" '$1 == c && $2 == a { n++; s += $4; if ($xc != "") v[++k] = $xc + 0 }
        END { if (!n) exit; asort(v); m = k ? ((k % 2) ? v[(k + 1) / 2] : (v[k / 2] + v[k / 2 + 1]) / 2) : ""
              printf "\\textbf{%s} & \\num{%.4g} & %s & %d run%s \\\\\n", l, s / n, (m == "" ? "--" : sprintf("\\num{%.3g}", m)), n, (n > 1 ? "s" : "") }' "$out/champions.csv"
    done
    echo "\\midrule"
    gawk -F, -v c="$name" -v xc="$bc" '$1 == c && $5 != "" { printf "%s,%s,%s,%s,%s\n", $5, $2, $xc, $4, $3 }' "$out/baselines.csv" | sort -t, -k1,1g | while IFS=, read -r acc model res feas n; do
      printf '%s & \\num{%.4g} & %s & %s \\\\\n' "$(pretty "$model")" "$acc" "$( [ -n "$res" ] && printf '\\num{%.3g}' "$res" || echo --)" "$( [ "$structural" = 1 ] && echo "not scorable (no program)" || { [ "$feas" = "$n" ] && echo "feasible ($feas/$n)"; } || echo "\\textcolor{bad}{infeasible ($feas/$n)}")"
    done
  } > "$out/$num-table.tex"
done
# Derived figures the narrative cites.
g() { gawk -F, "$@" "$out/baselines.csv"; }
macro CaseThreeAGBytesMB "$(g '$1 ~ /^03_/ && $2 == "autogluon_best" { printf "%.0f", $9 / 1e6 }')"
macro CaseThreeLibPNinetyNineMin "$(g '$1 ~ /^03_/ && $2 != "autogluon_best" && $2 !~ /tabpfn|kumo/ && $6 != "" { if (m == "" || $6 + 0 < m + 0) m = $6 } END { printf "%.0f", m }')"
macro CaseThreeLibPNinetyNineMax "$(g '$1 ~ /^03_/ && $2 != "autogluon_best" && $2 !~ /tabpfn|kumo|extra|random|knn/ && $6 != "" { if ($6 + 0 > m + 0) m = $6 } END { printf "%.0f", m }')"
macro CaseThreeXGBAcc "$(g '$1 ~ /^03_/ && $2 == "xgboost_default" { printf "%.6g", $5 }')"
macro CaseOneAGFit "$(g '$1 ~ /^01_/ && $2 == "autogluon_best" { printf "%.0f", $8 }')"
macro CaseOneCatFit "$(g '$1 ~ /^01_/ && $2 == "catboost_default" { printf "%.2g", $8 }')"
macro CaseOneAGAcc "$(g '$1 ~ /^01_/ && $2 == "autogluon_best" { printf "%.5g", $5 }')"
macro CaseTwoXGBCpu "$(g '$1 ~ /^02_/ && $2 == "xgboost_default" { printf "%.3g", $6 }')"
macro CaseTwoXGBAcc "$(g '$1 ~ /^02_/ && $2 == "xgboost_default" { printf "%.5g", $5 }')"
macro CaseSevenCatAcc "$(g '$1 ~ /^07_/ && $2 == "catboost_default" { printf "%.5g", $5 }')"
macro CaseSevenCatRps "$(g '$1 ~ /^07_/ && $2 == "catboost_default" { printf "%.0f", $6 }')"
macro CaseNineHGBAcc "$(g '$1 ~ /^09_/ && $2 == "sklearn_hgb" { printf "%.5g", $5 }')"
macro CaseNineHGBBytes "$(g '$1 ~ /^09_/ && $2 == "sklearn_hgb" { printf "%.0f", $9 / 1e3 }')"
macro CaseNineOursBytes "$(gawk -F, '$1 ~ /^09_/ && $2 == "A" && $7 != "" { v[++n] = $7 / 1e3 } END { asort(v); printf "%.0f", (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2 }' "$out/champions.csv")"
macro CaseTwoBigModelMB "$(gawk -F, '$1 ~ /^02_/ && $2 == "A" && $7 != "" { if ($7 + 0 > m + 0) m = $7 } END { printf "%.0f", m / 1e6 }' "$out/champions.csv")"
macro CaseOneXGBAcc "$(g '$1 ~ /^01_/ && $2 == "xgboost_default" { printf "%.5g", $10 }')"
macro CaseOneEVsXGB "$(gawk -F, -v x="$(g '$1 ~ /^01_/ && $2 == "xgboost_default" { print $10 }')" '$1 ~ /^01_/ && $2 == "E" { s += $6; n++ } END { if (n && x) printf "%+.1f", 100 * (s / n / x - 1) }' "$out/champions.csv")"
macro CaseOneTabAcc "$(g '$1 ~ /^01_/ && $2 == "tabpfn_v2" { printf "%.5g", $5 }')"
macro CaseOneTabMsRow "$(g '$1 ~ /^01_/ && $2 == "tabpfn_v2" { printf "%.0f", $7 / 1000 }')"
macro CaseOneTabFit "$(g '$1 ~ /^01_/ && $2 == "tabpfn_v2" { printf "%.2g", $8 }')"
macro CaseThreeTabAcc "$(g '$1 ~ /^03_/ && $2 == "tabpfn_v2" { printf "%.5g", $5 }')"
macro CaseThreeTabMsRow "$(g '$1 ~ /^03_/ && $2 == "tabpfn_v2" { printf "%.0f", $7 / 1000 }')"
macro CaseTwoTabObj "$(g '$1 ~ /^02_/ && $2 == "tabpfn_v2" { printf "%.4g", $10 }')"
macro CaseNineAGAcc "$(g '$1 ~ /^09_/ && $2 == "autogluon_best" { printf "%.4g", $5 }')"
macro CaseNineAGMB "$(g '$1 ~ /^09_/ && $2 == "autogluon_best" { printf "%.0f", $9 / 1e6 }')"
macro CaseNineCatAcc "$(g '$1 ~ /^09_/ && $2 == "catboost_default" { printf "%.4g", $5 }')"
macro CaseNineVsHGB "$(gawk -F, -v h="$(g '$1 ~ /^09_/ && $2 == "sklearn_hgb" { print $5 }')" '$1 ~ /^09_/ && $2 == "A-extended" { s += $4; n++ } END { if (n && h) printf "%+.1f", 100 * (s / n / h - 1) }' "$out/champions.csv")"
macro CaseNineXNodes "$(gawk -F, '$1 ~ /^09_/ && $2 == "A-extended" { printf "%s%d", (n++ ? "--" : ""), $5 }' "$out/champions.csv" | gawk -F"--" '{ mn = $1; mx = $1; for (i = 2; i <= NF; i++) { if ($i < mn) mn = $i; if ($i > mx) mx = $i } print (mn == mx ? mn : mn "--" mx) }')"
macro CaseNineXKB "$(gawk -F, '$1 ~ /^09_/ && $2 == "A-extended" { v[++n] = $7 / 1e3 } END { asort(v); printf "%.0f", (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2 }' "$out/champions.csv")"


# Off-scale alternatives: points worse than 2x the best accuracy on the chart are left off the plot
# and named in the caption instead (Case<K>OffScale), so one very weak model cannot flatten the rest.
for num in $featured; do
  K=$(echo "$num" | sed 's/^0*//' | gawk '{ split("One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen", w, " "); print w[$1] }')
  cap=$(cat "$out/$num"-ours-*.csv "$out/$num"-baselines-feasible.csv "$out/$num"-baselines-infeasible.csv | gawk -F, '$1 != "x" && $2 != "" { if (m == "" || $2 + 0 < m) m = $2 + 0 } END { print 2 * m }')
  off=""
  for feas in feasible infeasible; do
    f=$out/$num-baselines-$feas.csv
    off=$off$(gawk -F, -v cap="$cap" 'NR > 1 && $2 + 0 > cap { printf "%s \\num{%.3g}; ", $3, $2 }' "$f")
    gawk -F, -v cap="$cap" 'NR == 1 || $2 + 0 <= cap' "$f" > "$f.kept"; mv "$f.kept" "$f"
  done
  macro "Case${K}OffScale" "$( [ -n "$off" ] && echo "Off the chart (worse than 2\$\\times\$ the best): ${off%; }." )"
done

# Margins versus the best feasible alternative under the same business objective (mean objective).
for num in $featured; do
  name=$(basename "$(ls -d "$cases_dir/${num}"_*)")
  K=$(echo "$num" | sed 's/^0*//' | gawk '{ split("One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen", w, " "); print w[$1] }')
  bestobj=$(gawk -F, -v c="$name" '$1 == c && $4 == $3 && $10 != "" { if (b == "" || $10 + 0 < b + 0) b = $10 } END { print b }' "$out/baselines.csv")
  for arm in A E A-extended; do
    tag=$(echo "$arm" | tr -d '-')
    read -r obj pct < <(gawk -F, -v c="$name" -v a="$arm" -v b="$bestobj" '$1 == c && $2 == a && $6 != "" { s += $6; n++ }
      END { if (!n) { print "n/a n/a"; exit } m = s / n; printf "%.5g %s\n", m, (b == "" ? "n/a" : sprintf("%+.1f", 100 * (m / b - 1))) }' "$out/champions.csv")
    macro "Case${K}${tag}Obj" "$obj"; macro "Case${K}${tag}VsBest" "$pct"
  done
done

# The full leaderboard (all 14 cases), from the repository's own generator, re-set for LaTeX.
"$cases_dir/bin/leaderboard.sh" --extended "$(ls -d "$evidence"/*/extended-2* 2>/dev/null | grep -v smoke | sed 's|.*/extended-||' | sort -u | tr '\n' ' ')" 2>/dev/null \
  | gawk -F'|' "$short"'
    function esc(t) { gsub(/\\/, "\\textbackslash{}", t); gsub(/_/, "\\_", t); gsub(/%/, "\\%", t); gsub(/&/, "\\&", t); return t }
    function trim(t) { gsub(/^ +| +$/, "", t); return t }
    NR > 2 && NF > 9 {
      for (i = 2; i <= 9; i++) $i = trim($i)
      cs = $3; sub(/_/, " ", cs); gsub(/_/, " ", cs)
      risk = ($4 == "HIGH") ? "\\textcolor{bad}{HIGH}" : (($4 == "MEDIUM") ? "\\textcolor{limit}{MED}" : "LOW")
      ours = $5; if (ours ~ /^–/ || ours == "") ours = "pending"
      else if (match(ours, /^([^ ]+) \((.*)\)$/, m)) ours = sprintf("\\num{%s} {\\tiny(%s)}", m[1], esc(m[2]))
      alt = $6; if (match(alt, /^([^ ]+) \((.*)\)$/, m)) alt = sprintf("%s \\num{%s}", ((m[1] in s) ? s[m[1]] : esc(m[1])), m[2]); else alt = esc(alt)
      mg = $7; sub(/ \(accuracy only\)/, "\\textsuperscript{a}", mg); sub(/ \(native proxy\)/, "\\textsuperscript{n}", mg)
      fe = $8; sub(/ \(objective prices program structure\)/, "", fe); sub(/ \(native proxy\)/, "\\textsuperscript{n}", fe)
      ac = $9; sub(/ \(ours better\)/, "", ac); sub(/ \(alternative better\)/, "", ac)
      printf "%s & %s & %s & %s & %s & %s & %s & %s \\\\\n", $2, esc(cs), risk, ours, alt, esc(mg), esc(fe), esc(ac) }' \
  | sed 's/\\textbackslash{}textsuperscript{\([an]\)}/\\textsuperscript{\1}/g; s/\\\\_/\\_/g' > "$out/leaderboard-rows.tex"

# Coverage of the foundation-model and AutoML alternatives across all 14 cases (global summary).
cov() { gawk -F, -v m="$1" '$2 == m && $7 != "" { split($1, p, "_"); c[p[1]] = 1 } END { n = asorti(c, k); for (i = 1; i <= n; i++) printf "%s%s", (i > 1 ? ", " : ""), k[i]; if (!n) printf "none yet" }' "$baselines/summary.csv"; }
[ -f "$baselines/summary.csv" ] && note "$baselines/summary.csv"
macro KumoCases "$(cov kumo_tabular_small)"; macro TabPFNCases "$(cov tabpfn_v2)"; macro AutoGluonCases "$(cov autogluon_best)"

# Provenance: every run the brief reads, by case, arm, stamp, host, commit and binary.
gawk -F, 'NR > 1 { split($1, p, "_"); sc = $5; b = ""; if ($4 !~ /^xbabe/) $4 = "xbabe4 (arm64 container)"; if (match(sc, /\+bundle:(.*)$/, m)) { b = m[1]; sub(/\+bundle:.*/, "", sc) }
    k = p[1] SUBSEP $2 SUBSEP $3; if (k in seen) next; seen[k] = 1
    st = $3; gsub(/_/, "\\_", st)
    printf "%s & %s & \\texttt{%s} & %s & \\texttt{%s}%s & \\texttt{%s} \\\\\n", p[1], $2, st, $4, substr(sc, 1, 12), (b != "" ? " + bundle \\texttt{" b "}" : ""), substr($6, 1, 12) }' "$out/provenance.csv" > "$out/provenance-rows.tex"
macro BriefHosts "$(gawk -F, 'NR > 1 && $4 ~ /^xbabe/ { h[$4] = 1 } END { n = asorti(h, k); for (i = 1; i <= n; i++) printf "%s%s", (i > 1 ? ", " : ""), k[i] }' "$out/provenance.csv")"  # x86 hosts; the xbabe4 container records its container id
macro BriefHasArm "$(gawk -F, 'NR > 1 && $4 !~ /^xbabe/ { n++ } END { print (n ? "yes" : "no") }' "$out/provenance.csv")"
macro BriefEvidenceCommit "$(gawk -F, 'NR > 1 { sc = $5; sub(/\+bundle:.*/, "", sc); c[substr(sc, 1, 12)] = 1 } END { n = asorti(c, k); for (i = 1; i <= n; i++) printf "%s%s", (i > 1 ? ", " : ""), k[i] }' "$out/provenance.csv")"
macro BriefJopeSha "$(gawk -F, 'NR > 1 { c[substr($6, 1, 12)] = 1 } END { n = asorti(c, k); for (i = 1; i <= n; i++) printf "%s%s", (i > 1 ? ", " : ""), k[i] }' "$out/provenance.csv")"


# ---------------------------------------------------------------------------------------------
# Evolution of each case's business KPI (objective.json loss) per GP generation, for EVERY case.
# Series: A / B / E main runs (96 evaluations or 600 s), and extended chains (X = arm A, Y = arm E),
# whose generation index runs on across rounds. For each run: gen_best = the best measured
# (feasible) objective among that generation's first measurements (remeasures excluded);
# best_so_far = the running minimum, i.e. the global best the search holds after that generation.
# Every main run's final best_so_far is checked against receipt.json .search.best.validation_objective.
evo_obs() {  # <run dir> <series> <seed> <generation offset>
  note "$1/search/cut-0/observations.json"; note "$1/receipt.json"
  jq -r --arg s "$2" --arg seed "$3" --argjson off "$4" '
    map(select(.origin != "remeasure")) as $o | ($o | map(.generation) | max) as $last
    | range(0; $last + 1) as $g
    | [$s, $seed, ($off + $g), ($o | map(select(.generation == $g and .status == "measured" and .objective_loss != null) | .objective_loss) | min)] | @csv' \
    "$1/search/cut-0/observations.json" | tr -d '"'
}
mkdir -p "$out/evo"
echo "case,series,seed,generation,gen_best,best_so_far" > "$out/evolution.csv"
echo "case,series,seed,run,final_best_so_far,receipt_best,match" > "$out/evolution-checks.csv"
for cdir in "$cases_dir"/[0-9][0-9]_*/; do
  cdir=${cdir%/}; name=$(basename "$cdir"); num=${name%%_*}
  raw=$out/evo/$num.long.csv; : > "$raw"  # per-case working series (kept; overwritten each build)
  for arm in A B E; do
    run=$(latest_with_arm "$name" "$arm"); [ -n "$run" ] || continue
    while read -r seed dir; do
      evo_obs "$dir" "$arm" "$seed" 0 >> "$raw"
      fin=$(gawk -F, -v s="$arm" -v d="$seed" '$1 == s && $2 == d && $4 != "" { if (b == "" || $4 + 0 < b + 0) b = $4 } END { print b }' "$raw")
      rec=$(jq -r '.search.best.validation_objective // empty' "$dir/receipt.json")
      gawk -v c="$name" -v s="$arm" -v d="$seed" -v r="$dir" -v f="$fin" -v e="$rec" 'BEGIN { m = (f != "" && e != "" && (f - e) * (f - e) <= 1e-24 * e * e) ? "yes" : "NO"; print c "," s "," d "," r "," f "," e "," m }' >> "$out/evolution-checks.csv"
    done < <(jq -r --arg a "$arm" 'select(.status == "verified" and .arm == $a) | "\(.rng_seed) \(.dir | sub("^.*/jope-data/case_study/"; $ENV.JOPE_EV + "/"))"' "$run/runs.jsonl")
  done
  for xarm in A E; do  # the extended stamp with the most verified rounds for this arm
    x=$({ ls -d "$evidence/$name"/extended-2* 2>/dev/null || true; } | { grep -v smoke || true; } | while read -r d; do
          n=$(cat "$d/$xarm"/s*/rounds.jsonl 2>/dev/null | grep -c '"status":"verified"' || true); if [ "$n" -gt 0 ]; then echo "$n $d"; fi; done | sort -n | tail -1 | cut -d' ' -f2)
    [ -n "$x" ] || continue
    ser=$([ "$xarm" = A ] && echo X || echo Y)
    for s in "$x/$xarm"/s*; do
      seed=${s##*/s}; off=0; note "$s/rounds.jsonl"
      while read -r rdir; do
        evo_obs "$rdir" "$ser" "$seed" "$off" >> "$raw"
        off=$(( off + $(jq '[.[] | select(.origin != "remeasure") | .generation] | max + 1' "$rdir/search/cut-0/observations.json") ))
      done < <(jq -r 'select(.status == "verified") | (.dir | sub("^.*/jope-data/case_study/"; $ENV.JOPE_EV + "/"))' "$s/rounds.jsonl")
      last=$(jq -r 'select(.status == "verified") | (.dir | sub("^.*/jope-data/case_study/"; $ENV.JOPE_EV + "/"))' "$s/rounds.jsonl" | tail -1); [ -n "$last" ] || continue
      fin=$(gawk -F, -v s="$ser" -v d="$seed" '$1 == s && $2 == d && $4 != "" { if (b == "" || $4 + 0 < b + 0) b = $4 } END { print b }' "$raw")
      rec=$(jq -r '.search.best.validation_objective // empty' "$last/receipt.json")
      gawk -v c="$name" -v s="$ser" -v d="$seed" -v r="$last" -v f="$fin" -v e="$rec" 'BEGIN { m = (f != "" && e != "" && (f - e) * (f - e) <= 1e-24 * e * e) ? "yes" : ((e + 0 > f + 0) ? "earlier-round" : "NO"); print c "," s "," d "," r "," f "," e "," m }' >> "$out/evolution-checks.csv"
    done
  done
  # best_so_far per (series, seed), then a wide table per case for pgfplots (nan = no data).
  gawk -F, -v c="$name" '{ k = $1 SUBSEP $2; if ($4 != "" && (!(k in b) || $4 + 0 < b[k] + 0)) b[k] = $4; print c "," $1 "," $2 "," $3 "," $4 "," ((k in b) ? b[k] : "") }' "$raw" >> "$out/evolution.csv"
done
cols="A7 A11 A13 B7 B11 B13 E7 E11 E13 X7 X11 X13 Y7 Y11 Y13"
gawk -F, -v cols="$cols" -v o="$out/evo" 'NR > 1 { split($1, p, "_"); n = p[1]; seen[n] = 1; k = $2 $3; if ($6 != "") v[n, $4, k] = $6; if ($5 != "") gb[n, $4, k] = $5; if ($4 + 0 > mx[n]) mx[n] = $4 + 0; have[n, k] = 1 }
  END { nc = split(cols, C, " "); for (n in seen) { f = o "/" n ".csv"; h = "g"; for (i = 1; i <= nc; i++) h = h "," C[i] ",g" C[i]; print h > f
          for (g = 0; g <= mx[n]; g++) { line = g; for (i = 1; i <= nc; i++) { k = C[i]; line = line "," (((n, g, k) in v) ? v[n, g, k] : "nan") "," (((n, g, k) in gb) ? gb[n, g, k] : "nan") } print line > f }
          close(f) } }' "$out/evolution.csv"


# Evolution figures. Reference line per case = the leaderboard's best alternative under the SAME
# business objective (native-serving proxy where the leaderboard used it); none where the
# objective cannot score alternatives (structure-priced) or none is feasible.
"$cases_dir/bin/leaderboard.sh" --extended "$(ls -d "$evidence"/*/extended-2* 2>/dev/null | grep -v smoke | sed 's|.*/extended-||' | sort -u | tr '\n' ' ')" > "$out/leaderboard.md" 2>/dev/null
gawk -F'|' "$short"'
  function trim(t) { gsub(/^ +| +$/, "", t); return t }
  NR > 2 && NF > 9 { c = trim($3); split(c, p, "_"); alt = trim($6); mg = trim($7); fe = trim($8); v = ""; lab = ""
    if (mg !~ /accuracy only/ && match(alt, /^([^ ]+) \(([^)]*)\)$/, m)) { v = m[2]; lab = ((m[1] in s) ? s[m[1]] : m[1]); if (fe ~ /native proxy/) lab = lab ", native-serving proxy" }
    print p[1] "|" v "|" lab "|" trim($4) }' "$out/leaderboard.md" > "$out/evo-ref.txt"
# Styles: colour = arm, dash = RNG seed.
evo_style() { case $1 in A*) c=ours;; B*) c=alt;; E*) c=ourse;; X*) c=oursext;; Y*) c=oursy;; esac
  case $1 in *11) d="dashed";; *13) d="densely dotted";; *) d="solid";; esac; echo "$c,$d"; }
evo_name() { case $1 in A) echo "DL seeds (A)";; B) echo "random seeds (B)";; E) echo "best-practice seeds (E)";; X) echo "DL seeds, extended";; Y) echo "best-practice seeds, extended";; esac; }
casetitle() { jq -r '.name | sub("^[0-9]+_"; "") | gsub("_"; " ")' "$(ls -d "$cases_dir/$1"_*)/case.json"; }
: > "$out/evo-gain-rows.tex"
for f in "$out"/evo/[0-9][0-9].csv; do  # wide tables only (not *.long.csv)
  num=$(basename "$f" .csv)
  IFS="|" read -r _ ref reflab risk < <(grep "^$num|" "$out/evo-ref.txt" || echo "$num|||")
  # Axis range from the plotted best-so-far values and the reference; log scale when they span > 8x.
  read -r ymin ymax logy < <(gawk -F, -v r="$ref" 'NR == 1 { for (i = 2; i <= NF; i++) if ($i !~ /^g/) b[i] = 1; next }
    { for (i in b) if ($i != "nan") { v = $i + 0; if (mn == "" || v < mn) mn = v; if (mx == "" || v > mx) mx = v } }
    END { if (r != "") { if (r + 0 < mn) mn = r + 0; if (r + 0 > mx) mx = r + 0 }
          lg = (mn > 0 && mx / mn > 8) ? "log" : "normal"; pad = (mx - mn) * 0.06; if (pad == 0) pad = mx * 0.01
          if (lg == "log") printf "%g %g log\n", mn / 1.5, mx * 1.5; else printf "%g %g normal\n", mn - pad, mx + pad }' "$f")
  hdr=$(head -1 "$f")
  xmax=$(gawk -F, 'END { print $1 }' "$f"); [ "$xmax" -ge 4 ] || xmax=4
  have() { gawk -F, -v col="$1" 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == col) k = i; next } k && $k != "nan" { found = 1; exit } END { exit !found }' "$f"; }
  {
    echo "% generated by build-data.sh from data/evolution.csv; best-so-far of the business objective per generation"
    for col in $(echo "$hdr" | tr ',' '\n' | grep -v '^g'); do
      have "$col" || continue
      st=$(evo_style "$col")
      [ "${col#X}" != "$col" ] && have "g$col" && echo "\\addplot[only marks,mark=*,mark size=0.7pt,color=${st%%,*},opacity=0.35,forget plot,evodots] table[col sep=comma,x=g,y=g$col]{data/evo/$num.csv};"
      pts=$(gawk -F, -v col="$col" 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == col) k = i; next } $k != "nan" { n++ } END { print n + 0 }' "$f")
      mk=$( [ "$pts" -le 1 ] && echo ",mark=*,mark size=1.6pt" || true )  # a run that ended in generation 0 is a single point
      echo "\\addplot[${st%%,*},${st#*,},thick,const plot,forget plot$mk] table[col sep=comma,x=g,y=$col]{data/evo/$num.csv};"
    done
    [ -n "$ref" ] && echo "\\addplot[bad,dashed,thick,mark=none,forget plot] coordinates {(0,$ref) ($xmax,$ref)};"
    true
  } > "$out/evo-$num-lines.tex"
  {  # legend entries for the series this case actually has
    for ser in X Y A E B; do
      if have "${ser}7" || have "${ser}11" || have "${ser}13"; then st=$(evo_style "${ser}7"); echo "\\addlegendimage{${st%%,*},thick}\\addlegendentry{$(evo_name "$ser")}"; fi
    done
    echo "\\addlegendimage{black,thick,solid}\\addlegendentry{RNG seed 7}\\addlegendimage{black,thick,dashed}\\addlegendentry{seed 11}\\addlegendimage{black,thick,densely dotted}\\addlegendentry{seed 13}"
    [ -n "$ref" ] && echo "\\addlegendimage{bad,dashed}\\addlegendentry{$(echo "$reflab" | sed 's/_/\\_/g') \\num{$ref}}"
    true
  } > "$out/evo-$num-legend.tex"
  ylab=$( [ "$logy" = log ] && echo ",y tick label style={/pgf/number format/sci,/pgf/number format/precision=1}" || echo ",scaled y ticks=false,y tick label style={/pgf/number format/fixed,/pgf/number format/precision=5}" )
  printf '\\pgfplotsset{evo%s/.style={ymin=%s,ymax=%s,xmin=0,xmax=%s%s}}\n' "$num" "$ymin" "$ymax" "$xmax" "$ylab" >> "$out/macros.tex"
  printf '\\expandafter\\def\\csname evoymode%s\\endcsname{%s}\n' "$num" "$logy" >> "$out/macros.tex"
  printf '\\expandafter\\def\\csname evotitle%s\\endcsname{%s %s}\n' "$num" "$num" "$(casetitle "$num")" >> "$out/macros.tex"
  printf '\\expandafter\\def\\csname evorisk%s\\endcsname{%s}\n' "$num" "$risk" >> "$out/macros.tex"
  printf '\\expandafter\\def\\csname evoylabel%s\\endcsname{%s}\n' "$num" "$(jq -r 'if .metric == "constrained" then "\(.fitness.metric | gsub("_"; " ")), feasible programs" elif .metric == "composite" then "composite objective" else "\(.metric) objective" end' "$(ls -d "$cases_dir/$num"_*)/objective.json")" >> "$out/macros.tex"
  # Evolution gain per series: (generation-0 best - final global best) / generation-0 best, median over seeds.
  gawk -F, -v n="$num" -v t="$(casetitle "$num")" -v risk="$risk" -v ref="$ref" -v cols="X A E B" 'NR == 1 { for (i = 2; i <= NF; i++) h[i] = $i; next }
    { for (i in h) if (h[i] !~ /^g/ && $i != "nan") { if (!(i in f0)) f0[i] = $i + 0; fl[i] = $i + 0; gn[i] = $1 } }
    function med(a, k,   m) { m = asort(a); return (m % 2) ? a[(m + 1) / 2] : (a[m / 2] + a[m / 2 + 1]) / 2 }
    END { line = n " & " t " & " (risk == "HIGH" ? "\\textcolor{bad}{HIGH}" : (risk == "MEDIUM" ? "\\textcolor{limit}{MED}" : risk))
      split(cols, C, " ")
      for (j = 1; j <= 4; j++) { delete gg; delete gs; delete fv; k = 0
        for (i in h) if (substr(h[i], 1, 1) == C[j] && h[i] !~ /^g/ && (i in f0)) { k++; gg[k] = (f0[i] > 0 ? 100 * (f0[i] - fl[i]) / f0[i] : 0); gs[k] = gn[i]; fv[k] = fl[i] }
        line = line " & " (k ? sprintf("%.1f\\%% {\\tiny(%d gen, \\num{%.4g})}", med(gg), med(gs), med(fv)) : "--") }
      print line " & " (ref != "" ? sprintf("\\num{%.4g}", ref) : "--") " \\\\" }' "$f" >> "$out/evo-gain-rows.tex"
done


# The all-cases evolution grid (one panel per case; a case without a completed run says so).
{
  for n in 01 02 03 04 05 06 07 08 09 10 11 12 13 14; do
    t=$(casetitle "$n"); r=$(grep "^$n|" "$out/evo-ref.txt" | cut -d'|' -f4)
    if [ -f "$out/evo/$n.csv" ]; then
      printf '\\nextgroupplot[evo%s,ymode=%s,title={%s %s \\textcolor{alt}{\\tiny %s}}]\n\\csname @@input\\endcsname data/evo-%s-lines.tex\n' "$n" "$(grep -o "evoymode$n.endcsname{[a-z]*}" "$out/macros.tex" | tail -1 | sed 's/.*{\(.*\)}/\1/')" "$n" "$t" "$r" "$n"
    else
      printf '\\nextgroupplot[title={%s %s \\textcolor{alt}{\\tiny %s}},xmin=0,xmax=1,ymin=0,ymax=1,xtick=\\empty,ytick=\\empty]\n\\node[font=\\scriptsize,align=center,text=alt] at (axis cs:0.5,0.5) {no completed run\\\\yet (50k rows x 384)};\n' "$n" "$t" "$r"
    fi
  done
} > "$out/evo-grid.tex"
macro EvoMedianXGain "$(gawk -F'&' 'match($4, /([-0-9.]+)\\%/, m) { v[++n] = m[1] + 0 } END { if (!n) exit; asort(v); printf "%.1f", (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2 }' "$out/evo-gain-rows.tex")"
macro EvoXCases "$(gawk -F'&' '$4 ~ /%/ { n++ } END { print n + 0 }' "$out/evo-gain-rows.tex")"
macro EvoXImproved "$(gawk -F'&' 'match($4, /([-0-9.]+)\\%/, m) && m[1] + 0 > 0 { n++ } END { print n + 0 }' "$out/evo-gain-rows.tex")"
macro EvoCases "$(wc -l < "$out/evo-gain-rows.tex")"
macro EvoMainRuns "$(gawk -F, 'NR > 1 && $2 ~ /^[ABE]$/ { n++ } END { print n + 0 }' "$out/evolution-checks.csv")"
macro EvoMainMatch "$(gawk -F, 'NR > 1 && $2 ~ /^[ABE]$/ && $7 == "yes" { n++ } END { print n + 0 }' "$out/evolution-checks.csv")"
macro EvoExtRuns "$(gawk -F, 'NR > 1 && $2 ~ /^[XY]$/ { n++ } END { print n + 0 }' "$out/evolution-checks.csv")"
macro EvoExtMatch "$(gawk -F, 'NR > 1 && $2 ~ /^[XY]$/ && $7 == "yes" { n++ } END { print n + 0 }' "$out/evolution-checks.csv")"
# Gain macros for the narrative: \evogain{NN}{X|A|E|B} = median gain over seeds from generation 0.
gawk -F'&' '{ n = $1; gsub(/ /, "", n); for (j = 4; j <= 7; j++) { v = $j; ser = substr("XAEB", j - 3, 1); if (match(v, /([-0-9.]+)\\%/, m)) printf "\\expandafter\\def\\csname evogain%s%s\\endcsname{%s}\n", n, ser, m[1]; else printf "\\expandafter\\def\\csname evogain%s%s\\endcsname{--}\n", n, ser } }' "$out/evo-gain-rows.tex" >> "$out/macros.tex"

stamp=$(date -u +%FT%TZ)
inputs_sha=$(sort -u "$out/inputs.txt" | while read -r f; do [ -f "$f" ] && sha256sum "$f"; done | sha256sum | cut -c1-16)
macro BriefDataStamp "$stamp"
macro BriefInputsSha "$inputs_sha"
macro BriefInputCount "$(sort -u "$out/inputs.txt" | wc -l)"
macro BriefSourceCommit "$(git -C "$root" rev-parse --short=12 HEAD)"
echo "build-data: $(wc -l < "$out/champions.csv") champion rows, $(wc -l < "$out/baselines.csv") baseline rows, $(($(wc -l < "$out/evolution.csv") - 1)) evolution points, $(grep -c ",yes$" "$out/evolution-checks.csv") of $(($(wc -l < "$out/evolution-checks.csv") - 1)) runs match their receipt; inputs sha $inputs_sha"
