#!/usr/bin/env bash
# build-winners.sh : the numbers behind the plain-language brief (brief.tex), from cross-validated
# evidence only.
#
# Our side: the CV-EVO chains (extend_case.sh --cuts 5): every program is scored by its mean over
# the 5 folds of train.csv (prepared.json cuts 0-4). test.csv is never read.
# Alternatives: ${JOPE_CASE_BASELINES:-~/jope-data/resfit-2071/baselines}/<case>/summary_cv.csv,
# one row per (model, fold, seed), scored by jopedime fitness evaluate on the same folds.
# Outputs: winners-data/ (CSV series, LaTeX macros, and inputs.txt listing every file read).
# bash, jq and gawk only.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd)
cases_dir=$root/case_study; out=$here/winners-data
evidence=${JOPE_CASE_OUT:-$HOME/jope-data/case_study}
baselines=${JOPE_CASE_BASELINES:-$HOME/jope-data/resfit-2071/baselines}
winners=${BRIEF_WINNERS:-"09 02 12"}; seeds=${BRIEF_SEEDS:-"7 11 13"}
cvstamp=${BRIEF_CV_STAMP:-$({ ls -d "$evidence"/*/extended-*-cv5 2>/dev/null || true; } | sed 's|.*/extended-||' | sort | tail -1)}
[ -n "$cvstamp" ] || { echo "build-winners: no cross-validated (…-cv5) runs under $evidence" >&2; exit 2; }
export JOPE_EV=$evidence
mkdir -p "$out/work"; : > "$out/inputs.txt"; : > "$out/macros.tex"
note() { printf '%s\n' "$1" >> "$out/inputs.txt"; }
macro() { printf '\\newcommand{\\%s}{%s}\n' "$1" "$2" >> "$out/macros.tex"; }
word() { echo "$1" | sed 's/^0*//' | gawk '{ split("One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen", w, " "); print w[$1] }'; }
tex() { sed 's/\\/\\textbackslash{}/g; s/_/\\_/g; s/%/\\%/g; s/&/\\&/g; s/#/\\#/g'; }
# Short display names of the alternatives.
short='BEGIN { split("xgboost_default XGBoost lightgbm_default LightGBM catboost_default CatBoost catboost_strong CatBoost-1000 sklearn_hgb HistGB autogluon_best AutoGluon ridge_cv Ridge random_forest RandomForest extra_trees ExtraTrees knn10 kNN tabpfn_v2 TabPFN-v2 kumo_tabular_small Kumo-Tabular xgboost_tuned XGBoost-tuned lightgbm_tuned LightGBM-tuned pls_cv PLS elasticnet_cv ElasticNet lasso_cv Lasso svr_rbf_cv SVR kernel_ridge_cv KernelRidge", w, " "); for (i = 1; i < length(w); i += 2) s[w[i]] = w[i + 1] }'

# Every program of one round measured on all 5 folds: 5-fold mean objective and RMSE, per-fold
# values, mean resource readings, and the generation it was first created in (its "birth").
cvp() {
  jq -sc --argjson round "$2" --argjson off "$3" '[to_entries[] | .key as $c | .value[]
      | select(.status == "measured" and .objective_loss != null)
      | {g: .genome, c: $c, l: .objective_loss, r: .rmse, gen: .generation, o: .origin, res: (.resources // {})}]
    | group_by(.g) | map(
        ([.[] | .res]) as $rs
        | {genome: .[0].g, born: (map(select(.o != "remeasure") | .gen) | min), cov: (map(.c) | unique | length),
           folds: (group_by(.c) | map(map(.l) | add / length)), fold_rmse: (group_by(.c) | map(map(.r) | add / length)),
           res: (reduce (($rs | map(keys) | add // []) | unique)[] as $k ({}; .[$k] = ($rs | map(.[$k] // empty) | add / length)))})
    | map(select(.cov == 5 and .born != null)
          | . + {obj: (.folds | add / length), rmse: (.fold_rmse | add / length), round: $round, gg: ($off + .born)})
    | .[]' "$1"/search/cut-*/observations.json
}

# Plain words for the program's building blocks, in order (unknown operators keep their name).
plain() {
  gawk '{ s = $0; gsub(/[()]/, " ", s); n = split(s, t, " "); out = ""
    m["histgbm"] = "boosted decision trees"; m["histgbm2"] = "bagged boosted decision trees"; m["ridge"] = "linear (ridge) regression"
    m["ridge_gcv"] = "self-tuning linear regression"; m["cheb"] = "polynomial curves"; m["quantile_bins"] = "value bins"
    m["blom"] = "rank transform"; m["robust_scale"] = "robust scaling"; m["standardize"] = "standardising"; m["winsor_clip"] = "outlier clipping"
    m["poly_interact"] = "pairwise interactions"; m["pair_prod"] = "selected pairwise products"; m["ksmooth"] = "nearest-neighbour averaging"
    m["gpr"] = "Gaussian-process regression"; m["rff_cos"] = "random cosine features"; m["rff_sin"] = "random sine features"; m["pca"] = "principal components"
    m["hinge"] = "hinge (kink) features"; m["oblique_hinge"] = "angled hinge features"; m["ensemble"] = "linear + tree ensemble"; m["row_stats"] = "row summary statistics"
    m["target_bin_encode"] = "target-binned encoding"; m["lasso"] = "sparse linear regression"; m["elastic_net"] = "elastic-net regression"; m["select_k"] = "feature selection"
    m["mlp"] = "small neural network"; m["knn"] = "nearest neighbours"; m["log1p"] = "log transform"; m["yeo_johnson"] = "power transform"
    for (i = 1; i <= n; i++) { w = t[i]; if (w == "pipe" || w == "X" || w ~ /=/) continue; out = out (out == "" ? "" : ", then ") ((w in m) ? m[w] : w) }
    print out }'
}

# What evolution changed between two programs, in plain words (learner settings, added/removed steps).
ops() {  # <genome>: one line per operator, "name<TAB>params", in order
  gawk '{ s = $0; sub(/^\(pipe X */, "", s); sub(/\)$/, "", s)
    while (match(s, /\(([a-z0-9_]+)([^()]*)\)/, m)) { p = m[2]; sub(/^ +/, "", p); print m[1] "\t" p; s = substr(s, RSTART + RLENGTH) } }' <<<"$1"
}
changes() {  # <start genome> <end genome>
  local a b la lb words="" steps_a steps_b added removed
  a=$(ops "$1"); b=$(ops "$2"); la=$(tail -1 <<<"$a" | cut -f1); lb=$(tail -1 <<<"$b" | cut -f1)
  if [ "$la" = "$lb" ]; then
    words=$(gawk -F'\t' -v pa="$(tail -1 <<<"$a" | cut -f2)" -v pb="$(tail -1 <<<"$b" | cut -f2)" 'BEGIN {
        n["depth"] = "tree depth"; n["rounds"] = "number of trees"; n["lr"] = "learning rate"; n["bins"] = "histogram bins"
        n["min_leaf"] = "minimum leaf size"; n["lambda"] = "regularisation"; n["leaves"] = "leaves per tree"; n["bags"] = "bagged copies"
        split(pa, A, " "); split(pb, B, " "); for (i in A) { split(A[i], kv, "="); va[kv[1]] = kv[2]; keys[kv[1]] = 1 } for (i in B) { split(B[i], kv, "="); vb[kv[1]] = kv[2]; keys[kv[1]] = 1 }
        m = asorti(keys, K); out = ""
        for (i = 1; i <= m; i++) { k = K[i]; x = (k in va) ? va[k] : "default"; y = (k in vb) ? vb[k] : "default"; if (x != y) { if (x ~ /^[0-9.]+$/ && length(x) > 6) x = sprintf("%.3g", x); if (y ~ /^[0-9.]+$/ && length(y) > 6) y = sprintf("%.3g", y); out = out (out == "" ? "" : ", ") ((k in n) ? n[k] : k) " " x " $\\to$ " y } }
        print out }')
    words="the same kind of model ($(echo "($la)" | plain)) with new settings${words:+: $words}"
  else
    words="a different model: $(echo "$2" | plain)"
  fi
  steps_a=$(head -n -1 <<<"$a" | cut -f1 | sort -u); steps_b=$(head -n -1 <<<"$b" | cut -f1 | sort -u)
  added=$(comm -13 <(echo "$steps_a" | gawk 'NF') <(echo "$steps_b" | gawk 'NF') | while read -r o; do echo "($o)" | plain; done | paste -sd, - | sed 's/,/, /g')
  removed=$(comm -23 <(echo "$steps_a" | gawk 'NF') <(echo "$steps_b" | gawk 'NF') | while read -r o; do echo "($o)" | plain; done | paste -sd, - | sed 's/,/, /g')
  [ -n "$added" ] && words="$words; added step: $added"
  [ -n "$removed" ] && words="$words; removed step: $removed"
  echo "$words"
}

echo "case,model,folds,feasible_folds,proxy_feasible_folds,objective,accuracy,proxy_objective,fit_cpu_seconds,fit_seconds,predict_us_per_row,row_latency_p99_us,model_bytes" > "$out/alternatives.csv"
for num in $winners; do
  cdir=$(ls -d "$cases_dir/${num}"_*); name=$(basename "$cdir"); K=$(word "$num"); note "$cdir/case.json"; note "$cdir/objective.json"
  structural=$(jq -e '[.. | objects | (.name? // .resource? // empty)] | any(. == "program_nodes" or . == "stage_count")' "$cdir/objective.json" >/dev/null 2>&1 && echo 1 || echo 0)
  x=$evidence/$name/extended-$cvstamp
  echo "arm,seed,rounds,evaluations,start_obj,start_rmse,end_obj,end_rmse,end_round,end_gen,final_obj,final_rmse,gens" > "$out/$num-chains.csv"
  echo "arm,seed,global_generation,gen_best,best_so_far,best_so_far_rmse" > "$out/$num-evo-long.csv"
  for arm in A E; do for seed in $seeds; do
    c=$x/$arm/s$seed; [ -f "$c/rounds.jsonl" ] || continue
    note "$c/rounds.jsonl"; w=$out/work/$num-$arm$seed.jsonl; : > "$w"; off=0; last_rd=""; nr=0; evals=0
    while read -r rd r; do
      [ -f "$rd/receipt.json" ] || continue
      for f in "$rd"/search/cut-*/observations.json; do note "$f"; done; note "$rd/receipt.json"
      cvp "$rd" "$r" "$off" >> "$w"
      lastg=$(jq -s '[.[][] | .generation] | max' "$rd"/search/cut-*/observations.json)
      jq -nc --argjson r "$r" --argjson off "$off" --argjson last "$lastg" '{meta: true, round: $r, off: $off, last: $last}' >> "$w"
      off=$(( off + $(jq -s '[.[][] | select(.origin != "remeasure") | .generation] | max + 1' "$rd"/search/cut-*/observations.json) ))
      evals=$(( evals + $(jq '[.search.cuts[].completed_evaluations] | add' "$rd/receipt.json") )); last_rd=$rd; nr=$r
    done < <(jq -r 'select(.status == "verified") | "\(.dir | sub("^.*/jope-data/case_study/"; $ENV.JOPE_EV + "/")) \(.round)"' "$c/rounds.jsonl")
    [ -n "$last_rd" ] || continue
    # The evolution curve: per global generation, the best new program and the best so far.
    jq -s -r --arg a "$arm" --arg s "$seed" '(map(select(.meta)) | map(range(.off; .off + .last + 1)) | flatten) as $gs
      | (map(select(.meta | not))) as $p
      | foreach $gs[] as $g ({b: null}; ($p | map(select(.gg == $g)) | if length > 0 then min_by(.obj) else null end) as $pg
          | .gb = ($pg.obj // null) | (if $pg != null and (.b == null or $pg.obj < .b) then .b = $pg.obj | .br = $pg.rmse else . end) | .g = $g;
          [$a, $s, .g, .gb, .b, .br] | @csv)' "$w" | tr -d '"' >> "$out/$num-evo-long.csv"
    # Start (best of the first generation), end (global best) and the final round's champion.
    jq -s -r --arg a "$arm" --arg s "$seed" --argjson nr "$nr" --argjson ev "$evals" --slurpfile rc "$last_rd/receipt.json" '
      map(select(.meta | not)) as $p | ($p | map(select(.round == 1 and .born == 0)) | min_by(.obj)) as $st | ($p | min_by(.obj)) as $en
      | (map(select(.meta)) | map(.off + .last) | max) as $gens
      | [$a, $s, $nr, $ev, $st.obj, $st.rmse, $en.obj, $en.rmse, $en.round, $en.gg, $rc[0].search.best.validation_objective, $rc[0].search.best.validation_rmse, $gens] | @csv' "$w" | tr -d '"' >> "$out/$num-chains.csv"
    # end = the final champion (the program the last round selected and would deploy), with its 5-fold
    # scores from that round, and the generation where that program was first created.
    jq -s -c --arg a "$arm" --arg s "$seed" --argjson nr "$nr" --arg fg "$(jq -r .search.selected_genome "$last_rd/receipt.json")" 'map(select(.meta | not)) as $p
      | {arm: $a, seed: $s, start: ($p | map(select(.round == 1 and .born == 0)) | min_by(.obj)),
         end: (($p | map(select(.round == $nr and .genome == $fg)) | first) + {gg: ($p | map(select(.genome == $fg) | .gg) | min)})}' "$w" > "$out/work/$num-$arm$seed.ends.json"
  done; done
  # Wide table for the evolution plot (one column per chain; nan = no data).
  cols=$(for a in A E; do for s in $seeds; do printf "%s%s " "$a" "$s"; done; done)
  gawk -F, -v cols="$cols" 'NR > 1 { k = $1 $2; seen[k] = 1; v[$3, k] = ($5 == "" ? "nan" : $5); if ($3 + 0 > mx) mx = $3 + 0 }
    END { n = split(cols, C, " "); h = "g"; for (i = 1; i <= n; i++) h = h "," C[i]; print h
          for (g = 0; g <= mx; g++) { l = g; for (i = 1; i <= n; i++) l = l "," (((g, C[i]) in v) ? v[g, C[i]] : "nan"); print l } }' "$out/$num-evo-long.csv" > "$out/$num-evo.csv"
  gawk -F, -v cols="$cols" 'NR > 1 { k = $1 $2; v[$3, k] = ($6 == "" ? "nan" : $6); if ($3 + 0 > mx) mx = $3 + 0 }
    END { n = split(cols, C, " "); h = "g"; for (i = 1; i <= n; i++) h = h "," C[i]; print h
          for (g = 0; g <= mx; g++) { l = g; for (i = 1; i <= n; i++) l = l "," (((g, C[i]) in v) ? v[g, C[i]] : "nan"); print l } }' "$out/$num-evo-long.csv" > "$out/$num-evo-rmse.csv"

  macro "Win${K}ChainCols" "$(head -1 "$out/$num-evo.csv" | cut -d, -f2-)"

  # Alternatives: means over the 5 folds; feasible only if feasible on every fold. For latency limits
  # the native-serving proxy credits each library with its amortized batch latency.
  csv_raw=$baselines/$name/summary_cv.csv
  csv=$out/work/$num-alt-folds.csv
  if [ -f "$csv_raw" ]; then
    # One row per (model, fold): numeric readings averaged over that fold's seeds; a fold counts as
    # feasible only if every seed was; not-applicable and failed runs are left out.
    gawk -F, -v OFS=, 'NR == 1 { print; for (i = 1; i <= NF; i++) { col[$i] = i; h[i] = $i } nf = NF; next }
      { st = $col["status"]; if (st == "not_applicable" || st ~ /fail|error|oom|skipped/) next
        if ($col["rmse"] == "" && $col["baseline_loss"] == "") next
        k = $col["model"] SUBSEP $col["fold"]; if (!(k in seen)) { seen[k] = 1; order[++nk] = k; base[k] = $0; feas[k] = 1 }
        ns[k]++; if ($col["feasible"] != "true") feas[k] = 0
        for (i = 1; i <= nf; i++) if (h[i] !~ /^(case|model|seed|fold|feasible|status|error)$/ && $i != "") { s[k, i] += $i; c[k, i]++ } }
      END { for (j = 1; j <= nk; j++) { k = order[j]; split(base[k], f, ","); f[col["seed"]] = ns[k]; f[col["feasible"]] = (feas[k] ? "true" : "false")
              for (i = 1; i <= nf; i++) if (h[i] !~ /^(case|model|seed|fold|feasible|status|error)$/) f[i] = ((k, i) in c) ? s[k, i] / c[k, i] : ""
              if (!feas[k]) f[col["objective_loss"]] = ""; f[col["error"]] = ""
              line = f[1]; for (i = 2; i <= nf; i++) line = line OFS f[i]; print line } }' "$csv_raw" > "$csv"
  fi
  if [ -f "$csv" ]; then
    note "$csv_raw"
    gawk -F, -v c="$name" -v lim="$(jq -c '[.limits[]? | {r: .resource, max, min}]' "$cdir/objective.json" | tr -d ' ')" '
      NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; nl = split(lim, L, "},{"); next }
      { m = $col["model"]; f = $col["fold"]; k = m SUBSEP f; if (k in seen) next; seen[k] = 1; n[m]++
        if ($col["feasible"] == "true") { fe[m]++; so[m] += $col["objective_loss"] }
        a = ($col["baseline_loss"] != "" ? $col["baseline_loss"] : $col["rmse"]); sa[m] += a
        pf = 1; for (j = 1; j <= nl; j++) { if (!match(L[j], /"r":"([a-z0-9_]+)"/, rr)) continue; r = rr[1]; v = (r ~ /^row_latency/) ? $col["predict_us_per_row"] : $col[r]
          mx = ""; mn = ""; if (match(L[j], /"max":([0-9.eE+-]+)/, q)) mx = q[1]; if (match(L[j], /"min":([0-9.eE+-]+)/, q)) mn = q[1]
          if (v == "" || (mx != "" && v + 0 > mx + 0) || (mn != "" && v + 0 < mn + 0)) pf = 0 }
        if (pf) { pfe[m]++; spo[m] += a }
        split("fit_cpu_seconds fit_seconds predict_us_per_row row_latency_p99_us model_bytes", R, " ")
        for (j = 1; j <= 5; j++) if ($col[R[j]] != "") { sr[m, j] += $col[R[j]]; nr2[m, j]++ } }
      END { for (m in n) { line = c "," m "," n[m] "," fe[m] + 0 "," pfe[m] + 0 "," (fe[m] == n[m] ? so[m] / n[m] : "") "," sa[m] / n[m] "," (pfe[m] == n[m] ? spo[m] / n[m] : "")
              for (j = 1; j <= 5; j++) line = line "," ((m, j) in nr2 ? sr[m, j] / nr2[m, j] : ""); print line } }' "$csv" >> "$out/alternatives.csv"
  fi

  # Headline numbers. Ours = the median over chains of the final champion (and the best chain).
  # Structure-priced objectives cannot score library models, so those cases compare accuracy (RMSE).
  col=$([ "$structural" = 1 ] && echo 12 || echo 11)   # final_rmse or final_obj in chains.csv
  read -r med best nch < <(gawk -F, -v k="$col" 'NR > 1 && $k != "" { v[++n] = $k + 0 } END { if (!n) { print "n/a n/a 0"; exit } asort(v); printf "%.6g %.6g %d\n", (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2, v[1], n }' "$out/$num-chains.csv")
  read -r smed < <(gawk -F, -v k="$([ "$structural" = 1 ] && echo 6 || echo 5)" 'NR > 1 && $k != "" { v[++n] = $k + 0 } END { if (!n) { print "n/a"; exit } asort(v); printf "%.6g\n", (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2 }' "$out/$num-chains.csv")
  altcol=$([ "$structural" = 1 ] && echo 7 || echo 6)
  latency=$(jq -e '[.limits[]? | .resource] | any(startswith("row_latency"))' "$cdir/objective.json" >/dev/null 2>&1 && echo 1 || echo 0)
  [ "$latency" = 1 ] && altcol=8
  read -r balt bmodel nalt < <(gawk -F, -v c="$name" -v k="$altcol" '$1 == c && $3 == 5 && $k != "" { n++; if (b == "" || $k + 0 < b + 0) { b = $k; m = $2 } } END { if (!n) print "n/a none 0"; else print b, m, n }' "$out/alternatives.csv")
  margin=$(gawk -v x="$med" -v b="$balt" 'BEGIN { if (x !~ /^[0-9.eE+-]+$/ || b !~ /^[0-9.eE+-]+$/ || b == 0) print "n/a"; else printf "%+.1f", 100 * (x - b) / b }')
  macro "Win${K}Better" "$(gawk -v x="$med" -v b="$balt" 'BEGIN { if (x ~ /^[0-9.eE+-]+$/ && b ~ /^[0-9.eE+-]+$/ && b > 0) printf "%.0f", 100 * (b - x) / b; else print "n/a" }')"
  macro "Win${K}BestBetter" "$(gawk -v x="$best" -v b="$balt" 'BEGIN { if (x ~ /^[0-9.eE+-]+$/ && b ~ /^[0-9.eE+-]+$/ && b > 0) printf "%.0f", 100 * (b - x) / b; else print "n/a" }')"
  macro "Win${K}OursMedianRmse" "$(gawk -F, 'NR > 1 && $12 != "" { v[++n] = $12 + 0 } END { if (!n) { print "n/a"; exit } asort(v); printf "%.5g", (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2 }' "$out/$num-chains.csv")"
  macro "Win${K}ArmEnds" "$(gawk -F, 'NR > 1 { if ($1 == "A") { a = a (a ? ", " : "") sprintf("%.3f", $11) } else { e = e (e ? ", " : "") sprintf("%.3f", $11) } } END { print "E: " e "; A: " a }' "$out/$num-chains.csv")"
  macro "Win${K}ArmStarts" "$(gawk -F, 'NR > 1 { if ($1 == "A") { a = a (a ? ", " : "") sprintf("%.3f", $5) } else { e = e (e ? ", " : "") sprintf("%.3f", $5) } } END { print "E: " e "; A: " a }' "$out/$num-chains.csv")"
  macro "Win${K}Name" "$(jq -r '.dataset.display_name // .dataset.id' "$cdir/case.json" | tex)"
  macro "Win${K}Rows" "$(jq -r .dataset.rows "$cdir/case.json")"; macro "Win${K}Features" "$(jq -r .dataset.features "$cdir/case.json")"
  macro "Win${K}Ours" "$med"; macro "Win${K}OursBest" "$best"; macro "Win${K}Chains" "$nch"; macro "Win${K}Start" "$smed"
  macro "Win${K}Alt" "$balt"; macro "Win${K}AltModel" "$(echo "$bmodel" | tex)"; macro "Win${K}AltCount" "$nalt"; macro "Win${K}Margin" "$margin"
  # The best alternative's accuracy and training CPU (5-fold means), and the full-coverage count.
  macro "Win${K}AltAcc" "$(gawk -F, -v c="$name" -v m="$bmodel" '$1 == c && $2 == m { printf "%.5g", $7 }' "$out/alternatives.csv")"
  macro "Win${K}AltCpu" "$(gawk -F, -v c="$name" -v m="$bmodel" '$1 == c && $2 == m { printf "%.3g", $9 }' "$out/alternatives.csv")"
  macro "Win${K}AltFull" "$(gawk -F, -v c="$name" '$1 == c && $3 == 5 { n++ } END { print n + 0 }' "$out/alternatives.csv")"
  macro "Win${K}AltPartial" "$(gawk -F, -v c="$name" '$1 == c && $3 < 5 { n++ } END { print n + 0 }' "$out/alternatives.csv")"
  macro "Win${K}Gens" "$(gawk -F, 'NR > 1 { if ($13 + 0 > m) m = $13 + 0 } END { print m + 0 }' "$out/$num-chains.csv")"
  macro "Win${K}Evals" "$(gawk -F, 'NR > 1 { s += $4 } END { printf "%.0f", s }' "$out/$num-chains.csv")"
  # Best chain: start and end programs, in plain words and as written.
  bestends=$(for f in "$out"/work/$num-*.ends.json; do [ -f "$f" ] && cat "$f"; done | jq -s -c --argjson st "$structural" 'map(select(.end != null)) | min_by(if $st == 1 then .end.rmse else .end.obj end)')
  if [ -n "$bestends" ] && [ "$bestends" != "null" ]; then
    for side in start end; do
      g=$(jq -r --arg s "$side" '.[$s].genome' <<<"$bestends")
      cap=$(echo "$side" | sed 's/^./\U&/')
      macro "Win${K}${cap}Plain" "$(echo "$g" | plain | tex)"
      macro "Win${K}${cap}Program" "$(echo "$g" | tex)"
      macro "Win${K}${cap}Obj" "$(jq -r --arg s "$side" '.[$s].obj | . * 1e6 | round / 1e6' <<<"$bestends")"
      macro "Win${K}${cap}Rmse" "$(jq -r --arg s "$side" '.[$s].rmse | . * 1e6 | round / 1e6' <<<"$bestends")"
      for rkey in fit_cpu_seconds fit_seconds row_latency_p99_us model_bytes program_nodes predict_us_per_row; do
        v=$(jq -r --arg s "$side" --arg k "$rkey" '.[$s].res[$k] // empty' <<<"$bestends")
        macro "Win${K}${cap}$(echo "$rkey" | sed -E 's/p99/pninetynine/; s/(^|_)([a-z0-9])/\U\2/g; s/[0-9]//g')" "${v:-n/a}"
      done
      macro "Win${K}${cap}Gen" "$(jq -r --arg s "$side" '.[$s].gg' <<<"$bestends")"
      macro "Win${K}${cap}ModelKB" "$(jq -r --arg s "$side" '.[$s].res.model_bytes // empty | . / 1000 | round' <<<"$bestends")"
    done
    macro "Win${K}BestChain" "$(jq -r '"\(.arm)\(.seed)"' <<<"$bestends")"
    macro "Win${K}Changes" "$(changes "$(jq -r .start.genome <<<"$bestends")" "$(jq -r .end.genome <<<"$bestends")" | sed 's/_/\\_/g')"
    macro "Win${K}CpuRatio" "$(gawk -v a="$(gawk -F, -v c="$name" -v m="$bmodel" '$1 == c && $2 == m { print $9 }' "$out/alternatives.csv")" -v o="$(jq -r '.end.res.fit_cpu_seconds // empty' <<<"${bestends:-null}" 2>/dev/null)" 'BEGIN { if (a + 0 > 0 && o + 0 > 0) printf "%.0f", a / o; else print "n/a" }')"
    # Fold by fold: the best chain's final program against the best alternative.
    if [ -f "$csv" ] && [ "$bmodel" != none ]; then
      key=$([ "$structural" = 1 ] && echo fold_rmse || echo folds)
      ours_f=$(jq -r --arg k "$key" '.end[$k] | map(tostring) | join(" ")' <<<"$bestends")
      alt_f=$(gawk -F, -v m="$bmodel" -v st="$structural" -v lat="$latency" 'NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
        $col["model"] == m && !($col["fold"] in seen) { seen[$col["fold"]] = 1; a = ($col["baseline_loss"] != "" ? $col["baseline_loss"] : $col["rmse"])
          v[$col["fold"]] = (st == 1 || lat == 1) ? a : $col["objective_loss"] } END { for (f = 0; f < 5; f++) printf "%s ", v[f] }' "$csv")
      wins=$(gawk -v o="$ours_f" -v a="$alt_f" 'BEGIN { n = split(o, O, " "); split(a, A, " "); for (i = 1; i <= n; i++) if (A[i] != "" && O[i] + 0 < A[i] + 0) w++; print w + 0 }')
      macro "Win${K}FoldWins" "$wins"
      paste -d, <(echo "$ours_f" | tr ' ' '\n') <(echo "$alt_f" | tr ' ' '\n' | gawk 'NF') | gawk -F, 'BEGIN { print "fold,ours,alt" } { print NR - 1 "," $1 "," $2 }' > "$out/$num-folds.csv"
    fi
  fi
  # Alternatives scored on fewer than 5 folds (too slow to repeat): compared with our best run's
  # final program on exactly the same folds, never with our 5-fold mean.
  if [ -f "$csv" ] && [ -n "${bestends:-}" ] && [ "$bestends" != "null" ]; then
    key=$([ "$structural" = 1 ] && echo fold_rmse || echo folds)
    gawk -F, -v c="$name" '$1 == c && $3 < 5 { print $2 }' "$out/alternatives.csv" | while read -r pm; do
      gawk -F, -v m="$pm" -v st="$structural" -v lat="$latency" -v o="$(jq -r --arg k "$key" '.end[$k] | map(tostring) | join(" ")' <<<"$bestends")" "$short"'
        NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; split(o, O, " "); next }
        $col["model"] == m && !($col["fold"] in seen) { seen[$col["fold"]] = 1; f = $col["fold"]
          a = ($col["baseline_loss"] != "" ? $col["baseline_loss"] : $col["rmse"]); v = (st == 1 || lat == 1) ? a : $col["objective_loss"]
          if (v == "" || v + 0 == 0) next; n++; sa += v; so += O[f + 1]; fl = fl (fl == "" ? "" : ", ") (f + 1) }
        END { if (n) printf "%s (fold %s): %.4g against our %.4g on the same fold%s; ", ((m in s) ? s[m] : m), fl, sa / n, so / n, (n > 1 ? "s" : "") }' "$csv"
    done | sed "s/; $//" > "$out/$num-partial.txt"
    macro "Win${K}PartialNote" "$(sed 's/; $//' "$out/$num-partial.txt" | tex)"
    macro "Win${K}PartialList" "$(gawk -F, -v c="$name" '$1 == c && $3 < 5 { print $2 }' "$out/alternatives.csv" | while read -r pm; do gawk -F, -v m="$pm" -v st="$structural" -v lat="$latency" "$short"' NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next } $col["model"] == m && !d { a = ($col["baseline_loss"] != "" ? $col["baseline_loss"] : $col["rmse"]); v = (st == 1 || lat == 1) ? a : $col["objective_loss"]; if (v == "" || v + 0 == 0) next; d = 1; printf "%s %.3g, ", ((m in s) ? s[m] : m), v }' "$csv"; done | sed 's/, $//' | tex)"
  fi
  # Named single-fold comparisons for the text: accuracy and training CPU on fold 0, ours vs theirs.
  if [ -f "$csv" ] && [ -n "${bestends:-}" ] && [ "$bestends" != "null" ]; then
    macro "Win${K}OursFoldOneRmse" "$(jq -r '.end.fold_rmse[0] | . * 1e6 | round / 1e6' <<<"$bestends")"
    macro "Win${K}OursFoldOneObj" "$(jq -r '.end.folds[0] | . * 1e4 | round / 1e4' <<<"$bestends")"
    for pm in autogluon_best tabpfn_v2 kumo_tabular_small xgboost_tuned lightgbm_tuned; do
      tag=$(case $pm in autogluon_best) echo AutoGluon;; tabpfn_v2) echo TabPFN;; kumo_tabular_small) echo Kumo;; xgboost_tuned) echo XGBTuned;; lightgbm_tuned) echo LGBTuned;; esac)
      read -r acc cpu obj < <(gawk -F, -v m="$pm" 'NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
        $col["model"] == m && $col["fold"] == 0 && !d { d = 1; a = ($col["baseline_loss"] != "" ? $col["baseline_loss"] : $col["rmse"]); printf "%.4g %.4g %.4g\n", a, $col["fit_cpu_seconds"], $col["objective_loss"] } END { if (!d) print "n/a n/a n/a" }' "$csv")
      macro "Win${K}${tag}FoldOneAcc" "$acc"; macro "Win${K}${tag}FoldOneCpu" "$cpu"; macro "Win${K}${tag}FoldOneObj" "$obj"
      macro "Win${K}${tag}Mean" "$(gawk -F, -v c="$name" -v m="$pm" '$1 == c && $2 == m && $3 == 5 { printf "%.4g", $6 }' "$out/alternatives.csv")"
    done
  fi
  # The benchmark chart: every alternative's 5-fold mean, our start and our end, best at the top.
  {
    echo "label,value,kind"
    gawk -F, -v c="$name" -v k="$altcol" "$short"' $1 == c && $3 == 5 { lab = (($2 in s) ? s[$2] : $2); if ($k != "") print lab "," $k ",alt"; else print lab " (fails a limit)," ($7 != "" ? $7 : 0) ",bad" }' "$out/alternatives.csv"
    sk=$([ "$structural" = 1 ] && echo rmse || echo obj)
    [ -n "${bestends:-}" ] && [ "$bestends" != null ] && echo "JopeDime best run: start,$(jq -r --arg k "$sk" '.start[$k]' <<<"$bestends"),start"
    [ "$med" != n/a ] && echo "JopeDime typical run (median of $nch),$med,ours"
    [ "$best" != n/a ] && echo "JopeDime best run: end,$best,best"
  } | { read -r h; sort -t, -k2,2gr; } | gawk -F, '{ v[NR] = $2 + 0; l[NR] = $1; k[NR] = $3; if ($3 == "start" || $3 == "ours" || $3 == "best") { if (v[NR] > ref) ref = v[NR] } }
      END { cap = 2.5 * ref; print "label,alt,ours,start,bad,best,shown,text"
            for (i = 1; i <= NR; i++) { s = (v[i] > cap ? cap : v[i]); t = sprintf("%.4g", v[i]) (v[i] > cap ? " (off scale)" : "")
              printf "%s,%s,%s,%s,%s,%s,%s,%s\n", l[i], (k[i] == "alt" ? s : "nan"), (k[i] == "ours" ? s : "nan"), (k[i] == "start" ? s : "nan"), (k[i] == "bad" ? s : "nan"), (k[i] == "best" ? s : "nan"), s, t } }' > "$out/$num-bench.csv"
  macro "Win${K}BenchMax" "$(gawk -F, 'NR > 1 && $7 + 0 > m { m = $7 + 0 } END { printf "%.4g", m * 1.32 }' "$out/$num-bench.csv")"
  macro "Win${K}BenchRows" "$(($(wc -l < "$out/$num-bench.csv") - 1))"
  macro "Win${K}BenchLast" "$(($(wc -l < "$out/$num-bench.csv") - 2))"
done

stamp=$(date -u +%FT%TZ)
macro BriefCvStamp "$(echo "$cvstamp" | tex)"
macro BriefDataStamp "$stamp"
macro BriefInputCount "$(sort -u "$out/inputs.txt" | wc -l)"
macro BriefInputsSha "$(sort -u "$out/inputs.txt" | while read -r f; do [ -f "$f" ] && sha256sum "$f"; done | sha256sum | cut -c1-16)"
echo "build-winners: cv stamp $cvstamp; cases $winners; $(sort -u "$out/inputs.txt" | wc -l) inputs"
