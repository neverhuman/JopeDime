#!/usr/bin/env bash
# Refresh the current regression table from its checked analysis; no fits or dataset reads.
set -euo pipefail
root=${JOPE_PERFORMANCE_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
mode=${1:---check}
case "$mode" in --check|--write) ;; *) echo 'usage: render.sh [--check|--write]' >&2; exit 2 ;; esac
cd "$root"
results=assets/automl-elo/v1/RESULTS.json
summary=docs/paper/innovationzero/evidence/elo-summary.json
figures=assets/automl-elo/v1/FIGURE_VALIDATION.json
fail() { echo "README performance: $*" >&2; exit 1; }
expected=$(jq -er '.source.sha256' "$summary")
actual=$(sha256sum "$results"); actual=${actual%% *}
[[ "$actual" == "$expected" ]] || fail 'results changed since the paper evidence was bound; reconcile the analysis and provenance first'
jq -e '
 .state == "COMPLETE" and .complete_pairwise_coverage == true and
 .cluster == "family" and .bootstrap_samples == 10000 and
 .seeds_repeated_within_family == [41,73,109] and
 .methods == ["GPT6_Astra_Max","GPT6p1_Sol_Max","Gemini4_argon_high","Opus5p5_Extra"] and
 ([.elo | keys[]] == .methods) and
 all(.elo[]; (.rating | isfinite) and (.ci95 | length == 2) and
     all(.ci95[]; isfinite) and .ci95[0] <= .rating and .rating <= .ci95[1]) and
 (.pairwise | length == 6) and all(.pairwise[]; .paired_families == 64 and
     (.geometric_rmse_ratio | isfinite) and .geometric_rmse_ratio > 0 and
     (.ratio_ci95 | length == 2) and all(.ratio_ci95[]; isfinite) and
     .ratio_ci95[0] > 0 and .ratio_ci95[0] <= .geometric_rmse_ratio and
     .geometric_rmse_ratio <= .ratio_ci95[1])
' "$results" >/dev/null || fail 'incomplete or incompatible comparison'
jq -e '
 .elo.Opus5p5_Extra.rating == ([.elo[].rating] | max) and
 all(.pairwise[]; if .a == "GPT6p1_Sol_Max" then .geometric_rmse_ratio < 1
 elif .b == "GPT6p1_Sol_Max" then .geometric_rmse_ratio > 1 else true end) and
 ([.pairwise[] | select(.holm_gemini_three_p != null)] | length) == 3 and
 all(.pairwise[] | select(.holm_gemini_three_p != null);
     (.holm_gemini_three_p | isfinite) and .holm_gemini_three_p >= 0.05 and .holm_gemini_three_p <= 1) and
 all(.pairwise[] | select(.a == "GPT6p1_Sol_Max" and
     (.b == "Opus5p5_Extra" or .b == "Gemini4_argon_high"));
     .ratio_ci95[0] <= 1 and .ratio_ci95[1] >= 1) and
 ([.pairwise[] | [.a,.b]] | sort) ==
 [["GPT6_Astra_Max","GPT6p1_Sol_Max"],["GPT6_Astra_Max","Gemini4_argon_high"],
 ["GPT6_Astra_Max","Opus5p5_Extra"],["GPT6p1_Sol_Max","Gemini4_argon_high"],
 ["GPT6p1_Sol_Max","Opus5p5_Extra"],["Gemini4_argon_high","Opus5p5_Extra"]]
' "$results" >/dev/null || fail 'winner, error-size interpretation or pair inventory changed; review the prose'
jq -e '.figures | keys == ["algorithm-source-size.png","algorithm-source-size.svg","automl-elo.png","automl-elo.svg","automl-pairwise.png","automl-pairwise.svg"]' "$figures" >/dev/null || fail 'expected the complete six-figure inventory'
jq -e '.panel.families == 64 and .panel.folds == 3 and .panel.cells_per_entrant == 576 and
 .panel.seeds == [41,73,109] and .panel.complete_pairwise_coverage == true and
 .panel.role == "existing training families; prepared inner-CV contexts"' "$summary" >/dev/null || fail 'panel contract changed'
# The plotted values must be the same evidence as the table, with every figure byte pinned.
[[ $(jq -er '.inputs["FOUR_MODEL_RESULTS.json"]' "$figures") == "$actual" ]] || fail 'figure input differs from table'
while IFS=$'\t' read -r name hash; do
 actual_figure=$(sha256sum "assets/automl-elo/v1/figures/$name"); actual_figure=${actual_figure%% *}
 [[ "$actual_figure" == "$hash" ]] || fail "figure $name differs from its recorded rendering"
done < <(jq -er '.figures | to_entries[] | [.key,.value.sha256] | @tsv' "$figures")
matrix=$(sha256sum assets/automl-elo/v1/match-matrix.txt); matrix=${matrix%% *}
[[ "$matrix" == $(jq -er '.inputs["four-match-matrix.txt"]' "$figures") ]] || fail 'pairwise plot input changed'
begin='<!-- current-performance:begin -->'
end='<!-- current-performance:end -->'
[[ $(grep -Fxc "$begin" README.md || true) == 1 && $(grep -Fxc "$end" README.md || true) == 1 ]] || fail 'expected one current-performance marker pair'
block=$(
 printf '%s\n\n' "$begin"
 cat <<'TEXT'
### Latest complete native-regressor ranking

**64 regression families × 3 folds × 3 seeds; 576 fits per entrant.** These are four
LLM-authored regression algorithms, fitted on the same prepared inner-CV inputs.
The names identify the submitted algorithms. This table does not measure the
LLMs at inference time or the neural program-proposal policy.

![Four native regression algorithms: Elo point estimates and 95% family-bootstrap intervals on the 64-family panel](assets/automl-elo/v1/figures/automl-elo.svg)

| Rank | Algorithm | Elo | 95% family-bootstrap interval |
| ---: | --- | ---: | ---: |
TEXT
 jq -r '.elo | to_entries | sort_by(-.value.rating) | to_entries[] |
 [(.key+1),.value.key,.value.value.rating,.value.value.ci95[0],.value.value.ci95[1]] | @tsv' "$results" |
 while IFS=$'\t' read -r rank method rating low high; do
  LC_ALL=C printf '| %d | `%s` | %.1f | %.1f–%.1f |\n' "$rank" "$method" "$rating" "$low" "$high"
 done
 cat <<'TEXT'

**Opus has the highest Elo; Sol has the lowest geometric-mean RMSE point estimate.**
Elo rewards how often a method wins, while RMSE ratios also reflect error size.
TEXT
 for opponent in Opus5p5_Extra Gemini4_argon_high; do
  jq -r --arg b "$opponent" '.pairwise[] | select(.a == "GPT6p1_Sol_Max" and .b == $b) |
  [.b,.geometric_rmse_ratio,.ratio_ci95[0],.ratio_ci95[1]] | @tsv' "$results" |
  while IFS=$'\t' read -r name ratio low high; do
   LC_ALL=C printf 'Sol / `%s` RMSE: **%.3f [%.3f, %.3f]**.\n' "$name" "$ratio" "$low" "$high"
  done
 done
 cat <<'TEXT'
Both 95% intervals include equal error. None of Gemini's three RMSE comparisons
is significant after Holm adjustment.

![Pairwise match scores among the four native regressors, averaged over three seeds within each family](assets/automl-elo/v1/figures/automl-pairwise.svg)

The heatmap cells are **seed-averaged family match scores** (ties count as 0.5),
not integer majority-family win rates. All six pairs cover all 64 families;
10,000 family bootstrap resamples retain the three seeds within each family.
These are existing training families with historical exposure. Stored preprocessing
predates the original split, and seed 41 was seen before seeds 73 and 109 were
fixed. This is exploratory development evidence, with no untouched-family claim.
Historical and Gemini fits used different hosts; fit times are not a controlled
speed ranking. Elo values from different opponent pools cannot be compared.

Source: [results and uncertainty](assets/automl-elo/v1/RESULTS.json),
[protocol and fit custody](assets/automl-elo/v1/SOURCES.md), and
[exposure audit](docs/paper/innovationzero/evidence/elo-summary.json).
TEXT
 printf '\n%s' "$end"
)
current=$(cat README.md)
[[ "$current" == *"$begin"*"$end"* ]] || fail 'markers are out of order'
prefix=${current%%"$begin"*}
suffix=${current#*"$end"}
updated=$prefix$block$suffix
if [[ "$mode" == --check ]]; then
 [[ "$updated" == "$current" ]] || fail 'table/prose is stale; run bash assets/readme-performance/v1/render.sh --write'
 echo 'README current-performance table, analysis identity and all six figure hashes agree.'
else
 printf '%s\n' "$updated" > README.md
 echo 'README current-performance block refreshed from committed evidence.'
fi
