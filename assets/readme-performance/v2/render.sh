#!/usr/bin/env bash
# Refresh the README's current regression ranking from committed evidence; no fits or dataset reads.
# v2 renders the six-entrant AutoML Elo v3 panel and kimik3p1_max's rank on every panel it was
# scored on. v1 (assets/readme-performance/v1) rendered the four-entrant panel the paper binds.
set -euo pipefail
root=${JOPE_PERFORMANCE_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
mode=${1:---check}
case "$mode" in --check|--write) ;; *) echo 'usage: render.sh [--check|--write]' >&2; exit 2 ;; esac
cd "$root"
results=assets/automl-elo/v3/RESULTS.json
record=crates/jope-automl/submissions/kimik3p1_max/verification.json
# The prose below was reviewed against these exact bytes. If either file changes, reconcile the
# prose with it first, then update its pin.
results_sha256=d6466e2c70b29f209253862ab8555da6d14a41ade08c8721f77e8c3dd95bfd60
record_sha256=749b2dafbf2fc10301ccc99207a6c7a0507235f33b7ff25d307c6b30987c0909
# The research-branch commit that holds the reference-board ladders the record summarises.
boards_commit=7f7ff32833de92327c71ed6abcd683d0decf4660
boards_url=https://github.com/neverhuman/JopeDime/blob/$boards_commit/research/elo/reference-board/boards
fail() { echo "README performance: $*" >&2; exit 1; }
sha() { local h; h=$(sha256sum "$1"); printf '%s' "${h%% *}"; }
[[ $(sha "$results") == "$results_sha256" ]] || fail "$results changed since the prose was reviewed; reconcile the prose, then update its pin"
[[ $(sha "$record") == "$record_sha256" ]] || fail "$record changed since the prose was reviewed; reconcile the prose, then update its pin"
jq -e '
 .state == "COMPLETE" and .complete_pairwise_coverage == true and
 .cluster == "family" and .bootstrap_samples == 10000 and
 .seeds_repeated_within_family == [41,73,109] and
 .methods == ["GPT6_Astra_Max","GPT6p1_Sol_Max","Gemini4_argon_high","Opus5p5_Extra","grok4p7_xhigh","kimik3p1_max"] and
 ([.elo | keys[]] == .methods) and
 all(.elo[]; (.rating | isfinite) and (.ci95 | length == 2) and
     all(.ci95[]; isfinite) and .ci95[0] <= .rating and .rating <= .ci95[1]) and
 (.pairwise | length == 15) and all(.pairwise[]; .paired_families == 64 and
     (.geometric_rmse_ratio | isfinite) and .geometric_rmse_ratio > 0 and
     (.ratio_ci95 | length == 2) and all(.ratio_ci95[]; isfinite) and
     .ratio_ci95[0] > 0 and .ratio_ci95[0] <= .geometric_rmse_ratio and
     .geometric_rmse_ratio <= .ratio_ci95[1]) and
 ([.pairwise[] | [.a,.b]] | sort) ==
 ([.methods as $m | range(0; $m | length) as $i | range($i + 1; $m | length) as $j | [$m[$i], $m[$j]]] | sort)
' "$results" >/dev/null || fail 'incomplete or incompatible comparison'
# Every sentence of interpretation below is one of these conditions.
jq -e '
 def rating(m): .elo[m].rating;
 def within(x; m): .elo[m].ci95[0] <= x and x <= .elo[m].ci95[1];
 [.elo | to_entries | sort_by(-.value.rating)[] | .key] as $order |
 $order[0] == "Opus5p5_Extra" and $order[2] == "kimik3p1_max" and
 within(rating("GPT6p1_Sol_Max"); "kimik3p1_max") and within(rating("kimik3p1_max"); "GPT6p1_Sol_Max") and
 all(.pairwise[]; if .a == "GPT6p1_Sol_Max" then .geometric_rmse_ratio < 1
 elif .b == "GPT6p1_Sol_Max" then .geometric_rmse_ratio > 1 else true end) and
 ([.pairwise[] | select(has("holm_kimi_five_p")) | .a] | sort) ==
 ([.methods[] | select(. != "kimik3p1_max")] | sort) and
 all(.pairwise[] | select(has("holm_kimi_five_p")); .b == "kimik3p1_max" and
     (.holm_kimi_five_p | isfinite) and .holm_kimi_five_p >= 0 and .holm_kimi_five_p <= 1) and
 ([.pairwise[] | select(has("holm_kimi_five_p") and .holm_kimi_five_p < 0.05) |
   [.a, (.geometric_rmse_ratio < 1)]] | sort) == [["Gemini4_argon_high", true], ["grok4p7_xhigh", false]]
' "$results" >/dev/null || fail 'winner, Kimi rank, Sol/Kimi overlap, error-size or Holm interpretation changed; review the prose'
jq -e --slurpfile res "$results" '
 $res[0].elo.kimik3p1_max as $k |
 .elo.results == "assets/automl-elo/v3/RESULTS.json" and .elo.cells_completed == 576 and
 .elo.rating == $k.rating and .elo.ci95 == $k.ci95 and .elo.rank == "3 of 6" and
 (.reference_boards.note | contains("v1/next 7f7ff32833")) and
 (.reference_boards | del(.note, .wave) | keys) == ["dev91-audit-v1","dev91-s1-two-fold-v1","p201b-train-audit-v1"] and
 all(.reference_boards | del(.note, .wave)[];
     (.rank | test("^[1-9][0-9]* of [1-9][0-9]*$")) and
     (.rank | split(" of ") | map(tonumber) | .[0] <= .[1]) and
     (.elo | isfinite) and (.ci95 | length == 2) and all(.ci95[]; isfinite) and
     .ci95[0] <= .elo and .elo <= .ci95[1])
' "$record" >/dev/null || fail 'the Kimi record disagrees with the results, or its board inventory changed'
begin='<!-- current-performance:begin -->'
end='<!-- current-performance:end -->'
[[ $(grep -Fxc "$begin" README.md || true) == 1 && $(grep -Fxc "$end" README.md || true) == 1 ]] || fail 'expected one current-performance marker pair'
panel_row() { # label, description, rank, rating, low, high
 LC_ALL=C printf '| %s | %s | %s | %.1f | %.1f–%.1f |\n' "$1" "$2" "$3" "$4" "$5" "$6"
}
board_row() { # board id, description
 local rank rating low high
 IFS=$'\t' read -r rank rating low high < <(jq -r --arg b "$1" '.reference_boards[$b] | [.rank,.elo,.ci95[0],.ci95[1]] | @tsv' "$record")
 panel_row "[\`$1\`]($boards_url/$1/LADDER.md)" "$2" "$rank" "$rating" "$low" "$high"
}
block=$(
 printf '%s\n\n' "$begin"
 cat <<'TEXT'
### Latest complete native-regressor ranking

**64 regression families × 3 folds × 3 seeds; 576 fits per entrant.** These are six
LLM-authored regression algorithms, fitted on the same prepared inner-CV inputs.
The names identify the submitted algorithms. This table does not measure the
LLMs at inference time or the neural program-proposal policy.

| Rank | Algorithm | Elo | 95% family-bootstrap interval |
| ---: | --- | ---: | ---: |
TEXT
 jq -r '.elo | to_entries | sort_by(-.value.rating) | to_entries[] |
 [(.key+1),.value.key,.value.value.rating,.value.value.ci95[0],.value.value.ci95[1]] | @tsv' "$results" |
 while IFS=$'\t' read -r rank method rating low high; do
  LC_ALL=C printf '| %d | `%s` | %.1f | %.1f–%.1f |\n' "$rank" "$method" "$rating" "$low" "$high"
 done
 cat <<'TEXT'

**Opus has the highest Elo. `kimik3p1_max`, added on 2026-11-07, ranks third, level
with Sol within both 95% intervals. Sol still has the lowest geometric-mean RMSE
point estimate.** Elo rewards how often a method wins, while RMSE ratios also
reflect error size.

| Opponent | Opponent / `kimik3p1_max` RMSE | 95% family-bootstrap interval | Holm-adjusted p |
| --- | ---: | ---: | ---: |
TEXT
 jq -r '.elo as $e | [.pairwise[] | select(.b == "kimik3p1_max")] | sort_by(-$e[.a].rating)[] |
 [.a,.geometric_rmse_ratio,.ratio_ci95[0],.ratio_ci95[1],.holm_kimi_five_p] | @tsv' "$results" |
 while IFS=$'\t' read -r name ratio low high holm; do
  LC_ALL=C printf '| `%s` | %.3f | %.3f–%.3f | %.3f |\n' "$name" "$ratio" "$low" "$high" "$holm"
 done
 cat <<'TEXT'

A ratio below 1 means the opponent's error is lower. The Holm adjustment covers
Kimi's five comparisons. Two differences remain significant at 0.05 after it:
`Gemini4_argon_high` has lower error than Kimi, and Kimi has lower error than
`grok4p7_xhigh`.

All 15 pairs cover all 64 families; 10,000 family bootstrap resamples retain the
three seeds within each family. These are existing training families with
historical exposure. Stored preprocessing predates the original split, and seed 41
was seen before seeds 73 and 109 were fixed. This is exploratory development
evidence, with no untouched-family claim. The entrants were fitted on different
hosts; fit times are not a controlled speed ranking. Elo values from different
opponent pools cannot be compared. The `grok4p7_xhigh` cells come from its open
pull request; its source is not on `main` yet.

#### `kimik3p1_max` on every panel it was scored on

| Panel | What it scores | Kimi rank | Elo | 95% interval |
| --- | --- | ---: | ---: | ---: |
TEXT
 jq -r '.elo | [.rank,.rating,.ci95[0],.ci95[1]] | @tsv' "$record" |
 while IFS=$'\t' read -r rank rating low high; do
  panel_row '[AutoML Elo v3](assets/automl-elo/v3/SOURCES.md)' '64 training families × 3 folds × 3 seeds' "$rank" "$rating" "$low" "$high"
 done
 board_row dev91-audit-v1 'DL-Val91: 91 development contexts in 19 families, AUDIT split'
 board_row dev91-s1-two-fold-v1 'the same 91 contexts, two-fold split'
 board_row p201b-train-audit-v1 '201 training contexts, one per family, AUDIT split'
 cat <<'TEXT'

Each panel has its own opponents and anchor, so Elo values cannot be compared
across rows. The reference boards also rate other kinds of model: foundation
models on both DL-Val91 boards, and classical baselines and JopeDime's own search
models on `dev91-audit-v1` and `p201b-train-audit-v1`. Their full ladders are on
the research branch `v1/next`, linked above. Kimi was scored once on each board,
and no other entrant was rerun.

Source: [results and uncertainty](assets/automl-elo/v3/RESULTS.json),
[protocol and fit custody](assets/automl-elo/v3/SOURCES.md), and
[Kimi's verification record](crates/jope-automl/submissions/kimik3p1_max/verification.json).
The paper's evidence binds the earlier four-entrant panel and its figures,
[`assets/automl-elo/v1`](assets/automl-elo/v1/SOURCES.md).
TEXT
 printf '\n%s' "$end"
)
current=$(cat README.md)
[[ "$current" == *"$begin"*"$end"* ]] || fail 'markers are out of order'
prefix=${current%%"$begin"*}
suffix=${current#*"$end"}
updated=$prefix$block$suffix
if [[ "$mode" == --check ]]; then
 [[ "$updated" == "$current" ]] || fail 'table/prose is stale; run bash assets/readme-performance/v2/render.sh --write'
 echo 'README current-performance tables and their pinned evidence (v3 results, Kimi record) agree.'
else
 printf '%s\n' "$updated" > README.md
 echo 'README current-performance block refreshed from committed evidence.'
fi
