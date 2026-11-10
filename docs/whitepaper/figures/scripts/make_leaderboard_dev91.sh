#!/usr/bin/env bash
# Writes the dev91-audit-v1 leaderboard exhibits (fig:leaderboard, tab:headtohead) from two
# committed v1/next files, read as Git objects:
#   - the board ladder (Elo, 95% interval) at 7f7ff32833;
#   - the H414 FINAL study (names, classes and XSTACK_2L's paired contrasts) at 45008196.
# No entrant is fitted, scored or rerun. Bash + jq only; no network.
#
# Usage: make_leaderboard_dev91.sh [--check|--write] [WHITEPAPER_DIR]
#   JOPE_REPO selects the Git repository that holds both commits (default: this checkout).
set -euo pipefail

mode=${1:---check}
case "$mode" in --check|--write) ;; *) echo "usage: $0 [--check|--write] [DIR]" >&2; exit 2 ;; esac
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
dir=${2:-$here}
repo=${JOPE_REPO:-$(git -C "$here" rev-parse --show-toplevel)}

board_commit=7f7ff32833de92327c71ed6abcd683d0decf4660
board_path=research/elo/reference-board/boards/dev91-audit-v1/ladder.json
board_sha256=92ebf1f6ea2645d6eb6295f2a16b2c404cbfc38a9ae61d2c9c808721f62c601d
study_commit=45008196814c9a0730fe11dc32bbff2c74f9bb75
study_path=research/flywheel_selfplay/h414_kimi_k3p1_max/final_study.json
study_sha256=a48ac7d40923d74458161e1eb0b56fd23a3e0abe2e2229c272251dd1e4d5aed1

blob() { git -C "$repo" show "$1:$2"; }
for pin in "$board_commit $board_path $board_sha256" "$study_commit $study_path $study_sha256"; do
  set -- $pin
  got=$(blob "$1" "$2" | sha256sum | cut -d' ' -f1)
  [ "$got" = "$3" ] || { echo "make_leaderboard_dev91: $2 at ${1:0:10} has sha256 $got, expected $3" >&2; exit 1; }
done
board_blob=$(git -C "$repo" rev-parse "$board_commit:$board_path")
study_blob=$(git -C "$repo" rev-parse "$study_commit:$study_path")

lib='
def fx(d): . as $x | pow(10; d) as $m | (($x * $m) | round) as $r | ($r | fabs) as $a
  | (if $r < 0 then "-" else "" end) + (($a / $m) | floor | tostring)
  + (if d > 0 then "." + ((($a % $m) + $m) | tostring | .[1:]) else "" end);
def num(d): "\\num{" + fx(d) + "}";
def pval: if . < 0.001 then "$<$\\num{0.001}" elif . < 0.01 then num(3)
          elif . > 0.99 then "$>$\\num{0.99}" else num(2) end;
def tex: gsub("_"; "\\_");
def dname: .name | sub("^h[0-9]+\\."; "") | tex;
'

data=$(jq -n --slurpfile b <(blob "$board_commit" "$board_path") --slurpfile s <(blob "$study_commit" "$study_path") \
  --arg bc "$board_commit" --arg bp "$board_path" --arg bb "$board_blob" --arg bs "$board_sha256" \
  --arg sc "$study_commit" --arg sp "$study_path" --arg sb "$study_blob" --arg ss "$study_sha256" "$lib"'
  $b[0] as $B | $s[0] as $S
  | if $S.board != $B.board or $S.board_sha256 != $B.board_sha256 then error("study and board disagree on the board") else . end
  | if ($B.incomplete | length) != 0 then error("board has incomplete entrants") else . end
  | ($B.elo.elo | del(.anchor)) as $E
  | if ($S.ladder | length) != ($E | length) then error("entrant counts differ") else . end
  | if any($S.ladder[]; .elo != $E[.id].elo or .ci95 != $E[.id].ci95) then error("study ladder differs from board ladder") else . end
  | if $S.ours != "h364.XSTACK_2L" then error("unexpected focus") else . end
  | if ($S.ours_vs | length) != ($E | length) - 1 then error("missing contrasts") else . end
  | {schema: "eurekazero.leaderboard-dev91/v1",
     note: "Stored rows only: every entrant was fitted on the SEARCH rows and scored once on the AUDIT rows. dev91-audit-v1 is a reuse cohort (H257 tuned on it; H381 and H410 read five of its contexts).",
     sources: {board: {commit: $bc, path: $bp, git_blob: $bb, sha256: $bs},
               study: {commit: $sc, path: $sp, git_blob: $sb, sha256: $ss}},
     board: $B.board, board_sha256: $B.board_sha256, anchor: $S.anchor,
     n_contexts: $B.n_contexts, n_families: $B.n_families,
     statistics: $S.statistics,
     ours: ["h364.XSTACK_2L", "h335.H335_AGG"], focus: $S.ours,
     entrants: [$S.ladder[] | {rank, id, name, class, elo, ci95}],
     contrasts: ($S.ours_vs | with_entries(.value |= {
       families_won: .primary_clipped.families_won, families_tied: .primary_clipped.families_tied,
       families_lost: .primary_clipped.families_lost,
       gain: .headline.family_mean_gain, gain_ci95: .gain_ci95_family_bootstrap,
       rmse_cut: .headline.pct_rmse_reduction,
       signflip_p: .primary_clipped.signflip_p, mirror_signflip_p: .primary_clipped.mirror_signflip_p,
       lofo_min: .primary_clipped.lofo_min,
       verdict: (if .primary_clipped.pass then "better" elif .primary_clipped.inferior then "worse" else "not separated" end)}))}')

# The tabular of tab:headtohead (caption and label stay in 07_results.tex): grouped by class,
# ordered by rank. The whole environment is generated so that no \input sits inside the alignment.
rows=$(jq -r "$lib"'
  . as $D | ($D.entrants | map({(.id): .}) | add) as $by
  | def row(id): ($by[id]) as $e | $D.contrasts[id] as $c
      | "    \($e | dname) & \($e.rank) & \($c.families_won)--\($c.families_tied)--\($c.families_lost) & \($c.gain | num(3)) [\($c.gain_ci95[0] | num(3)), \($c.gain_ci95[1] | num(3))] & \($c.rmse_cut * 100 | num(1))\\% & \($c.signflip_p | pval) & \($c.lofo_min | num(3)) & \($c.verdict) \\\\";
    def group(title; ids): "    \\multicolumn{8}{@{}l}{\\emph{" + title + "}} \\\\", (ids[] | row(.));
  "% Generated by figures/scripts/make_leaderboard_dev91.sh from figures/data/leaderboard_dev91.json; do not edit.",
  "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}Xrcrrrrl@{}}",
  "    \\toprule",
  "    Comparator & Elo rank & Families W--T--L & Gain [95\\% interval] & RMSE & $p$ & LOFO & Verdict \\\\",
  "    \\midrule",
  group("Tabular foundation models"; ["kumo_tabular_large_e8.v1", "tabfm_ensemble.v1", "tabpfn_v3p5.v1", "mitra_v2_zeroshot.v1", "tabicl_v2.v1"]),
  "    \\addlinespace",
  group("Frontier-designed regressors"; [$D.entrants[] | select(.class == "frontier LLM") | .id]),
  "    \\addlinespace",
  group("AutoML and classic learners"; ["autogluon_300s.v1", "catboost.v1"]),
  "    \\bottomrule",
  "\\end{tabularx}"
' <<<"$data")

# Figure: one row per entrant, Elo and 95% interval; marker shape encodes the class.
fig=$(jq -r "$lib"'
  . as $D
  | 950 as $lo | 2350 as $hi | 3.10 as $x0 | 8.05 as $x1 | 0.27 as $dy
  | def X: ((. - $lo) / ($hi - $lo) * ($x1 - $x0) + $x0) | fx(3);
    def Y(r): (-(r - 1) * $dy) | fx(3);
    def mark(cls; x; y):
      if cls == "foundation model" then "\\draw[black,fill=white,line width=0.4pt] (\(x),\(y)) circle (0.065);"
      elif cls == "frontier LLM" then "\\fill[black] (\(x),\(y)) ++(0,0.08) -- ++(-0.075,-0.13) -- ++(0.15,0) -- cycle;"
      elif cls == "AutoML" then "\\draw[black,fill=white,line width=0.4pt] (\(x),\(y)) ++(0,0.085) -- ++(0.085,-0.085) -- ++(-0.085,-0.085) -- ++(-0.085,0.085) -- cycle;"
      elif cls == "classic learner" then "\\draw[black,fill=white,line width=0.4pt] (\(x),\(y)) ++(-0.06,-0.06) rectangle ++(0.12,0.12);"
      else "\\fill[black] (\(x),\(y)) circle (0.07);" end;
    ($D.entrants | length) as $n | (-(($n - 1) * $dy) - 0.22) as $ybot
  | "% Generated by figures/scripts/make_leaderboard_dev91.sh from figures/data/leaderboard_dev91.json; do not edit.",
    "\\begin{tikzpicture}[font=\\scriptsize]",
    ([1000, 1250, 1500, 1750, 2000, 2250][] as $t
      | "\\draw[\(if $t == 1500 then "black!55,dashed" else "black!12" end),line width=0.4pt] (\($t | X),0.22) -- (\($t | X),\($ybot | fx(3)));",
        "\\node[anchor=north] at (\($t | X),\($ybot | fx(3))) {\($t)};"),
    "\\draw[black!55,line width=0.4pt] (\($x0 | fx(2)),\($ybot | fx(3))) -- (\($x1 | fx(2)),\($ybot | fx(3)));",
    "\\node[anchor=north] at (\((($x0 + $x1) / 2) | fx(2)),\(($ybot - 0.32) | fx(3))) {Elo on \\texttt{dev91-audit-v1} (dashed: anchor, 1500)};",
    ($D.entrants[] as $e | ($D.ours | index($e.id)) as $ours
      | "\\node[anchor=east] at (2.95,\(Y($e.rank))) {\($e.rank).\\ \(if $ours then "\\textbf{\($e | dname)}" else ($e | dname) end)};",
        "\\draw[black!45,line width=0.7pt] (\($e.ci95[0] | X),\(Y($e.rank))) -- (\($e.ci95[1] | X),\(Y($e.rank)));",
        mark($e.class; $e.elo | X; Y($e.rank))),
    (($ybot - 0.95) | fx(3)) as $ly
    | ([["foundation model", "foundation model", 0.10], ["frontier LLM", "frontier LLM", 2.60], ["AutoML", "AutoML", 4.45],
        ["classic learner", "classic learner", 5.75], ["JopeDime", "JopeDime (bold: ours, locked)", 0.10]][] as [$cls, $txt, $lx]
       | (if $cls == "JopeDime" then (($ybot - 1.30) | fx(3)) else $ly end) as $yy
       | mark($cls; ($lx + 0.12) | fx(2); $yy),
         "\\node[anchor=west] at (\(($lx + 0.25) | fx(2)),\($yy)) {\($txt)};"),
    "\\end{tikzpicture}"
' <<<"$data")

json=$(jq . <<<"$data")
emit() { # path content
  if [ "$mode" = --write ]; then
    mkdir -p "$(dirname "$dir/$1")"
    printf '%s\n' "$2" > "$dir/$1"
  elif ! cmp -s <(printf '%s\n' "$2") "$dir/$1"; then
    echo "make_leaderboard_dev91: $1 is stale; rerun with --write" >&2
    return 1
  fi
}
status=0
emit figures/data/leaderboard_dev91.json "$json" || status=1
emit figures/data/leaderboard_dev91_table.tex "$rows" || status=1
emit figures/fig_leaderboard_dev91.tex "$fig" || status=1
[ "$status" -eq 0 ] && echo "dev91-audit-v1 leaderboard exhibits ${mode#--}: $(jq '.entrants | length' <<<"$data") entrants, $(jq '.contrasts | length' <<<"$data") contrasts, sources verified"
exit "$status"
