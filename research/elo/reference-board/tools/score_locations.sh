#!/bin/bash
# Regenerate SCORE_LOCATIONS.json, the machine-readable form of README section 1, from the files stored in this
# directory (BOARD.json, scores/*.jsonl, ARCHIVE.json, external/*). Reads git files only; touches no host or S3.
# Usage: tools/score_locations.sh > SCORE_LOCATIONS.json
set -euo pipefail
RB=$(cd "$(dirname "$0")/.." && pwd)
GIT=research/elo/reference-board

entrants() {  # board -> {id: {records, completed}}
  for f in "$RB/boards/$1/scores/"*.jsonl; do
    jq -n --arg id "$(basename "$f" .jsonl)" --argjson s "$(jq -s '[length, (map(select(.status == "completed")) | length)]' "$f")" \
      '{($id): {records: $s[0], completed: $s[1]}}'
  done | jq -s 'add'
}
host_root() {
  case $1 in
    dev91-audit-v1) echo "xbabe2:/home/ubuntu/jope-data/h335-stagev/refboard/dev91-audit-v1";;
    p201b-train-audit-v1) echo "xbabe2:/home/ubuntu/jope-data/h335-cl/refboard/p201b-train-audit-v1";;
    dev91-s1-two-fold-v1) echo "xbabe2:/home/ubuntu/jope-data/h257/refboard/dev91-s1-two-fold-v1";;
  esac
}
boards=$(for d in "$RB"/boards/*/; do
  b=$(basename "$d")
  jq -n --arg b "$b" --arg git "$GIT/boards/$b/" --arg host "$(host_root "$b")" \
    --slurpfile bd "$d/BOARD.json" --slurpfile ar "$d/ARCHIVE.json" --argjson en "$(entrants "$b")" \
    '{($b): {git: $git, host_root: $host, board_sha256: $bd[0].board_sha256, n_contexts: $bd[0].n_contexts,
             n_families: $bd[0].n_families, split: $bd[0].split, s3_archive: $ar[0], entrants: $en}}'
done | jq -s 'add')
S3S=$RB/external/dev91-s3-single-split-v1.jsonl
s3single=$(jq -s --arg git "$GIT/external/dev91-s3-single-split-v1.jsonl" '
  {git: $git, records: length,
   note: "The original objects stay where their campaigns wrote them; every line has s3_uri + version_id.",
   models: (group_by(.model_id) | map({key: .[0].model_id,
            value: {contexts: length, protocol: .[0].protocol,
                    s3_prefixes: (map(.s3_uri | sub("/[^/]*$"; "") | sub("^(?<p>s3://[^/]+/experiments/[^/]+/[^/]+/[^/]+/[^/]+).*"; "\(.p)")) | unique)}})
            | from_entries)}' "$S3S")
panel=$(jq -n --arg git "$GIT/external/panel_v1/" --slurpfile ar "$RB/external/panel_v1/ARCHIVE.json" \
  --slurpfile e "$RB/external/panel_v1/elo_panel_v1.json" \
  '{git: $git, s3_archive: $ar[0], regression_elo: ($e[0].regression.elo // null), binary_elo: ($e[0].binary.elo // null)}')
jq -n --argjson boards "$boards" --argjson s3single "$s3single" --argjson panel "$panel" '
{schema: "jope.reference-board.score-locations/v1",
 generated_by: "research/elo/reference-board/tools/score_locations.sh",
 rule: "Never re-run a reference model. Read these stored scores. Numbers from different splits are never compared.",
 boards: $boards,
 external: {"dev91-s3-single-split-v1": $s3single, panel_v1: $panel,
            sotarena_golden45: {git: "assets/regression/ (sotarena-tabular-regression-v2/results/tabular-regression-v2/scores.csv.zst, leaderboards; SOURCES.md)"}},
 models: [
  {model: "TabFM 1.0", two_fold: "h257.TabFM", audit: "tabfm.v1"},
  {model: "TabFM 1.0 ensemble", two_fold: "h257.TabFM_Ensemble", audit: "tabfm_ensemble.v1"},
  {model: "Kumo-Tabular large", two_fold: "h257.Kumo_Tabular_L", audit: "kumo_tabular_large_e8.v1",
   s3_single_split: "s3.Kumo_Tabular_L_S3_Query1Split", panel_v1: "kumo_tabular_large_e8"},
  {model: "Kumo-Tabular small", two_fold: "h257.Kumo_Tabular_S", audit: "kumo_tabular_small_e8.v1", s3_single_split: "s3.Kumo_Tabular_S_S3_Query1Split"},
  {model: "TabPFN 3.5", two_fold: "h257.PriorLabs_TabPFN_3p5", audit: "tabpfn_v3p5.v1",
   s3_single_split: "s3.PriorLabs_TabPFN_3p5_S3_OuterTest"},
  {model: "TabPFN 3.5 (1 estimator)", two_fold: "h257.PriorLabs_TabPFN_3p5_Single", audit: "tabpfn_v3p5_e1.v1"},
  {model: "TabPFN v3", audit: "tabpfn_v3.v1"},
  {model: "TabICL v2", two_fold: "h257.TabICL_v2", audit: "tabicl_v2.v1", s3_single_split: "s3.TabICL_v2_S3_Query1Split"},
  {model: "TabDPT 1.3", two_fold: "h257.TabDPT_1p3", audit: "tabdpt_1p3.v1", s3_single_split: "s3.TabDPT_1p3_S3_Query1Split"},
  {model: "Mitra v2", two_fold: "h257.Mitra_v2", audit: "mitra_v2_zeroshot.v1", s3_single_split: "s3.Mitra_v2_S3_Query1Split"},
  {model: "AutoGluon (300 s)", audit: "autogluon_300s.v1", panel_v1: "autogluon"},
  {model: "CatBoost", audit: "catboost.v1", panel_v1: "catboost"},
  {model: "XGBoost", audit: "xgboost.v1", panel_v1: "xgboost"},
  {model: "LightGBM", audit: "lightgbm.v1", panel_v1: "lightgbm"},
  {model: "Random forest", audit: "random_forest.v1", panel_v1: "random_forest"},
  {model: "Ridge", audit: "ridge.v1", panel_v1: "ridge"},
  {model: "Opus5p5_Extra (LLM regressor)", two_fold: "h257.Opus5p5_Extra", audit: "h335.Opus5p5_Extra"},
  {model: "Gemini4_argon_high (LLM regressor)", two_fold: "h257.Gemini4_argon_high", audit: "h335.Gemini4_argon_high"},
  {model: "GPT6p1_Sol_Max (LLM regressor)", two_fold: "h257.GPT6p1_Sol_Max", audit: "h335.GPT6p1_Sol_Max"},
  {model: "GPT6_Astra_Max (LLM regressor)", two_fold: "h257.GPT6_Astra_Max", audit: "h335.GPT6_Astra_Max"},
  {model: "grok4p7_xhigh (LLM regressor)", audit: "h412.grok4p7_xhigh"},
  {model: "kimik3p1_max (LLM regressor)", two_fold: "h414.kimik3p1_max", audit: "h414.kimik3p1_max"}
 ]}'
