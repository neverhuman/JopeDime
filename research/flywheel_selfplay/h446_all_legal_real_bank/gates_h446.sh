#!/bin/bash
# H446 integrity & holdout-disjointness gates. Create-once.
#   gates_h446.sh PREREG_COMMIT -> $EV/results/H446.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H446
STAGE=$(cat "$EV/logs/pub_stage")
R=${WAVE_OUT:-/home/ubuntu/jope-data/h335-cl/h446/$(basename "$STAGE")}
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h446_all_legal_real_bank
OUT=$EV/results/H446.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }

ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }

ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
first=$(find "$R" -type f -printf '%T@\n' | cut -d. -f1 | sort -n | awk 'NR==1')
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H446.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage_hash=$(ok bash -c "cd '$STAGE/code' && sha256sum --quiet -c ../CODE.sha256")
view=$(ok bash -c "[ \"\$(find /home/ubuntu/jope-data/h335-cl/apex/tables -maxdepth 1 -type f | wc -l)\" = 661 ]")

jq -n \
  --slurpfile t "$EV/logs/run_times.json" \
  --slurpfile un "$R/selftest.json" \
  --slurpfile fs "$R/fetch_s3.json" \
  --slurpfile sm "$R/banks/H446_SUMMARY.json" \
  --argjson ptime "$ptime" --argjson first "$first" \
  --argjson anc "$anc" --argjson prereg "$prereg" --argjson stage_hash "$stage_hash" \
  --argjson view "$view" --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" '
  $t[0] as $t | $un[0] as $un | $fs[0] as $fs | $sm[0] as $sm |
  {schema: "jope.h446.gates/v1", at: $at,
   checks: {
     prereg_commit_time: $ptime, first_output_time: $first, stage_mode: $mode,
     run_times: $t,
     fit_icl_tables: $sm.banks.fit_icl_wide400.tables,
     excl_p201b_tables: $sm.banks.all_legal_excl_p201b.tables,
     all_legal_tables: $sm.banks.all_legal.tables,
     all_legal_wide_gt_100: $sm.banks.all_legal.wide_gt_100_feats,
     dropped_s4: ($sm.screen_audit.dropped_confirm_or_dev91_s4 | length),
     dropped_dev_h425: ($sm.screen_audit.dropped_dev_h425_s1_s4 | length),
     delta_refusal_log_lines: $sm.delta_refusal_log_lines,
     s3_write_operations: $fs.s3_write_operations
   },
   gates: {
     g1_prereg_before_output: ($anc and $prereg and $first > $ptime),
     g2_pinned_stage: ($stage_hash and $mode == "pub"),
     g3_selftest: ($un.all_passed == true),
     g4_zero_holdout_and_audit_leakage: (
       $sm.banks.all_legal.confirm_table_opens == 0 and
       $sm.banks.all_legal.dev91_table_opens == 0 and
       $sm.banks.all_legal.p201b_audit_rows_written == 0 and
       $sm.banks.all_legal.fitsel_audit_rows_written == 0 and
       $sm.delta_refusal_log_lines == 0 and
       $sm.guard.stage_v_unlocked == false and
       $fs.s3_write_operations == 0 and
       $view
     ),
     g5_fit_icl_strict_cohort_and_scale: (
       ($sm.banks.fit_icl_wide400.cohorts | keys) == ["FIT-ICL"] and
       $sm.banks.fit_icl_wide400.tables >= 950 and
       $sm.banks.all_legal_excl_p201b.tables >= 1400 and
       $sm.banks.all_legal.tables >= 1650 and
       $sm.banks.all_legal.max_cols_observed <= 401 and
       $sm.banks.all_legal.wide_gt_100_feats >= 50
     ),
     g6_cpu_budget: ($t.wall_s <= 7200 and $t.procs <= 4)
   }}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"
chmod a-w "$OUT"
jq '{gates, pass, checks}' "$OUT"
