#!/bin/bash
# H440 integrity gates (PREREG section 4). Run AFTER driver_h440.sh and BEFORE P201b readout. Create-once.
#   gates_h440.sh PREREG_COMMIT -> $EV/results/H440.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H440
STAGE=$(cat "$EV/logs/pub_stage")
R=${WAVE_OUT:-/home/ubuntu/jope-data/h335-cl/h440/$(basename "$STAGE")}
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h440_guard_and_duan_matern_cols
OUT=$EV/results/H440.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }

SHA_BF=f8849ccb7343cfec1c0fd6688aaafef7580b85001046f2707fab58c5ecebe069
SHA_BP=da29e8ec64f623e0fa2ab29b31bbd40b91bef36adf8ddc8e2e773f51c96d01af

ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
first=$(find "$R" -type f -printf '%T@\n' | cut -d. -f1 | sort -n | awk 'NR==1')
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H440.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
noread=$([ ! -e "$R/READOUT_P201B.json" ] && echo true || echo false)
view=$(ok bash -c "[ \"\$(ls /home/ubuntu/jope-data/h335-cl/apex/tables | wc -l)\" = 661 ]")

jq -n \
  --slurpfile t "$EV/logs/run_times.json" \
  --slurpfile un "$R/unit_tests.json" \
  --slurpfile af "$R/APPLY_FITSEL.json" \
  --slurpfile ap "$R/APPLY_P201B.json" \
  --argjson ptime "$ptime" --argjson first "$first" \
  --argjson anc "$anc" --argjson prereg "$prereg" --argjson stage "$stage" \
  --argjson noread "$noread" --argjson view "$view" \
  --arg sbf "$SHA_BF" --arg sbp "$SHA_BP" \
  --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" '
  $t[0] as $t | $un[0] as $un | $af[0] as $af | $ap[0] as $ap |
  {schema: "jope.h440.gates/v1", at: $at,
   checks: {
     prereg_commit_time: $ptime, first_output_time: $first, stage_mode: $mode,
     run_times: $t,
     fitsel_columns_ok: $af.columns_ok, p201b_columns_ok: $ap.columns_ok,
     fitsel_h430_ok: $af.h430_ok, p201b_h430_ok: $ap.h430_ok,
     fitsel_h433_ok: $af.h433_ok, p201b_h433_ok: $ap.h433_ok,
     fitsel_replay_w_diff: $af.replay_max_w_diff, p201b_replay_w_diff: $ap.replay_max_w_diff,
     fitsel_replay_u_diff: $af.replay_max_u_diff, p201b_replay_u_diff: $ap.replay_max_u_diff
   },
   gates: {
     g1_prereg_before_output: ($anc and $prereg and $first > $ptime and $noread),
     g2_pinned_banks: ($af.bank_sha256 == $sbf and $ap.bank_sha256 == $sbp and $af.aligned_all == true and $ap.aligned_all == true),
     g3_replay_exact: ($af.replay_max_w_diff < 1e-12 and $ap.replay_max_w_diff < 1e-12 and $af.replay_max_u_diff < 1e-12 and $ap.replay_max_u_diff < 1e-12),
     g4_cpu_budget: ($t.wall_s <= 14400 and $t.procs <= 16),
     g5_columns_complete: ($af.columns_ok >= 101 and $ap.columns_ok >= 191),
     g6_unit_and_smoke: ($un.all_passed == true),
     g7_confirm_untouched: $view,
     g8_pub_stage: ($stage and $mode == "pub")
   }}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"
chmod a-w "$OUT"
jq '{gates, pass, checks}' "$OUT"
