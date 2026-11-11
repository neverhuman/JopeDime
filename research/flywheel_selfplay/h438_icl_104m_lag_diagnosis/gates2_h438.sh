#!/bin/bash
# H438 gates after addendum 1 (code fix before any result). Run BEFORE the analysis. Create-once output.
#   gates2_h438.sh PREREG_COMMIT ADDENDUM_COMMIT      -> $EV/results/H438.gates2.json
set -euo pipefail
PC=${1:?commit that published the PREREG and code}; AC=${2:?commit that published addendum 1}
EV=/home/ubuntu/jope_plan/evidence/H438
STAGE=$(cat "$EV/logs/pub_stage2"); REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h438_icl_104m_lag_diagnosis
OUT=$EV/results/H438.gates2.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
pulled=$(stat -c %Y "$EV/pulled_logs/SHA256SUMS")
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
anc2=$(ok git -C "$REPO" merge-base --is-ancestor "$AC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H438.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
add=$(ok cmp -s "$STAGE/code/PREREG_H438_ADDENDUM_1.md" <(git -C "$REPO" show "$AC:$WDIR/PREREG_ADDENDUM_1.md"))
code=$(ok cmp -s "$STAGE/code/lag_h438.py" <(git -C "$REPO" show "$AC:$WDIR/lag_h438.py"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
logs=$(ok bash -c "cd $EV/pulled_logs && sha256sum --quiet -c SHA256SUMS")
noread=$([ ! -e "$EV/results/H438.lag.json" ] && [ ! -s "$EV/results/H438.lag.txt" ] && echo true || echo false)
view=$(ok bash -c "[ \"\$(ls /home/ubuntu/jope-data/h335-cl/apex/tables | wc -l)\" = 661 ]")
changed=$(git -C "$REPO" diff --numstat "$PC" "$AC" -- "$WDIR/lag_h438.py" | gawk '{print $1 + $2}')
jq -n --slurpfile a1 "$EV/logs/run_times_attempt1.json" --argjson ptime "$ptime" --argjson pulled "$pulled" --argjson now "$(date +%s)" \
  --argjson anc "$anc" --argjson anc2 "$anc2" --argjson prereg "$prereg" --argjson add "$add" --argjson code "$code" --argjson stage "$stage" \
  --argjson logs "$logs" --argjson noread "$noread" --argjson view "$view" --argjson changed "${changed:-0}" --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" '
  {schema: "jope.h438.gates2/v1", at: $at,
   checks: {prereg_commit_time: $ptime, logs_pulled_time: $pulled, gate_time: $now, stage_mode: $mode, first_attempt: $a1[0],
            code_lines_changed_by_addendum: $changed},
   gates: {
     g1_prereg_before_read: ($anc and $prereg and $now > $ptime and $noread),
     g2_log_copies_intact_and_older_than_prereg: ($logs and $pulled < $ptime),
     g3_confirm_untouched: $view,
     g4_pub_stage: ($stage and $mode == "pub"),
     g5_addendum_published_no_result: ($anc2 and $add and $code and $noread and $a1[0].exit != 0),
     g6_small_code_change: ($changed <= 8)}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"; jq '{gates, pass, changed: .checks.code_lines_changed_by_addendum}' "$OUT"
jq -e '.pass' "$OUT" > /dev/null
