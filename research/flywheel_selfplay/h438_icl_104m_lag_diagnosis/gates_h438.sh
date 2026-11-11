#!/bin/bash
# H438 integrity gates (PREREG section 8). The wave is one read of stored logs, so the gates run BEFORE it:
# the driver runs this script and starts the analysis only if every gate passes. Create-once output.
#   gates_h438.sh PREREG_COMMIT      -> $EV/results/H438.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published the PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H438
STAGE=$(cat "$EV/logs/pub_stage"); REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h438_icl_104m_lag_diagnosis
OUT=$EV/results/H438.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
pulled=$(stat -c %Y "$EV/pulled_logs/SHA256SUMS")
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H438.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
logs=$(ok bash -c "cd $EV/pulled_logs && sha256sum --quiet -c SHA256SUMS")
noread=$([ ! -e "$EV/results/H438.lag.json" ] && echo true || echo false)
view=$(ok bash -c "[ \"\$(ls /home/ubuntu/jope-data/h335-cl/apex/tables | wc -l)\" = 661 ]")
jq -n --argjson ptime "$ptime" --argjson pulled "$pulled" --argjson now "$(date +%s)" --argjson anc "$anc" --argjson prereg "$prereg" \
  --argjson stage "$stage" --argjson logs "$logs" --argjson noread "$noread" --argjson view "$view" --arg mode "$(cat "$STAGE/mode")" \
  --arg at "$(date -u +%FT%TZ)" --arg sums "$(sha256sum "$EV/pulled_logs/SHA256SUMS" | cut -d' ' -f1)" --argjson files "$(wc -l < "$EV/pulled_logs/SHA256SUMS")" '
  {schema: "jope.h438.gates/v1", at: $at,
   checks: {prereg_commit_time: $ptime, logs_pulled_time: $pulled, gate_time: $now, stage_mode: $mode, log_files: $files, log_sums_sha256: $sums},
   gates: {
     g1_prereg_before_read: ($anc and $prereg and $now > $ptime and $noread),
     g2_log_copies_intact_and_older_than_prereg: ($logs and $files == 8 and $pulled < $ptime),
     g3_confirm_untouched: $view,
     g4_pub_stage: ($stage and $mode == "pub")}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"; jq '{gates, pass}' "$OUT"
jq -e '.pass' "$OUT" > /dev/null
