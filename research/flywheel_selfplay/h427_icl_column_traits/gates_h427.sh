#!/bin/bash
# H427 checks (analysis only): the PREREG is older than the output, the inputs are the stored H424 / H425 files, the
# sandbox view still holds no CONFIRM table, the budget was kept, published stage. Create-once output.
#   gates_h427.sh PREREG_COMMIT      -> $EV/results/H427.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published the PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H427; E=/home/ubuntu/jope_plan/evidence
R=$EV/results/H427_DEV.analysis.json; STAGE=$(cat "$EV/logs/pub_stage")
REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h427_icl_column_traits
OUT=$EV/results/H427.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC"); rtime=$(stat -c %Y "$R")
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H427.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
inputs=$(ok bash -c "[ \"\$(jq -r '.inputs_sha256[\"H424_DEV.apply.json\"]' $R)\" = \"\$(jq -r .evidence.apply_sha256 /home/ubuntu/apex-stage/h424/code/metrics.json)\" ] && [ \"\$(jq -r '.inputs_sha256[\"bank_DEV.npz\"]' $R)\" = \"\$(jq -r .sha256 /home/ubuntu/jope-data/h335-cl/h424/pub/bank_DEV.npz.json)\" ] && cmp -s $E/H425/results/EXCLUSION_DEV.json <(git -C $REPO show origin/v1/next:research/flywheel_selfplay/h425_contamination_screen/EXCLUSION_DEV.json)")
view=$(ok bash -c "[ \"\$(ls /home/ubuntu/jope-data/h335-cl/apex/tables | wc -l)\" = 661 ]")
jq -n --slurpfile t "$EV/logs/run_times.json" --argjson ptime "$ptime" --argjson rtime "$rtime" --argjson anc "$anc" \
  --argjson prereg "$prereg" --argjson stage "$stage" --argjson inputs "$inputs" --argjson view "$view" \
  --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" '
  $t[0] as $t | {schema: "jope.h427.gates/v1", at: $at, checks: {run_times: $t, prereg_commit_time: $ptime, output_time: $rtime, stage_mode: $mode},
   gates: {g1_prereg_before_output: ($anc and $prereg and $rtime > $ptime), g2_stored_inputs: $inputs, g3_confirm_untouched: $view,
           g4_budget: (($t.end - $t.start) <= 3600 and $t.threads <= 8), g5_pub_stage: ($stage and $mode == "pub")}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"; jq '{gates, pass}' "$OUT"
