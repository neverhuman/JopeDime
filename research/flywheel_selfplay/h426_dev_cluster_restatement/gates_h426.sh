#!/bin/bash
# H426 gates (PREREG section 5). The restatement script itself stops unless every stored original is reproduced and the
# screen file is the one the frozen H425 list was cut from; this script checks the rest. Create-once output.
#   gates_h426.sh PREREG_COMMIT      -> $EV/results/H426.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published the PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H426; E=/home/ubuntu/jope_plan/evidence
OUTR=$EV/results/H426_DEV.restatement.json
STAGE=$(cat "$EV/logs/pub_stage")
REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h426_dev_cluster_restatement
OUT=$EV/results/H426.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC"); rtime=$(stat -c %Y "$OUTR")
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H426.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
lists=$(ok cmp -s "$E/H425/results/EXCLUSION_DEV.json" <(git -C "$REPO" show origin/v1/next:research/flywheel_selfplay/h425_contamination_screen/EXCLUSION_DEV.json))
man=$(ok bash -c "cd $E/H420/pulled/H420_DEV && sha256sum --quiet -c ../H420_DEV.sha256 && cd $E/H421/pulled/H421_DEV && sha256sum --quiet -c ../H421_DEV.sha256")
view=$(ok bash -c "[ \"\$(ls /home/ubuntu/jope-data/h335-cl/apex/tables | wc -l)\" = 661 ]")
jq -n --slurpfile r "$OUTR" --slurpfile t "$EV/logs/run_times.json" --argjson ptime "$ptime" --argjson rtime "$rtime" \
  --argjson anc "$anc" --argjson prereg "$prereg" --argjson stage "$stage" --argjson lists "$lists" --argjson man "$man" \
  --argjson view "$view" --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" '
  $r[0] as $r | $t[0] as $t |
  {schema: "jope.h426.gates/v1", at: $at,
   checks: {contexts: $r.contexts, units: $r.units, originals_reproduced_max_abs_diff: $r.originals_reproduced_max_abs_diff,
            rows: ($r.rows | length), run_times: $t, prereg_commit_time: $ptime, output_time: $rtime, stage_mode: $mode,
            sandbox_view_table_files_661_no_confirm: $view},
   gates: {
     g1_prereg_before_output: ($anc and $prereg and $rtime > $ptime),
     g2_stored_inputs: ($lists and $man and $r.contexts == 115),
     g3_originals_reproduced: ($r.originals_reproduced_max_abs_diff <= 1e-12 and ($r.rows | length) >= 10),
     g4_confirm_untouched: $view,
     g5_budget: (($t.end - $t.start) <= 3600 and $t.threads <= 8),
     g6_pub_stage: ($stage and $mode == "pub")}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"
jq '{gates, pass}' "$OUT"
