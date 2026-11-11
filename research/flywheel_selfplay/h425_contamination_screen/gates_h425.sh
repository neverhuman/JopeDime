#!/bin/bash
# H425 integrity gates (PREREG section 7). Run AFTER the screen and the publication of the frozen lists and BEFORE the
# sensitivity read. Create-once output.
#   gates_h425.sh PREREG_COMMIT LISTS_COMMIT      -> $EV/results/H425.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published the PREREG and code}
LC=${2:?commit on origin/v1/next that published the frozen lists}
EV=/home/ubuntu/jope_plan/evidence/H425
SCREEN=/home/ubuntu/jope-data/h335-cl/h425/pub/SCREEN.json
STAGE=$(cat "$EV/logs/pub_stage")
REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h425_contamination_screen
OUT=$EV/results/H425.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC"); stime=$(stat -c %Y "$SCREEN")
panc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
lanc=$(ok git -C "$REPO" merge-base --is-ancestor "$LC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H425.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
exl=$(ok cmp -s "$EV/results/EXCLUSION_DEV.json" <(git -C "$REPO" show "$LC:$WDIR/EXCLUSION_DEV.json"))
cfl=$(ok cmp -s "$EV/results/CONFIRM_FLAGS.json" <(git -C "$REPO" show "$LC:$WDIR/CONFIRM_FLAGS.json"))
nosens=$([ ! -e "$EV/results/H425_DEV.sensitivity.json" ] && echo true || echo false)
h424=$(ok bash -c "[ \"\$(sha256sum /home/ubuntu/jope_plan/evidence/H424/results/H424_DEV.apply.json | cut -d' ' -f1)\" = \"\$(jq -r .evidence.apply_sha256 /home/ubuntu/apex-stage/h424/code/metrics.json)\" ]")
jq -n --slurpfile s "$SCREEN" --slurpfile e "$EV/results/EXCLUSION_DEV.json" --slurpfile c "$EV/results/CONFIRM_FLAGS.json" \
  --argjson ptime "$ptime" --argjson stime "$stime" --argjson panc "$panc" --argjson lanc "$lanc" --argjson prereg "$prereg" \
  --argjson stage "$stage" --argjson exl "$exl" --argjson cfl "$cfl" --argjson nosens "$nosens" --argjson h424 "$h424" \
  --arg mode "$(cat "$STAGE/mode")" --arg ssha "$(sha256sum "$SCREEN" | cut -d' ' -f1)" --arg at "$(date -u +%FT%TZ)" '
  $s[0] as $s | $e[0] as $e | $c[0] as $c |
  {schema: "jope.h425.gates/v1", at: $at, screen_sha256: $ssha,
   checks: {queries: $s.queries, fit_icl_tables: $s.fit_icl_tables, summary: $s.summary, calibration: $s.fit_icl_calibration_flags,
            confirm_targets_keyed: $s.confirm_targets_keyed, guard_installed: $s.guard.installed, wall_s: $s.wall_s, procs: $s.procs,
            thresholds: $s.thresholds, prereg_commit_time: $ptime, screen_file_time: $stime,
            exclusion_contexts: ($e.contexts | length), confirm_contexts: ($c.contexts | length),
            lists_screen_sha256: [$e.screen_sha256, $c.screen_sha256], stage_mode: $mode},
   gates: {
     g1_prereg_before_screen: ($panc and $prereg and $stime > $ptime),
     g2_cohorts_and_sandbox: ($s.queries == ["DEV", "CONFIRM"] and $s.fit_icl_tables == 440 and $s.summary.DEV.contexts == 115
         and $s.summary.CONFIRM.contexts == 241 and $s.summary.CONFIRM.core == 142 and $s.guard.installed == true),
     g3_no_confirm_target: ($s.confirm_targets_keyed == false and ([$s.per_context[] | select(.cohort == "CONFIRM") | .b] | all(. == null))),
     g4_lists_are_the_screens: ($e.screen_sha256 == $ssha and $c.screen_sha256 == $ssha and ($e.contexts | length) == 115
         and ($c.contexts | length) == 241),
     g5_lists_published_before_sensitivity: ($lanc and $exl and $cfl and $nosens),
     g6_h424_values_unchanged: $h424,
     g7_budget: ($s.wall_s <= 7200 and $s.procs <= 8),
     g8_pub_stage: ($stage and $mode == "pub")}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"
jq '{gates, pass, summary: .checks.summary}' "$OUT"
