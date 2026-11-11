#!/bin/bash
# H425 integrity gates (PREREG section 7 as changed by addendum 1: the feature screen covers DEV only; CONFIRM gets the
# manifest-only screen). Run AFTER the screen and the publication of the frozen lists and BEFORE the sensitivity read.
# Create-once output.
#   gates_h425.sh PREREG_COMMIT ADDENDUM_COMMIT LISTS_COMMIT      -> $EV/results/H425.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published the PREREG}
AC=${2:?commit on origin/v1/next that published addendum 1 and the changed code}
LC=${3:?commit on origin/v1/next that published the frozen lists}
EV=/home/ubuntu/jope_plan/evidence/H425
SCREEN=/home/ubuntu/jope-data/h335-cl/h425/pub/SCREEN.json
STAGE=$(cat "$EV/logs/pub_stage")
REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h425_contamination_screen
OUT=$EV/results/H425.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC"); atime=$(git -C "$REPO" log -1 --format=%ct "$AC"); stime=$(stat -c %Y "$SCREEN")
anc=$(ok bash -c "git -C $REPO merge-base --is-ancestor $PC origin/v1/next && git -C $REPO merge-base --is-ancestor $AC origin/v1/next && git -C $REPO merge-base --is-ancestor $LC origin/v1/next")
prereg=$(ok bash -c "cmp -s $STAGE/code/PREREG_H425.md <(git -C $REPO show $PC:$WDIR/PREREG.md) && cmp -s $STAGE/code/PREREG_H425_ADDENDUM_1.md <(git -C $REPO show $AC:$WDIR/PREREG_ADDENDUM_1.md)")
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
exl=$(ok cmp -s "$EV/results/EXCLUSION_DEV.json" <(git -C "$REPO" show "$LC:$WDIR/EXCLUSION_DEV.json"))
cfl=$(ok cmp -s "$EV/results/CONFIRM_META.json" <(git -C "$REPO" show "$LC:$WDIR/CONFIRM_META.json"))
nosens=$([ ! -e "$EV/results/H425_DEV.sensitivity.json" ] && echo true || echo false)
h424=$(ok bash -c "[ \"\$(sha256sum /home/ubuntu/jope_plan/evidence/H424/results/H424_DEV.apply.json | cut -d' ' -f1)\" = \"\$(jq -r .evidence.apply_sha256 /home/ubuntu/apex-stage/h424/code/metrics.json)\" ]")
noconf=$(ok bash -c "[ \"\$(ls /home/ubuntu/jope-data/h335-cl/apex/tables | wc -l)\" = 661 ]")
jq -n --slurpfile s "$SCREEN" --slurpfile e "$EV/results/EXCLUSION_DEV.json" --slurpfile c "$EV/results/CONFIRM_META.json" \
  --argjson ptime "$ptime" --argjson atime "$atime" --argjson stime "$stime" --argjson anc "$anc" --argjson prereg "$prereg" \
  --argjson stage "$stage" --argjson exl "$exl" --argjson cfl "$cfl" --argjson nosens "$nosens" --argjson h424 "$h424" \
  --argjson noconf "$noconf" --arg mode "$(cat "$STAGE/mode")" --arg ssha "$(sha256sum "$SCREEN" | cut -d' ' -f1)" --arg at "$(date -u +%FT%TZ)" '
  $s[0] as $s | $e[0] as $e | $c[0] as $c |
  {schema: "jope.h425.gates/v2", at: $at, screen_sha256: $ssha,
   checks: {queries: $s.queries, fit_icl_tables: $s.fit_icl_tables, summary: $s.summary, calibration: $s.fit_icl_calibration_flags,
            guard_installed: $s.guard.installed, wall_s: $s.wall_s, procs: $s.procs, thresholds: $s.thresholds,
            prereg_commit_time: $ptime, addendum_commit_time: $atime, screen_file_time: $stime,
            exclusion_contexts: ($e.contexts | length), confirm_meta: ($c | del(.contexts)),
            sandbox_view_table_files_661_no_confirm: $noconf, stage_mode: $mode},
   gates: {
     g1_prereg_and_addendum_before_screen: ($anc and $prereg and $stime > $ptime and $stime > $atime),
     g2_dev_only_in_sandbox: ($s.queries == ["DEV"] and $s.fit_icl_tables == 440 and $s.summary.DEV.contexts == 115
         and $s.guard.installed == true and ([$s.per_context[] | .cohort] | unique) == ["DEV"]),
     g3_confirm_not_opened: ($noconf and ($c.feature_screens | startswith("not run")) and $c.n == 241 and $c.core == 142
         and $c.manifest_sha256 == $s.manifest_sha256),
     g4_list_is_the_screens: ($e.screen_sha256 == $ssha and ($e.contexts | length) == 115),
     g5_lists_published_before_sensitivity: ($exl and $cfl and $nosens),
     g6_h424_values_unchanged: $h424,
     g7_budget: ($s.wall_s <= 7200 and $s.procs <= 8),
     g8_pub_stage: ($stage and $mode == "pub")}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"
jq '{gates, pass, summary: .checks.summary, confirm_meta: .checks.confirm_meta.counts}' "$OUT"
