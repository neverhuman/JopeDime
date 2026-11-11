#!/bin/bash
# H432 integrity gates (PREREG section 6). Run AFTER the weight study and BEFORE the two readouts. Create-once output.
#   gates_h432.sh PREREG_COMMIT      -> $EV/results/H432.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published the PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H432; R=/home/ubuntu/jope-data/h432/pub
STAGE=$(cat "$EV/logs/pub_stage"); REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h432_narrow_only_column
E4=/home/ubuntu/jope_plan/evidence/H424/results
BANK=/home/ubuntu/jope-data/h335-cl/h424/pub/bank_DEV.npz
OUT=$EV/results/H432.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
first=$(stat -c %Y "$R"/need/*.json | sort -n | head -n 1)
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H432.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
noread=$([ ! -e "$EV/results/H432_DEV.narrow.json" ] && [ ! -e "$EV/results/H432_DEV.need.json" ] && echo true || echo false)
view=$(ok bash -c "[ \"\$(ls /home/ubuntu/jope-data/h335-cl/apex/tables | wc -l)\" = 661 ]")
jq -n --slurpfile t "$EV/logs/run_times.json" --slurpfile bj "$BANK.json" --slurpfile st <(cat "$R"/need/run_status_*.status) \
  --slurpfile nd <(cat "$R"/need/*.json | jq -c '{replay_abs_diff, wide, cpu_s}') \
  --argjson ptime "$ptime" --argjson first "$first" --argjson anc "$anc" --argjson prereg "$prereg" --argjson stage "$stage" \
  --argjson noread "$noread" --argjson view "$view" --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" \
  --arg a20 "$(sha "$E4/H424_DEV.apply.json")" --arg a24 "$(sha "$E4/H424_DEV_h024.apply.json")" \
  --arg scr "$(sha /home/ubuntu/jope_plan/evidence/H425/results/SCREEN_DEV.json)" --arg cpuset "$(cat "$EV/logs/cpu_set.txt")" '
  $t[0] as $t |
  {schema: "jope.h432.gates/v1", at: $at,
   checks: {contexts: ($nd | length), wide: ([$nd[] | select(.wide)] | length), replay_max_abs_diff: ([$nd[].replay_abs_diff] | max),
            processes: ($st | length), threads: ([$st[].threads] | add), torch_loaded_any: ([$st[].torch_loaded] | any),
            cpu_s: ([$st[].cpu_s] | add), run_times: $t, cpu_set: $cpuset, prereg_commit_time: $ptime, first_output_time: $first,
            stage_mode: $mode, inputs: {apply: $a20, apply_h024: $a24, screen: $scr, bank: $bj[0].sha256}},
   gates: {
     g1_prereg_before_output: ($anc and $prereg and $first > $ptime and $noread),
     g2_pinned_inputs: ($a20 == "ce0eb88749ebfb5c92aa6c10cd621563b1ab4a041756cc6e882b282be824faca"
         and $a24 == "aacbcd6894de3dda9e248f8c4cc47020e204e93656e690ddf886beb4499808f7"
         and $scr == "fa8edd8a08a9aef30af98b8240f3e3b1f40d11e9681b157d7190cb00ff58891e"
         and $bj[0].sha256 == "dccf9beb3312f42e5a28c32aba33389c13f9c582add85084b0fc7d236f613c9b"),
     g3_all_contexts_and_replay: (($nd | length) == 115 and ([$nd[] | select(.wide)] | length) == 35 and ([$nd[].replay_abs_diff] | max) == 0),
     g4_cpu_only_8_threads_no_model: (([$st[].threads] | add) <= 8 and ([$st[].torch_loaded] | any | not)),
     g5_one_hour_cap: (($t.run_end - $t.run_start) <= 3600),
     g6_confirm_untouched: $view,
     g7_pub_stage: ($stage and $mode == "pub")}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"; jq '{gates, pass, checks: (.checks | {contexts, wide, replay_max_abs_diff, processes, cpu_s, cpu_set})}' "$OUT"
