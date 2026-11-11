#!/bin/bash
# H435 integrity gates (PREREG section 9). Run AFTER the per-table fits and BEFORE the readout. Create-once output.
#   gates_h435.sh PREREG_COMMIT      -> $EV/results/H435.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published the PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H435; R=/home/ubuntu/jope-data/h435/pub
STAGE=$(cat "$EV/logs/pub_stage"); REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h435_weight_screen_prospective
OUT=$EV/results/H435.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
first=$(stat -c %Y "$R"/tables/*.json | sort -n | head -n 1)
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H435.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
noread=$([ ! -e "$EV/results/H435_FITSEL.readout.json" ] && echo true || echo false)
view=$(ok bash -c "[ \"\$(ls /home/ubuntu/jope-data/h335-cl/apex/tables | wc -l)\" = 661 ]")
h430now=$(stat -c %Y /home/ubuntu/jope-data/h430/pub/cols_FITSEL /home/ubuntu/jope-data/h430/pub /home/ubuntu/jope_plan/evidence/H430/results | paste -sd,)
jq -n --slurpfile t "$EV/logs/run_times.json" --slurpfile st <(cat "$R"/tables/run_status_*.status) \
  --slurpfile nd <(cat "$R"/tables/*.json | jq -c '{aligned_h12, wide, ncols: (.columns | length), cols: (.columns | keys)}') \
  --argjson ptime "$ptime" --argjson first "$first" --argjson anc "$anc" --argjson prereg "$prereg" --argjson stage "$stage" \
  --argjson noread "$noread" --argjson view "$view" --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" \
  --arg scr "$(sha /home/ubuntu/jope-data/h335-cl/h425/dev/screen_FITSEL_2.json)" --arg val "$(sha /home/ubuntu/jope-data/apex-icl/bank/fitsel_search_val.npz)" \
  --arg oof "$(sha /home/ubuntu/jope-data/apex-icl/diag/mon/oof_M_r3_h012.npz)" --arg smoke "$(sha /home/ubuntu/jope-data/h424/dev/apply_smoke1.json)" \
  --arg h430 "$(sha /home/ubuntu/jope_plan/evidence/H430/results/APPLY_FITSEL.json)" --arg h430before "$(cat "$EV/logs/h430_mtimes_before.txt")" \
  --arg h430now "$h430now" --arg cpuset "$(cat "$EV/logs/cpu_set.txt")" '
  $t[0] as $t |
  {schema: "jope.h435.gates/v1", at: $at,
   checks: {tables: ($nd | length), wide: ([$nd[] | select(.wide)] | length), aligned_h12: ([$nd[] | select(.aligned_h12)] | length),
            columns_scored: ([$nd[].ncols] | add), with_h20: ([$nd[] | select(.cols | index("icl_h20_ridgeres"))] | length),
            with_h430: ([$nd[] | select(.cols | index("h430_gbm_plain"))] | length),
            processes: ($st | length), threads: ([$st[].threads] | add), torch_loaded_any: ([$st[].torch_loaded] | any),
            cpu_s: ([$st[].cpu_s] | add), run_times: $t, cpu_set: $cpuset, prereg_commit_time: $ptime, first_output_time: $first, stage_mode: $mode,
            inputs: {fitsel_screen: $scr, val_bank: $val, oof_h12: $oof, h424_smoke: $smoke, h430_apply: $h430}},
   gates: {
     g1_prereg_before_output: ($anc and $prereg and $first > $ptime and $noread),
     g2_pinned_inputs: ($scr == "d1c91046c567e5e314623f4d00ebf528d59550cec42e31d51fa606e6315ad24e"
         and $val == "8e517921b7d5bb46c223a12262e7e0c64531db3fad7dede601d002a09840bba0"
         and $oof == "199754684f58895c98a606e200db85753a6a1f4d86404dd214f21204cfd4117f"
         and $smoke == "8e916f30068747cb1fb7852651522c1915dc878ca56b25652c2e0c7461eb64ff"
         and $h430 == "155eb59d215736c0dde0877ba3aa95ada323c65600fc0092f43653b456ccb384"),
     g3_all_fitsel_tables: (($nd | length) == 106 and ([$nd[] | select(.wide)] | length) == 33),
     g4_cpu_only_4_threads_no_model: (([$st[].threads] | add) <= 4 and ([$st[].torch_loaded] | any | not)),
     g5_45_minute_cap: (($t.run_end - $t.run_start) <= 2700),
     g6_confirm_untouched: $view,
     g7_other_session_folders_unchanged: ($h430before == $h430now),
     g8_pub_stage: ($stage and $mode == "pub")}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"; jq '{gates, pass, checks: (.checks | {tables, wide, aligned_h12, columns_scored, with_h20, with_h430, processes, cpu_s, cpu_set})}' "$OUT"
