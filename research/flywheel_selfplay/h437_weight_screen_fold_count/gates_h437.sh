#!/bin/bash
# H437 integrity gates (PREREG section 7). Run AFTER the per-fold sums and BEFORE the readout. Create-once output.
#   gates_h437.sh PREREG_COMMIT      -> $EV/results/H437.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published the PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H437
STAGE=$(cat "$EV/logs/pub_stage"); REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h437_weight_screen_fold_count
STATS=$EV/results/H437_FITSEL.fold_sums.json
OUT=$EV/results/H437.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
first=$(stat -c %Y "$STATS")
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H437.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
noread=$([ ! -e "$EV/results/H437_FITSEL.readout.json" ] && echo true || echo false)
view=$(ok bash -c "[ \"\$(ls /home/ubuntu/jope-data/h335-cl/apex/tables | wc -l)\" = 661 ]")
jq -n --slurpfile t "$EV/logs/run_times.json" --slurpfile st <(jq -c 'del(.rows)' "$STATS") \
  --argjson ptime "$ptime" --argjson first "$first" --argjson anc "$anc" --argjson prereg "$prereg" --argjson stage "$stage" \
  --argjson noread "$noread" --argjson view "$view" --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" \
  --arg scr "$(sha /home/ubuntu/jope-data/h335-cl/h425/dev/screen_FITSEL_2.json)" --arg val "$(sha /home/ubuntu/jope-data/apex-icl/bank/fitsel_search_val.npz)" \
  --arg oof "$(sha /home/ubuntu/jope-data/apex-icl/diag/mon/oof_M_r3_h012.npz)" \
  --arg h435t "$(sha /home/ubuntu/jope_plan/evidence/H435/results/H435_FITSEL.per_table.jsonl)" \
  --arg h435r "$(sha /home/ubuntu/jope_plan/evidence/H435/results/H435_FITSEL.readout.json)" '
  $t[0] as $t | $st[0] as $st |
  {schema: "jope.h437.gates/v1", at: $at,
   checks: {stats: $st, run_times: $t, prereg_commit_time: $ptime, stats_file_time: $first, stage_mode: $mode,
            inputs: {fitsel_screen: $scr, val_bank: $val, oof_h12: $oof, h435_per_table: $h435t, h435_readout: $h435r}},
   gates: {
     g1_prereg_before_output: ($anc and $prereg and $first > $ptime and $noread),
     g2_pinned_inputs: ($scr == "d1c91046c567e5e314623f4d00ebf528d59550cec42e31d51fa606e6315ad24e"
         and $val == "8e517921b7d5bb46c223a12262e7e0c64531db3fad7dede601d002a09840bba0"
         and $oof == "199754684f58895c98a606e200db85753a6a1f4d86404dd214f21204cfd4117f"
         and $h435t == "024cdf09aa0842fa96d05e903b936dba9c5e2b31863acb2c4e492ed62e2f8564"
         and $h435r == "dcbe898f6911f27447902dc84a069be909a24f0f841b986ffa509d26759976cd"),
     g3_all_tables_and_replay: ($st.tables == 106 and $st.replay_max_abs_diff <= 1e-9),
     g4_one_thread_no_model_no_weight_builder: ($st.threads == 1 and ($st.torch_loaded | not) and ($st.weight_builder_loaded | not)),
     g5_30_minute_cap: (($t.run_end - $t.run_start) <= 1800),
     g6_confirm_untouched: $view,
     g7_pub_stage: ($stage and $mode == "pub")}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"; jq '{gates, pass, checks: (.checks.stats | {tables, replay_max_abs_diff, cpu_s})}' "$OUT"
jq -e '.pass' "$OUT" > /dev/null
