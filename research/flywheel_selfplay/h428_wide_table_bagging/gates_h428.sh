#!/bin/bash
# H428 integrity gates (PREREG section 6). Run AFTER inference and BEFORE the readout. Create-once output.
#   gates_h428.sh PREREG_COMMIT      -> $EV/results/H428.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published the PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H428; R=/home/ubuntu/jope-data/h428/pub/bag
STAGE=$(cat "$EV/logs/pub_stage"); REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h428_wide_table_bagging
CK=/home/ubuntu/jope-data/h424/ICL_M_r3_ckpt_h020.pt; CKSHA=e3c8fd9972ce0c1684630ff06b159e950f6f498c4461d0454e0c6a6f44d38dd7
BANKSHA=8e517921b7d5bb46c223a12262e7e0c64531db3fad7dede601d002a09840bba0
OUT=$EV/results/H428.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
first=$(find "$R" -name '*.json' ! -name 'bag_status_*' -printf '%T@\n' | sort -n | head -n 1 | cut -d. -f1)
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H428.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
ck=$(ok bash -c "[ \"\$(sha256sum $CK | cut -d' ' -f1)\" = $CKSHA ]")
noread=$([ ! -e "$EV/results/H428_FITSEL.readout.json" ] && echo true || echo false)
view=$(ok bash -c "[ \"\$(ls /home/ubuntu/jope-data/h335-cl/apex/tables | wc -l)\" = 661 ]")
jq -n --slurpfile t "$EV/logs/run_times.json" --argjson ptime "$ptime" --argjson first "$first" --argjson anc "$anc" \
  --argjson prereg "$prereg" --argjson stage "$stage" --argjson ck "$ck" --argjson noread "$noread" --argjson view "$view" \
  --arg cks "$CKSHA" --arg bs "$BANKSHA" --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" \
  --argjson files "$(find "$R" -name '*.json' ! -name 'bag_status_*' | wc -l)" \
  --slurpfile st <(cat "$R"/bag_status_*.json) '
  $t[0] as $t |
  {schema: "jope.h428.gates/v1", at: $at,
   checks: {context_files: $files, processes: ($st | length), threads: ([$st[].threads] | add), devices: ([$st[].device] | unique),
            cuda_visible_any: ([$st[].cuda_visible] | any), bank_sha256: ([$st[].bank_sha256] | unique), ckpt_sha256: ([$st[].ckpt_sha256] | unique),
            wide_contexts: ([$st[].wide_contexts] | unique), written: ([$st[].written] | add), budget_stops: ([$st[] | select(.budget_stop)] | length),
            cpu_s: ([$st[].cpu_s] | add), run_times: $t, prereg_commit_time: $ptime, first_output_time: $first, stage_mode: $mode},
   gates: {
     g1_prereg_before_output: ($anc and $prereg and $first > $ptime and $noread),
     g2_pinned_checkpoint_and_monitor_bank: ($ck and ([$st[].ckpt_sha256] | unique) == [$cks] and ([$st[].bank_sha256] | unique) == [$bs]),
     g3_fit_sel_wide_contexts: (([$st[].wide_contexts] | unique) == [33] and $files <= 33),
     g4_cpu_only_16_threads: (([$st[].device] | unique) == ["cpu"] and ([$st[].cuda_visible] | any | not) and ([$st[].threads] | add) <= 16),
     g5_four_hour_cap: (($t.end - $t.start) <= 14400),
     g6_confirm_untouched: $view,
     g7_pub_stage: ($stage and $mode == "pub")}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"; jq '{gates, pass, checks: (.checks | {context_files, written, budget_stops, cpu_s})}' "$OUT"
