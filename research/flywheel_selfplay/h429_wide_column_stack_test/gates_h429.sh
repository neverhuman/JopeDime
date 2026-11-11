#!/bin/bash
# H429 integrity gates (PREREG section 6). Run AFTER the stack step and BEFORE the readout. Create-once output.
#   gates_h429.sh PREREG_COMMIT      -> $EV/results/H429.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published the PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H429; R=/home/ubuntu/jope-data/h429/pub; H4=/home/ubuntu/jope-data/h424/pub
STAGE=$(cat "$EV/logs/pub_stage"); REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h429_wide_column_stack_test
CKSHA=e3c8fd9972ce0c1684630ff06b159e950f6f498c4461d0454e0c6a6f44d38dd7
BANK=/home/ubuntu/jope-data/h335-cl/h424/pub/bank_DEV.npz
OUT=$EV/results/H429.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
first=$(for c in $(cat "$EV/logs/wide_cids.txt"); do stat -c %Y "$R/cols_T1/$c.npz"; done | sort -n | head -n 1)
h24=$(stat -c %Y /home/ubuntu/jope_plan/evidence/H424/results/H424_DEV_h024.readout2.json 2>/dev/null || echo 0)
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H429.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
noread=$([ ! -e "$EV/results/H429_DEV.readout.json" ] && echo true || echo false)
view=$(ok bash -c "[ \"\$(ls /home/ubuntu/jope-data/h335-cl/apex/tables | wc -l)\" = 661 ]")
same=0; for c in $(comm -23 <(jq -r '.per_context | keys[]' "$R/APPLY.json" | sort) <(sort "$EV/logs/wide_cids.txt")); do cmp -s "$R/cols_T1/$c.npz" "$H4/cols_DEV/$c.npz" && same=$((same + 1)); done
jq -n --slurpfile t "$EV/logs/run_times.json" --slurpfile ap "$R/APPLY.json" --slurpfile fb "$EV/logs/fallback.json" --slurpfile bj "$BANK.json" \
  --slurpfile st <(cat "$R"/cols_T1/cols_status_*.json) --argjson ptime "$ptime" --argjson first "$first" --argjson h24 "$h24" \
  --argjson anc "$anc" --argjson prereg "$prereg" --argjson stage "$stage" --argjson noread "$noread" --argjson view "$view" \
  --argjson same "$same" --argjson wide "$(wc -l < "$EV/logs/wide_cids.txt")" --arg cks "$CKSHA" --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" '
  $t[0] as $t | $ap[0] as $ap | ($fb[0].fallback | length) as $nfb |
  {schema: "jope.h429.gates/v1", at: $at,
   checks: {wide: $wide, written: ([$st[].written] | add), fallback: $nfb, narrow_files_identical_to_h424: $same,
            processes: ($st | length), threads: ([$st[].threads] | add), devices: ([$st[].device] | unique),
            cuda_visible_any: ([$st[].cuda_visible] | any), cpu_s: ([$st[].cpu_s] | add), budget_stops: ([$st[] | select(.budget_stop)] | length),
            apply: ($ap | del(.per_context, .cols_status)), run_times: $t, prereg_commit_time: $ptime, first_column_time: $first,
            h424_second_readout_time: $h24, stage_mode: $mode},
   gates: {
     g1_prereg_before_output: ($anc and $prereg and $first > $ptime and $noread),
     g2_after_h424_second_readout: ($h24 > 0 and $first > $h24),
     g3_pinned_checkpoint_and_bank: (([$st[].ckpt_sha256] | unique) == [$cks] and $ap.ckpt_sha256 == [$cks]
         and ([$st[].bank_sha256] | unique) == [$bj[0].sha256] and $ap.bank_sha256 == $bj[0].sha256),
     g4_cpu_only_16_threads: (([$st[].device] | unique) == ["cpu"] and ([$st[].cuda_visible] | any | not) and ([$st[].threads] | add) <= 16),
     g5_four_hour_cap: (($t.apply_end - $t.run_start) <= 14400),
     g6_stack_step: ($ap.cohort == "DEV" and $ap.n == 115 and $ap.aligned_all == true and $ap.columns_ok == 115
         and $ap.replay_max_w_diff == 0 and $ap.replay_max_u_diff <= 1e-9),
     g7_wide_rebuilt_narrow_untouched: ($wide == 35 and (([$st[].written] | add) + $nfb) == 35 and $same == 80),
     g8_confirm_untouched: $view,
     g9_pub_stage: ($stage and $mode == "pub")}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"; jq '{gates, pass, checks: (.checks | {written, fallback, narrow_files_identical_to_h424, cpu_s})}' "$OUT"
