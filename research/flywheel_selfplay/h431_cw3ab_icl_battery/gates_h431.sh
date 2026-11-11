#!/bin/bash
# H431 integrity gates (PREREG section 4). Run AFTER driver_h431.sh and BEFORE readout_h431.py. Create-once output.
#   gates_h431.sh PREREG_COMMIT -> $EV/results/H431.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H431
R=/home/ubuntu/jope-data/h335-cl/h431/pub
STAGE=$(cat "$EV/logs/pub_stage")
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h431_cw3ab_icl_battery
OUT=$EV/results/H431.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }

SHA_M20=e3c8fd9972ce0c1684630ff06b159e950f6f498c4461d0454e0c6a6f44d38dd7
SHA_M24=7fbca0e702bee00b4df0a58fac4062f550ceb71e9319fbbed5356a9b326ef176
SHA_L18=153117c5fd76dde38fc3d4d7de0997dd9f4bb92ba4eef42bf741ff7530124121
SHA_VAL=8e517921b7d5bb46c223a12262e7e0c64531db3fad7dede601d002a09840bba0
SHA_ICB=$(jq -r .sha256 "$R/ICB_v1.npz.json")

ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
first=$(find "$R" -type f -printf '%T@\n' | cut -d. -f1 | sort -n | head -n 1)
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H431.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
noread=$([ ! -e "$EV/results/H431_READOUT.json" ] && echo true || echo false)
view=$(ok bash -c "[ \"\$(ls /home/ubuntu/jope-data/h335-cl/apex/tables | wc -l)\" = 661 ]")

count_ok() { jq -s '[.[] | select(.ok == true)] | length' "$1"/*.json; }
b_m20=$(jq -s '[.[] | select(.task_id != null and .ok == true)] | length' "$R/bat_M_h020"/task_*.json)
b_m24=$(jq -s '[.[] | select(.task_id != null and .ok == true)] | length' "$R/bat_M_h024"/task_*.json)
b_l18=$(jq -s '[.[] | select(.task_id != null and .ok == true)] | length' "$R/bat_L_h018"/task_*.json)
m_m20=$(jq -s '[.[] | select(.cid != null and .ok == true)] | length' "$R/mon_M_h020"/ctx_*.json)
m_m24=$(jq -s '[.[] | select(.cid != null and .ok == true)] | length' "$R/mon_M_h024"/ctx_*.json)
m_l18=$(jq -s '[.[] | select(.cid != null and .ok == true)] | length' "$R/mon_L_h018"/ctx_*.json)

jq -n \
  --slurpfile t "$EV/logs/run_times.json" \
  --slurpfile ex "$R/EXCLUSION_FITSEL.json" \
  --slurpfile un "$R/unit_tests.json" \
  --slurpfile st <(cat "$R"/bat_*/status_*.json "$R"/mon_*/status_*.json) \
  --argjson ptime "$ptime" --argjson first "$first" \
  --argjson anc "$anc" --argjson prereg "$prereg" --argjson stage "$stage" \
  --argjson noread "$noread" --argjson view "$view" \
  --argjson bm20 "$b_m20" --argjson bm24 "$b_m24" --argjson bl18 "$b_l18" \
  --argjson mm20 "$m_m20" --argjson mm24 "$m_m24" --argjson ml18 "$m_l18" \
  --arg sm20 "$SHA_M20" --arg sm24 "$SHA_M24" --arg sl18 "$SHA_L18" \
  --arg sval "$SHA_VAL" --arg sicb "$SHA_ICB" \
  --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" '
  $t[0] as $t | $ex[0] as $ex | $un[0] as $un |
  {schema: "jope.h431.gates/v1", at: $at,
   checks: {
     prereg_commit_time: $ptime, first_output_time: $first, stage_mode: $mode,
     run_times: $t, fitsel_screen_n: $ex.n, fitsel_tiers: $ex.tier_counts,
     battery_ok: {M_h020: $bm20, M_h024: $bm24, L_h018: $bl18},
     monitor_ok: {M_h020: $mm20, M_h024: $mm24, L_h018: $ml18},
     status_files: ($st | length),
     max_threads_per_pass: ([$st | group_by(.mode + "_" + .ckpt_sha256)[] | ([.[].threads] | add)] | max),
     devices: ([$st[].device] | unique),
     cuda_visible_any: ([$st[].cuda_visible] | any),
     total_cpu_s: ([$st[].cpu_s] | add)
   },
   gates: {
     g1_prereg_before_output: ($anc and $prereg and $first > $ptime and $noread),
     g2_pinned_inputs: (
       (([$st[] | .ckpt_sha256] | unique | sort) == ([$sm20, $sm24, $sl18] | sort)) and
       (([$st[] | select(.mode == "battery") | .icb_sha256] | unique) == [$sicb]) and
       (([$st[] | select(.mode == "monitor") | .val_sha256] | unique) == [$sval])
     ),
     g3_cpu_only_16_threads: (
       ([$st[].device] | unique) == ["cpu"] and
       ([$st[].cuda_visible] | any | not) and
       (([$st | group_by(.mode + "_" + .ckpt_sha256)[] | ([.[].threads] | add)] | max) <= 16)
     ),
     g4_four_hour_cap: ($t.wall_s <= 14400),
     g5_battery_and_screen_complete: (
       $ex.n == 106 and $bm20 == 340 and $bm24 == 340 and $bl18 == 340 and
       $mm20 == 106 and $mm24 == 106 and $ml18 == 106
     ),
     g6_partb_equivalence: ($un.all_passed == true),
     g7_confirm_untouched: $view,
     g8_pub_stage: ($stage and $mode == "pub")
   }}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"
chmod a-w "$OUT"
jq '{gates, pass, checks}' "$OUT"
