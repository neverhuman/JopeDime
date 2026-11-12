#!/bin/bash
# H447 integrity gates (PREREG §3). Run AFTER driver_h447.sh apply and BEFORE eval_h447.py readout. Create-once.
#   gates_h447.sh PREREG_COMMIT -> $EV/results/H447.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H447
STAGE=$(cat "$EV/logs/pub_stage")
R=${WAVE_OUT:-/home/ubuntu/jope-data/h335-cl/h447/$(basename "$STAGE")}
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h447_legal_icl_104m_24m_r4e8
OUT=$EV/results/H447.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }

SHA_BASE=672c99372a969dda
SHA_BF=f8849ccb7343cfec1c0fd6688aaafef7580b85001046f2707fab58c5ecebe069
SHA_BP=da29e8ec64f623e0fa2ab29b31bbd40b91bef36adf8ddc8e2e773f51c96d01af
SHA_L=637bc123db18ca027fb46f5113a8f778f3aafb703fecb8c0edc91da7faeebffc
SHA_M=eb3a0903a7da136e2cdedd1fa3f1863fd351121d49f50a4b49e5b0e6d57e9ae4
SHA_M20=e3c8fd9972ce0c1684630ff06b159e950f6f498c4461d0454e0c6a6f44d38dd7

ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
first=$(find "$R" -type f -printf '%T@\n' | cut -d. -f1 | sort -n | awk 'NR==1')
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H447.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage_base=$(ok bash -c "[ \"\$(head -c16 '$STAGE/base_CODE.sha256')\" = '$SHA_BASE' ]")
stage_hash=$(ok bash -c "cd '$STAGE/code' && sha256sum --quiet -c ../CODE.sha256")
noread=$([ ! -e "$EV/results/READOUT_H447.json" ] && [ ! -e "$EV/results/FROZEN_FAIR_CHAMPION.json" ] && echo true || echo false)
view=$(ok bash -c "[ \"\$(find /home/ubuntu/jope-data/h335-cl/apex/tables -maxdepth 1 -type f | wc -l)\" = 661 ]")
no_d91_lock=$([ ! -e "/home/ubuntu/jope_plan/ns/receipts/H448_LOCK_sha256.txt" ] && echo true || echo false)
ckpts_ok=$(ok bash -c "[ \"\$(sha256sum /home/ubuntu/jope-data/h447/inputs/ICL_L_g1_w2_ckpt_h030.pt | cut -d' ' -f1)\" = '$SHA_L' ] && [ \"\$(sha256sum /home/ubuntu/jope-data/h447/inputs/ICL_M_r3_ckpt_h040.pt | cut -d' ' -f1)\" = '$SHA_M' ] && [ \"\$(sha256sum /home/ubuntu/jope-data/h424/ICL_M_r3_ckpt_h020.pt | cut -d' ' -f1)\" = '$SHA_M20' ]")

jq -n \
  --slurpfile t "$EV/logs/run_times.json" \
  --slurpfile st "$R/selftests.json" \
  --slurpfile af "$R/APPLY_FITSEL.json" \
  --slurpfile ap "$R/APPLY_P201B.json" \
  --argjson ptime "$ptime" --argjson first "$first" \
  --argjson anc "$anc" --argjson prereg "$prereg" \
  --argjson stage_base "$stage_base" --argjson stage_hash "$stage_hash" \
  --argjson noread "$noread" --argjson view "$view" \
  --argjson no_d91_lock "$no_d91_lock" --argjson ckpts_ok "$ckpts_ok" \
  --arg sbf "$SHA_BF" --arg sbp "$SHA_BP" \
  --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" '
  $t[0] as $t | $st[0] as $st | $af[0] as $af | $ap[0] as $ap |
  {schema: "jope.h447.gates/v1", at: $at,
   checks: {
     prereg_commit_time: $ptime, first_output_time: $first, stage_mode: $mode,
     run_times: $t,
     fitsel: {n: $af.n, r4_e8_base_ok: $af.r4_e8_base_ok, m20_ok: $af.m20_ok, replay_max_w_diff: $af.replay_max_w_diff, replay_max_u_diff: $af.replay_max_u_diff},
     p201b: {n: $ap.n, r4_e8_base_ok: $ap.r4_e8_base_ok, m20_ok: $ap.m20_ok, replay_max_w_diff: $ap.replay_max_w_diff, replay_max_u_diff: $ap.replay_max_u_diff}
   },
   gates: {
     g1_prereg_before_output: ($anc and $prereg and $first > $ptime and $noread),
     g2_pinned_stage_and_inputs: ($stage_base and $stage_hash and $mode == "pub" and $ckpts_ok and $af.bank_sha256 == $sbf and $ap.bank_sha256 == $sbp),
     g3_selftests: ($st.all_passed == true),
     g4_exact_replays_and_completeness: ($af.n == 106 and $af.aligned_all == true and $af.replay_max_w_diff == 0 and $af.replay_max_u_diff <= 1e-9 and $af.r4_e8_base_ok >= 101 and $af.m20_ok >= 101 and $ap.n == 201 and $ap.aligned_all == true and $ap.replay_max_w_diff == 0 and $ap.replay_max_u_diff <= 1e-9 and $ap.r4_e8_base_ok >= 191 and $ap.m20_ok >= 191),
     g5_zero_holdout_opens: ($view and $no_d91_lock),
     g6_pre_readout_ready: $noread
   }}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"
chmod a-w "$OUT"
jq '{gates, pass, checks}' "$OUT"
