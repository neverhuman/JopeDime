#!/bin/bash
# H445 post-apply integrity gates (PREREG section 4). Run AFTER driver_h445.sh apply and BEFORE readout. Create-once.
#   gates_h445.sh PREREG_COMMIT REVIEW_RECEIPT_PATH -> $EV/results/H445.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published PREREG and code}
REV=${2:?path to cl_fwriter review receipt}
EV=/home/ubuntu/jope_plan/evidence/H445
STAGE=$(cat "$EV/logs/pub_stage")
R=${WAVE_OUT:-/home/ubuntu/jope-data/h335-cl/h445/$(basename "$STAGE")}
CL_OUT=$(cat "$EV/logs/cl_cols_root")
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h445_joint_guarded_dev_stack
PRE_IN=$EV/results/H445.pre_apply.json
OUT=$EV/results/H445.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }

SHA_BASE=672c99372a969dda
SHA_DEV=dccf9beb3312f42e5a28c32aba33389c13f9c582add85084b0fc7d236f613c9b
SHA_SCR=fa8edd8a08a9aef30af98b8240f3e3b1f40d11e9681b157d7190cb00ff58891e
SHA_H424=ce0eb88749ebfb5c92aa6c10cd621563b1ab4a041756cc6e882b282be824faca
SHA_FRZ=c55ca3389ebf24e2ce4362db2ce9aa50a38f3215fb2cc4662486b6a75bd0f027

ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
sha() { sha256sum "$1" | cut -d' ' -f1; }

ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
pc_short=${PC:0:10}
rev_ok=$(ok bash -c "[ -f '$REV' ] && grep -Eq '$PC|$pc_short' '$REV' && grep -qi 'NO BLOCKING FINDING' '$REV'")
rev_time=$([ -f "$REV" ] && stat -c %Y "$REV" || echo 9999999999)
first_our=$(find "$R" -type f -printf '%T@\n' | cut -d. -f1 | sort -n | awk 'NR==1')
first_cl=$(find "$CL_OUT" -type f -printf '%T@\n' | cut -d. -f1 | sort -n | awk 'NR==1')
first=$(( first_our < first_cl ? first_our : first_cl ))

anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H445.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage_base=$(ok bash -c "[ \"\$(head -c16 '$STAGE/base_CODE.sha256')\" = '$SHA_BASE' ]")
stage_hash=$(ok bash -c "cd '$STAGE/code' && sha256sum --quiet -c ../CODE.sha256")
noread=$([ ! -e "$EV/results/READOUT_DEV.json" ] && echo true || echo false)
pre_ok=$(ok bash -c "[ -f '$PRE_IN' ] && [ \"\$(jq -r .pass '$PRE_IN')\" = true ]")
view=$(ok bash -c "[ \"\$(find /home/ubuntu/jope-data/h335-cl/apex/tables -maxdepth 1 -type f | wc -l)\" = 661 ]")
excl_ok=$(ok cmp -s "/home/ubuntu/jope_plan/evidence/H425/results/EXCLUSION_DEV.json" <(git -C "$REPO" show origin/v1/next:research/flywheel_selfplay/h425_contamination_screen/EXCLUSION_DEV.json))
h420_ok=$(ok bash -c "cd /home/ubuntu/jope_plan/evidence/H420/pulled/H420_DEV && sha256sum --quiet -c ../H420_DEV.sha256")
cl_man_ok=$(ok bash -c "cd '$CL_OUT' && sha256sum --quiet -c MANIFEST.sha256 && grep -Eq '^prereg_commit ($PC|$pc_short)' RUN_INFO.txt && grep -q 'c84af3af9fae4f32' RUN_INFO.txt && grep -q '436737c85ad4d398' RUN_INFO.txt")
bank_no_aud=$(ok bash -c "[ \"\$(jq -r .audit_targets_written /home/ubuntu/jope-data/h335-cl/h424/pub/bank_DEV.npz.json)\" = false ]")

jq -n \
  --slurpfile t "$EV/logs/run_times.json" \
  --slurpfile un "$R/selftest.json" \
  --slurpfile ap "$R/APPLY_DEV.json" \
  --argjson ptime "$ptime" --argjson rev_time "$rev_time" --argjson first "$first" \
  --argjson anc "$anc" --argjson prereg "$prereg" --argjson rev_ok "$rev_ok" \
  --argjson stage_base "$stage_base" --argjson stage_hash "$stage_hash" \
  --argjson noread "$noread" --argjson pre_ok "$pre_ok" --argjson view "$view" \
  --argjson excl_ok "$excl_ok" --argjson h420_ok "$h420_ok" \
  --argjson cl_man_ok "$cl_man_ok" --argjson bank_no_aud "$bank_no_aud" \
  --arg sdev "$SHA_DEV" --arg sscr "$(sha /home/ubuntu/jope_plan/evidence/H425/results/SCREEN_DEV.json)" \
  --arg escr "$SHA_SCR" --arg sh424 "$(sha /home/ubuntu/jope_plan/evidence/H424/results/H424_DEV.apply.json)" \
  --arg eh424 "$SHA_H424" --arg sfrz "$SHA_FRZ" \
  --arg mode "$(cat "$STAGE/mode")" --arg rev "$REV" --arg at "$(date -u +%FT%TZ)" '
  $t[0] as $t | $un[0] as $un | $ap[0] as $ap |
  {schema: "jope.h445.gates/v1", at: $at, review_receipt: $rev,
   checks: {
     prereg_commit_time: $ptime, review_time: $rev_time, first_output_time: $first, stage_mode: $mode,
     pre_apply_check_passed: $pre_ok,
     run_times: $t, n_contexts: $ap.n, promote_cw5: $ap.promote_cw5,
     matern_ok: $ap.matern_ok, h430_ok: $ap.h430_ok, h433_ok: $ap.h433_ok, icl_ok: $ap.icl_ok,
     icl_source_counts: $ap.icl_source_counts,
     replay_max_w_diff: $ap.replay_max_w_diff, replay_max_u_diff: $ap.replay_max_u_diff,
     h424_max_b_diff: $ap.h424_b_diff, h424_max_i_diff: $ap.h424_max_i_diff
   },
   gates: {
     g1_prereg_and_review_before_output: ($anc and $prereg and $rev_ok and $first > $ptime and $first >= $rev_time and $noread and $pre_ok),
     g2_pinned_stage_and_banks: ($stage_base and $stage_hash and $mode == "pub" and $ap.bank_sha256 == $sdev and $bank_no_aud and $sscr == $escr and $excl_ok and $h420_ok and $sh424 == $eh424 and $ap.frozen_cw5_sha256 == $sfrz),
     g3_two_hands_cl_manifest: $cl_man_ok,
     g4_all_115_aligned_and_exact_replays: ($ap.n == 115 and $ap.aligned_all == true and $ap.replay_max_w_diff == 0 and $ap.replay_max_u_diff <= 1e-9 and $ap.h424_max_b_diff == 0 and $ap.h424_max_i_diff <= 1e-9),
     g5_columns_complete: ($ap.matern_ok >= 109 and $ap.h430_ok >= 109 and $ap.h433_ok >= 109 and $ap.icl_ok >= 109),
     g6_cpu_budget: ($t.compute_wall_s <= 36000 and $t.procs <= 16),
     g7_selftest: ($un.all_passed == true),
     g8_confirm_untouched: $view
   }}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"
chmod a-w "$OUT"
jq '{gates, pass, checks}' "$OUT"
