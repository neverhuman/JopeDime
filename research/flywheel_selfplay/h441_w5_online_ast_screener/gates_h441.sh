#!/bin/bash
# H441 integrity gates (PREREG section 3). Run AFTER driver_h441.sh and BEFORE readout. Create-once.
#   gates_h441.sh PREREG_COMMIT -> $EV/results/H441.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H441
STAGE=$(cat "$EV/logs/pub_stage")
R=${WAVE_OUT:-/home/ubuntu/jope-data/h335-cl/h441/$(basename "$STAGE")}
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h441_w5_online_ast_screener
OUT=$EV/results/H441.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }

SHA_BF=f8849ccb7343cfec1c0fd6688aaafef7580b85001046f2707fab58c5ecebe069
SHA_EXCL=be87f957c1ab0089b389e5bee5ae2bea5ad141d40aa9568973fd3c611f4154da

ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
first=$(find "$R" -type f -printf '%T@\n' | cut -d. -f1 | sort -n | awk 'NR==1')
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H441.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
noread=$([ ! -e "$EV/results/READOUT_W5.json" ] && [ ! -e "$R/READOUT_W5.json" ] && echo true || echo false)
view=$(ok bash -c "[ \"\$(ls /home/ubuntu/jope-data/h335-cl/apex/tables | wc -l)\" = 661 ]")

jq -n \
  --slurpfile t "$EV/logs/run_times.json" \
  --slurpfile un "$R/unit_tests.json" \
  --slurpfile hm "$R/h01a_scores_FITSEL.meta.json" \
  --slurpfile ap "$R/APPLY_W5.json" \
  --argjson ptime "$ptime" --argjson first "$first" \
  --argjson anc "$anc" --argjson prereg "$prereg" --argjson stage "$stage" \
  --argjson noread "$noread" --argjson view "$view" \
  --arg sbf "$SHA_BF" --arg sexcl "$SHA_EXCL" \
  --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" '
  $t[0] as $t | $un[0] as $un | $hm[0] as $hm | $ap[0] as $ap |
  {schema: "jope.h441.gates/v1", at: $at,
   checks: {
     prereg_commit_time: $ptime, first_output_time: $first, stage_mode: $mode,
     run_times: $t, n_total: $ap.n_total, n_clean: $ap.n_clean,
     h01a_candidates: $hm.total_candidates, cuda_visible: $hm.cuda_visible
   },
   gates: {
     g1_prereg_before_output: ($anc and $prereg and $first > $ptime and $noread),
     g2_pinned_inputs: ($ap.bank_sha256 == $sbf and $ap.excl_sha256 == $sexcl and $ap.n_total == 106 and $ap.n_clean == 88),
     g3_cpu_only: ($hm.cuda_visible == "" and $t.wall_s <= 3600),
     g4_unit_tests: ($un.all_passed == true),
     g5_confirm_untouched: $view,
     g6_pub_stage: ($stage and $mode == "pub")
   }}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"
chmod a-w "$OUT"
jq '{gates, pass, checks}' "$OUT"
