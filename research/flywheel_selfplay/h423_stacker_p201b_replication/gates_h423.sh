#!/bin/bash
# H423 integrity gates (PREREG section 6), from the step outputs of the published run. Written before any stacker
# value on P201b exists; run AFTER the apply step and BEFORE the readout. Create-once output.
#   gates_h423.sh PREREG_COMMIT      -> $EV/results/H423_P201B.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published the PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H423
R=/home/ubuntu/jope-data/h423/pub
STAGE=$(cat "$EV/logs/pub_stage")
REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h423_stacker_p201b_replication
W422=research/flywheel_selfplay/h422_learned_stacker; FROZEN_COMMIT=3b30a9051e7c412ca07dfb478a4f646f6958f109
OUT=$EV/results/H423_P201B.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
inp=$(ok bash -c "cd $EV/pulled/H416_CONFIRM && sha256sum --quiet -c ../H416_CONFIRM.sha256")
src=$(ok bash -c "grep -q '^source_equal true$' $EV/logs/pull_check.txt && grep -q '^filelist_equal true$' $EV/logs/pull_check.txt")
fpub=$(git -C "$REPO" show "$FROZEN_COMMIT:$W422/FROZEN.json" | sha256sum | cut -d' ' -f1)
floc=$(sha256sum "$R/FROZEN.json" | cut -d' ' -f1)
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
first=$(find "$R/P201B" -name '*.npz' -printf '%T@\n' | sort -n | head -n 1 | cut -d. -f1)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H423.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
board=$(ok cmp -s <(jq -cS . "$EV/logs/BOARD_p201b-train-audit-v1.json") <(git -C "$REPO" show origin/v1/next:research/elo/reference-board/boards/p201b-train-audit-v1/BOARD.json | jq -cS .))
jq -n --slurpfile c "$R/P201B/cache_P201B.json" --slurpfile ap "$R/APPLY.json" --slurpfile dj "$EV/results/dev91_disjoint.json" \
  --slurpfile t "$EV/results/tests.json" --argjson inp "$inp" --argjson src "$src" --arg fpub "$fpub" --arg floc "$floc" \
  --argjson ptime "$ptime" --argjson anc "$anc" --argjson first "$first" --argjson prereg "$prereg" --argjson stage "$stage" \
  --argjson board "$board" --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" --arg pc "$PC" '
  $c[0] as $c | $ap[0] as $ap | $dj[0] as $dj | $t[0] as $t | ($c.cpu_s + $ap.cpu_s) as $cpu |
  {schema: "jope.h423.gates/v1", at: $at, prereg_commit: $pc,
   checks: {input_copy_matches_manifest: $inp, input_copy_matched_source_at_pull: $src, cache: ($c | del(.run_dir)),
            apply: ($ap | del(.per_context)), dev91_disjoint: $dj, tests: $t, frozen_published_sha256: $fpub,
            frozen_used_sha256: $floc, prereg_commit_time: $ptime, prereg_commit_on_v1_next: $anc,
            first_cache_file_time: $first, staged_prereg_equals_published: $prereg, stage_hash_ok: $stage,
            stage_mode: $mode, board_copy_equals_published: $board, cpu_s_total: $cpu},
   gates: {
     g1_inputs: ($inp and $src),
     g2_p201b_only_and_no_closed_board_overlap: ($c.cohorts == ["DL-Train"] and $c.contexts == 201 and $c.written == 201
         and $ap.n == 201 and $board and $dj.disjoint == true and $ap.closed_overlap == 0
         and ($ap.closed_boards_checked | index("dev91-audit-v1") != null)),
     g3_replay: ($c.records_with_stored_weights == 201 and $c.max_w_diff_vs_record == 0 and $c.max_u_diff_vs_record <= 1e-9
         and $ap.replay_max_abs_diff <= 1e-9 and $ap.lam0_max_abs_diff <= 1e-6),
     g4_frozen_model_is_h422s: ($fpub == $floc and $ap.frozen_sha256 == $floc
         and $floc == "a43739b6469e2964da21186dcdbd272ea65e0e0a63e13f38f3b9cc25c7d69e00"),
     g5_prereg_before_p201b_values: ($anc and $prereg and $first > $ptime),
     g6_no_audit_target_in_predictions: ($ap.label_free == true),
     g7_pub_stage: ($stage and $mode == "pub"),
     g8_deploy_code_tests: ($t.ok == true and $t.failures == 0 and $t.errors == 0 and $t.tests == 12),
     g9_budget: ($cpu <= 21600)}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"
jq '{gates, pass, cpu_s_total: .checks.cpu_s_total, tests: .checks.tests, apply: .checks.apply}' "$OUT"
