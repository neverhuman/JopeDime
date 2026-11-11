#!/bin/bash
# H422 integrity gates (PREREG section 6), from the step outputs of the published run. Written before any DEV matrix
# is opened; run AFTER the apply step and BEFORE the readout. Create-once output.
#   gates_h422.sh FROZEN_COMMIT      -> $EV/results/H422_DEV.gates.json
set -euo pipefail
FC=${1:?commit on origin/v1/next that published FROZEN.json}
EV=/home/ubuntu/jope_plan/evidence/H422
R=/home/ubuntu/jope-data/h422/pub
STAGE=$(cat "$EV/logs/pub_stage")
REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h422_learned_stacker
E420=/home/ubuntu/jope_plan/evidence/H420/pulled; E417=/home/ubuntu/jope_plan/evidence/H417/pulled
OUT=$EV/results/H422_DEV.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
inputs=$(jq -n \
  --argjson dev "$(ok bash -c "cd $E420/H420_DEV && sha256sum --quiet -c ../H420_DEV.sha256")" \
  --argjson fit "$(ok bash -c "cd $E420/H420_FIT-SEL && sha256sum --quiet -c ../H420_FIT-SEL.sha256")" \
  --argjson base "$(ok bash -c "cd $E417/APEX_BASE && grep -E '  \./APEX_BASE/(config_BASE.json|stages/H416_BASE/|h416_BASE_npz/)' ../APEX_BASE.sha256 | sha256sum --quiet -c -")" \
  --argjson p201e "$(ok bash -c "cd /home/ubuntu/jope_plan/evidence/H421/pulled && sha256sum --quiet -c $EV/logs/inputs_P201E.sha256")" \
  '{h420_dev: $dev, h420_fitsel: $fit, h417_base: $base, p201e: $p201e}')
frozen_pub=$(git -C "$REPO" show "$FC:$WDIR/FROZEN.json" | sha256sum | cut -d' ' -f1)
frozen_loc=$(sha256sum "$R/fit/FROZEN.json" | cut -d' ' -f1)
ftime=$(git -C "$REPO" log -1 --format=%ct "$FC")
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$FC" origin/v1/next)
dev_first=$(find "$R/DEV_TBOTH" -name '*.npz' -printf '%T@\n' | sort -n | head -n 1 | cut -d. -f1)
prereg=$(ok cmp -s "$STAGE/code/PREREG_H422.md" <(git -C "$REPO" show "origin/v1/next:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
jq -n --slurpfile p "$R/P201E/cache_P201E.json" --slurpfile f "$R/FITSEL_TBOTH/cache_TBOTH.json" \
  --slurpfile d "$R/DEV_TBOTH/cache_TBOTH.json" --slurpfile cv "$R/fit/CROSSFIT.json" --slurpfile ap "$R/APPLY.json" \
  --argjson inputs "$inputs" --arg fpub "$frozen_pub" --arg floc "$frozen_loc" --argjson ftime "$ftime" \
  --argjson anc "$anc" --argjson devfirst "$dev_first" --argjson prereg "$prereg" --argjson stage "$stage" \
  --arg mode "$(cat "$STAGE/mode")" --arg at "$(date -u +%FT%TZ)" --arg fc "$FC" '
  $p[0] as $p | $f[0] as $f | $d[0] as $d | $cv[0] as $cv | $ap[0] as $ap |
  ($p.cpu_s + $f.cpu_s + $d.cpu_s + $cv.cpu_s + $ap.cpu_s) as $cpu |
  {schema: "jope.h422.gates/v1", at: $at, frozen_commit: $fc,
   checks: {inputs: $inputs, design: {p201e: ($p | del(.run_dir)), fitsel: ($f | del(.run_dir))}, dev: ($d | del(.run_dir)),
            crossfit: {n: $cv.n, gate: $cv.gate, lam0_max_abs_diff: $cv.lam0_max_abs_diff},
            apply: ($ap | del(.per_context)), frozen_published_sha256: $fpub, frozen_local_sha256: $floc,
            frozen_commit_time: $ftime, frozen_commit_on_v1_next: $anc, first_dev_cache_file_time: $devfirst,
            staged_prereg_equals_published: $prereg, stage_hash_ok: $stage, stage_mode: $mode, cpu_s_total: $cpu},
   gates: {
     g1_inputs_match_manifests: ([$inputs[]] | all),
     g2_design_only: ($p.cohorts == ["DL-Train"] and $f.cohorts == ["FIT-SEL"] and $p.contexts == 185 and $f.contexts == 106
         and $p.written == 185 and $f.written == 106 and $cv.n == 291
         and $f.records_with_stored_weights == 106 and $f.max_w_diff_vs_record == 0 and $f.max_u_diff_vs_record <= 1e-9),
     g3_train_gate_passed: ($cv.gate.pass == true and $cv.lam0_max_abs_diff <= 1e-6),
     g4_frozen_before_dev: ($fpub == $floc and $ap.frozen_sha256 == $floc and $anc and $devfirst > $ftime),
     g5_dev_replay: ($d.cohorts == ["DEV"] and $d.contexts == 115 and $d.written == 115 and $ap.n == 115
         and $d.records_with_stored_weights == 115 and $d.max_w_diff_vs_record == 0 and $d.max_u_diff_vs_record <= 1e-9
         and $ap.replay_max_abs_diff <= 1e-9 and $ap.lam0_max_abs_diff <= 1e-6),
     g6_no_audit_target_in_predictions: ($ap.label_free == true),
     g7_pub_stage: ($prereg and $stage and $mode == "pub"),
     g8_budget: ($cpu <= 21600)}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"
jq '{gates, pass, cpu_s_total: .checks.cpu_s_total, crossfit_gate: .checks.crossfit.gate, apply: .checks.apply}' "$OUT"
