#!/bin/bash
# H424 integrity gates (PREREG section 6), from the step outputs of the published run. Written before any DEV table is
# opened by this wave; run AFTER the apply step and BEFORE the readout. Create-once output.
#   gates_h424.sh PREREG_COMMIT      -> $EV/results/H424_DEV.gates.json
set -euo pipefail
PC=${1:?commit on origin/v1/next that published the PREREG and code}
EV=/home/ubuntu/jope_plan/evidence/H424
R=/home/ubuntu/jope-data/h424/pub
BANK=/home/ubuntu/jope-data/h335-cl/h424/pub/bank_DEV.npz
STAGE=$(cat "$EV/logs/pub_stage")
REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h424_icl_stack_column
CKPT_SHA=$(cut -d' ' -f1 "$EV/logs/ckpt_h020.sha256")
OUT=$EV/results/H424_DEV.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
ok() { "$@" >/dev/null 2>&1 && echo true || echo false; }
inp=$(ok bash -c "cd /home/ubuntu/jope_plan/evidence/H420/pulled/H420_DEV && sha256sum --quiet -c ../H420_DEV.sha256")
ck=$(ok bash -c "[ \"\$(sha256sum /home/ubuntu/jope-data/h424/ICL_M_r3_ckpt_h020.pt | cut -d' ' -f1)\" = $CKPT_SHA ] && grep -q \"^$CKPT_SHA  xbabe1\" $EV/logs/ckpt_h020.source.sha256")
ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
anc=$(ok git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next)
btime=$(stat -c %Y "$BANK")
prereg=$(ok cmp -s "$STAGE/code/PREREG_H424.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md"))
stage=$(ok bash -c "cd $STAGE/code && sha256sum --quiet -c ../CODE.sha256")
jq -n --slurpfile b "$BANK.json" --slurpfile ap "$R/APPLY.json" --slurpfile t "$EV/logs/run_times.json" \
  --argjson inp "$inp" --argjson ck "$ck" --arg cks "$CKPT_SHA" --argjson ptime "$ptime" --argjson anc "$anc" \
  --argjson btime "$btime" --argjson prereg "$prereg" --argjson stage "$stage" --arg mode "$(cat "$STAGE/mode")" \
  --arg at "$(date -u +%FT%TZ)" --arg pc "$PC" '
  $b[0] as $b | $ap[0] as $ap | $t[0] as $t | $ap.cols_status as $st |
  {schema: "jope.h424.gates/v1", at: $at, prereg_commit: $pc,
   checks: {input_copy_matches_manifest: $inp, checkpoint_is_the_pinned_one: $ck, checkpoint_sha256: $cks,
            bank: ($b | del(.file)), apply: ($ap | del(.per_context, .cols_status)), run_times: $t,
            inference: {processes: ($st | length), threads_max: ([$st[].threads] | max), devices: ([$st[].device] | unique),
                        cuda_visible_any: ([$st[].cuda_visible] | any), ckpt: ([$st[].ckpt_sha256] | unique),
                        written: ([$st[].written] | add), budget_stops: ([$st[] | select(.budget_stop)] | length),
                        cpu_s: ([$st[].cpu_s] | add), wall_s_max: ([$st[].wall_s] | max)},
            prereg_commit_time: $ptime, prereg_commit_on_v1_next: $anc, bank_file_time: $btime,
            staged_prereg_equals_published: $prereg, stage_hash_ok: $stage, stage_mode: $mode},
   gates: {
     g1_inputs: ($inp and $ck),
     g2_dev_only_no_audit_target_in_bank: ($b.panel == "DEV" and $b.cohorts == ["DEV"] and $b.contexts == 115
         and $b.audit_targets_written == false and $b.guard.installed == true and $ap.cohort == "DEV" and $ap.n == 115
         and $ap.bank_sha256 == $b.sha256),
     g3_alignment_and_replay: ($ap.aligned_all == true and $ap.replay_max_w_diff == 0 and $ap.replay_max_u_diff <= 1e-9),
     g4_prereg_before_dev_tables: ($anc and $prereg and $btime > $ptime),
     g5_cpu_only_at_most_16_threads: (([$st[].device] | unique) == ["cpu"] and ([$st[].cuda_visible] | any | not)
         and ([$st[].threads] | add) <= 16 and ([$st[].ckpt_sha256] | unique) == [$cks] and $ap.ckpt_sha256 == [$cks]),
     g6_four_hour_cap: (($t.inference_end - $t.run_start) <= 14400 and ($t.apply_end - $t.run_start) <= 14400),
     g7_pub_stage: ($stage and $mode == "pub")}}
  | .pass = ([.gates[]] | all)' > "$OUT.part"
mv -n "$OUT.part" "$OUT"; chmod a-w "$OUT"
jq '{gates, pass, inference: .checks.inference, apply: .checks.apply}' "$OUT"
