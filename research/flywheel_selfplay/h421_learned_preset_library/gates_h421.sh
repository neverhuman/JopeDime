#!/bin/bash
# H421 integrity gates (PREREG section 7, addenda 1-2), from the pulled, sha-compared copies of the DEV run directory
# and of the fill run, the driver log and the published files. Written before the DEV deploys ended; run BEFORE the
# readout. Create-once output.
#   gates_h421.sh            -> $EV/results/H421_DEV.gates.json
set -euo pipefail
EV=/home/ubuntu/jope_plan/evidence/H421
P=$EV/pulled/H421_DEV/H420_DEV
F=$EV/pulled/H421_FILL/H421_FILL
STAGE=/home/ubuntu/apex-stage/h421/pub-q2
REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h421_learned_preset_library
SEL_COMMIT=c8a3726472fbfef99449c87b289ae3b05ad520e8      # selection + preset files on origin/v1/next
OUT=$EV/results/H421_DEV.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
(cd "$EV/pulled/H421_DEV" && sha256sum --quiet -c ../H421_DEV.sha256)
(cd "$EV/pulled/H421_FILL" && sha256sum --quiet -c ../H421_FILL.sha256)
T=$(mktemp -d "${TMPDIR:-/tmp}/h421_gates.XXXXXX")   # scratch lists; left in place
last() { tail -n 1 "$1"; }
declare -A FILE=( [TPORT]=LIB_PORT [TRAND]=LIB_RAND [TCOV]=LIB_COV )
per=$(for v in TPORT TRAND TCOV; do
  jq -r '.cids[]' "$P/config_$v.json" | sort > "$T/cids"
  ls "$P/stages/H416_$v" | sed 's/__.*//' | sort > "$T/r16"; ls "$P/stages/H415_$v" | sed 's/__.*//' | sort > "$T/r15"
  recs=$(for c in $(cat "$T/cids"); do last "$P/stages/H416_$v/${c}__confirm.jsonl"; done | jq -s '
    {results: (map(select(.type == "result")) | length), errors: (map(select(.error != null)) | length),
     sxb_missing_or_invalid: (map(select((.outputs["P3+LIB3.SXB"] == null) or (.outputs["P3+LIB3.SXB"].invalid == true))) | length),
     recomputed_units: (map(select(.recomputed_as != null)) | length)}')
  st=$(jq -s '{status_files: length, units: (map(.units) | add),
               guard_installed_all: (map(.guard.installed) | all),
               guard_refusals_in_process: (map([.guard | to_entries[] | select(.key | startswith("refused")) | .value] | add) | add),
               same_family_bank_hits: (map(.bank_counters.same_family_bank_hits) | add),
               lang_selfcheck_stock_equal_all: (map(.lang_selfcheck.stock_stream_equal) | all),
               engine: {physical_runs: (map(.engine.physical_runs) | add), timeouts: (map(.engine.timeouts) | add),
                        failures: (map(.engine.failures) | add), pauses: (map(.engine.pauses) | add)}}' "$P"/status_${v}_*.json)
  pub=$(git -C "$REPO" show "$SEL_COMMIT:$WDIR/${FILE[$v]}.json" | sha256sum | cut -d' ' -f1)
  jq -n --arg v "$v" --argjson n "$(wc -l < "$T/cids")" --argjson recs "$recs" --argjson st "$st" \
    --argjson same16 "$(cmp -s "$T/cids" "$T/r16" && echo true || echo false)" \
    --argjson same15 "$(cmp -s "$T/cids" "$T/r15" && echo true || echo false)" \
    --arg man "$(jq -r .manifest_sha256 "$P/config_$v.json")" --arg pre "$(jq -r .presets_sha256 "$P/config_$v.json")" \
    --arg pub "$pub" --arg selsha "$(jq -r ".${FILE[$v]}_sha256" "$EV/results/selection2/selection.json")" \
    --argjson cfg_mtime "$(stat -c %Y "$P/config_$v.json")" \
    --arg code "$(jq -cS .code_sha256 "$P/config_$v.json" | sha256sum | cut -d' ' -f1)" \
    '{($v): {contexts: $n, h416_records_equal_cids: $same16, h415_records_equal_cids: $same15, records: $recs,
             status: $st, manifest_sha256: $man, presets_sha256: $pre, published_presets_sha256: $pub,
             selection_presets_sha256: $selsha, config_created: $cfg_mtime, code_manifest_digest: $code}}'
done | jq -s add)
replay=$(grep '"worker_runs"' "$EV/logs/driver_dev.log" | tail -n 1)
fill=$(jq -s '{status_files: length, contexts: (map(.contexts_done) | add), kinds: (map(.physical_by_kind | keys) | add | unique),
               worker_s: (map(.engine.wall_s_total) | add), timeouts: (map(.engine.timeouts) | add),
               budget_stops: (map(select(.budget_stop)) | length), guard_installed_all: (map(.guard.installed) | all),
               guard_refusals_in_process: (map([.guard | to_entries[] | select(.key | startswith("refused")) | .value] | add) | add),
               pool_sha256: (map(.pool_sha256) | unique)}' "$F"/status_fill_*.json)
jq -r '.cids[]' "$F/candidates.json" | sort > "$T/design"; jq -r '.cids[]' "$P/config_TPORT.json" | sort > "$T/dev"
overlap=$(comm -12 "$T/design" "$T/dev" | wc -l)
sel=$(jq '{gate_pass: .gate.pass, gate_mean_clipped: .gate.mean_clipped, gate_p: .gate.signflip_p_one_sided, pool_sha256}' "$EV/results/selection2/selection.json")
selpub=$(git -C "$REPO" show "$SEL_COMMIT:$WDIR/selection.json" | sha256sum | cut -d' ' -f1)
seltime=$(git -C "$REPO" log -1 --format=%ct "$SEL_COMMIT")
# the baseline records (H420 TBOTH) are still the ones H420 read out
base=$(cd "$EV/pulled/H421_DEV" && grep -E 'H420_DEV/(config_TBOTH.json|stages/H41[56]_TBOTH/|h416_TBOTH_npz/)' /home/ubuntu/jope_plan/evidence/H420/pulled/H420_DEV.sha256 \
  | sha256sum --quiet -c - >/dev/null 2>&1 && echo true || echo false)
prereg=$(cmp -s "$STAGE/code/PREREG_H421.md" <(git -C "$REPO" show "origin/v1/next:$WDIR/PREREG.md") && echo true || echo false)
stage=$( (cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256) >/dev/null 2>&1 && echo true || echo false)
jq -n --argjson per "$per" --argjson replay "$replay" --argjson fill "$fill" --argjson overlap "$overlap" --argjson sel "$sel" \
  --arg selpub "$selpub" --arg sellocal "$(sha256sum "$EV/results/selection2/selection.json" | cut -d' ' -f1)" \
  --argjson seltime "$seltime" --argjson base "$base" --argjson prereg "$prereg" --argjson stage "$stage" \
  --arg mode "$(cat "$STAGE/mode")" --argjson memos "$(wc -l < "$EV/logs/fill_memo_seed.sha256")" \
  --argjson design "$(wc -l < "$T/design")" \
  --arg sha "$(sha256sum "$EV/pulled/H421_DEV.sha256" | cut -d' ' -f1)" --arg at "$(date -u +%FT%TZ)" '
  {schema: "jope.h421.gates/v1", at: $at, pulled_manifest_sha256: $sha,
   checks: {variants: $per, replaycheck: $replay, fill: $fill, design_contexts: $design, design_dev_overlap: $overlap,
            selection: $sel, selection_published_sha256: $selpub, selection_local_sha256: $sellocal,
            selection_commit_time: $seltime, baseline_records_unchanged: $base, seeded_memos: $memos,
            staged_prereg_equals_published: $prereg, stage_hash_ok: $stage, stage_mode: $mode},
   gates: {
     g1_records_and_units: ([$per[] | (.h416_records_equal_cids and .h415_records_equal_cids and .records.results == .contexts
         and .records.errors == 0 and .status.units == .contexts)] | all),
     g2_guard_and_bank: (([$per[] | (.status.guard_installed_all and .status.guard_refusals_in_process == 0
         and .status.same_family_bank_hits == 0)] | all) and $fill.guard_installed_all and $fill.guard_refusals_in_process == 0),
     g3_selfcheck_and_replay: (([$per[] | .status.lang_selfcheck_stock_equal_all] | all) and $replay.worker_runs == 0 and $replay.ok),
     g4_design_only: ($overlap == 0 and $fill.kinds == ["search"] and $fill.contexts == $design and $memos == $design
         and ($fill.pool_sha256 | length) == 1 and $fill.pool_sha256[0] == $sel.pool_sha256),
     g5_train_gate_passed: ($sel.gate_pass == true),
     g6_presets_published_before_dev: (($selpub == $sellocal)
         and ([$per[] | (.presets_sha256 == .published_presets_sha256 and .presets_sha256 == .selection_presets_sha256
              and .config_created > $seltime)] | all)),
     g7_baseline_unchanged: $base,
     g8_pub_stage_and_one_config: ($prereg and $stage and $mode == "pub"
         and ([$per[] | .manifest_sha256] | unique | length) == 1 and ([$per[] | .code_manifest_digest] | unique | length) == 1)}}
  | .pass = ([.gates[]] | all)' > "$T/out.json"
mv -n "$T/out.json" "$OUT"; chmod a-w "$OUT"
jq '{gates, pass, variants: (.checks.variants | map_values({records, engine: .status.engine})), fill: .checks.fill,
     replay: .checks.replaycheck.worker_runs, overlap: .checks.design_dev_overlap}' "$OUT"
