#!/bin/bash
# H420 integrity gates (PREREG section 5, addenda 1-2) for one panel, from the pulled, sha-compared run copy, the
# seed manifests and the saved chain / driver log lines. Written and run BEFORE the readout. Create-once output.
#   gates_h420.sh PANEL        -> $EV/results/H420_<PANEL>.gates.json
set -euo pipefail
PANEL=${1:?panel}
EV=/home/ubuntu/jope_plan/evidence/H420
P=$EV/pulled/H420_$PANEL/H420_$PANEL
STAGE=/home/ubuntu/apex-stage/h420/pub-p4
REPO=/home/ubuntu/JopeDime; WDIR=research/flywheel_selfplay/h420_lang1_grammar_exposure
OUT=$EV/results/H420_$PANEL.gates.json
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
(cd "$EV/pulled/H420_$PANEL" && sha256sum --quiet -c "../H420_$PANEL.sha256")
T=$(mktemp -d "${TMPDIR:-/tmp}/h420_gates.XXXXXX")   # scratch lists; left in place
last() { tail -n 1 "$1"; }
per=$(for v in TBOTH TLIB TGX; do
  jq -r '.cids[]' "$P/config_$v.json" | sort > "$T/cids"
  ls "$P/stages/H416_$v" | sed 's/__.*//' | sort > "$T/r16"; ls "$P/stages/H415_$v" | sed 's/__.*//' | sort > "$T/r15"
  recs=$(for c in $(cat "$T/cids"); do last "$P/stages/H416_$v/${c}__confirm.jsonl"; done | jq -s '
    {results: (map(select(.type == "result")) | length), errors: (map(select(.error != null)) | length),
     sxb_missing_or_invalid: (map(select((.outputs["P3+LIB3.SXB"] == null) or (.outputs["P3+LIB3.SXB"].invalid == true))) | length),
     recomputed_units: (map(select(.recomputed_as != null)) | length)}')
  st=$(jq -s '{status_files: length, units: (map(.units) | add),
               guard_installed_all: (map(.guard.installed) | all),
               guard_refusals_in_process: (map([.guard | to_entries[] | select(.key | startswith("refused")) | .value] | add) | add),
               guard_cross_process_log_lines_max: (map(.guard.cross_process_refusal_log_lines) | max),
               same_family_bank_hits: (map(.bank_counters.same_family_bank_hits) | add),
               lang_selfcheck_stock_equal_all: (map(.lang_selfcheck.stock_stream_equal) | all),
               engine: {physical_runs: (map(.engine.physical_runs) | add), timeouts: (map(.engine.timeouts) | add),
                        failures: (map(.engine.failures) | add), pauses: (map(.engine.pauses) | add),
                        memo_env_mismatch_refused: (map(.engine.memo_env_mismatch_refused) | add)}}' "$P"/status_${v}_*.json)
  jq -n --arg v "$v" --argjson n "$(wc -l < "$T/cids")" --argjson recs "$recs" --argjson st "$st" \
    --argjson same16 "$(cmp -s "$T/cids" "$T/r16" && echo true || echo false)" \
    --argjson same15 "$(cmp -s "$T/cids" "$T/r15" && echo true || echo false)" \
    --arg man "$(jq -r .manifest_sha256 "$P/config_$v.json")" \
    --arg code "$(jq -cS .code_sha256 "$P/config_$v.json" | sha256sum | cut -d' ' -f1)" \
    '{($v): {contexts: $n, h416_records_equal_cids: $same16, h415_records_equal_cids: $same15, records: $recs,
             status: $st, manifest_sha256: $man, code_manifest_digest: $code}}'
done | jq -s add)
replay=$(grep '"worker_runs"' "$EV/logs/replaycheck_$PANEL.json" | tail -n 1)
ident=$(grep 'seeded smoke: worker runs' "$EV/logs/driver_identity.log" | tail -n 1 \
  | gawk '{for (i = 1; i <= NF; i++) { if ($i == "runs") r = $(i + 1); if ($i == "difference") d = $(i + 1) }; gsub(",", "", r);
           printf "{\"worker_runs\": %s, \"worst_seeded_difference\": %s}", r, d}')
seed=$(jq -n --argjson cids "$(wc -l < "$EV/logs/seed_H420_${PANEL}_$(cat "$EV/logs/seed_host_$PANEL").cids")" \
  --argjson memos "$(grep -c "H420_$PANEL/gproot_AGG" "$EV/logs/seed_H420_${PANEL}_$(cat "$EV/logs/seed_host_$PANEL").sha256")" \
  --argjson src "$(wc -l < "$EV/logs/seed_H420_${PANEL}_$(cat "$EV/logs/seed_host_$PANEL").src.sha256")" \
  '{cids: $cids, concatenated_memos: $memos, source_memo_hashes: $src}')
prereg=$(cmp -s "$STAGE/code/PREREG_H420.md" <(git -C "$REPO" show "origin/v1/next:$WDIR/PREREG.md") && echo true || echo false)
stage=$( (cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256) >/dev/null 2>&1 && echo true || echo false)
jq -n --arg panel "$PANEL" --argjson per "$per" --argjson replay "$replay" --argjson ident "$ident" --argjson seed "$seed" \
  --argjson prereg "$prereg" --argjson stage "$stage" --arg mode "$(cat "$STAGE/mode")" \
  --arg sha "$(sha256sum "$EV/pulled/H420_$PANEL.sha256" | cut -d' ' -f1)" --arg at "$(date -u +%FT%TZ)" '
  {schema: "jope.h420.gates/v1", at: $at, panel: $panel, pulled_manifest_sha256: $sha,
   checks: {variants: $per, replaycheck: $replay, seeded_smoke_identity: $ident, seed: $seed,
            staged_prereg_equals_published: $prereg, stage_hash_ok: $stage, stage_mode: $mode},
   gates: {
     g1_records_and_units: ([$per[] | (.h416_records_equal_cids and .h415_records_equal_cids and .records.results == .contexts
         and .records.errors == 0 and .status.units == .contexts)] | all),
     g2_guard_and_bank: ([$per[] | (.status.guard_installed_all and .status.guard_refusals_in_process == 0
         and .status.same_family_bank_hits == 0)] | all),
     g3_lang_selfcheck: (([$per[] | .status.lang_selfcheck_stock_equal_all] | all) and $replay.lang_selfcheck_stock_equal),
     g4_replaycheck_zero_worker_runs: ($replay.worker_runs == 0 and $replay.ok),
     g5_seeded_identity: ($ident.worker_runs == 0 and $ident.worst_seeded_difference <= 1e-9),
     g6_seed_manifests: ($seed.cids == ([$per[] | .contexts] | max) and $seed.concatenated_memos == $seed.cids),
     g7_pub_stage_and_one_config: ($prereg and $stage and $mode == "pub"
         and ([$per[] | .manifest_sha256] | unique | length) == 1 and ([$per[] | .code_manifest_digest] | unique | length) == 1)}}
  | .pass = ([.gates[]] | all)' > "$T/out.json"
mv -n "$T/out.json" "$OUT"; chmod a-w "$OUT"
jq '{gates, pass, variants: (.checks.variants | map_values({records, engine: .status.engine, cross: .status.guard_cross_process_log_lines_max})),
     replay: .checks.replaycheck.worker_runs, identity: .checks.seeded_smoke_identity, seed: .checks.seed}' "$OUT"
