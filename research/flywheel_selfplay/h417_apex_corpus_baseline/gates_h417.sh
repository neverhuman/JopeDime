#!/bin/bash
# H417 harness gates on the BASE run (PREREG section 7, gates 1-4), from the pulled, sha-compared deploy copy, the
# three search arms' stage records on xbabe0 and the view-encoding logs. Create-once output.
#   gates_h417.sh            -> $EV/results/APEX_BASE_BASE.gates.json
# Written at closeout, after the readout had already been run (the published stage had no gate step; see README).
set -euo pipefail
DATA=/home/ubuntu/jope-data/h335-cl
EV=/home/ubuntu/jope_plan/evidence/H417
P=$EV/pulled/APEX_BASE/APEX_BASE
OUT=$EV/results/APEX_BASE_BASE.gates.json
DOPE=c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
(cd "$EV/pulled/APEX_BASE" && sha256sum --quiet -c ../APEX_BASE.sha256)
last() { tail -n 1 "$1"; }
N=$(jq '.cids | length' "$P/config_BASE.json")
jq -r '.cids[]' "$P/config_BASE.json" | sort > "$EV/results/.gates_cids.$$"
# gate 1: one stage record per context in each search arm; every deploy unit completed
arms=$(for a in GS GE GL; do
  ssh -o BatchMode=yes xbabe0 "ls $DATA/runs/APEX_BASE_$a/stages/E417_AGG" | sed 's/__.*//' | sort > "$EV/results/.gates_arm.$$"
  jq -n --arg a "$a" --argjson n "$(wc -l < "$EV/results/.gates_arm.$$")" \
    --argjson uniq "$(sort -u "$EV/results/.gates_arm.$$" | wc -l)" \
    --argjson same "$(cmp -s "$EV/results/.gates_cids.$$" "$EV/results/.gates_arm.$$" && echo true || echo false)" \
    '{($a): {records: $n, distinct_contexts: $uniq, equals_config_cids: $same}}'
done | jq -s add)
n16=0; err=0; multi=0; kmax=0; n15=0
for c in $(cat "$EV/results/.gates_cids.$$"); do
  f=$P/stages/H416_BASE/${c}__confirm.jsonl
  [ -f "$f" ] || continue
  n16=$((n16 + 1)); [ "$(wc -l < "$f")" = 1 ] || multi=$((multi + 1))
  [ -f "$P/stages/H415_BASE/${c}__pool.jsonl" ] && n15=$((n15 + 1))
done
recs=$(for c in $(cat "$EV/results/.gates_cids.$$"); do last "$P/stages/H416_BASE/${c}__confirm.jsonl"; done \
  | jq -s '{errors: (map(select(.error != null)) | length),
            sxb_missing_or_invalid: (map(select((.outputs["P3+LIB3.SXB"] == null) or (.outputs["P3+LIB3.SXB"].invalid == true))) | length),
            kcar100_bag_vs_kfb_max_abs_delta: (map((.kcar100_bag_u - .kfb_u) | fabs) | max),
            recomputed_units: (map(select(.recomputed_as != null)) | length)}')
# resumed units are recomputed under H415_BASEr from memo hits: their values must equal the stored H415_BASE record
rmax=$(for f in "$P"/stages/H415_BASEr/*__pool.jsonl; do
  c=$(basename "$f" __pool.jsonl)
  jq -n --slurpfile a <(last "$P/stages/H415_BASE/${c}__pool.jsonl") --slurpfile b <(last "$f") \
    '[$a[0].outputs | to_entries[] | select(.value.value != null) | . as $e
      | ($b[0].outputs[$e.key].value // null) as $v | select($v != null) | (($e.value.value - $v) | fabs)] | max // 0'
done | jq -s 'max // 0')
st=$(jq -s '{status_files: length,
             units_final_shards: (map(select(.shard | endswith("/8")) | .units) | add),
             same_family_bank_hits: (map(.bank_counters.same_family_bank_hits) | add),
             guard_installed_all: (map(.guard.installed) | all),
             guard_refusals_in_process: (map([.guard | to_entries[] | select(.key | startswith("refused")) | .value] | add) | add),
             guard_cross_process_log_lines_max: (map(.guard.cross_process_refusal_log_lines) | max),
             audit_ledger: {requests: (map(.audit_ledger.audit_requests) | add), outputs: (map(.audit_ledger.outputs) | add)},
             akf_ledger: {requests: (map(.akf_ledger.member_requests) | add), outputs: (map(.akf_ledger.outputs) | add)}}' \
  "$P"/status_BASE_*.json)
armst=$(ssh -o BatchMode=yes xbabe0 "cat $DATA/runs/APEX_BASE_G[SEL]/shards/status_*.json" \
  | jq -s '{status_files: length, same_family_bank_hits: (map(.bank_counters.same_family_bank_hits) | add),
            guard_refusals_in_process: (map([.guard | to_entries[] | select(.key | startswith("refused")) | .value] | add) | add)}')
dope=$(for l in "$EV/logs/encode_BASE.log" "$EV/logs/encode_SMK2.log"; do head -n 1 "$l"; done \
  | jq -s --arg d "$DOPE" '{encodes: length, all_before_after_equal_pin: (map(.dope_sha256_before == $d and .dope_sha256_after == $d) | all)}')
jq -n --argjson n "$N" --argjson arms "$arms" --argjson n16 "$n16" --argjson n15 "$n15" --argjson multi "$multi" \
  --argjson recs "$recs" --argjson rmax "$rmax" --argjson st "$st" --argjson armst "$armst" --argjson dope "$dope" \
  --arg sha "$(sha256sum "$EV/pulled/APEX_BASE.sha256" | cut -d' ' -f1)" --arg at "$(date -u +%FT%TZ)" '
  {schema: "jope.h417.gates/v1", at: $at, pulled_manifest_sha256: $sha, contexts: $n,
   checks: {arms: $arms, deploy_records_h416: $n16, deploy_records_h415: $n15, records_with_more_than_one_line: $multi,
            records: $recs, recomputed_vs_stored_max_abs_delta: $rmax, deploy_status: $st, arm_status: $armst, dope: $dope},
   gates: {
     g1_records_and_units: (([$arms[] | (.records == $n and .distinct_contexts == $n and .equals_config_cids)] | all)
        and $n16 == $n and $n15 == $n and $multi == 0 and $recs.errors == 0 and $recs.sxb_missing_or_invalid == 0
        and $st.units_final_shards == $n),
     g2_bank_ledgers_guard: ($st.same_family_bank_hits == 0 and $armst.same_family_bank_hits == 0
        and $st.guard_installed_all and $st.guard_refusals_in_process == 0 and $armst.guard_refusals_in_process == 0
        and $st.audit_ledger.outputs > 0 and $st.akf_ledger.outputs > 0),
     g3_kcar100_bag_equals_kfb: ($recs.kcar100_bag_vs_kfb_max_abs_delta <= 1e-9),
     g4_dope_unchanged: $dope.all_before_after_equal_pin}}
  | .pass = ([.gates[]] | all)' > "$OUT.tmp.$$"
mv -n "$OUT.tmp.$$" "$OUT"; chmod a-w "$OUT"
jq '{gates, pass, checks: (.checks | del(.arms))}' "$OUT"
