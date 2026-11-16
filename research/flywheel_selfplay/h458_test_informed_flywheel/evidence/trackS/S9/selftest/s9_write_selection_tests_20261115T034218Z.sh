#!/bin/bash
# Tests for s9_write_selection.sh (W1 fix) on fake receipts. Each case builds a fresh receipt dir, runs the tool, and
# compares the exit code (and, for passes, the record) with what PREREG_S9 section 5 and AMENDMENT_6 want.
set -uo pipefail
TOOL=${1:-/home/ubuntu/h458s-stage/code/s9_write_selection.sh.new}
B=${S9SEL_TEST_DIR:-/tmp/claude-1000/-home-ubuntu-JopeDime/7afc9e1c-0a35-41de-8b38-8892ec985b0e/scratchpad/s9sel_test}
C80=$(tr -d '\n' < /home/ubuntu/jope-data/h458-quarantine/gp_chain/p201b_v1/CIDS_80.txt | jq -R -c 'split(",")')
GOOD=7b90ae79f83b7f3540bbfaaebd90c0812c2c7a602f7da37eabad445863b8bc47
pass=0; fail=0
# fake <dir> <file name> <tier> <arms json: [{arm, R1, T, desc}]> [n_contexts] [scorer sha] [ctrl prefix]
fake() {
  local d=$1 f=$2 t=$3 arms=$4 n=${5:-} sha=${6:-$GOOD} cp=${7:-a0}
  [ -n "$n" ] || { [ "$t" = ultra ] && n=20 || n=80; }
  jq -n --argjson c80 "$C80" --argjson arms "$arms" --argjson n "$n" --arg sha "$sha" --arg cp "$cp" '
    ($arms | map(select(.desc != true))) as $el
    | {schema: "jope.h458s.s9_board_stack/v1", board: "P201B", written_utc: "2026-11-15T05:00:00Z",
       n_contexts: $n, contexts: ($c80[0:$n] | reverse),
       controls: {a: "\($cp)a", b: "\($cp)b", dominating: "\($cp)x"},
       columns: (["\($cp)a", "\($cp)b", "\($cp)x"] + ($arms | map(.arm))),
       descriptive: ($arms | map(select(.desc == true) | .arm)),
       code_sha256: {"__main__": {path: "/home/ubuntu/h458s-stage/code/s9_p201b_stack.py", sha256: $sha},
                     h335_stats: {path: "/home/ubuntu/x/h335_stats.py", sha256: "ee3b4543"}},
       rules: ($arms | map({key: .arm, value: {selection_eligible: (.desc != true), T_over_ctrl_a: .T, R1: .R1,
                                                holm_p: {}}}) | from_entries),
       r1_passing_eligible_by_T: ($el | map(select(.R1 == true)) | sort_by(-.T) | map(.arm)),
       r1_holm_within_run: ($el | map({key: "\(.arm) - \($cp)a", value: (.p // 0.01)},
                                      {key: "\(.arm) - \($cp)b", value: (.p // 0.01)}) | from_entries),
       reads: {gp_alone: ($el | map({key: "\(.arm) - \($cp)a", value: {signflip_p: (.p // 0.01)}},
                                     {key: "\(.arm) - \($cp)b", value: {signflip_p: (.p // 0.01)}}) | from_entries)},
       noise: {gp_any: false, stack_any: false}}' > "$d/$f"
  chmod 0444 "$d/$f"
}
base() {  # a standard directory: medium (cl_h458 + A5), high (cl_h458 + A5)
  local d=$B/case_$1; mkdir -p "$d"; chmod -R u+w "$d" 2>/dev/null
  find "$d" -mindepth 1 -maxdepth 1 -name 'S9_*' -exec mv -t "$B/trash" {} + 2>/dev/null
  fake "$d" S9_STACK_P201B_medium_v1.json medium '[{"arm":"a2","R1":true,"T":0.02},{"arm":"a4","R1":false,"T":0.01},{"arm":"a0s","R1":null,"T":0.0,"desc":true}]'
  fake "$d" S9_STACK_P201B_medium_a5_prod_pure.json medium '[{"arm":"a5_medium_pure_shrunk","R1":true,"T":0.03},{"arm":"a5_medium_pure_ratchet","R1":false,"T":0.0}]' "" "" a0_medium_pure_s
  fake "$d" S9_STACK_P201B_high_v1.json high '[{"arm":"a2","R1":true,"T":0.05,"p":0.001},{"arm":"a4","R1":true,"T":0.03,"p":0.004},{"arm":"a1l_s1","R1":true,"T":0.09,"desc":true}]'
  fake "$d" S9_STACK_P201B_high_a5_prod_pure.json high '[{"arm":"a5_high_pure_shrunk","R1":true,"T":0.04},{"arm":"a5_high_pure_ratchet","R1":false,"T":0.06}]' "" "" a0_high_pure_s
  echo "$d"
}
STD=(S9_STACK_P201B_medium_v1.json S9_STACK_P201B_medium_a5_prod_pure.json S9_STACK_P201B_high_v1.json S9_STACK_P201B_high_a5_prod_pure.json)
expect() {  # expect <case> <want rc> <dir> <tier> <arm> <expected names...>
  local name=$1 want=$2 d=$3 tier=$4 arm=$5; shift 5
  local out=$d/SEL_$name.json
  S9_OUT=$d bash "$TOOL" "$out" "$tier" "$arm" "$@" > "$d/$name.out" 2> "$d/$name.err"; local rc=$?
  if [ "$rc" -eq "$want" ]; then pass=$((pass+1)); echo "PASS $name rc=$rc $(head -c 160 "$d/$name.err" | tr '\n' ' ')"
  else fail=$((fail+1)); echo "FAIL $name rc=$rc want $want: $(cat "$d/$name.err" | tr '\n' ' ')"; fi
}
mkdir -p "$B/trash"
d=$(base ok); expect high_argmax 0 "$d" high a2 "${STD[@]}"
r=$d/SEL_high_argmax.json
if jq -e '.selected_arm == "a2" and .tier == "high" and .selected_from == "S9_STACK_P201B_high_v1.json" and .T_over_ctrl_a == 0.05
          and (.passers_at_tier_by_T | map(.arm)) == ["a2", "a5_high_pure_shrunk", "a4"] and (.receipts | length) == 4
          and .holm_pooled_over_every_P201B_R1_test.m == 16 and .board == "P201B"
          and ((.holm_pooled_over_every_P201B_R1_test.adjusted_p["S9_STACK_P201B_high_v1.json::a2 - a0a"] - 0.016) | fabs) < 1e-12
          and (.receipts | all(.scorer_sha256 == "'$GOOD'"))' "$r" >/dev/null && [ ! -w "$r" ]; then
  pass=$((pass+1)); echo "PASS record_contents"; else fail=$((fail+1)); echo "FAIL record_contents"; jq -c '{selected_arm, selected_from, T_over_ctrl_a, p: (.passers_at_tier_by_T | map(.arm)), m: .holm_pooled_over_every_P201B_R1_test.m, a: .holm_pooled_over_every_P201B_R1_test.adjusted_p}' "$r"; fi
S9_OUT=$d bash "$TOOL" "$r" high a2 "${STD[@]}" >/dev/null 2>"$d/create_once.err"; rc=$?; if [ $rc -eq 2 ]; then pass=$((pass+1)); echo "PASS create_once rc=2 $(head -c 120 "$d/create_once.err")"; else fail=$((fail+1)); echo "FAIL create_once rc=$rc"; fi
d=$(base notmax); expect not_argmax 3 "$d" high a4 "${STD[@]}"
d=$(base desc); expect descriptive_never_selected 3 "$d" high a1l_s1 "${STD[@]}"
d=$(base nonpass); expect r1_false_arm 3 "$d" high a5_high_pure_ratchet "${STD[@]}"
d=$(base med); expect medium_while_high_passes 3 "$d" medium a5_medium_pure_shrunk "${STD[@]}"
d=$(base medok); find "$d" -name 'S9_STACK_P201B_high_*' -exec chmod u+w {} + ; \
  fake "$d" S9_STACK_P201B_high_v1.json high '[{"arm":"a2","R1":false,"T":0.05},{"arm":"a4","R1":false,"T":0.03}]'; \
  fake "$d" S9_STACK_P201B_high_a5_prod_pure.json high '[{"arm":"a5_high_pure_shrunk","R1":false,"T":0.04}]' "" "" a0_high_pure_s
expect medium_after_null_high 0 "$d" medium a5_medium_pure_shrunk "${STD[@]}"
expect medium_not_argmax 3 "$d" medium a2 "${STD[@]}"
d=$(base nohigh); mv "$d"/S9_STACK_P201B_high_* "$B/trash/"; expect medium_without_high 3 "$d" medium a5_medium_pure_shrunk S9_STACK_P201B_medium_v1.json S9_STACK_P201B_medium_a5_prod_pure.json
d=$(base unexp); expect unexpected_receipt 3 "$d" high a2 S9_STACK_P201B_medium_v1.json S9_STACK_P201B_high_v1.json S9_STACK_P201B_high_a5_prod_pure.json
d=$(base missing); expect missing_expected 3 "$d" high a2 "${STD[@]}" S9_STACK_P201B_high_extra.json
d=$(base notier); fake "$d" S9_STACK_P201B_cycle_medium_a5_pure.json medium '[{"arm":"a5_medium_pure_stg1","R1":null,"T":0.0,"desc":true}]' "" "" a0_medium_pure_s
expect untiered_name 2 "$d" high a2 "${STD[@]}" S9_STACK_P201B_cycle_medium_a5_pure.json
d=$(base dup); fake "$d" S9_STACK_P201B_high_v1b.json high '[{"arm":"a2","R1":true,"T":0.07}]'
expect duplicate_arm_in_tier 3 "$d" high a2 "${STD[@]}" S9_STACK_P201B_high_v1b.json
d=$(base tie); chmod u+w "$d/S9_STACK_P201B_high_a5_prod_pure.json"; fake "$d" S9_STACK_P201B_high_a5_prod_pure.json high '[{"arm":"a5_high_pure_shrunk","R1":true,"T":0.05}]' "" "" a0_high_pure_s
expect tie 3 "$d" high a2 "${STD[@]}"
d=$(base writable); chmod u+w "$d/S9_STACK_P201B_high_v1.json"; expect writable_receipt 2 "$d" high a2 "${STD[@]}"
d=$(base ctx); chmod u+w "$d/S9_STACK_P201B_medium_v1.json"; fake "$d" S9_STACK_P201B_medium_v1.json medium '[{"arm":"a2","R1":true,"T":0.02}]' 79
expect wrong_contexts 3 "$d" high a2 "${STD[@]}"
d=$(base tok); chmod u+w "$d/S9_STACK_P201B_high_a5_prod_pure.json"; fake "$d" S9_STACK_P201B_high_a5_prod_pure.json high '[{"arm":"a5_medium_pure_shrunk","R1":true,"T":0.04}]' "" "" a0_high_pure_s
expect other_tier_column 3 "$d" high a2 "${STD[@]}"
d=$(base ultraok); fake "$d" S9_STACK_P201B_ultra_v1.json ultra '[{"arm":"a2","R1":null,"T":0.08}]'
expect ultra_null_read_ok 0 "$d" high a2 "${STD[@]}" S9_STACK_P201B_ultra_v1.json
d=$(base ultrapass); fake "$d" S9_STACK_P201B_ultra_v1.json ultra '[{"arm":"a2","R1":true,"T":0.08}]'
expect ultra_with_passer 3 "$d" high a2 "${STD[@]}" S9_STACK_P201B_ultra_v1.json
d=$(base ultractx); fake "$d" S9_STACK_P201B_ultra_v1.json ultra '[{"arm":"a2","R1":null,"T":0.08}]' 80
expect ultra_wrong_contexts 3 "$d" high a2 "${STD[@]}" S9_STACK_P201B_ultra_v1.json
d=$(base ultratier); expect tier_ultra_refused 2 "$d" ultra a2 "${STD[@]}"
d=$(base scorer); chmod u+w "$d/S9_STACK_P201B_high_v1.json"; fake "$d" S9_STACK_P201B_high_v1.json high '[{"arm":"a2","R1":true,"T":0.05}]' "" deadbeef
expect wrong_scorer 3 "$d" high a2 "${STD[@]}"
d=$(base raw); cp "$d/S9_STACK_P201B_high_v1.json" "$d/S9_STACK_P201B_high_v1.raw.json"; expect raw_ignored 0 "$d" high a2 "${STD[@]}"
d=$(base nopass); for f in "$d"/S9_STACK_P201B_*; do chmod u+w "$f"; done
  fake "$d" S9_STACK_P201B_medium_v1.json medium '[{"arm":"a2","R1":false,"T":0.02}]'
  fake "$d" S9_STACK_P201B_medium_a5_prod_pure.json medium '[{"arm":"a5_medium_pure_shrunk","R1":false,"T":0.03}]' "" "" a0_medium_pure_s
  fake "$d" S9_STACK_P201B_high_v1.json high '[{"arm":"a2","R1":false,"T":0.05}]'
  fake "$d" S9_STACK_P201B_high_a5_prod_pure.json high '[{"arm":"a5_high_pure_shrunk","R1":false,"T":0.04}]' "" "" a0_high_pure_s
expect null_everywhere 3 "$d" medium a2 "${STD[@]}"
d=$(base board); chmod u+w "$d/S9_STACK_P201B_medium_v1.json"; jq '.board = "FITSEL"' "$d/S9_STACK_P201B_medium_v1.json" > "$d/x.tmp" && cat "$d/x.tmp" > "$d/S9_STACK_P201B_medium_v1.json" && mv "$d/x.tmp" "$B/trash/x.$RANDOM"; chmod 0444 "$d/S9_STACK_P201B_medium_v1.json"
expect wrong_board 3 "$d" high a2 "${STD[@]}"
echo "tests: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
