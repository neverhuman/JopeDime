#!/bin/bash
# H425: the two frozen lists, cut from the screen's output by the pre-registered tier rule (create-once).
#   lists_h425.sh SCREEN.json OUT_DIR   -> OUT_DIR/EXCLUSION_DEV.json (115 DEV contexts), OUT_DIR/CONFIRM_FLAGS.json (241)
set -euo pipefail
S=${1:?SCREEN.json}; O=${2:?out dir}
[ ! -e "$O/EXCLUSION_DEV.json" ] && [ ! -e "$O/CONFIRM_FLAGS.json" ] || { echo "lists exist (create-once)" >&2; exit 3; }
SHA=$(sha256sum "$S" | cut -d' ' -f1)
cut_list() {  # COHORT
  jq --arg co "$1" --arg sha "$SHA" '
    def row: {tier, core, family_key, rows, width,
              flags: {s1: .s1.flag, s1_copy: .s1.flag_copy, s2: .s2.flag, s3: .s3.flag, s4: .s4.flag, b: (if .b then .b.flag else null end)},
              scores: {s1_specific: .s1.specific_columns, s1_matched: .s1.matched, s1_strong: .s1.matched_strong, s2_row_share: .s2.share,
                       s3_matched: .s3.matched, s3_rare: .s3.rare_pairs, b_share: (if .b then .b.share else null end)},
              matched_fit_icl: ([.s1, .s2, .s3, .s4, .b] | map(select(. != null and .flag == true) | {cid: .fit, family_key: .fit_family_key}) | unique)};
    (.per_context | with_entries(select(.value.cohort == $co) | .value |= row)) as $c |
    {schema: "jope.h425.list/v1", cohort: $co, screen_sha256: $sha, thresholds, n: ($c | length),
     tier_counts: ([$c[] | .tier] | group_by(.) | map({(.[0]): length}) | add),
     tier_counts_core: ([$c[] | select(.core) | .tier] | group_by(.) | map({(.[0]): length}) | add),
     contexts: $c}' "$S"
}
cut_list DEV > "$O/EXCLUSION_DEV.json.part"
cut_list CONFIRM > "$O/CONFIRM_FLAGS.json.part"
mv -n "$O/EXCLUSION_DEV.json.part" "$O/EXCLUSION_DEV.json"; mv -n "$O/CONFIRM_FLAGS.json.part" "$O/CONFIRM_FLAGS.json"
chmod a-w "$O/EXCLUSION_DEV.json" "$O/CONFIRM_FLAGS.json"
jq -c '{cohort, n, tier_counts, tier_counts_core}' "$O/EXCLUSION_DEV.json" "$O/CONFIRM_FLAGS.json"
