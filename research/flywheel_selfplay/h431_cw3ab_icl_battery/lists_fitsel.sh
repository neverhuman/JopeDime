#!/bin/bash
# H431: Cut EXCLUSION_FITSEL.json (106 FIT-SEL contexts) from SCREEN_FITSEL.json using H425's exact tier rule.
# Usage: lists_fitsel.sh SCREEN_FITSEL.json OUT_DIR
set -euo pipefail
S=${1:?SCREEN_FITSEL.json}; O=${2:?out dir}
[ ! -e "$O/EXCLUSION_FITSEL.json" ] || { echo "$O/EXCLUSION_FITSEL.json exists (create-once)" >&2; exit 3; }
mkdir -p "$O"
SHA=$(sha256sum "$S" | cut -d' ' -f1)
jq --arg co "FIT-SEL" --arg sha "$SHA" '
  def row: {tier, core, family_key, rows, width,
            flags: {s1: .s1.flag, s1_copy: .s1.flag_copy, s2: .s2.flag, s3: .s3.flag, s4: .s4.flag, b: (if .b then .b.flag else null end)},
            scores: {s1_specific: .s1.specific_columns, s1_matched: .s1.matched, s1_strong: .s1.matched_strong, s2_row_share: .s2.share,
                     s3_matched: .s3.matched, s3_rare: .s3.rare_pairs, b_share: (if .b then .b.share else null end)},
            matched_fit_icl: ([.s1, .s2, .s3, .s4, .b] | map(select(. != null and .flag == true) | {cid: .fit, family_key: .fit_family_key}) | unique)};
  (.per_context | with_entries(select(.value.cohort == $co) | .value |= row)) as $c |
  {schema: "jope.h425.list/v1", wave: "H431", cohort: $co, screen_sha256: $sha, thresholds, n: ($c | length),
   tier_counts: ([$c[] | .tier] | group_by(.) | map({(.[0]): length}) | add),
   tier_counts_core: ([$c[] | select(.core) | .tier] | group_by(.) | map({(.[0]): length}) | add),
   contexts: $c}' "$S" > "$O/EXCLUSION_FITSEL.json.part"
mv -n "$O/EXCLUSION_FITSEL.json.part" "$O/EXCLUSION_FITSEL.json"
chmod a-w "$O/EXCLUSION_FITSEL.json"
jq -c '{cohort, n, tier_counts}' "$O/EXCLUSION_FITSEL.json"
