#!/usr/bin/env bash
# check_sealed.sh <dataset_id>... : refuse any dataset in a sealed or held-out family.
#
# Exit 0 only when every id is clear of:
#   - the sealed meta-test identity manifest (dataset hashes, catalog ids, view-id prefixes)
#     research/flywheel_selfplay/h150_w5_provenance_unblock/sealed_family_manifest.json;
#   - the owner-approved JRB-01 exclusion's unkeyed catalog ids, when that host file exists;
#   - the regression-10 private benchmark panel (crates/jope-datasets/releases/regression-10-v1),
#     matched by name against the catalog display name and provenance id.
# Prints one line per id: CLEAR or SEALED with the reason. Read-only.
set -euo pipefail
root=${JOPE_CASE_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
manifest=$root/research/flywheel_selfplay/h150_w5_provenance_unblock/sealed_family_manifest.json
panel=$root/crates/jope-datasets/releases/regression-10-v1/release.json
jrb=${JOPE_JRB01_EXCLUSION:-/home/ubuntu/jope_plan/master/approved/JRB01_SEALED_FAMILY_EXCLUSION.json}
catalog=${JOPE_CASE_CATALOG:-$HOME/.cache/jope/datasets/catalog/v246/catalog.jsonl}
[ "$#" -gt 0 ] || { echo "usage: $0 <dataset_id>..." >&2; exit 2; }
[ -f "$manifest" ] || { echo "check_sealed: missing $manifest" >&2; exit 2; }
status=0
for id in "$@"; do
  [[ "$id" =~ ^[0-9a-f]{16}$ ]] || { echo "$id SEALED? not a 16-hex catalog id"; status=1; continue; }
  reason=$(jq -r --arg id "$id" '
      if (.dataset_hashes | index($id)) then "sealed_family_manifest.dataset_hashes"
      elif (.catalog_sealed_ids | index($id)) then "sealed_family_manifest.catalog_sealed_ids"
      elif any(.view_ids[]; startswith($id + "__")) then "sealed_family_manifest.view_ids"
      else empty end' "$manifest")
  if [ -z "$reason" ] && [ -f "$jrb" ]; then
    reason=$(jq -r --arg id "$id" '[.. | strings | select(. == $id)] | if length > 0 then "JRB01_SEALED_FAMILY_EXCLUSION" else empty end' "$jrb")
  fi
  if [ -z "$reason" ] && [ -f "$catalog" ]; then
    names=$(grep -F "\"$id\"" "$catalog" | head -1 | jq -r '[.display_name, .provenance.dataset_id, .name] | map(select(. != null)) | map(ascii_downcase) | join("|")' 2>/dev/null || true)
    if [ -n "$names" ]; then
      reason=$(jq -r --arg names "$names" '[.members[].dataset | ascii_downcase | select(. as $m | $names | contains($m))] | if length > 0 then "regression-10 private panel: " + join(",") else empty end' "$panel")
    fi
  fi
  if [ -n "$reason" ]; then echo "$id SEALED $reason"; status=1; else echo "$id CLEAR"; fi
done
exit $status
