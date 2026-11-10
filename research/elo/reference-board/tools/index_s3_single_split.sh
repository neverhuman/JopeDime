#!/bin/bash
# Index the precomputed single-split DL-Val91 records that already live in S3 (TabPFN 3.5 outer-test; Kumo-Tabular L/S,
# TabICL v2, TabDPT 1.3 and Mitra v2 query encodings). Read-only: one HEAD per object, nothing is downloaded or scored.
#
# Input: the H257 lookup s3_precomputed_tabular_dl_val91_cache.json ({model: {cid: {s3_key, protocol, utilities}}}),
#        stored verbatim in boards/dev91-s1-two-fold-v1/sources/ on the board root, and the dev91 BOARD.json (families).
# Usage: index_s3_single_split.sh LOOKUP_JSON BOARD_JSON > external/dev91-s3-single-split-v1.jsonl
# Each line: model_id, cid, family_id, protocol, s3_uri, version_id, content_length, last_modified, and the utilities
# H257 derived from the object. Exit 4 if any object is missing.
set -euo pipefail
LK=$1; BJ=$2
BUCKET=veox-jopedime-learning-use2
export AWS_PROFILE=${AWS_PROFILE:-veox-jepson-prod}
W=$(mktemp -d "${TMPDIR:-/tmp}/s3idx.XXXXXX")  # scratch only; never deleted by this tool
jq -c '[.contexts[] | {key:.cid, value:.family_id}] | from_entries' "$BJ" > "$W/fam.json"
jq -r 'to_entries[] | .value | to_entries[] | .value.s3_key' "$LK" | sort -u > "$W/keys"
xargs -a "$W/keys" -P 16 -I{} sh -c '
  r=$(aws s3api head-object --bucket '"$BUCKET"' --key "{}" --query "[ContentLength,VersionId,LastModified]" --output text 2>/dev/null) \
    && printf "%s\t%s\n" "{}" "$r" || printf "%s\tMISSING\n" "{}"' > "$W/head.tsv"
if grep -q $'\tMISSING$' "$W/head.tsv"; then grep $'\tMISSING$' "$W/head.tsv" >&2; exit 4; fi
jq -R -s 'split("\n") | map(select(length > 0) | split("\t")) | map({key:.[0], value:{len:(.[1]|tonumber), ver:.[2], mod:.[3]}}) | from_entries' \
  "$W/head.tsv" > "$W/head.json"
jq -c --arg b "$BUCKET" --slurpfile h "$W/head.json" --slurpfile f "$W/fam.json" '
  to_entries[] | .key as $m | .value | to_entries[] | .value as $r | $h[0][$r.s3_key] as $o
  | {model_id:("s3." + $m), cid:.key, family_id:$f[0][.key], protocol:$r.protocol,
     s3_uri:("s3://" + $b + "/" + $r.s3_key), version_id:$o.ver, content_length:$o.len, last_modified:$o.mod,
     utilities:($r | del(.cid, .s3_key, .protocol))}' "$LK" | sort
echo "scratch: $W" >&2
