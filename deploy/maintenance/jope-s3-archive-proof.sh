#!/usr/bin/env bash
# Publish one already verified encrypted USB shard to the versioned archive bucket.
set -euo pipefail
archive=${1:?usage: jope-s3-archive-proof.sh ARCHIVE BUCKET PREFIX RECEIPT_DIR}
bucket=${2:?usage: jope-s3-archive-proof.sh ARCHIVE BUCKET PREFIX RECEIPT_DIR}
prefix=${3:?usage: jope-s3-archive-proof.sh ARCHIVE BUCKET PREFIX RECEIPT_DIR}
receipt_dir=${4:?usage: jope-s3-archive-proof.sh ARCHIVE BUCKET PREFIX RECEIPT_DIR}
test -f "$archive"
test -f "$archive.receipt.json"
test -f "$archive.sha256"
[[ "$prefix" != /* && "$prefix" != *..* ]]
jq -e --arg archive "$archive" '.archive == $archive and .full_readback == true and .source_deleted == false' \
  "$archive.receipt.json" >/dev/null
(cd "$(dirname "$archive")" && sha256sum -c "$(basename "$archive").sha256") >/dev/null
read -r sha _ < "$archive.sha256"
[[ "$sha" =~ ^[0-9a-f]{64}$ ]]
bytes=$(stat -c %s "$archive")
key="$prefix/sha256/$sha.tar.zst.gpg"
mkdir -p -- "$receipt_dir"
export AWS_PROFILE=${AWS_PROFILE:-veox-jepson-prod}
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
if ! aws s3api head-object --bucket "$bucket" --key "$key" >/dev/null 2>&1; then
  aws s3 cp --only-show-errors "$archive" "s3://$bucket/$key" >/dev/null
fi
remote_sha=$(aws s3 cp --only-show-errors "s3://$bucket/$key" - | sha256sum | cut -d' ' -f1)
test "$remote_sha" = "$sha"
remote_bytes=$(aws s3api head-object --bucket "$bucket" --key "$key" --query ContentLength --output text)
test "$remote_bytes" = "$bytes"
version=$(aws s3api head-object --bucket "$bucket" --key "$key" --query VersionId --output text)
test "$version" != None
stamp=$(date -u +%Y%m%dT%H%M%SZ)
jq -n --arg archive "$archive" --arg bucket "$bucket" --arg key "$key" \
  --arg sha "$sha" --arg version "$version" --arg stamp "$stamp" --argjson bytes "$bytes" \
  '{schema:"jope.s3-archive-proof/v1",at:$stamp,archive:$archive,bucket:$bucket,
    key:$key,version:$version,sha256:$sha,bytes:$bytes,full_s3_readback:true}' \
  > "$receipt_dir/$sha.json"
cat "$receipt_dir/$sha.json"
