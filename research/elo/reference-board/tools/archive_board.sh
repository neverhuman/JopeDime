#!/bin/bash
# Archive one stored result set to S3, create-only, then GET that exact version back and verify every member.
#
#   archive_board.sh ROOT NAME --dry-run|--upload
#
# NAME is the S3 subpath under s3://veox-jopedime-learning-use2/experiments/reference-board/ (a board id, or
# external/<set>). Members:
#   * a board root (ROOT/BOARD.json present; NAME must equal its board id): BOARD.json, scores/*.jsonl,
#     preds/<model>/<cid>.f64, sources/*, ladder.json, LADDER.md (whichever exist);
#   * any other result set: every regular file under ROOT except ROOT/archive/.
# NEVER an envelope (envs/, *.jtt1): TRAIN / DL-Val rows stay on the hosts.
# Writes ROOT/archive/{members,SHA256SUMS}-<UTC>, the bundle, the exact-version GET, its verified extraction and
# archive-<UTC>.json (the receipt to copy into git). Nothing is ever deleted or overwritten: put-object is create-only.
set -euo pipefail
BR=$(readlink -f "$1"); NAME=$2; MODE=${3:?--dry-run|--upload}
case $NAME in */../*|../*|*/..|/*|'') echo "bad NAME $NAME" >&2; exit 2;; esac
if [ -f "$BR/BOARD.json" ]; then
  [ "$(jq -r .board "$BR/BOARD.json")" = "$NAME" ] || { echo "NAME must be the board id" >&2; exit 2; }
fi
UTC=$(date -u +%Y%m%dT%H%M%SZ)
BUCKET=veox-jopedime-learning-use2
KEY=experiments/reference-board/$NAME/results-$UTC.tar.zst
export AWS_PROFILE=${AWS_PROFILE:-veox-jepson-prod}
OUT=$BR/archive; mkdir -p "$OUT"
L=$OUT/members-$UTC.txt
(
  cd "$BR"
  if [ -f BOARD.json ]; then
    echo BOARD.json
    ls scores/*.jsonl
    for d in preds sources; do if [ -d "$d" ]; then find "$d" -type f | sort; fi; done
    for f in ladder.json LADDER.md; do if [ -f "$f" ]; then echo "$f"; fi; done
  else
    find . -path ./archive -prune -o -type f -print | sed 's#^\./##' | sort
  fi
) > "$L"
if grep -E '(^|/)envs/|\.jtt1$' "$L"; then echo "REFUSING: an envelope is in the member list" >&2; exit 3; fi
(cd "$BR" && xargs -a "$L" -d '\n' sha256sum) > "$OUT/SHA256SUMS-$UTC"
echo "set=$NAME members=$(wc -l < "$L") bytes=$(cd "$BR" && xargs -a "$L" -d '\n' stat -c %s | paste -sd+ | bc)"
[ "$MODE" = --upload ] || { echo "dry run: nothing uploaded"; exit 0; }
B=$OUT/results-$UTC.tar.zst
tar -C "$BR" -cf - -T "$L" | zstd -q -T8 -12 -o "$B"
SHA=$(sha256sum "$B" | cut -d' ' -f1)
VER=$(aws s3api put-object --bucket "$BUCKET" --key "$KEY" --body "$B" --if-none-match '*' \
        --checksum-algorithm SHA256 --query VersionId --output text)
G=$OUT/get-$VER.tar.zst
aws s3api get-object --bucket "$BUCKET" --key "$KEY" --version-id "$VER" "$G" >/dev/null
[ "$(sha256sum "$G" | cut -d' ' -f1)" = "$SHA" ] || { echo "GET sha mismatch" >&2; exit 5; }
X=$OUT/verify-$VER; mkdir -p "$X"
zstd -dc "$G" | tar -C "$X" -xf -
(cd "$X" && sha256sum --quiet -c "$OUT/SHA256SUMS-$UTC")
[ "$(cd "$X" && find . -type f | wc -l)" = "$(wc -l < "$L")" ] || { echo "extracted member count differs" >&2; exit 6; }
jq -n --arg n "$NAME" --arg uri "s3://$BUCKET/$KEY" --arg v "$VER" --arg s "$SHA" --arg m "$(wc -l < "$L")" \
  --arg u "$UTC" --arg sums "$(sha256sum "$OUT/SHA256SUMS-$UTC" | cut -d' ' -f1)" \
  --arg bs "$( [ -f "$BR/BOARD.json" ] && jq -r .board_sha256 "$BR/BOARD.json" || true)" \
  '{schema:"jope.reference-board.archive/v1", set:$n, board_sha256:(if $bs == "" then null else $bs end),
    uri:$uri, version_id:$v, sha256:$s, members:($m|tonumber), member_sha256sums_sha256:$sums,
    verified:"EXACT_VERSION_GET_AND_MEMBERS_VERIFIED", utc:$u}' | tee "$OUT/archive-$UTC.json"
