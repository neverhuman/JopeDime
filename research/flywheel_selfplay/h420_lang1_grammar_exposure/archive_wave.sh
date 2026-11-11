#!/bin/bash
# Closeout archive of one APEX wave (H415 / H416 / H417 pattern): one create-only tar.zst bundle, exact-version GET,
# bundle sha + member check. Members:
#   - the wave's evidence folder: top-level files, pulled/ (run copies with stage records and npz matrices), results/,
#     logs/;
#   - of each extra run directory named on the command line: top-level files and shard logs (no memo, no stages,
#     no npz);
#   - the cohort manifest, the wave's ns receipts and the stage CODE.sha256 / mode files.
# Not archived: envelopes / views / tables, the memos, any sealed / holdout / dev91 path (refused by name).
# Usage: archive_wave.sh WAVE DATE <--dry-run|--upload> [RUN_DIR_NAME ...]      e.g. archive_wave.sh H420 20261111 --dry-run H420_GX_DEV
set -euo pipefail
WAVE=${1:?wave}; DATE=${2:?date}; MODE=${3:?--dry-run|--upload}; shift 3
w=${WAVE,,}
EV=/home/ubuntu/jope_plan/evidence/$WAVE
OUT=/home/ubuntu/jope-data/$w-closeout
BUCKET=veox-jopedime-learning-use2
KEY=experiments/rd_breakthrough_20261030/$WAVE/closeout-$DATE/${w}_closeout_bundle.tar.zst
D0=/home/ubuntu/jope-data/h335-cl
mkdir -p "$OUT"
export AWS_PROFILE=veox-jepson-prod
B=$OUT/${w}_closeout_bundle.tar.zst
L=$OUT/bundle_members.txt
{
  for r in "$@"; do
    [ -d "$D0/runs/$r" ] || { echo "missing run dir $r" >&2; exit 2; }
    find "$D0/runs/$r" -type f ! -path '*/gproot_*' ! -path '*/stages/*' ! -path '*_npz/*' ! -name 'memo*' | sort
  done
  echo "$D0/apex/APEX_COHORTS.json"
  find "$EV/pulled" -type f | sort
  find /home/ubuntu/jope_plan/ns/receipts -maxdepth 1 -name "${WAVE}_*" -type f ! -name '*.lock' | sort
  find "/home/ubuntu/apex-stage/$w" -mindepth 2 -maxdepth 2 \( -name CODE.sha256 -o -name mode -o -name source_commit \) \
    -type f | sort
  find "$EV" -maxdepth 1 -type f \( -name '*.md' -o -name '*.txt' -o -name '*.sh' -o -name '*.json' -o -name '*.diff' \
    -o -name '*.sha256' \) | sort
  find "$EV/results" "$EV/logs" -type f ! -name '.*' 2>/dev/null | sort
} | sort -u | sed 's#^/##' > "$L"
if grep -E '\.jtt1$|views-search/|/views/|dlval-view|/_work/|sealed|holdout|dl-?test|refboard|dev91-audit' "$L"; then
  echo "REFUSING: an envelope / view / sealed / holdout-board path is in the member list" >&2; exit 3
fi
(cd / && xargs -a "$L" -d '\n' sha256sum) > "$OUT/bundle_SHA256SUMS"
echo "members: $(wc -l < "$L")  bytes: $(cd / && xargs -a "$L" -d '\n' stat -c %s | gawk '{s += $1} END {print s}')"
[ "$MODE" = --upload ] || { echo "dry run: nothing uploaded"; exit 0; }
[ ! -e "$B" ] || { echo "bundle $B exists; refusing to overwrite" >&2; exit 4; }
tar -C / -cf - -T "$L" | zstd -q -T8 -12 -o "$B"
SHA=$(sha256sum "$B" | cut -d' ' -f1)
VER=$(aws s3api put-object --bucket "$BUCKET" --key "$KEY" --body "$B" --if-none-match '*' \
        --checksum-algorithm SHA256 --query VersionId --output text)
G=$OUT/get_$VER.tar.zst
aws s3api get-object --bucket "$BUCKET" --key "$KEY" --version-id "$VER" "$G" >/dev/null
[ "$(sha256sum "$G" | cut -d' ' -f1)" = "$SHA" ] || { echo "GET sha mismatch" >&2; exit 5; }
X=$OUT/verify_$VER; mkdir -p "$X"
zstd -dc "$G" | tar -C "$X" -xf -
(cd "$X" && sha256sum --quiet -c "$OUT/bundle_SHA256SUMS")
jq -n --arg uri "s3://$BUCKET/$KEY" --arg v "$VER" --arg s "$SHA" --arg n "$(wc -l < "$L")" --arg wave "$w" \
  --arg b "$(stat -c %s "$B")" --arg u "$(date -u +%FT%TZ)" \
  '{schema:("jope." + $wave + ".archive-manifest/v1"), uri:$uri, version_id:$v, sha256:$s, bytes:($b|tonumber), members:($n|tonumber),
    members_file:"bundle_SHA256SUMS", verified:"EXACT_VERSION_GET_AND_MEMBERS_VERIFIED", utc:$u}' > "$OUT/archive_manifest.json"
cat "$OUT/archive_manifest.json"
