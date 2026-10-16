#!/usr/bin/env bash
# deploy/ci/retain-local-evidence.sh <name> <dir> — keep CI evidence on the self-hosted runner's
# persistent local disk instead of GitHub artifact storage. Destination:
#   ${JOPE_LOCAL_EVIDENCE_ROOT:-/var/cache/jope/target/ci-evidence-retained}/<repo>/<name>/
# (the per-lane persistent target volume). The host, lane, path and a sha256 manifest are printed
# to the job log and step summary so every result stays traceable to a file. If AWS credentials
# and JOPE_EVIDENCE_S3_URI are present on the runner, the same tree is also copied to S3.
set -euo pipefail
name=${1:?name required}; src=${2:?evidence dir required}
[[ $name =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid evidence name: $name" >&2; exit 1; }
[ -d "$src" ] || { echo "no evidence directory: $src" >&2; exit 1; }
root=${JOPE_LOCAL_EVIDENCE_ROOT:-/var/cache/jope/target/ci-evidence-retained}
repo=${GITHUB_REPOSITORY:-local}; dest="$root/${repo//\//_}/$name"
manifest=$(mktemp)
trap 'rm -f "$manifest"' EXIT
(cd "$src" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum) > "$manifest"
# The cache is only the hand-off, not durable custody. A missing or corrupt local
# copy must fail the job rather than report an unavailable location as success.
mkdir -p "$dest"
cp -a "$src/." "$dest/"
cp "$manifest" "$dest/MANIFEST.sha256"
(cd "$dest" && sha256sum --check MANIFEST.sha256)
where="${JOPE_RUNNER_HOST:-$(hostname -s)}:${JOPE_RUNNER_LANE:-unknown}:$dest"
s3=""
# S3 is optional, but once JOPE_EVIDENCE_S3_URI is set a failed copy is a failed job.
if [ -n "${JOPE_EVIDENCE_S3_URI:-}" ]; then
  command -v aws >/dev/null 2>&1 || { echo "JOPE_EVIDENCE_S3_URI set but aws CLI missing" >&2; exit 1; }
  aws sts get-caller-identity >/dev/null || { echo "JOPE_EVIDENCE_S3_URI set but no usable AWS credentials" >&2; exit 1; }
  s3="$JOPE_EVIDENCE_S3_URI/${repo//\//_}/$name/"
  aws s3 cp --recursive --no-follow-symlinks --only-show-errors "$src" "$s3"
fi
# Bounded disk: retained evidence older than JOPE_LOCAL_EVIDENCE_DAYS (default 14) is pruned,
# and every removal is logged.
days=${JOPE_LOCAL_EVIDENCE_DAYS:-14}
[[ $days =~ ^[0-9]+$ ]] || { echo "invalid JOPE_LOCAL_EVIDENCE_DAYS: $days" >&2; exit 1; }
find "$root/${repo//\//_}" -mindepth 1 -maxdepth 1 -type d -mtime "+$days" -print -exec rm -rf -- {} + | sed 's/^/pruned retained evidence older than '"$days"'d: /'
echo "evidence $name retained at $where ${s3:+and $s3}"
echo "sha256 manifest:"; cat "$manifest"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  { echo "### Evidence \`$name\`"; echo "- local: \`$where\`"; [ -z "$s3" ] || echo "- s3: \`$s3\`"
    echo '```'; cat "$manifest"; echo '```'; } >> "$GITHUB_STEP_SUMMARY"
fi
