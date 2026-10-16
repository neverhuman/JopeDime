#!/usr/bin/env bash
# deploy/ci/retain-local-evidence.sh <name> <dir> — keep CI evidence on the self-hosted runner's
# persistent local disk instead of GitHub artifact storage. Destination:
#   ${JOPE_LOCAL_EVIDENCE_ROOT:-/var/cache/jope/target/ci-evidence-retained}/<repo>/<name>/
# (the per-lane persistent target volume). The host, lane, path and a sha256 manifest are printed
# to the job log and step summary so every result stays traceable to a file. If AWS credentials
# and JOPE_EVIDENCE_S3_URI are present on the runner, the same tree is also copied to S3.
set -euo pipefail
name=${1:?name required}; src=${2:?evidence dir required}
[ -d "$src" ] || { echo "no evidence directory: $src" >&2; exit 1; }
root=${JOPE_LOCAL_EVIDENCE_ROOT:-/var/cache/jope/target/ci-evidence-retained}
repo=${GITHUB_REPOSITORY:-local}; dest="$root/${repo//\//_}/$name"
manifest=$(mktemp)
(cd "$src" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum) > "$manifest"
if mkdir -p "$dest" 2>/dev/null && cp -a "$src/." "$dest/"; then
  cp "$manifest" "$dest/MANIFEST.sha256"
  where="${JOPE_RUNNER_HOST:-$(hostname -s)}:${JOPE_RUNNER_LANE:-unknown}:$dest"
else
  where="(local retention unavailable on this runner)"
fi
s3=""
if [ -n "${JOPE_EVIDENCE_S3_URI:-}" ] && command -v aws >/dev/null 2>&1 && aws sts get-caller-identity >/dev/null 2>&1; then
  aws s3 cp --recursive --only-show-errors "$src" "$JOPE_EVIDENCE_S3_URI/${repo//\//_}/$name/" && s3="$JOPE_EVIDENCE_S3_URI/${repo//\//_}/$name/"
fi
echo "evidence $name retained at $where ${s3:+and $s3}"
echo "sha256 manifest:"; cat "$manifest"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  { echo "### Evidence \`$name\`"; echo "- local: \`$where\`"; [ -z "$s3" ] || echo "- s3: \`$s3\`"
    echo '```'; cat "$manifest"; echo '```'; } >> "$GITHUB_STEP_SUMMARY"
fi
rm -f "$manifest"
