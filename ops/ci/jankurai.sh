#!/usr/bin/env bash
# Full default-policy audit of immutable, complete tracked product input.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $# == 1 ]] || { echo 'usage: jankurai.sh NEW_REPORT_DIRECTORY' >&2; exit 2; }
auditor=${JANKURAI_BIN:-jankurai}
[[ "$("$auditor" --version)" == 'jankurai 1.7.5' ]] || { echo 'audit requires pinned jankurai 1.7.5' >&2; exit 2; }
output=$(realpath -m -- "$1")
[[ ! -e "$output" && ! -L "$output" ]] || { echo 'audit report directory must be fresh' >&2; exit 2; }
mkdir -p -- "$output"
export GIT_OPTIONAL_LOCKS=0
python3 deploy/ci/audit-source.py --repo "$PWD" --output "$output/input"
status=0
"$auditor" audit "$output/input/source" --full --read-only --no-badge --no-score-history --mode standard \
  --json "$output/repo-score.json" --md "$output/repo-score.md" \
  --timings-json "$output/timings.json" \
  --github-step-summary "$output/summary.md" \
  --repair-queue-jsonl "$output/repair-queue.jsonl" || status=$?
if [[ -f "$output/summary.md" && -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  cat "$output/summary.md" >> "$GITHUB_STEP_SUMMARY"
fi
exit "$status"
