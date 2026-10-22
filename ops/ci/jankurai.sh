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
read -r manifest_sha256 _ < <(sha256sum -- "$output/input/SOURCE_MANIFEST.json")
status=0
"$auditor" audit "$output/input/source" --full --read-only --no-badge --no-score-history --mode standard \
  --json "$output/repo-score.json" --md "$output/repo-score.md" \
  --timings-json "$output/timings.json" \
  --github-step-summary "$output/summary.md" \
  --repair-queue-jsonl "$output/repair-queue.jsonl" || status=$?
verification_status=0
python3 -B deploy/ci/verify-audit-report.py "$output" --audit-exit "$status" \
  --expected-manifest-sha256 "$manifest_sha256" || verification_status=$?
if [[ -f "$output/summary.md" && -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  cat "$output/summary.md" >> "$GITHUB_STEP_SUMMARY"
fi
# JOPE_AUDIT_REPORT_ONLY=1 (trusted main, until the score reaches the floor) passes only an
# auditor policy failure (exit 1) below the 85 floor whose reports and captured input verified;
# the failed decision stays in the reports. A failure at or above the floor (for example hard
# findings), and every refusal, inconsistency or tool error, still fails. The captured source
# stays read-only here so retained evidence replays; the workflow unlocks it after upload.
score=-1
if [[ "$status" == 1 && "$verification_status" == 0 && "${JOPE_AUDIT_REPORT_ONLY:-}" == 1 ]]; then
  score=$(python3 -B -c 'import json, sys; print(json.load(open(sys.argv[1]))["score"])' "$output/audit-verification.json")
fi
if (( score >= 0 && score < 85 )); then
  printf '{"schema": "jope.audit-report-only/v1", "audit_exit": 1, "audit_passed": false, "score": %s, "minimum_score": 85, "job_result": "report-only pass"}\n' "$score" > "$output/report-only.json"
  echo "::warning title=Audit below floor (report-only)::score $score/85; the failed decision is retained in repo-score.json"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    printf '\n**Report-only:** score %s/85 is below the floor; this job passes until the floor is reached.\n' "$score" >> "$GITHUB_STEP_SUMMARY"
  fi
  exit 0
fi
if [[ "$status" != 0 ]]; then exit "$status"; fi
exit "$verification_status"
