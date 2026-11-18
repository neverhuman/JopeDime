#!/usr/bin/env bash
# Keep the live GITHUB_STEP_SUMMARY within GitHub's 1024 KB job-summary limit.
# Call this after the full bytes have been copied into published evidence.
# Run 44364383884 aggregate (job 135065988632) reported 3478 KB and the upload aborted.
set -euo pipefail
limit=900000
summary=${GITHUB_STEP_SUMMARY:-}
[ -n "$summary" ] && [ -f "$summary" ] || exit 0
bytes=$(wc -c < "$summary" | tr -d ' ')
if (( bytes <= limit )); then
  exit 0
fi
run=${GITHUB_RUN_ID:-}
attempt=${GITHUB_RUN_ATTEMPT:-}
where='`step-summary.md` in the ci-aggregate evidence'
if [ -n "$run" ] && [ -n "$attempt" ]; then
  where="\`step-summary.md\` in \`ci-aggregate-${run}-${attempt}\`"
fi
tmp=$(mktemp)
{
  echo '### Aggregate step summary'
  echo "The complete summary is ${bytes} bytes, above GitHub's 1024 KB limit, and is retained as ${where}."
  echo
  echo '_(Live step summary replaced so the job summary upload stays under 900 KB.)_'
} > "$tmp"
cat "$tmp" > "$summary"
rm -f "$tmp"
