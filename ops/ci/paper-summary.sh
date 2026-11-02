#!/usr/bin/env bash
set -euo pipefail
{
  echo '### InnovationZero publication source checks'
  echo 'The source-bound receipt contains archive, citation, derived-table and refusal-check output. CI does not typeset PDFs or install TeX.'
  echo 'Generate the three publication PDFs explicitly with just paper; PDF rendering is separate from product CI.'
  echo 'Evidence uses the existing run/attempt-scoped Actions cache transport; GitHub artifact storage is not used.'
  echo 'The upload step records the cache publication and SHA-256 manifest. Restore it with the repository download-ci-evidence action using the same name and path.'
  echo 'Runner-local copies are temporary; cache availability is subject to eviction.'
} >> "$GITHUB_STEP_SUMMARY"
