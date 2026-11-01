#!/usr/bin/env bash
set -euo pipefail
{
  echo '### InnovationZero publication outputs'
  echo 'The source-bound receipt contains the three PDFs, TeX logs, build environment, and check output.'
  echo 'Evidence uses the existing run/attempt-scoped Actions cache transport; GitHub artifact storage is not used.'
  echo 'The upload step records the cache publication and SHA-256 manifest. Restore it with the repository download-ci-evidence action using the same name and path.'
  echo 'Runner-local copies are temporary; cache availability is subject to eviction.'
} >> "$GITHUB_STEP_SUMMARY"
