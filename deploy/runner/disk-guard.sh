#!/usr/bin/env bash
# deploy/runner/disk-guard.sh — refuse to start a runner when the docker root or the state
# directory has less than JOPE_RUNNER_MIN_FREE_GB free (default 50 GB). Exit 0 = enough space.
set -euo pipefail
# shellcheck source=deploy/runner/common.sh
# shellcheck disable=SC1091  # sourced relative to this file; run shellcheck -x to follow
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

docker_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
status=0
for path in "$docker_root" "${JOPE_RUNNER_STATE}" /tmp; do
  [[ -d $path ]] || continue
  free_kb=$(df -Pk "$path" | awk 'NR==2{print $4}')
  free_gb=$((free_kb / 1024 / 1024))
  if (( free_gb < JOPE_RUNNER_MIN_FREE_GB )); then
    echo "disk-guard: $path has ${free_gb} GB free < ${JOPE_RUNNER_MIN_FREE_GB} GB — refusing to start (inspect storage and follow the custody cleanup procedure)" >&2
    status=1
  else
    echo "disk-guard: $path ${free_gb} GB free (min ${JOPE_RUNNER_MIN_FREE_GB})"
  fi
done
exit $status
