#!/usr/bin/env bash
# deploy/runner/prune.sh — nightly reclaim (jope-runner-prune.timer): dangling images and build
# cache older than 7 days, stopped runner containers, and `cargo sweep --time 7` on the shared
# target volume. Never touches the SDK or models volumes.
set -euo pipefail
# shellcheck source=deploy/runner/common.sh
# shellcheck disable=SC1091  # sourced relative to this file; run shellcheck -x to follow
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "prune: $(date -u +%FT%TZ)"
docker container prune -f --filter "label=jope.runner=1" --filter "until=24h"
docker image prune -f --filter "until=168h"
docker builder prune -f --filter "until=168h"
# cargo sweep inside the runner image so the sweep sees the same rustc versions the lanes used.
docker run --rm --user "$JOPE_RUNNER_UID" --network none \
  --cap-drop ALL --security-opt no-new-privileges --read-only --tmpfs /tmp:size=1g \
  -v jope-cargo:/var/cache/jope/cargo -v jope-target:/var/cache/jope/target \
  --entrypoint cargo "$JOPE_RUNNER_IMAGE" sweep --time 7 /var/cache/jope/target || echo "prune: cargo sweep failed (non-fatal)"
mkdir -p "$JOPE_RUNNER_STATE"
date -u +%FT%TZ > "$JOPE_RUNNER_STATE/last-prune"
docker system df
