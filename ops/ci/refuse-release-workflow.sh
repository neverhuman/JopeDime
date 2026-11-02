#!/usr/bin/env bash
# Releases are cut by deploy/release/release.sh; this legacy workflow is fail-closed.
set -euo pipefail
echo "::error title=release.yml is not the release path::DISPATCH-ONLY. THIS IS NOT THE RELEASE PATH. Releases are cut on xbabe2 by deploy/release/release.sh (owner decision D-D); docs/release/README.md explains how to verify one. This workflow stays only as a reference until the self-hosted release lane lands."
exit 1
