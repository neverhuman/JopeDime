#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:-}" in
  prepare)
    : "${NODE_VERSION:?NODE_VERSION is required}"
    test "$(node --version)" = "v$NODE_VERSION"
    bash deploy/ci/typescript/compiler-tools.sh
    npm ci --prefix deploy/ci/typescript --ignore-scripts --no-audit --no-fund
    ;;
  source-check)
    git diff --check
    cargo fmt --all --check
    cargo check --locked -p feat-math -p jope-cli -p jope-pipeline --no-default-features --all-targets
    ;;
  qualify) bash deploy/ci/typescript/qualify.sh ;;
  browser) bash deploy/ci/typescript/browser-tools.sh ;;
  evidence) node deploy/ci/typescript/evidence.mjs create "${2:?evidence directory is required}" ;;
  *) echo 'usage: typescript.sh prepare|source-check|qualify|browser|evidence [directory]' >&2; exit 2 ;;
esac
