#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck source=ops/ci/lib.sh
source ops/ci/lib.sh
jope_ci_environment
[[ $# == 1 ]] || { echo 'usage: fast-lane.sh LANE' >&2; exit 2; }
case "$1" in
  changes|hygiene|rust-check|rust-test|emitters|inputs|python|e2e-smoke) ;;
  *) echo "unknown fast CI lane: $1" >&2; exit 2 ;;
esac
exec bash deploy/ci/fast-lane.sh "$1"
