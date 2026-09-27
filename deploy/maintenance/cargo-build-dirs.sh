#!/usr/bin/env bash
# Report candidates only. Cargo markers never establish that contents are disposable.
# The report core is `jope-ci build-report`, installed beside this wrapper (README.md).
set -euo pipefail
exec "$(dirname -- "$(realpath -- "$0")")/jope-ci" build-report scan "$@"
