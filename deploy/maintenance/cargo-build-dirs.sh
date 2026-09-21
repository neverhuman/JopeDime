#!/usr/bin/env bash
# Report candidates only. Cargo markers never establish that contents are disposable.
set -euo pipefail
exec python3 "$(dirname -- "$(realpath -- "$0")")/jope_report.py" scan "$@"
