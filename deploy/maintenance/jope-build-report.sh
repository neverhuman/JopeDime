#!/usr/bin/env bash
# Save a report and its diagnostics outside working trees. No reclamation operation.
set -euo pipefail
exec python3 "$(dirname -- "$(realpath -- "$0")")/jope_report.py" report "$@"
