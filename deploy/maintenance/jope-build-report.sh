#!/usr/bin/env bash
# Save a report and its diagnostics outside working trees. No reclamation operation.
# The report core is `jope-ci build-report`, installed beside this wrapper (README.md); the
# receipt records this directory's wrapper checksums. The two approved Cargo roots are host
# layout, so they live here (as they did in the retired jope_report.py), not in crates/
# (ADR-0003); the report refuses any configuration that does not name exactly these two.
set -euo pipefail
here=$(dirname -- "$(realpath -- "$0")")
exec "$here/jope-ci" build-report report --tools-dir "$here" \
  --approved-root /home/ubuntu/JopeDime \
  --approved-root /home/ubuntu/jain-split/jope-dime \
  "$@"
