#!/bin/bash
# H354 library sandbox (bwrap, xbabe2): no network, own PID namespace, / read-only, /home/ubuntu and /tmp EMPTY tmpfs.
# Bound back read-only: ~/.local (system-python libraries: catboost, lightgbm, xgboost, scikit-learn), the reference-
# board source (score_reference.py + models.json), this H354 source dir, the pinned frontier binary, PHASE_ROOT/inputs.
# Read-write: PHASE_ROOT/lib only. Nothing else under /home/ubuntu is visible: no H335 run dir, no DL-Val / dev91 /
# sealed / holdout path, no other board.
# Usage: sandbox_h354.sh PHASE_ROOT <command...>
set -euo pipefail
PR=$(readlink -f "$1"); shift
SRCDIR=$(dirname "$(readlink -f "$0")")
RB=${H354_REFBOARD:-/home/ubuntu/jope_plan/evidence/REFBOARD/source/research/elo/reference-board}
BIN=${H354_FRONTIER_BIN:-/home/ubuntu/jope-data/h335-cl/build/frontier_audit_h335}
[ -s "$PR/inputs/MANIFEST.sha256" ] || { echo "no inputs/MANIFEST.sha256 under $PR" >&2; exit 2; }
mkdir -p "$PR/lib"
TMPFS=()
for d in /mnt /media /srv /app /review /Users /opt/work; do [ -d "$d" ] && TMPFS+=(--tmpfs "$d"); done
bwrap --ro-bind / / --unshare-net true 2>/dev/null || { echo "bwrap with a private network is required" >&2; exit 4; }
exec bwrap --ro-bind / / \
  --tmpfs /home/ubuntu --tmpfs /tmp "${TMPFS[@]}" \
  --dev /dev --proc /proc \
  --ro-bind /home/ubuntu/.local /home/ubuntu/.local \
  --ro-bind "$RB" "$RB" \
  --ro-bind "$SRCDIR" "$SRCDIR" \
  --ro-bind "$BIN" "$BIN" \
  --ro-bind "$PR/inputs" "$PR/inputs" \
  --bind "$PR/lib" "$PR/lib" \
  --unshare-net --unshare-pid --die-with-parent \
  --setenv HOME /home/ubuntu --setenv PYTHONDONTWRITEBYTECODE 1 \
  --setenv H354_REFBOARD_TOOLS "$RB/tools" --setenv H354_FRONTIER_BIN "$BIN" \
  "$@"
# Thread counts come from models.json (8 per classical fit), exactly as in the reference-board runs (run_board.sh sets
# no OpenMP/BLAS variables either).
