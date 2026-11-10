#!/bin/bash
# H420 deploy chain for one panel in one run directory: replay check, then variants B, TGX, TLIB, TBOTH in order.
#   h420_chain.sh STAGE bwrap|docker CPUSET MEM PANEL RUN_DIR JOBS [REPLAY_LIMIT]
set -euo pipefail
STAGE=${1:?}; KIND=${2:?}; CPUS=${3:?}; MEM=${4:?}; PANEL=${5:?}; R=${6:?}; JOBS=${7:?}; LIMIT=${8:-4}
run() {
  if [ "$KIND" = bwrap ]; then bash "$STAGE/code/run_h420_local.sh" "$STAGE" "$CPUS" "$@"
  else bash "$STAGE/code/run_apex_xbabe0.sh" "$STAGE" "$CPUS" "$MEM" "$@"; fi
}
run train_h420.py replaycheck --run-dir "$R" --panel "$PANEL" --cores "$CPUS" --limit "$LIMIT" --jobs 12
for v in ${H420_VARIANTS:-B TGX TLIB TBOTH}; do
  echo "$(date -u +%FT%TZ) variant $v"
  run train_h420.py deploy --run-dir "$R" --panel "$PANEL" --variant "$v" --cores "$CPUS" --jobs "$JOBS"
done
echo "$(date -u +%FT%TZ) chain done"
