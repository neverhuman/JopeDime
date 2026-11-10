#!/bin/bash
# H357 GP on xbabe0 (H357 PREREG §3): the frozen H335 AGG recipe on P201c, seed 357101, g50; 12 AGG shards x 8 workers.
# Staged into the GP code root by stage_gpc_h357.sh; launched by `run_c_h357.sh launch` as
#   systemd-run --user --unit h357-C-gp -p MemoryMax=100G -p MemorySwapMax=0 <gpc>/code/run_gp_c_xbabe0.sh
set -euo pipefail
STAGE=$(dirname "$(dirname "$(readlink -f "$0")")")
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256)
[ -s "$STAGE/PREREG_ON_V1NEXT" ] || { echo "no PREREG_ON_V1NEXT marker (run_c_h357.sh stage writes it)" >&2; exit 3; }
export H335_GROUP=AGG H335_STAGE=$STAGE H335_CPUSET=16-63,80-127 H335_MEMORY=96g
echo "start $(date -u +%FT%TZ) stage $STAGE CODE.sha256 $(sha256sum "$STAGE/CODE.sha256" | cut -d' ' -f1)"
exec "$STAGE/code/sandbox_h335_docker.sh" python3 -u gp_c_h357.py launch \
  --run-dir /home/ubuntu/jope-data/h335-cl/runs/H357_C_xbabe0 --cpus 16-63,80-127 --shards 12 --workers-per-shard 8 \
  --jobs 12
