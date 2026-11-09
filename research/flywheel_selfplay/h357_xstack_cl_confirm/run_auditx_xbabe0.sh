#!/bin/bash
# H357 Amendment 1 on xbabe0: AUDIT predictions for GP members that X_CL uses but X0 gave zero weight (gp_c_h357.py
# audit-extra), same frozen engine / worker / timeout as the GP run. Launched by `run_c_h357.sh auditx` as
#   systemd-run --user --unit h357-C-auditx -p MemoryMax=100G -p MemorySwapMax=0 <gpc>/code/run_auditx_xbabe0.sh
set -euo pipefail
STAGE=$(dirname "$(dirname "$(readlink -f "$0")")")
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256)
[ -s "$STAGE/PREREG_ON_V1NEXT" ] || { echo "no PREREG_ON_V1NEXT marker" >&2; exit 3; }
DATA=/home/ubuntu/jope-data/h335-cl
export H335_GROUP=AGG H335_STAGE=$STAGE H335_CPUSET=16-63,80-127 H335_MEMORY=96g
echo "start $(date -u +%FT%TZ) stage $STAGE CODE.sha256 $(sha256sum "$STAGE/CODE.sha256" | cut -d' ' -f1)"
exec "$STAGE/code/sandbox_h335_docker.sh" python3 -u gp_c_h357.py audit-extra \
  --request "$DATA/h357/audit_extra_request.json" --gp-run-dir "$DATA/runs/H357_C_xbabe0" \
  --run-dir "$DATA/runs/H357_C_xbabe0_auditx" --cores 16-63,80-127 --jobs 48
