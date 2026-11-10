#!/bin/bash
# Owner-authorized resume (2026-11-05 07:2xZ) on dev91-audit-v1 ONLY: the two unfinished CPU reference models.
# The 8 complete models (91/91) are not passed, and run_board.sh skips every (model, context) that already has a
# record, so nothing scored is run again. Launch on xbabe2:
#   systemd-run --user --unit refboard-dev91-resume-cpu -p CPUQuota=3200% -p MemoryMax=64G -p MemorySwapMax=0 \
#     /bin/bash <this file>
set -euo pipefail
RB=/home/ubuntu/jope_plan/evidence/REFBOARD/source/research/elo/reference-board
export REFBOARD_CPUS=38-53,102-117
"$RB/tools/run_board.sh" /home/ubuntu/jope-data/h335-stagev/refboard/dev91-audit-v1 4 \
  kumo_tabular_large_e8.v1 autogluon_300s.v1
echo DEV91_RESUME_CPU_DONE
