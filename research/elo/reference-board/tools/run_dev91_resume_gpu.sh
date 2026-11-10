#!/bin/bash
# Owner-authorized resume (2026-11-05 07:2xZ) on dev91-audit-v1 ONLY: the unfinished GPU reference model.
# tabfm.v1 is complete (91/91) and is not passed; run_board.sh skips every (model, context) already recorded.
# xbabe3 has no unprivileged bwrap -> REFBOARD_ALLOW_NOSANDBOX=1 (README section 8). Launch on xbabe3:
#   systemd-run --user --unit refboard-dev91-resume-gpu -p CPUQuota=800% -p MemoryMax=48G -p MemorySwapMax=0 \
#     /bin/bash <this file>
set -euo pipefail
RB=/home/ubuntu/jope_plan/evidence/REFBOARD/source/research/elo/reference-board
export REFBOARD_GPU=1 REFBOARD_ALLOW_NOSANDBOX=1 REFBOARD_CPUS=120-127
"$RB/tools/run_board.sh" /home/ubuntu/jope-data/h335-stagev/refboard/dev91-audit-v1 2 tabfm_ensemble.v1
echo DEV91_RESUME_GPU_DONE
