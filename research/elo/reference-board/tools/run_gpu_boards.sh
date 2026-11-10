#!/bin/bash
# Production GPU scoring (TabFM) of the reference boards on xbabe3: dev91 test board first, then the P201b TRAIN board.
# No bwrap on xbabe3 (unprivileged user namespaces blocked) -> REFBOARD_ALLOW_NOSANDBOX=1, documented in README.md.
set -euo pipefail
RB=/home/ubuntu/jope_plan/evidence/REFBOARD/source/research/elo/reference-board
export REFBOARD_GPU=1 REFBOARD_ALLOW_NOSANDBOX=1 REFBOARD_CPUS=120-127
"$RB/tools/run_board.sh" /home/ubuntu/jope-data/h335-stagev/refboard/dev91-audit-v1 2 tabfm.v1 tabfm_ensemble.v1
"$RB/tools/run_board.sh" /home/ubuntu/jope-data/h335-cl/refboard/p201b-train-audit-v1 2 tabfm.v1 tabfm_ensemble.v1
echo ALL_GPU_BOARDS_DONE
