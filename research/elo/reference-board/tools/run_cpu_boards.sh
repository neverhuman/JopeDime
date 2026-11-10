#!/bin/bash
# Production CPU scoring of the reference boards on xbabe2 (dev91 test board first, then the P201b TRAIN board).
set -euo pipefail
RB=/home/ubuntu/jope_plan/evidence/REFBOARD/source/research/elo/reference-board
M="ridge.v1 random_forest.v1 xgboost.v1 lightgbm.v1 catboost.v1 tabpfn_v3p5.v1 tabpfn_v3.v1 kumo_tabular_large_e8.v1 autogluon_300s.v1"
export REFBOARD_CPUS=22-53,86-117
"$RB/tools/run_board.sh" /home/ubuntu/jope-data/h335-stagev/refboard/dev91-audit-v1 7 $M
"$RB/tools/run_board.sh" /home/ubuntu/jope-data/h335-cl/refboard/p201b-train-audit-v1 7 $M
echo ALL_CPU_BOARDS_DONE
