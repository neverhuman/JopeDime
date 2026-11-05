#!/bin/bash
# Second xbabe2 CPU driver, started 2026-11-05 ~06:10Z so the P201b TRAIN board runs in parallel with the dev91 board
# on otherwise idle cores. The dev91 driver's later TRAIN step finds this driver's board lock and exits.
set -euo pipefail
RB=/home/ubuntu/jope_plan/evidence/REFBOARD/source/research/elo/reference-board
M="ridge.v1 random_forest.v1 xgboost.v1 lightgbm.v1 catboost.v1 tabpfn_v3p5.v1 tabpfn_v3.v1 kumo_tabular_large_e8.v1 autogluon_300s.v1"
export REFBOARD_CPUS=54-59,61,64-71,73,75-77,79-83,118-124,126,127
"$RB/tools/run_board.sh" /home/ubuntu/jope-data/h335-cl/refboard/p201b-train-audit-v1 4 $M
echo TRAIN_PARALLEL_DONE
