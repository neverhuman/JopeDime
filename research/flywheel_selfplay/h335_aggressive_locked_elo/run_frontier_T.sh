#!/bin/bash
# H335 Phase T (descriptive): the 4 LLM Rust regressors, fit SEARCH -> predict AUDIT, on P201b; xbabe2 bwrap sandbox.
# Launched as: systemd-run --user --unit h335-frontier-T -p MemoryMax=24G -p MemorySwapMax=0 -p CPUQuota=800% <this>
set -euo pipefail
SRC=/home/ubuntu/jope_plan/evidence/H335/source/research/flywheel_selfplay/h335_aggressive_locked_elo
cd "$SRC"
exec nice -n 19 taskset -c 54-57,118-121 ./sandbox_h335.sh env H335_GROUP=BASE python3 -u h335_frontier.py \
  --phase T --out-dir /home/ubuntu/jope-data/h335-cl/frontier/T --jobs 8
