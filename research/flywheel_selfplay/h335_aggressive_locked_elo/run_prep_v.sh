#!/bin/bash
# H335 Stage V prep on xbabe2 (stage_v_mounts_h335.md S4): only after the unlock (S2) and the V root build (S3).
# Runs from the frozen read-only code copy; prints counts only. Re-launch with --resume after an interruption (any
# number of times before FIRST_OUTCOME_READ; recorded steps, incl. the R882 regression result, are never recomputed).
# Launched as (a script file: no ${var} expansion by systemd):
#   systemd-run --user --unit h335-v-prep -p MemoryMax=32G -p MemorySwapMax=0 -p CPUQuota=2000% \
#     /home/ubuntu/h335-stage/code/run_prep_v.sh [--resume]
set -euo pipefail
export H335_CODE=/home/ubuntu/h335-stage/code
cd "$H335_CODE"
exec nice -n 19 taskset -c 38-53,102-117 ./sandbox_h335_stagev.sh v \
  python3 -u stage_v_h335.py prep --threads 4 --llm-jobs 16 "$@"
