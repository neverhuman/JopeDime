#!/bin/bash
# H335 Phase T on xbabe2 (PREREG §3.2): P201b contexts with sorted index % 4 == 0; 2 AGG + 2 BASE shards x 8 workers
# on cpus 38-53,102-117 (16 physical cores, both SMT siblings), nice 19, bwrap sandbox.
# Launched as: systemd-run --user --unit h335-T-xbabe2 -p MemoryMax=48G -p MemorySwapMax=0 -p CPUQuota=3200% \
#              <H335 source dir>/run_T_xbabe2.sh
set -euo pipefail
SRC=/home/ubuntu/jope_plan/evidence/H335/source/research/flywheel_selfplay/h335_aggressive_locked_elo
DATA=/home/ubuntu/jope-data/h335-cl
RUN=$DATA/runs/T_xbabe2
mkdir -p "$DATA/runs"
export H335_CODE=/home/ubuntu/h335-stage/code   # frozen read-only copy, identical to the xbabe0 stage
cd "$H335_CODE"
exec nice -n 19 ./sandbox_h335.sh python3 -u h335_lab.py launch --phase T --run-dir "$RUN" \
  --cpus 38-53,102-117 --shards-agg 2 --shards-base 2 --workers-per-shard 8 --host-part 0/4 --jobs 12
