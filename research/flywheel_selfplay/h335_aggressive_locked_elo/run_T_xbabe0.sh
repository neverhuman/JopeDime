#!/bin/bash
# H335 Phase T on xbabe0 (PREREG §3.2): P201b contexts with sorted index % 4 in {1,2,3}; 6 AGG + 6 BASE shards x 8.
# Launched as: systemd-run --user --unit h335-T-xbabe0 -p MemoryMax=100G -p MemorySwapMax=0 \
#              /home/ubuntu/h335-stage/code/run_T_xbabe0.sh
set -euo pipefail
CODE=/home/ubuntu/h335-stage/code
DATA=/home/ubuntu/jope-data/h335-cl
RUN=$DATA/runs/T_xbabe0
mkdir -p "$DATA/runs"
export H335_CPUSET=16-63,80-127 H335_MEMORY=96g
exec "$CODE/sandbox_h335_docker.sh" python3 -u h335_lab.py launch --phase T --run-dir "$RUN" \
  --cpus 16-63,80-127 --shards-agg 6 --shards-base 6 --workers-per-shard 8 --host-part 1,2,3/4 --jobs 12
