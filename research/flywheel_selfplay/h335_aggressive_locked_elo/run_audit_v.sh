#!/bin/bash
# H335 Stage V audit units on xbabe0 (stage_v_mounts_h335.md S6): only after `stage_v_xbabe0.sh prep` succeeded.
# One docker container (--stage-v); inside it one audit-shard process per (group, shard), H335_GROUP set per process.
# Prints counts only. After an interruption: the same unit with --resume (recorded units replay; same shard layout).
# Launched on xbabe0 as (a script file: no ${var} expansion by systemd):
#   systemd-run --user --unit h335-v-audit -p MemoryMax=100G -p MemorySwapMax=0 \
#     /home/ubuntu/h335-stage/code/run_audit_v.sh [--resume]
# MemoryMax caps the docker client; the container itself is capped by H335_MEMORY (docker --memory).
set -euo pipefail
CODE=/home/ubuntu/h335-stage/code
export H335_STAGE=/home/ubuntu/h335-stage H335_STAGE_V=/home/ubuntu/h335-stage-v
export H335_CPUSET=16-63,80-127 H335_MEMORY=96g
exec "$CODE/sandbox_h335_docker.sh" --stage-v python3 -u stage_v_h335.py audit \
  --cpus 16-63,80-127 --shards-agg 6 --shards-base 6 --workers-per-shard 8 --jobs 12 "$@"
