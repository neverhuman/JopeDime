#!/bin/bash
# One-off helper for the H450 epoch-2 dev91 eval (cl_fwriter, 2026-11-13 ~00:15Z).
#
# Two 3090s give byte-identical columns for the same context: the baseline context 016648dc, recomputed on xbabe3,
# matched the copy computed on xbabe2 (cmp). This helper therefore has xbabe3, idle since its shard 1/2 finished at
# 00:07:56Z, compute xbabe2's shard-0 contexts from the highest index down.
#
# It runs the published cols_h450_gpu.py (pub-r10 stage on xbabe3) one context at a time (--num-shards 91 --shard-idx I)
# and copies each result into the local column dir. The local shard-0 process skips any context whose file already
# exists. The helper stops at the first context the local process has already written.
set -euo pipefail
TAG=g02_r4_L_A_r4_L_B
OUTC=/home/ubuntu/jope-data/h450-quarantine/r4/cols/$TAG
CIDS=/tmp/claude-1000/-home-ubuntu-JopeDime/624cd2df-7bb6-4462-bf5a-9408d45ab70d/scratchpad/cids_sorted.txt
XE=/home/ubuntu/jope-data/h450-r4-x3/eval
CODE=/home/ubuntu/jope-data/h450-r4-x3/stage/pub-r10/code
for line in $(seq 91 -2 1); do
  idx=$((line - 1)); f=$(sed -n "${line}p" "$CIDS")
  [ -e "$OUTC/$f" ] && { echo "$(date -u +%T) idx $idx already local: stop"; break; }
  ssh -o BatchMode=yes xbabe3 "cd $CODE && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 nice -n 10 python3 -B cols_h450_gpu.py \
    --bank $XE/bank_d91_search_f64.npz --ckpt-la $XE/ckpts/$TAG/la.pt --ckpt-ma $XE/ckpts/$TAG/m.pt --ckpt-lb $XE/ckpts/$TAG/lb.pt \
    --ckpt-mb $XE/ckpts/$TAG/m.pt --out $XE/cols/$TAG --device cuda --num-shards 91 --shard-idx $idx" | tail -1
  rsync -a --ignore-existing -e "ssh -o BatchMode=yes" "xbabe3:$XE/cols/$TAG/$f" "$OUTC/"
  echo "$(date -u +%T) idx $idx ($f) from xbabe3"
done
