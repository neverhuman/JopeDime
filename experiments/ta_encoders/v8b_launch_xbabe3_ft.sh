#!/bin/bash
# v8b-ft on xbabe3 (moved from xbabe1 at 21:25Z: its full-SM trunk pass made hv8r2's guard throttle hv8r2 from ~260 to ~50 ctx/s).
# Waits for dv2r2 to finish so dv2r2 is not throttled either; everything under /dev/shm (no xbabe3 root writes).
cd /dev/shm/ta_encoders/ta-20261007T2030Z
while pgrep -u ubuntu -f "train_ta.py --model dv2 " >/dev/null || pgrep -u ubuntu -f "train_ta.py --model dv2$" >/dev/null || pgrep -u ubuntu -f "model dv2 --data" > /dev/null; do sleep 20; done
echo "$(date -u +%FT%TZ) dv2r2 gone; starting v8bft" >> v8bft.launch.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
nice -n 5 python3 code-v8b/train_ta.py --model hv8ft --out v8bft --epochs 100 --k-train 2 --trunk-lr 2e-5 --eval-batch 8 \
  --data data --probe-cids meta/probe_cids.json --probe-data /home/ubuntu/jopedime_probe_exp/20261007T1710Z/data \
  --probe-cache /dev/shm/ta_encoders/ta-20261007T2030Z/probe_cache --exclude meta/exclude_sealed.json --probe-threads 6 \
  --seed 0 --batch 16 --lr 0.0003 --eval-min 20 --pcap 128 --nmax 384 --max-hours 12 --guard-siblings hv8raw,hv8ft > v8bft.log 2>&1
