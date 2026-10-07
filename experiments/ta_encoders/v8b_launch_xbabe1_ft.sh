#!/bin/bash
# v8b-ft on xbabe1 (final placement). Waits for the hv8r2 process (pid 3568570, "--model hv8 --data ... --out hv8r2") to
# exit so hv8r2's guard is not tripped (on the shared 4090 the ft trunk pass throttled hv8r2 ~5x at 21:15-21:22Z), then runs
# next to v8b-raw (siblings ignore each other; encode queue keeps priority). xbabe3 was dropped: the B-b06 encode
# (hv7_sibling_runner b06spec-x3) took its GPU at ~21:35Z and dv2r2 is still running there.
cd /home/ubuntu/ta_encoders/ta-20261007T2030Z
while pgrep -u ubuntu -f "model hv8 --data data --out hv8r2" >/dev/null; do sleep 20; done
echo "$(date -u +%FT%TZ) hv8r2 gone; starting v8bft2" >> logs/v8bft2.launch.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
nice -n 5 python3 code-v8b/train_ta.py --model hv8ft --out v8bft2 --epochs 120 --k-train 2 --trunk-lr 2e-5 --eval-batch 8 \
  --data data --probe-cids meta/probe_cids.json --probe-data /home/ubuntu/jopedime_probe_exp/20261007T1710Z/data \
  --probe-cache /home/ubuntu/ta_encoders/ta-20261007T2030Z/probe_cache --exclude meta/exclude_sealed.json --probe-threads 6 \
  --seed 0 --batch 16 --lr 0.0003 --eval-min 20 --pcap 128 --nmax 384 --max-hours 12 --guard-siblings hv8raw,hv8ft > logs/v8bft2.out 2>&1
