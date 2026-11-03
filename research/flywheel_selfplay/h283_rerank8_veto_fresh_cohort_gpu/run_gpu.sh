#!/bin/bash
# H283 xbabe1 GPU step: wait for ~/qualification.lock (flock -n, retry every 5 min, never steal), then ONE process: the
# r3175 DL-Val91 regression and, only if exact, the fresh contexts (gpu_rescore.py). No retry.
set -u
H=/home/ubuntu/jope-data/h283
STOP=1793758500  # 2026-11-04T02:15:00Z
exec 9>>/home/ubuntu/qualification.lock
until flock -n 9; do
  if [ "$(date +%s)" -ge "$STOP" ]; then echo "$(date -u +%FT%TZ) BLOCKED_GPU_LOCK: lock never acquired by 02:15Z"; exit 4; fi
  echo "$(date -u +%FT%TZ) qualification.lock busy; retry in 300 s"; sleep 300
done
echo "$(date -u +%FT%TZ) LOCK ACQUIRED"
cd $H && nice -n 10 env PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/.venvs/flywheel-gp-fused/bin/python src/gpu_rescore.py \
  --ckpt inputs/round-3175.pt --stored inputs/round-3175.json --fresh-dir fresh --out attempt-001
echo "$(date -u +%FT%TZ) EXIT $?"
