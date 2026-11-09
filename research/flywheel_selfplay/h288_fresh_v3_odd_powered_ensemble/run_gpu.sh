#!/bin/bash
# H288 xbabe1 GPU step: wait for ~/qualification.lock (flock -n, retry every 5 min, never steal), then ONE process under
# a 3 GiB memory cap (MemAvailable >= 8 GiB first): the r3175 DL-Val91 regression and, only if exact, the cohort v3
# fresh contexts (H286's gpu_rescore.py, unchanged). No retry.
set -u
H=/home/ubuntu/jope-data/h288
STOP=1793764800  # 2026-11-04T04:00:00Z
exec 9>>/home/ubuntu/qualification.lock
until flock -n 9; do
  if [ "$(date +%s)" -ge "$STOP" ]; then echo "$(date -u +%FT%TZ) BLOCKED_GPU_LOCK: lock never acquired by 04:00Z"; exit 4; fi
  echo "$(date -u +%FT%TZ) qualification.lock busy; retry in 300 s"; sleep 300
done
echo "$(date -u +%FT%TZ) LOCK ACQUIRED"
cd $H || exit 5
[ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -ge $((8 * 1024 * 1024)) ] || { echo "$(date -u +%FT%TZ) STOP: MemAvailable < 8 GiB"; exit 6; }
systemd-run --user --scope --quiet -p MemoryMax=3G nice -n 10 env PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/.venvs/flywheel-gp-fused/bin/python src/gpu_rescore.py --ckpt inputs/round-3175.pt --stored inputs/round-3175.json \
  --round 3175 --sha256 1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b --fresh-dir fresh --out attempt-001
echo "$(date -u +%FT%TZ) EXIT $? status=$(jq -r .status attempt-001/rescore.json 2>/dev/null) outcome=$(jq -r .outcome attempt-001/rescore.json 2>/dev/null) fresh=$(jq -r .fresh.status attempt-001/rescore.json 2>/dev/null)"
