#!/bin/bash
# H282 xbabe1 GPU step: wait for ~/qualification.lock (flock -n, retry every 5 min, never steal), then re-score each
# eligible checkpoint once in its own process (H281 path). Any ABORT_* status (CUDA OOM, writer state, VRAM) stops the
# whole step: no retry.
set -u
H=/home/ubuntu/jope-data/h282
STOP=1793754000  # 2026-11-04T01:00:00Z
exec 9>>/home/ubuntu/qualification.lock
until flock -n 9; do
  if [ "$(date +%s)" -ge "$STOP" ]; then echo "$(date -u +%FT%TZ) BLOCKED_GPU_LOCK: lock never acquired by 01:00Z"; exit 4; fi
  echo "$(date -u +%FT%TZ) qualification.lock busy; retry in 300 s"; sleep 300
done
echo "$(date -u +%FT%TZ) LOCK ACQUIRED"
cd $H && mkdir attempt-001 || exit 5
while read -r n s; do
  if [ "$(date +%s)" -ge "$STOP" ]; then echo "$(date -u +%FT%TZ) COMPUTE STOP before r$n"; break; fi
  nice -n 10 env PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/.venvs/flywheel-gp-fused/bin/python src/gpu_rescore.py \
    --ckpt ckpt/round-$n.pt --stored ckpt/round-$n.json --round $n --sha256 $s --out attempt-001/r$n
  rc=$?
  st=$(jq -r .status attempt-001/r$n/rescore.json 2>/dev/null)
  echo "$(date -u +%FT%TZ) r$n rc=$rc status=$st outcome=$(jq -r .outcome attempt-001/r$n/rescore.json 2>/dev/null)"
  if [ "$st" != "COMPLETE" ]; then echo "$(date -u +%FT%TZ) STOP: r$n status $st (no retry)"; break; fi
done < src/eligible.txt
echo "$(date -u +%FT%TZ) EXIT"
