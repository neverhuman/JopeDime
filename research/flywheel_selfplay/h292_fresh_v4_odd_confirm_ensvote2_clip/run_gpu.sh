#!/bin/bash
# H292 xbabe1 GPU step (H288's path + H288's declared admission wait, made part of the prereg): every 5 min until
# 04:00Z, if nvidia-smi free VRAM >= 4.5 GiB AND MemAvailable >= 8 GiB AND flock -n ~/qualification.lock succeeds
# (never stolen), run ONE process under a 3 GiB memory cap: H288's gpu_rescore.py, unchanged (r3175 DL-Val91 regression
# in-process, then the cohort v4 fresh contexts). Single scientific attempt; an admission refusal by the scorer itself
# (NOT_ADMITTED_VRAM, before any forward pass) is recorded and the wait resumes into the next attempt directory.
# ~/jope_plan/qualification.lock is held by the live writer for its lifetime and is never touched.
set -u
H=/home/ubuntu/jope-data/h292
STOP=1793764800  # 2026-11-04T04:00:00Z
NEED_MIB=4608
n=1
while :; do
  if [ "$(date +%s)" -ge "$STOP" ]; then echo "$(date -u +%FT%TZ) BLOCKED_GPU: not admitted by 04:00Z"; exit 4; fi
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  mem=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$free" -ge "$NEED_MIB" ] && [ "$mem" -ge $((8 * 1024 * 1024)) ]; then
    exec 9>>/home/ubuntu/qualification.lock
    if flock -n 9; then
      echo "$(date -u +%FT%TZ) ADMITTED free=${free}MiB; LOCK ACQUIRED; attempt-00$n"
      cd $H && systemd-run --user --scope --quiet -p MemoryMax=3G nice -n 10 env PYTHONDONTWRITEBYTECODE=1 \
        /home/ubuntu/.venvs/flywheel-gp-fused/bin/python src/gpu_rescore.py --ckpt inputs/round-3175.pt --stored inputs/round-3175.json \
        --round 3175 --sha256 1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b --fresh-dir fresh --out attempt-00$n
      st=$(jq -r .status attempt-00$n/rescore.json 2>/dev/null)
      echo "$(date -u +%FT%TZ) EXIT status=$st outcome=$(jq -r .outcome attempt-00$n/rescore.json 2>/dev/null) fresh=$(jq -r .fresh.status attempt-00$n/rescore.json 2>/dev/null)"
      [ "$st" = "NOT_ADMITTED_VRAM" ] || exit 0
      n=$((n + 1)); exec 9>&-; cd - >/dev/null
    else
      exec 9>&-; echo "$(date -u +%FT%TZ) lock busy (free=${free}MiB)"
    fi
  else
    echo "$(date -u +%FT%TZ) waiting: free=${free}MiB MemAvailable=$((mem / 1024))MiB"
  fi
  sleep 300
done
