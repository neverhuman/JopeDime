#!/bin/bash
# H286 xbabe1 GPU step. Wait for ~/qualification.lock (flock -n, retry every 5 min, never steal), then in separate
# processes under one lock hold: r3175 (PRIMARY: DL-Val91 regression, then the fresh contexts), and, only if that is
# COMPLETE and time allows, r782 and r785 (DESCRIPTIVE distinct pools, same path). Each process runs under a 3 GiB
# memory cap; MemAvailable < 8 GiB before a process -> stop. Any non-COMPLETE status stops the step (no retry).
set -u
H=/home/ubuntu/jope-data/h286
STOP=1793761200  # 2026-11-04T03:00:00Z
memok() { [ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -ge $((8 * 1024 * 1024)) ]; }
exec 9>>/home/ubuntu/qualification.lock
until flock -n 9; do
  if [ "$(date +%s)" -ge "$STOP" ]; then echo "$(date -u +%FT%TZ) BLOCKED_GPU_LOCK: lock never acquired by 03:00Z"; exit 4; fi
  echo "$(date -u +%FT%TZ) qualification.lock busy; retry in 300 s"; sleep 300
done
echo "$(date -u +%FT%TZ) LOCK ACQUIRED"
cd $H || exit 5
run() {  # name round sha ckpt stored
  memok || { echo "$(date -u +%FT%TZ) STOP: MemAvailable < 8 GiB before $1"; return 1; }
  systemd-run --user --scope --quiet -p MemoryMax=3G nice -n 10 env PYTHONDONTWRITEBYTECODE=1 \
    /home/ubuntu/.venvs/flywheel-gp-fused/bin/python src/gpu_rescore.py --ckpt "$4" --stored "$5" --round "$2" \
    --sha256 "$3" --fresh-dir fresh --out "attempt-001/$1"
  st=$(jq -r .status "attempt-001/$1/rescore.json" 2>/dev/null)
  echo "$(date -u +%FT%TZ) $1 status=$st outcome=$(jq -r .outcome "attempt-001/$1/rescore.json" 2>/dev/null) fresh=$(jq -r .fresh.status "attempt-001/$1/rescore.json" 2>/dev/null)"
  [ "$st" = "COMPLETE" ]
}
mkdir attempt-001 || exit 6
run r3175 3175 1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b inputs/round-3175.pt inputs/round-3175.json || { echo "$(date -u +%FT%TZ) EXIT primary-not-complete"; exit 7; }
for spec in "r782 782 427b0d4608e94f60f5f97d79b5c1c4ce9a7efd59d17e7853f714e901f3a4af0c" "r785 785 ef6696f6369122a162d58b9b522fd83cae110a5576ef3760e3b43ba85d129e0a"; do
  set -- $spec
  if [ "$(date +%s)" -ge $((STOP - 900)) ]; then echo "$(date -u +%FT%TZ) SKIP descriptive $1 (time)"; continue; fi
  run "$1" "$2" "$3" "ckpt/round-$2.pt" "ckpt/round-$2.json" || { echo "$(date -u +%FT%TZ) STOP descriptive at $1"; break; }
done
echo "$(date -u +%FT%TZ) EXIT"
