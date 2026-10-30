#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
INPUT="$HOME/jope-data/h49/input"
CHECKPOINT="$HOME/jope-data/h49/ckpt/round-785.pt"
OUTPUT="$HOME/jope-data/h49/runs"
LOCK="$HOME/jope_plan/qualification.lock"
PY="$ROOT/research/flywheel_selfplay/h49/run.py"
DEVICE="cuda:0"
mkdir -p "$OUTPUT"

acquire_lock() {
  while true; do
    local avail_kb
    avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if (( avail_kb < 20971520 )); then
      echo "available RAM ${avail_kb} KiB below 20 GiB; pausing only H49"
      sleep 30
      continue
    fi
    exec {lockfd}>"$LOCK"
    if flock -n "$lockfd"; then return 0; fi
    exec {lockfd}>&-
    echo "qualification.lock busy; H49 waits without interrupting other work"
    sleep 30
  done
}

release_lock() {
  flock -u "$lockfd"
  exec {lockfd}>&-
}

acquire_lock
trap release_lock EXIT

python3 - "$OUTPUT/device-selection.json" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "device": "cuda:0",
    "input_checkpoint_sha256": "ef6696f6369122a162d58b9b522fd83cae110a5576ef3760e3b43ba85d129e0a",
    "reason": "H49 requires cuda:0 only; qualification.lock held by the H49 coordinator",
}, indent=2) + "\n")
PY

run_one() {
  local arm="$1" seed="$2" budget="${3:-141}"
  local out="$OUTPUT/H49-$arm/seed$seed"
  if [[ -s "$out/train-result.json" && -s "$out/selected-candidate.json" ]]; then
    echo "skip completed H49-$arm seed$seed"
    return 0
  fi
  mkdir -p "$out"
  local avail_kb gpu_free cpu_pct
  while true; do
    avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    cpu_pct=$(python3 -c 'import psutil; print(psutil.cpu_percent(interval=1))')
    if (( avail_kb >= 20971520 && gpu_free >= 20480 )) && python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= 85 else 1)' "$cpu_pct"; then
      break
    fi
    echo "preseed guard failed for H49-$arm seed$seed (RAM ${avail_kb} KiB, GPU ${gpu_free} MiB, CPU ${cpu_pct}%); releasing lock and pausing H49"
    release_lock
    sleep 30
    acquire_lock
  done
  echo "start H49-$arm seed$seed on cuda:0, budget=${budget}s, RAM_KiB=${avail_kb}"
  nice -n 10 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
    PYTHONPATH=/home/ubuntu/jope_work/telemetry-wheel-5c8a1671/site \
    python3 -u "$PY" --arm "$arm" --seed "$seed" --input "$INPUT" \
      --checkpoint "$CHECKPOINT" --output "$OUTPUT" --device "$DEVICE" \
      --budget-seconds "$budget" --lock-held >"$out/run.log" 2>&1
}

status=0
for seed in 0 1 2; do run_one 0 "$seed" 300 || { status=$?; break; }; done
if (( status == 0 )); then
  for seed in 0 1 2; do run_one SAM "$seed" 300 || { status=$?; break; }; done
fi
if (( status == 0 )); then
  for seed in 0 1 2; do run_one SWA "$seed" 300 || { status=$?; break; }; done
fi
if (( status == 0 )); then
  for seed in 3 4 5; do run_one ES2 "$seed" 141 || { status=$?; break; }; done
fi

release_lock
trap - EXIT
if (( status == 0 )); then
  echo "H49 train arms complete; launching terminal-only scorer"
  nice -n 10 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
    PYTHONPATH=/home/ubuntu/jope_work/telemetry-wheel-5c8a1671/site \
    python3 -u "$ROOT/research/flywheel_selfplay/h49/score.py" \
      >"$OUTPUT/terminal-score.log" 2>&1 || status=$?
fi
exit "$status"
