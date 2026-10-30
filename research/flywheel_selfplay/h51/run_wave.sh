#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
LOCK="$HOME/jope_plan/qualification.lock"
INPUT="$HOME/jope-data/h50/input"
CHECKPOINT="$HOME/jope-data/h50/ckpt/round-785.pt"
OUTPUT="$HOME/jope-data/h51/runs"
PY="$ROOT/research/flywheel_selfplay/h51/run.py"
mkdir -p "$OUTPUT"
DEVICE="cuda:0"
lock_wait_started=$(date +%s)

if [[ -s "$OUTPUT/device-selection.json" ]]; then
  DEVICE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["device"])' "$OUTPUT/device-selection.json")
fi

for seed in 9 10 11; do
  out="$OUTPUT/CONTROL/seed$seed"
  mkdir -p "$out"
  if [[ -s "$out/candidate-summary.json" ]]; then
    echo "skip completed H51-CONTROL seed$seed"
    continue
  fi
  while true; do
    if [[ "$DEVICE" == "cpu" ]]; then
      break
    fi
    exec {lockfd}>"$LOCK"
    if ! flock -n "$lockfd"; then
      exec {lockfd}>&-
      now=$(date +%s)
      if (( now - lock_wait_started >= 1200 )); then
        DEVICE="cpu"
        python3 - "$OUTPUT/device-selection.json" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "device": "cpu",
    "reason": "qualification.lock remained busy for at least 20 minutes before H51 training; CPU fallback per preregistration",
    "cpu_threads": 8,
}, indent=2) + "\n")
PY
        echo "GPU lock busy for 20 minutes before H51 start; using preregistered CPU fallback"
        break
      fi
      echo "qualification.lock busy; H51 waits"
      sleep 30
      continue
    fi
    available_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    cpu_pct=$(python3 -c 'import psutil; print(psutil.cpu_percent(interval=1))')
    gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if (( available_kb >= 20971520 && gpu_free >= 20480 )) && python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= 85 else 1)' "$cpu_pct"; then
      python3 - "$OUTPUT/device-selection.json" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "device": "cuda:0",
    "reason": "qualification.lock held by H51; GPU/RAM/CPU preflight passed",
    "cpu_threads": 4,
}, indent=2) + "\n")
PY
      break
    fi
    echo "resource guard failed: RAM=${available_kb}KiB CPU=${cpu_pct}% GPU_free=${gpu_free}MiB; release lock and retry"
    flock -u "$lockfd"
    exec {lockfd}>&-
    sleep 30
  done
  echo "start H51-CONTROL seed$seed on $DEVICE; RAM=${available_kb}KiB CPU=${cpu_pct}% GPU_free=${gpu_free:-not-applicable}MiB"
  set +e
  nice -n 15 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH=/home/ubuntu/jope_work/telemetry-wheel-5c8a1671/site \
    python3 -u "$PY" --arm CONTROL --seed "$seed" --input "$INPUT" \
      --checkpoint "$CHECKPOINT" --output "$OUTPUT" --device "$DEVICE" --lock-held \
      >"$out/run.log" 2>&1
  rc=$?
  set -e
  if [[ "$DEVICE" == "cuda:0" ]]; then
    flock -u "$lockfd"
    exec {lockfd}>&-
  fi
  if (( rc != 0 )); then
    echo "H51-CONTROL seed$seed failed with status $rc" >&2
    exit "$rc"
  fi
done
