#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
INPUT="$HOME/jope-data/h48/input"
CHECKPOINT="$HOME/jope-data/h48/ckpt/round-785.pt"
OUTPUT="$HOME/jope-data/h48/runs"
LOCK="$HOME/jope_plan/qualification.lock"
PY="$ROOT/research/flywheel_selfplay/h48/run.py"
DEVICE="cuda:0"
lock_wait_start=$(date +%s)
fallback_note="$OUTPUT/device-selection.json"
mkdir -p "$OUTPUT"

while true; do
  avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if (( avail_kb < 20971520 )); then
    echo "abort: available RAM below 20 GiB before qualification lock"
    exit 2
  fi
  exec {lockfd}>"$LOCK"
  if flock -n "$lockfd"; then
    break
  fi
  exec {lockfd}>&-
  now=$(date +%s)
  waited=$((now - lock_wait_start))
  if (( waited >= 1200 )) && [[ "$DEVICE" == "cuda:0" ]]; then
    DEVICE="cpu"
    python3 - "$fallback_note" "$waited" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "device": "cpu",
    "reason": "qualification.lock busy for at least 20 minutes before the H48 comparison wave started",
    "lock_wait_seconds": int(sys.argv[2]),
}, indent=2) + "\n")
PY
    echo "qualification lock busy for ${waited}s; all H48-0/PF/ES comparisons use CPU"
  else
    echo "qualification.lock busy; waiting for the serialized H48 wave"
  fi
  sleep 30
done

if [[ "$DEVICE" == "cuda:0" ]]; then
  gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if (( gpu_free < 20000 )); then
    DEVICE="cpu"
    echo "GPU has ${gpu_free} MiB free before the wave; all H48-0/PF/ES comparisons use CPU"
  fi
fi
python3 - "$fallback_note" "$DEVICE" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "device": sys.argv[2],
    "reason": "cuda:0 selected after the serialized lock became available" if sys.argv[2] == "cuda:0" else "CPU fallback selected before H48 comparisons because lock wait exceeded 20 minutes or GPU headroom was below 20 GiB",
}, indent=2) + "\n")
PY

run_one() {
  local arm="$1" seed="$2" budget="${3:-300}"
  local out="$OUTPUT/H48-$arm/seed$seed"
  mkdir -p "$out"
  if [[ -s "$out/train-result.json" && -s "$out/selected-candidate.json" ]]; then
    echo "skip completed H48-$arm seed$seed"
    return 0
  fi
  local avail_kb
  avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if (( avail_kb < 20971520 )); then
    echo "abort: available RAM below 20 GiB before H48-$arm seed$seed"
    return 2
  fi
  if [[ "$DEVICE" == "cpu" ]]; then
    local cpu_pct
    while true; do
      cpu_pct=$(python3 -c 'import psutil; print(psutil.cpu_percent(interval=1))')
      if python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= 85 else 1)' "$cpu_pct"; then
        break
      fi
      echo "host CPU ${cpu_pct}% exceeds 85%; waiting before H48-$arm seed$seed"
      sleep 15
    done
  else
    local gpu_free
    gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if (( gpu_free < 20000 )); then
      echo "GPU has ${gpu_free} MiB free before H48-$arm seed$seed; aborting to preserve device consistency"
      return 75
    fi
  fi
  echo "start H48-$arm seed$seed device=$DEVICE budget=${budget}s RAM_KiB=$avail_kb"
  nice -n 10 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
    PYTHONPATH=/home/ubuntu/jope_work/telemetry-wheel-5c8a1671/site \
    python3 -u "$PY" --input "$INPUT" --checkpoint "$CHECKPOINT" \
      --output "$OUTPUT" --arm "$arm" --seed "$seed" --device "$DEVICE" \
      --budget-seconds "$budget" --lock-held >"$out/run.log" 2>&1
  echo "complete H48-$arm seed$seed"
}

status=0
for arm in 0 PF; do
  for seed in 0 1 2; do
    run_one "$arm" "$seed" || { status=$?; break 2; }
  done
done

if (( status == 0 )) && [[ ! -s "$OUTPUT/h48-final-analysis.json" ]]; then
  budget=$(python3 - <<'PY'
import json, statistics
from pathlib import Path
rows = [json.loads((Path.home()/"jope-data/h48/runs"/"H48-PF"/f"seed{s}"/"train-result.json").read_text()) for s in range(3)]
print(max(1, int(statistics.median(row["wall_seconds"] for row in rows))))
PY
)
  for seed in 0 1 2; do
    run_one ES "$seed" "$budget" || { status=$?; break; }
  done
fi

flock -u "$lockfd"
exec {lockfd}>&-
if (( status == 0 )); then
  echo "all H48 training arms complete; starting one terminal DL-Val score per selected arm/seed"
  nice -n 10 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
    PYTHONPATH=/home/ubuntu/jope_work/telemetry-wheel-5c8a1671/site \
    python3 -u "$ROOT/research/flywheel_selfplay/h48/score.py" \
      >"$OUTPUT/terminal-score.log" 2>&1 || status=$?
fi
exit "$status"
