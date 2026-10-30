#!/usr/bin/env bash
# H47 serialized launcher: waits for H46 to finish, then one locked GPU seed at a time.
set -euo pipefail
INPUT="$HOME/jope-data/h46/input"
ROOT="${H47_RUN_ROOT:-$HOME/jope-data/h47/runs}"
LOCK="$HOME/jope_plan/qualification.lock"
PY="$PWD/research/flywheel_selfplay/h47/run.py"
while pgrep -f "flywheel_selfplay/h46/run" >/dev/null; do
  echo "waiting for H46 runs to finish"; sleep 60
done
for arm in R Z PW RP; do
  for seed in 0 1 2; do
    out="$ROOT/H47-$arm/seed$seed"
    [[ -f "$out/result.json" ]] && { echo "skip H47-$arm seed $seed"; continue; }
    mkdir -p "$out"
    while true; do
      avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
      if (( avail_kb < 20971520 )); then echo "abort: available RAM below 20 GiB"; exit 2; fi
      exec {lockfd}>"$LOCK"
      if flock -n "$lockfd"; then
        gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if (( gpu_free < 20000 )); then
          flock -u "$lockfd"; exec {lockfd}>&-
          echo "GPU has ${gpu_free} MiB free; retrying"; sleep 30; continue
        fi
        echo "start H47-$arm seed $seed $(date -u +%FT%TZ)"
        set +e
        nice -n 10 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
          PYTHONPATH=/home/ubuntu/jope_work/telemetry-wheel-5c8a1671/site \
          python3 "$PY" --input "$INPUT" --output "$out" --arm "$arm" --seed "$seed" --device cuda:0 \
          >"$out/run.log" 2>&1
        rc=$?
        set -e
        flock -u "$lockfd"; exec {lockfd}>&-
        if (( rc != 0 )); then echo "failed H47-$arm seed $seed rc=$rc"; exit "$rc"; fi
        echo "complete H47-$arm seed $seed $(date -u +%FT%TZ)"
        break
      fi
      exec {lockfd}>&-
      echo "qualification.lock busy; waiting for H47-$arm seed $seed"; sleep 30
    done
  done
done
echo "H47 wave complete $(date -u +%FT%TZ)"
