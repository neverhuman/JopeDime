#!/usr/bin/env bash
set -euo pipefail

INPUT="$HOME/jope-data/h46/input"
DEVICE="${H46_DEVICE:-cuda:0}"
ROOT="${H46_RUN_ROOT:-$HOME/jope-data/h46/runs}"
LOCK="$HOME/jope_plan/qualification.lock"
PY="$PWD/research/flywheel_selfplay/h46/run.py"
for arm in 0 A B C; do
  for seed in 0 1 2; do
    out="$ROOT/H46-$arm/seed$seed"
    if [[ -f "$out/result.json" ]]; then
      echo "skip completed H46-$arm seed $seed"
      continue
    fi
    mkdir -p "$out"
    if [[ "$DEVICE" == "cpu" ]]; then
      while true; do
        avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
        if (( avail_kb < 20971520 )); then
          echo "abort: available RAM below 20 GiB"
          exit 2
        fi
        cpu_pct=$(python3 -c 'import psutil; print(psutil.cpu_percent(interval=1))')
        if python3 -c "raise SystemExit(0 if float('$cpu_pct') <= 85 else 1)"; then
          break
        fi
        echo "host CPU ${cpu_pct}% exceeds 85%; waiting before CPU run"
        sleep 10
      done
      echo "start H46-$arm seed $seed device=cpu"
      nice -n 10 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
        PYTHONPATH=/home/ubuntu/jope_work/telemetry-wheel-5c8a1671/site \
        python3 "$PY" --input "$INPUT" --output "$out" --arm "$arm" --seed "$seed" --device cpu \
        >"$out/run.log" 2>&1
      echo "complete H46-$arm seed $seed"
      continue
    fi
    while true; do
      avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
      if (( avail_kb < 20971520 )); then
        echo "abort: available RAM below 20 GiB"
        exit 2
      fi
      exec {lockfd}>"$LOCK"
      if flock -n "$lockfd"; then
        gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if (( gpu_free < 20000 )); then
          flock -u "$lockfd"
          exec {lockfd}>&-
          echo "GPU has ${gpu_free} MiB free; released lock and retrying"
          sleep 30
          continue
        fi
        echo "start H46-$arm seed $seed device=cuda:0 gpu_free_mib=$gpu_free"
        set +e
        nice -n 10 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
          PYTHONPATH=/home/ubuntu/jope_work/telemetry-wheel-5c8a1671/site \
          python3 "$PY" --input "$INPUT" --output "$out" --arm "$arm" --seed "$seed" --device "$DEVICE" \
          >"$out/run.log" 2>&1
        rc=$?
        set -e
        flock -u "$lockfd"
        exec {lockfd}>&-
        if (( rc != 0 )); then
          echo "failed H46-$arm seed $seed rc=$rc; see $out/run.log"
          exit "$rc"
        fi
        echo "complete H46-$arm seed $seed"
        break
      fi
      exec {lockfd}>&-
      echo "qualification.lock busy; waiting for H46-$arm seed $seed"
      sleep 30
    done
  done
done
