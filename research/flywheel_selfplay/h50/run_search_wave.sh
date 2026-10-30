#!/usr/bin/env bash
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel)"
LOCK="$HOME/jope_plan/qualification.lock"
PY="$ROOT/research/flywheel_selfplay/h50/search.py"
INPUT="$HOME/jope-data/h50/input"
CHECKPOINT="$HOME/jope-data/h50/ckpt/round-785.pt"
OUTPUT="$HOME/jope-data/h50/runs"
mkdir -p "$OUTPUT"

for arm in ES3 CMA ESEMA; do
  for seed in 6 7 8; do
    out="$OUTPUT/H50-$arm/seed$seed"
    mkdir -p "$out"
    if [[ -s "$out/train-result.json" && -s "$out/selected-candidate.json" ]]; then
      echo "skip completed H50-$arm seed$seed"
      continue
    fi
    while true; do
      nice -n 10 flock -w 300 -E 75 "$LOCK" bash -c '
        set -euo pipefail
        PY="$0" INPUT="$1" CHECKPOINT="$2" OUTPUT="$3" ARM="$4" SEED="$5"
        export PY INPUT CHECKPOINT OUTPUT ARM SEED
        available_kb=$(awk "/MemAvailable:/ {print \$2}" /proc/meminfo)
        cpu_pct=$(python3 -c "import psutil; print(psutil.cpu_percent(interval=1))")
        gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d " ")
        if (( available_kb >= 20971520 && gpu_free >= 20480 )) && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 85 else 1)" "$cpu_pct"; then
          :
        else
          echo "resource guard failed: RAM=${available_kb}KiB CPU=${cpu_pct}% GPU=${gpu_free}MiB" >&2
          exit 76
        fi
        echo "start H50-$ARM seed$SEED on cuda:0, RAM_KiB=$available_kb CPU=$cpu_pct GPU_free_MiB=$gpu_free"
        exec env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
          PYTHONPATH=/home/ubuntu/jope_work/telemetry-wheel-5c8a1671/site \
          python3 -u "$PY" --arm "$ARM" --seed "$SEED" --input "$INPUT" \
            --checkpoint "$CHECKPOINT" --output "$OUTPUT" --lock-held
      ' -- "$PY" "$INPUT" "$CHECKPOINT" "$OUTPUT" "$arm" "$seed" >>"$out/run.log" 2>&1
      result=$?
      if (( result == 0 )); then
        break
      elif (( result == 75 || result == 76 )); then
        echo "H50-$arm seed$seed waiting 30s to retry lock/resource guard" | tee -a "$out/run.log"
        sleep 30
      else
        echo "H50-$arm seed$seed failed with exit $result" | tee -a "$out/run.log"
        exit "$result"
      fi
    done
  done
done

echo "H50 training arms complete; terminal scorer may start after all selected candidates are recorded."
