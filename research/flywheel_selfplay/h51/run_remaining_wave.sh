#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
LOCK="$HOME/jope_plan/qualification.lock"
INPUT="$HOME/jope-data/h50/input"
CHECKPOINT="$HOME/jope-data/h51/ckpt/round-882.pt"
OUTPUT="$HOME/jope-data/h51/runs"
PY="$ROOT/research/flywheel_selfplay/h51/remaining.py"
LOGROOT="$OUTPUT/remaining-wave"
mkdir -p "$LOGROOT"

for arm in DATA25 DATA50 DATA100 CLIPSMOOTH NULL; do
  for seed in 9 10 11; do
    result="$OUTPUT/$arm/seed$seed/candidate-summary.json"
    log="$LOGROOT/${arm}-seed${seed}.log"
    if [[ -s "$result" ]]; then
      echo "skip completed H51-$arm seed$seed"
      continue
    fi
    while true; do
      exec {lockfd}>"$LOCK"
      if ! flock -n "$lockfd"; then
        exec {lockfd}>&-
        echo "qualification.lock busy; waiting before H51-$arm seed$seed"
        sleep 15
        continue
      fi
      available_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
      cpu_pct=$(python3 -c 'import psutil; print(psutil.cpu_percent(interval=1))')
      gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
      if (( available_kb >= 20971520 && gpu_free >= 20480 )) && python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= 85 else 1)' "$cpu_pct"; then
        break
      fi
      echo "guard failed: MemAvailable=${available_kb}KiB CPU=${cpu_pct}% GPU_free=${gpu_free}MiB; releasing lock"
      flock -u "$lockfd"
      exec {lockfd}>&-
      sleep 15
    done
    printf '%s H51-%s seed%s start device=cuda:0 MemAvailable=%sKiB CPU=%s%% GPU_free=%sMiB\n' \
      "$(date -u +%FT%TZ)" "$arm" "$seed" "$available_kb" "$cpu_pct" "$gpu_free" | tee "$log"
    set +e
    nice -n 15 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
      PYTHONPATH=/home/ubuntu/jope_work/telemetry-wheel-5c8a1671/site \
      python3 -u "$PY" --arm "$arm" --seed "$seed" --input "$INPUT" \
        --checkpoint "$CHECKPOINT" --output "$OUTPUT" --device cuda:0 --lock-held >>"$log" 2>&1
    rc=$?
    set -e
    flock -u "$lockfd"
    exec {lockfd}>&-
    if (( rc != 0 )); then
      echo "H51-$arm seed$seed failed with status $rc" | tee -a "$log" >&2
      exit "$rc"
    fi
  done
done

python3 "$ROOT/research/flywheel_selfplay/h51/audit_remaining.py"
