#!/usr/bin/env bash
# Trains the slot arms of one family side by side, then probes every milestone checkpoint (SEARCH rows only).
# usage: pilot_driver.sh <family8> <init_ckpt> <init_key> ['{"json": "overrides"}'] [tag]
# env:   ARMS (default "rand canon")  VRAM_FRAC  WORKERS  CODE
set -uo pipefail
FAM="$1"; INIT="$2"; KEY="$3"; EXTRA="${4:-}"; TAG="${5:-p1}"
[ -n "$EXTRA" ] || EXTRA='{}'
CODE="${CODE:-$(cd "$(dirname "$0")" && pwd)}"
Q=/home/ubuntu/jope-data/h458-quarantine
ARMS="${ARMS:-rand canon}"
mkdir -p "$Q/runs" "$Q/probe"
for ARM in $ARMS; do
  "$CODE/launch_specialist.sh" "${TAG}_${FAM}_${ARM}" "$FAM" "$ARM" "$INIT" "$KEY" "$EXTRA" \
    > "$Q/runs/${TAG}_${FAM}_${ARM}.stdout" 2>&1 &
done
wait
for ARM in $ARMS; do
  RUN="$Q/runs/${TAG}_${FAM}_${ARM}"
  echo "$(date -u +%TZ) trained ${TAG}_${FAM}_${ARM}: $(ls "$RUN" | tr '\n' ' ')"
  for CK in "$RUN"/spec_e*.pt; do
    [ -f "$CK" ] || continue
    T="$(basename "$CK" .pt)"
    OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 19 taskset -c 32-63,80-127 \
      /usr/bin/python3 -B "$CODE/probe_h458.py" --ckpt "$CK" --ckpt-key ema --size M --arm "$ARM" --families "$FAM" \
      --device cuda --cuda-vram-frac 0.07 --threads 4 --label "${TAG}_${FAM}_${ARM}_${T}" \
      --out "$Q/probe/${TAG}_${FAM}_${ARM}_${T}.json" 2>&1 | tail -n 1
  done
done
echo "$(date -u +%TZ) pilot_driver done for $FAM"
