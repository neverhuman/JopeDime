#!/usr/bin/env bash
# Trains one specialist per family, PAR at a time, from the staged runtime copy; resumes a run that stopped early
# (up to 3 times); then probes its milestone checkpoints on SEARCH rows only.
# usage: queue_h458.sh <tag> <arm rand|canon> <init_ckpt> <init_key> '<json overrides>' <family8> [<family8> ...]
# env:   PAR (default 2)  VRAM_FRAC  WORKERS  REMOTE_WORKERS  CODE  PROBE_TAGS (default "01000 01500 02000")
set -uo pipefail
TAG="$1"; ARM="$2"; INIT="$3"; KEY="$4"; EXTRA="$5"; shift 5
[ -n "$EXTRA" ] || EXTRA='{}'
CODE="${CODE:-$(cd "$(dirname "$0")" && pwd)}"
Q=/home/ubuntu/jope-data/h458-quarantine
PAR="${PAR:-2}"
PROBE_TAGS="${PROBE_TAGS:-01000 01500 02000}"
mkdir -p "$Q/runs" "$Q/probe"

one_family() {
  local FAM="$1" NAME="${TAG}_${1}_${ARM}" TRY=0 RC=0
  local RUN="$Q/runs/$NAME"
  "$CODE/launch_specialist.sh" "$NAME" "$FAM" "$ARM" "$INIT" "$KEY" "$EXTRA" > "$Q/runs/$NAME.stdout" 2>&1 || RC=$?
  while [ ! -f "$RUN/FINAL.json" ] && [ -f "$RUN/ckpt_last.pt" ] && [ "$TRY" -lt 3 ]; do
    TRY=$((TRY + 1))
    echo "$(date -u +%TZ) $NAME stopped with exit $RC; resume $TRY of 3"
    RC=0
    RESUME=1 "$CODE/launch_specialist.sh" "$NAME" "$FAM" "$ARM" "$INIT" "$KEY" "$EXTRA" >> "$Q/runs/$NAME.stdout" 2>&1 || RC=$?
  done
  if [ ! -f "$RUN/FINAL.json" ]; then
    echo "$(date -u +%TZ) $NAME NOT FINISHED (exit $RC): $(ls "$RUN" 2>/dev/null | tr '\n' ' ')"
    return 1
  fi
  echo "$(date -u +%TZ) trained $NAME"
  local E CK
  for E in $PROBE_TAGS; do
    CK="$RUN/spec_e$E.pt"
    [ -f "$CK" ] || continue
    OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 19 taskset -c 32-63,80-127 \
      /usr/bin/python3 -B "$CODE/probe_h458.py" --ckpt "$CK" --ckpt-key ema --size M --arm "$ARM" --families "$FAM" \
      --device cuda --cuda-vram-frac 0.07 --threads 4 --label "${NAME}_e$E" \
      --out "$Q/probe/${NAME}_spec_e$E.json" 2>&1 | tail -n 1
  done
}

for FAM in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n || true; done
  one_family "$FAM" &
done
wait
echo "$(date -u +%TZ) queue $TAG done: $*"
