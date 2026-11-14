#!/usr/bin/env bash
# Launches one H458 specialist run from the staged runtime copy, with the resource caps the wave was admitted under.
# usage: launch_specialist.sh <run_name> <family8> <arm rand|canon> <init_ckpt> <init_key> ['{"json": "overrides"}']
# env:   VRAM_FRAC (default 0.13)  WORKERS (default 8)  CODE (default /home/ubuntu/h458-stage/code)
set -euo pipefail
NAME="$1"; FAM="$2"; ARM="$3"; INIT="$4"; KEY="$5"; EXTRA="${6:-}"
[ -n "$EXTRA" ] || EXTRA='{}'
CODE="${CODE:-/home/ubuntu/h458-stage/code}"
RUN="/home/ubuntu/jope-data/h458-quarantine/runs/$NAME"
mkdir -p "$RUN"
/usr/bin/python3 -B - "$RUN/config.json" "$FAM" "$ARM" "$EXTRA" <<'PY'
import json, sys
out, fam, arm, extra = sys.argv[1:5]
cfg = {"arm": arm, "size": "M", "families": [fam]}
cfg.update(json.loads(extra))
with open(out, "x") as fh:  # create-once: a run name is never reused
    fh.write(json.dumps(cfg, indent=1, sort_keys=True) + "\n")
PY
exec systemd-run --user --scope -q -p MemoryHigh=14G -p MemoryMax=18G choom -n 500 -- \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 JOPE_REF_CODE= PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  taskset -c 32-63,80-127 nice -n 19 /usr/bin/python3 -u -B "$CODE/train_h458.py" \
    --config "$RUN/config.json" --init "$INIT" --init-key "$KEY" --out "$RUN" \
    --device cuda --cuda-vram-frac "${VRAM_FRAC:-0.13}" --workers "${WORKERS:-8}"
