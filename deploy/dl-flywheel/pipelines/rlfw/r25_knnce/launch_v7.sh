#!/usr/bin/env bash
# R22 scratch-only, non-qualifying L_PI run, harness revision 7, inside the caller's capped scope (xbabe2 or xbabe3).
# Usage: LPI_UNIT=<scope unit> [LPI_CODE_ROOT=<dir with code/>] [LPI_PY=<python>] launch_v6.sh CONFIG_JSON OUT_DIR
# 1. Reads this scope's own properties back (the harness requires RuntimeMaxUSec <= 2h and >= wall + 120 s).
# 2. Samples GPU memory every 2 s when CUDA is visible (never when hidden). 3. Runs the harness; exits with its status.
set -u
X=${LPI_CODE_ROOT:-/home/ubuntu/claude-rlfw-lane/rlfw-takeover/lpi-scratch}
CONFIG=$1
OUT=$2
systemctl --user show "${LPI_UNIT:-unset}.scope" -p Id -p ActiveState -p Slice -p RuntimeMaxUSec \
  -p CPUQuotaPerSecUSec -p MemoryMax -p MemorySwapMax -p TasksMax > "$OUT.scope-properties.txt" 2>&1
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 2000 > "$OUT.vram.csv" 2>&1 &
  SMI=$!
else
  SMI=
fi
"${LPI_PY:-/home/ubuntu/jopedime_probe_exp/ofm-20261008/venv/bin/python}" -B "$X/code/lpi_scratch_train_v7.py" "$CONFIG" "$OUT" \
  > "$OUT.log" 2>&1
rc=$?
[ -n "$SMI" ] && kill "$SMI" 2>/dev/null
[ -n "$SMI" ] && wait "$SMI" 2>/dev/null
exit "$rc"
