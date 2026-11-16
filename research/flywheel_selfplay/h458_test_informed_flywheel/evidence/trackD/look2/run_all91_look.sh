#!/usr/bin/env bash
# Track D official look 2 of 2 (campaign look 2 of 4): arm h458d.RECALL_CANON_E2000_ALL91, pre-registered on ledger
# line 3 at 2026-11-15T03:14:56Z. Runs ONCE: it refuses if the ledger already holds a look entry for the arm.
set -euo pipefail
Q=/home/ubuntu/jope-data/h458-quarantine
ARM=h458d.RECALL_CANON_E2000_ALL91
LEDGER=/home/ubuntu/jope_plan/evidence/H458/LOOK_LEDGER.jsonl
cd /home/ubuntu/h458s-stage/code
[ "$(sha256sum score_h458s.py | cut -c1-16)" = 0a9adb2c702af628 ] || { echo "the scorer changed since the pre-registration"; exit 2; }
[ "$(sha256sum $Q/blocks/recall_canon_e2000_all91/SIDECAR.json | cut -c1-16)" = 8a1d512e87e67872 ] || { echo "the block's sidecar changed"; exit 2; }
grep -q "\"type\": \"pre\"" <(grep "$ARM" $LEDGER) || { echo "no pre entry"; exit 2; }
if grep "$ARM" $LEDGER | grep -q "\"type\": \"look\""; then echo "the look was spent already"; exit 2; fi
TS=$(date -u +%Y%m%dT%H%MZ); OUT=$Q/looks/all91_$TS
echo "$(date -u +%TZ) look starts, out $OUT"
systemd-run --user --scope -q -p MemoryMax=24G nice -n 19 taskset -c ${LOOK_CPUS:-112-127} env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/bin/python3 -B score_h458s.py --epoch-cols /home/ubuntu/jope-data/h450-quarantine/r4/cols/g44_r4_L_A_r4_L_B \
  --extra h458d=$Q/blocks/recall_canon_e2000_all91 --optset legacy --arm-optset legacy --arm $ARM --out "$OUT" --official --procs 8 > "$Q/looks/all91_$TS.log" 2>&1
echo "$(date -u +%TZ) look ended, exit $?"
tail -n 2 "$Q/looks/all91_$TS.log" | cut -c1-900
