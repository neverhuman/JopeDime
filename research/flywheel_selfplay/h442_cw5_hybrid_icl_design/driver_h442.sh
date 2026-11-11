#!/bin/bash
# H442 scored CPU run from the published stage:
#   0) test_cw5.py -> unit_tests.json
#   1) eval_cw5_fitsel.py run across P single-thread nice-19 shards on fitsel_search_val.npz
#   Stops BEFORE readout / FROZEN.json so gates_h442.sh runs first!
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

EV=/home/ubuntu/jope_plan/evidence/H442
ST=$(cat "$EV/logs/pub_stage")
R=${WAVE_OUT:-/home/ubuntu/jope-data/h335-cl/h442/$(basename "$ST")}
P=${WAVE_PROCS:-16}
CPUSET=${WAVE_CPUSET:-64-79}
[ "$P" -le 16 ] || { echo "P=$P exceeds 16-thread ceiling" >&2; exit 2; }

CK_M20=/home/ubuntu/jope-data/h424/ICL_M_r3_ckpt_h020.pt
SHA_M20=e3c8fd9972ce0c1684630ff06b159e950f6f498c4461d0454e0c6a6f44d38dd7
VAL=/home/ubuntu/jope-data/apex-icl/bank/fitsel_search_val.npz
SHA_VAL=8e517921b7d5bb46c223a12262e7e0c64531db3fad7dede601d002a09840bba0

log() { echo "$(date -u +%FT%TZ) $*"; }
(cd "$ST/code" && sha256sum --quiet -c ../CODE.sha256)
[ "$(cat "$ST/mode")" = pub ]
mkdir -p "$R/ctxs" "$EV/logs" "$EV/results"
cd "$ST/code"

T_START=$(date +%s)
log "run_start P=$P CPUSET=$CPUSET stage=$ST CODE.sha256=$(sha256sum "$ST/CODE.sha256" | cut -c1-16)"

if [ ! -e "$R/unit_tests.json" ]; then
  taskset -c "$CPUSET" nice -n 19 python3 -B test_cw5.py > "$R/unit_tests.raw.txt"
  head -n 1 "$R/unit_tests.raw.txt" > "$R/unit_tests.json"
  tail -n 1 "$R/unit_tests.raw.txt"
fi

rc=0
log "start CW-5 R0..R5 FIT-SEL evaluation ($P shards)"
for k in $(seq 0 $((P - 1))); do
  taskset -c "$CPUSET" nice -n 19 python3 -B eval_cw5_fitsel.py run \
    --val "$VAL" --val-sha256 "$SHA_VAL" --ckpt "$CK_M20" --ckpt-sha256 "$SHA_M20" \
    --size M --out "$R/ctxs" --shard "$k/$P" > "$EV/logs/cw5_shard_$k.log" 2>&1 &
done
for j in $(jobs -p); do wait "$j" || rc=$?; done
log "done CW-5 FIT-SEL shards (worst_rc=$rc)"
[ "$rc" -eq 0 ] || exit "$rc"

T_END=$(date +%s)
jq -n --argjson run_start "$T_START" --argjson run_end "$T_END" --argjson procs "$P" --arg cpuset "$CPUSET" \
  '{run_start:$run_start, run_end:$run_end, wall_s:($run_end-$run_start), procs:$procs, cpuset:$cpuset}' > "$EV/logs/run_times.json"
log "driver_h442 completed in $((T_END - T_START))s (ready for gates_h442.sh then readout)"
