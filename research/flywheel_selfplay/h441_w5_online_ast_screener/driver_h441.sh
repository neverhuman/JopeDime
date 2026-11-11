#!/bin/bash
# H441 scored CPU run from the published stage:
#   0) test_w5.py -> unit_tests.json
#   1) eval_w5_fitsel.py must-shas -> must_shas_FITSEL.json
#   2) score_h01a_fitsel.py inside H335 docker sandbox -> h01a_scores_FITSEL.npz
#   3) eval_w5_fitsel.py apply -> APPLY_W5.json (stops BEFORE READOUT_W5.json so gates_h441.sh runs first!)
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

EV=/home/ubuntu/jope_plan/evidence/H441
ST=$(cat "$EV/logs/pub_stage")
R=${WAVE_OUT:-/home/ubuntu/jope-data/h335-cl/h441/$(basename "$ST")}
CPUSET=${WAVE_CPUSET:-64-79}

BF=/home/ubuntu/jope-data/h335-cl/h424/dev/bank_FITSEL.npz
SHA_BF=f8849ccb7343cfec1c0fd6688aaafef7580b85001046f2707fab58c5ecebe069
EXCL=/home/ubuntu/jope-data/h335-cl/h431/pub-r2/EXCLUSION_FITSEL.json
SHA_EXCL=be87f957c1ab0089b389e5bee5ae2bea5ad141d40aa9568973fd3c611f4154da
RF=/home/ubuntu/jope_plan/evidence/H420/pulled/H420_FIT-SEL/H420_FIT-SEL

log() { echo "$(date -u +%FT%TZ) $*"; }
(cd "$ST/code" && sha256sum --quiet -c ../CODE.sha256)
[ "$(cat "$ST/mode")" = pub ]
[ "$(sha256sum "$BF" | cut -d' ' -f1)" = "$SHA_BF" ]
[ "$(sha256sum "$EXCL" | cut -d' ' -f1)" = "$SHA_EXCL" ]
mkdir -p "$R" "$EV/logs" "$EV/results"
cd "$ST/code"

T_START=$(date +%s)
log "run_start CPUSET=$CPUSET stage=$ST CODE.sha256=$(sha256sum "$ST/CODE.sha256" | cut -c1-16)"

if [ ! -e "$R/unit_tests.json" ]; then
  taskset -c "$CPUSET" nice -n 19 python3 -B test_w5.py > "$R/unit_tests.raw.txt"
  head -n 1 "$R/unit_tests.raw.txt" > "$R/unit_tests.json"
  tail -n 1 "$R/unit_tests.raw.txt"
fi

if [ ! -e "$R/must_shas_FITSEL.json" ]; then
  taskset -c "$CPUSET" nice -n 19 python3 -B eval_w5_fitsel.py must-shas --run-dir "$RF" --tag TBOTH --out "$R/must_shas_FITSEL.json"
fi

if [ ! -e "$R/h01a_scores_FITSEL.npz" ]; then
  log "scoring H0.1-A neural screener on FIT-SEL inside H335 docker sandbox"
  H335_GROUP=AGG H335_STAGE="$ST" H335_CPUSET="$CPUSET" "$ST/code/sandbox_h335_docker.sh" \
    python3 -B score_h01a_fitsel.py --must-shas-json "$R/must_shas_FITSEL.json" --out "$R/h01a_scores_FITSEL.npz" \
    > "$EV/logs/score_h01a.log" 2>&1
fi

if [ ! -e "$R/APPLY_W5.json" ]; then
  log "running 5-fold cross-family W5 evaluation on FIT-SEL"
  taskset -c "$CPUSET" nice -n 19 python3 -B eval_w5_fitsel.py apply \
    --run-dir "$RF" --tag TBOTH --bank "$BF" --h01a-npz "$R/h01a_scores_FITSEL.npz" --excl "$EXCL" --out "$R/APPLY_W5.json"
fi

T_END=$(date +%s)
jq -n --argjson run_start "$T_START" --argjson run_end "$T_END" --arg cpuset "$CPUSET" \
  '{run_start:$run_start, run_end:$run_end, wall_s:($run_end-$run_start), cpuset:$cpuset}' > "$EV/logs/run_times.json"
log "driver_h441 completed in $((T_END - T_START))s (ready for gates_h441.sh then readout)"
