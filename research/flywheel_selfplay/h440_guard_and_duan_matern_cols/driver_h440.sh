#!/bin/bash
# H440 scored CPU run from the published stage:
#   0) test_guard_matern.py + smoke_matern.py -> unit_tests.json
#   1) Stage A: FIT-SEL columns + apply + design-check readout (stops if harm or >5% failed)
#   2) Stage B: P201b columns + apply (stops BEFORE P201b readout so gates_h440.sh runs first!)
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

EV=/home/ubuntu/jope_plan/evidence/H440
ST=$(cat "$EV/logs/pub_stage")
R=${WAVE_OUT:-/home/ubuntu/jope-data/h335-cl/h440/$(basename "$ST")}
P=${WAVE_PROCS:-16}
CPUSET=${WAVE_CPUSET:-64-79}
[ "$P" -le 16 ] || { echo "P=$P exceeds 16-thread ceiling" >&2; exit 2; }

BF=/home/ubuntu/jope-data/h335-cl/h424/dev/bank_FITSEL.npz
SHA_BF=f8849ccb7343cfec1c0fd6688aaafef7580b85001046f2707fab58c5ecebe069
BP=/home/ubuntu/jope-data/h430/bank_P201B.npz
SHA_BP=da29e8ec64f623e0fa2ab29b31bbd40b91bef36adf8ddc8e2e773f51c96d01af
RF=/home/ubuntu/jope_plan/evidence/H420/pulled/H420_FIT-SEL/H420_FIT-SEL
RP=/home/ubuntu/jope_plan/evidence/H416/pulled/H416_CONFIRM/H416_CONFIRM

K430_F=/home/ubuntu/jope-data/h430/pub/cols_FITSEL
K430_P=/home/ubuntu/jope-data/h430/pub/cols_P201B
F433_F=/home/ubuntu/jope-data/h433/pub/cols_FITSEL
F433_P=/home/ubuntu/jope-data/h433/pub/cols_P201B
H436_F=/home/ubuntu/jope-data/h436/pub/cols_FITSEL
H436_P=/home/ubuntu/jope-data/h436/pub/cols_P201B

log() { echo "$(date -u +%FT%TZ) $*"; }
(cd "$ST/code" && sha256sum --quiet -c ../CODE.sha256)
[ "$(cat "$ST/mode")" = pub ]
[ "$(sha256sum "$BF" | cut -d' ' -f1)" = "$SHA_BF" ]
[ "$(sha256sum "$BP" | cut -d' ' -f1)" = "$SHA_BP" ]
mkdir -p "$R" "$EV/logs" "$EV/results"
cd "$ST/code"

T_START=$(date +%s)
log "run_start P=$P CPUSET=$CPUSET stage=$ST CODE.sha256=$(sha256sum "$ST/CODE.sha256" | cut -c1-16)"

if [ ! -e "$R/unit_tests.json" ]; then
  taskset -c "$CPUSET" nice -n 19 python3 -B test_guard_matern.py > "$R/unit_tests.raw.txt"
  head -n 1 "$R/unit_tests.raw.txt" > "$R/unit_tests.json"
  tail -n 1 "$R/unit_tests.raw.txt"
  taskset -c "$CPUSET" nice -n 19 python3 -B smoke_matern.py --dir "$R/smoke_synth"
fi

cols() { # bank, out_dir, label
  local rc=0
  log "start cols $3 ($P shards)"
  for k in $(seq 0 $((P - 1))); do
    taskset -c "$CPUSET" nice -n 19 python3 -B cols_matern.py \
      --bank "$1" --out "$2" --shard "$k/$P" --ctx-s 900 > "$EV/logs/cols_${3}_$k.log" 2>&1 &
  done
  for j in $(jobs -p); do wait "$j" || rc=$?; done
  log "$3 columns ended (worst_rc=$rc): $(ls "$2" | grep -Ec '^[0-9a-f]{64}\.npz$' || true) context files"
  return $rc
}

notok() {
  ls "$1" | grep -E '^[0-9a-f]{64}\.json$' | sed "s#^#$1/#" | xargs cat | jq -s '[.[] | select(.ok | not)] | length'
}

# ---- Stage A: FIT-SEL design check ----
cols "$BF" "$R/cols_FITSEL" FITSEL
EXTRA_F=()
[ -d "$K430_F" ] && EXTRA_F+=(--h430-cols "$K430_F")
if [ -d "$F433_F" ] && [ "$(ls "$F433_F" | grep -Ec '^[0-9a-f]{64}\.npz$' || true)" -ge 100 ]; then
  EXTRA_F+=(--h433-cols "$F433_F")
fi
if [ -d "$H436_F" ] && [ "$(ls "$H436_F" | grep -Ec '^[0-9a-f]{64}\.npz$' || true)" -ge 100 ]; then
  EXTRA_F+=(--h436-cols "$H436_F")
fi
if [ ! -e "$R/APPLY_FITSEL.json" ]; then
  taskset -c "$CPUSET" nice -n 19 python3 -B stack_matern.py apply \
    --run-dir "$RF" --tag TBOTH --panel FIT-SEL --bank "$BF" --cols "$R/cols_FITSEL" \
    "${EXTRA_F[@]}" --out "$R/APPLY_FITSEL.json" --procs "$P"
fi
if [ ! -e "$R/READOUT_FITSEL.json" ]; then
  taskset -c "$CPUSET" nice -n 19 python3 -B stack_matern.py readout \
    --apply "$R/APPLY_FITSEL.json" --out "$R/READOUT_FITSEL.json" | tee "$R/READOUT_FITSEL.txt"
fi
nf=$(notok "$R/cols_FITSEL")
if [ "$(jq -r .harm "$R/READOUT_FITSEL.json")" = true ]; then
  log "GUARD_MATERN_HARM_DESIGN: stopping before P201b"
  T_END=$(date +%s)
  jq -n --argjson run_start "$T_START" --argjson run_end "$T_END" --argjson procs "$P" --arg cpuset "$CPUSET" \
    '{run_start:$run_start, run_end:$run_end, wall_s:($run_end-$run_start), procs:$procs, cpuset:$cpuset, stopped_stage_a:true}' > "$EV/logs/run_times.json"
  exit 0
fi
if [ $((nf * 20)) -gt 106 ]; then
  log "FIT-SEL incomplete ($nf not ok > 5%): stopping before P201b"
  exit 2
fi
log "Stage A done ($nf not ok, harm=false); continuing to Stage B (P201b)"

# ---- Stage B: P201b columns + apply (stops BEFORE READOUT_P201B.json so gates_h440.sh runs first) ----
cols "$BP" "$R/cols_P201B" P201B
EXTRA_P=()
[ -d "$K430_P" ] && EXTRA_P+=(--h430-cols "$K430_P")
if [ -d "$F433_P" ] && [ "$(ls "$F433_P" | grep -Ec '^[0-9a-f]{64}\.npz$' || true)" -ge 191 ]; then
  EXTRA_P+=(--h433-cols "$F433_P")
fi
if [ -d "$H436_P" ] && [ "$(ls "$H436_P" | grep -Ec '^[0-9a-f]{64}\.npz$' || true)" -ge 191 ]; then
  EXTRA_P+=(--h436-cols "$H436_P")
fi
if [ ! -e "$R/APPLY_P201B.json" ]; then
  taskset -c "$CPUSET" nice -n 19 python3 -B stack_matern.py apply \
    --run-dir "$RP" --tag P201B --panel P201B --bank "$BP" --cols "$R/cols_P201B" \
    "${EXTRA_P[@]}" --out "$R/APPLY_P201B.json" --procs "$P"
fi

T_END=$(date +%s)
jq -n --argjson run_start "$T_START" --argjson run_end "$T_END" --argjson procs "$P" --arg cpuset "$CPUSET" \
  '{run_start:$run_start, run_end:$run_end, wall_s:($run_end-$run_start), procs:$procs, cpuset:$cpuset, stopped_stage_a:false}' > "$EV/logs/run_times.json"
log "driver_h440 completed in $((T_END - T_START))s (ready for gates_h440.sh then P201b readout)"
