#!/bin/bash
# H431 scored CPU run from the published stage:
#   0) test_cw3.py (6 synthetic & r9 byte-equivalence checks -> unit_tests.json)
#   1) screen_h425.py --queries FIT-SEL inside H335 docker sandbox + lists_fitsel.sh -> EXCLUSION_FITSEL.json
#   2) icb_gen.py -> ICB_v1.npz (340 tasks)
#   3) icb_run.py battery + monitor across P single-thread nice-19 CPU processes on M_h020, M_h024, L_h018
# Stops before readout_h431.py so gates_h431.sh runs first.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EV=/home/ubuntu/jope_plan/evidence/H431
ST=$(cat "$EV/logs/pub_stage")
R=${H431_OUT:-/home/ubuntu/jope-data/h335-cl/h431/$(basename "$ST")}
P=${H431_PROCS:-16}
CPUSET=${H431_CPUSET:-40-55}
[ "$P" -le 16 ] || { echo "P=$P exceeds 16-thread ceiling" >&2; exit 2; }

CK_M20=/home/ubuntu/jope-data/h424/ICL_M_r3_ckpt_h020.pt
SHA_M20=e3c8fd9972ce0c1684630ff06b159e950f6f498c4461d0454e0c6a6f44d38dd7
CK_M24=/home/ubuntu/jope-data/h424/ICL_M_r3_ckpt_h024.pt
SHA_M24=7fbca0e702bee00b4df0a58fac4062f550ceb71e9319fbbed5356a9b326ef176
CK_L18=/home/ubuntu/jope-data/apex-icl/diag/mon/L_w2_h018.pt
SHA_L18=153117c5fd76dde38fc3d4d7de0997dd9f4bb92ba4eef42bf741ff7530124121
VAL=/home/ubuntu/jope-data/apex-icl/bank/fitsel_search_val.npz
SHA_VAL=8e517921b7d5bb46c223a12262e7e0c64531db3fad7dede601d002a09840bba0

log() { echo "$(date -u +%FT%TZ) $*"; }
(cd "$ST/code" && sha256sum --quiet -c ../CODE.sha256)
[ "$(cat "$ST/mode")" = pub ]
mkdir -p "$R" "$EV/logs" "$EV/results"
cd "$ST/code"

T_START=${H431_T_START:-$(date +%s)}
log "run_start P=$P CPUSET=$CPUSET stage=$ST R=$R T_START=$T_START CODE.sha256=$(sha256sum "$ST/CODE.sha256" | cut -c1-16)"

# Step 0: Part B + reference unit tests
if [ ! -e "$R/unit_tests.json" ]; then
  taskset -c "$CPUSET" nice -n 19 python3 -B test_cw3.py > "$R/unit_tests.raw.txt"
  head -n 1 "$R/unit_tests.raw.txt" > "$R/unit_tests.json"
  tail -n 1 "$R/unit_tests.raw.txt"
fi

# Step 1: FIT-SEL feature contamination screen inside H335 docker sandbox
if [ ! -e "$R/SCREEN_FITSEL.json" ]; then
  log "running screen_h425.py --queries FIT-SEL in H335 docker sandbox"
  H335_STAGE="$ST" H335_CPUSET="$CPUSET" "$ST/code/sandbox_h335_docker.sh" \
    python3 -B screen_h425.py --queries FIT-SEL --out "$R/SCREEN_FITSEL.json" --procs 8 > "$EV/logs/screen_fitsel.log" 2>&1
fi
if [ ! -e "$R/EXCLUSION_FITSEL.json" ]; then
  "$ST/code/lists_fitsel.sh" "$R/SCREEN_FITSEL.json" "$R"
fi

# Step 2: Generate ICB_v1.npz (340 tasks)
if [ ! -e "$R/ICB_v1.npz" ]; then
  taskset -c "$CPUSET" nice -n 19 python3 -B icb_gen.py --out "$R/ICB_v1.npz" --seed 431001 > "$EV/logs/icb_gen.log"
fi
SHA_ICB=$(jq -r .sha256 "$R/ICB_v1.npz.json")
[ "$(sha256sum "$R/ICB_v1.npz" | cut -d' ' -f1)" = "$SHA_ICB" ]

run_sharded() {  # subcommand, label, size, ckpt, ckpt_sha, out_dir
  local sub=$1 lbl=$2 sz=$3 ck=$4 cksha=$5 odir=$6
  if [ -d "$odir" ] && [ "$(ls "$odir"/status_s*of${P}.json 2>/dev/null | wc -l)" = "$P" ]; then
    log "already complete $sub $lbl ($P status files)"
    return 0
  fi
  local rc=0
  log "start $sub $lbl ($P shards)"
  for k in $(seq 0 $((P - 1))); do
    if [ "$sub" = battery ]; then
      taskset -c "$CPUSET" nice -n 19 python3 -B icb_run.py battery \
        --icb "$R/ICB_v1.npz" --icb-sha256 "$SHA_ICB" --ckpt "$ck" --ckpt-sha256 "$cksha" \
        --size "$sz" --out "$odir" --shard "$k/$P" --threads 1 > "$EV/logs/${sub}_${lbl}_$k.log" 2>&1 &
    else
      taskset -c "$CPUSET" nice -n 19 python3 -B icb_run.py monitor \
        --val "$VAL" --val-sha256 "$SHA_VAL" --ckpt "$ck" --ckpt-sha256 "$cksha" \
        --size "$sz" --out "$odir" --shard "$k/$P" --threads 1 > "$EV/logs/${sub}_${lbl}_$k.log" 2>&1 &
    fi
  done
  for j in $(jobs -p); do wait "$j" || rc=$?; done
  log "done $sub $lbl (worst_rc=$rc)"
  return $rc
}

run_sharded_l() {  # subcommand, label, ckpt, ckpt_sha, out_dir (4 CCX shards x 4 threads = 16 threads)
  local sub=$1 lbl=$2 ck=$3 cksha=$4 odir=$5
  local ns=4 th=4 rc=0
  [ $((ns * th)) -le 16 ] || { echo "ns*th exceeds 16" >&2; exit 2; }
  if [ -d "$odir" ] && [ "$(ls "$odir"/status_s*of${ns}.json 2>/dev/null | wc -l)" = "$ns" ]; then
    log "already complete $sub $lbl ($ns status files)"
    return 0
  fi
  local ccx_sets=("$CPUSET" "$CPUSET" "$CPUSET" "$CPUSET")
  if [ "$CPUSET" = "64-79" ]; then
    ccx_sets=("64-67" "68-71" "72-75" "76-79")
  fi
  log "start $sub $lbl (${ns}x${th} CCX shards)"
  for k in 0 1 2 3; do
    local cset=${ccx_sets[$k]}
    if [ "$sub" = battery ]; then
      OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th \
        taskset -c "$cset" nice -n 19 python3 -B icb_run.py battery \
        --icb "$R/ICB_v1.npz" --icb-sha256 "$SHA_ICB" --ckpt "$ck" --ckpt-sha256 "$cksha" \
        --size L --out "$odir" --shard "$k/$ns" --threads "$th" --rep0-only > "$EV/logs/${sub}_${lbl}_4x4_$k.log" 2>&1 &
    else
      OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th \
        taskset -c "$cset" nice -n 19 python3 -B icb_run.py monitor \
        --val "$VAL" --val-sha256 "$SHA_VAL" --ckpt "$ck" --ckpt-sha256 "$cksha" \
        --size L --out "$odir" --shard "$k/$ns" --threads "$th" > "$EV/logs/${sub}_${lbl}_4x4_$k.log" 2>&1 &
    fi
  done
  for j in $(jobs -p); do wait "$j" || rc=$?; done
  log "done $sub $lbl (worst_rc=$rc)"
  return $rc
}

# Step 3: Score M_h020, M_h024 (full 340+106), and L_h018 (rep==0 68-cell battery + full 106 monitor)
run_sharded battery M_h020 M "$CK_M20" "$SHA_M20" "$R/bat_M_h020"
run_sharded monitor M_h020 M "$CK_M20" "$SHA_M20" "$R/mon_M_h020"
run_sharded battery M_h024 M "$CK_M24" "$SHA_M24" "$R/bat_M_h024"
run_sharded monitor M_h024 M "$CK_M24" "$SHA_M24" "$R/mon_M_h024"
run_sharded_l battery L_h018 "$CK_L18" "$SHA_L18" "$R/bat_L_h018"
run_sharded_l monitor L_h018 "$CK_L18" "$SHA_L18" "$R/mon_L_h018"

T_END=$(date +%s)
jq -n --argjson run_start "$T_START" --argjson run_end "$T_END" --argjson procs "$P" --arg cpuset "$CPUSET" \
  '{run_start: $run_start, run_end: $run_end, wall_s: ($run_end - $run_start), procs: $procs, cpuset: $cpuset}' > "$EV/logs/run_times.json"
log "driver_h431 completed in $((T_END - T_START))s"
