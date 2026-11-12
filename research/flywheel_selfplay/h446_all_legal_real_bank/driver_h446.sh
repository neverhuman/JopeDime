#!/bin/bash
# H446 CPU-only legal real-table bank builder (`cpus 104-107`, <= 4 procs @ nice 19, 0 GPU):
#   driver_h446.sh run -> selftest + fetch-s3 + build (fit_icl_wide400, all_legal_excl_p201b, all_legal)
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export H317_DATA=/home/ubuntu/jope-data/h335-cl

CMD=${1:?run}
EV=/home/ubuntu/jope_plan/evidence/H446
ST=$(cat "$EV/logs/pub_stage")
R=${WAVE_OUT:-/home/ubuntu/jope-data/h335-cl/h446/$(basename "$ST")}
BLOBS=/home/ubuntu/jope-data/h446/s3_train_blobs
P=${WAVE_PROCS:-4}
CPUSET=${WAVE_CPUSET:-104-107}
[ "$P" -le 4 ] || { echo "P=$P exceeds 4-core ceiling" >&2; exit 2; }

(cd "$ST/code" && sha256sum --quiet -c ../CODE.sha256)
[ "$(cat "$ST/mode")" = pub ]
mkdir -p "$EV/logs" "$EV/results" "$BLOBS"
cd "$ST/code"

if [ "$CMD" != "run" ]; then
  echo "unknown subcommand $CMD" >&2
  exit 2
fi

T0=$(date +%s)
mkdir -p "$R"
taskset -c "$CPUSET" nice -n 19 python3 -B icl_bank_all_legal.py selftest > "$R/selftest.raw.txt"
head -n 1 "$R/selftest.raw.txt" > "$R/selftest.json"
tail -n 1 "$R/selftest.raw.txt"

taskset -c "$CPUSET" nice -n 19 python3 -B icl_bank_all_legal.py fetch-s3 \
  --blob-dir "$BLOBS" --procs "$P" > "$R/fetch_s3.json"
cat "$R/fetch_s3.json"

rm -rf "$R/banks"
taskset -c "$CPUSET" nice -n 19 python3 -B icl_bank_all_legal.py build \
  --blob-dir "$BLOBS" --out-dir "$R/banks" --procs "$P" | tee "$EV/logs/build.log"

T_END=$(date +%s)
jq -n --argjson run_start "$T0" --argjson run_end "$T_END" --argjson procs "$P" --arg cpuset "$CPUSET" \
  '{run_start:$run_start, run_end:$run_end, wall_s:($run_end-$run_start), procs:$procs, cpuset:$cpuset}' > "$EV/logs/run_times.json"
echo "$(date -u +%FT%TZ) H446 build complete in $((T_END - T0))s"
