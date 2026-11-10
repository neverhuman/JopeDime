#!/bin/bash
# H420 deploy chain for one panel in one run directory: replay check, then the variants in order. A deploy process is
# bound by its Python thread, so every variant runs as SHARDS context-disjoint processes (train_h420.py --shard k/n),
# each on its own slice of the CPU list.
#   h420_chain.sh STAGE bwrap|docker CPUSET MEM PANEL RUN_DIR JOBS [REPLAY_LIMIT] [SHARDS]
#   env H420_VARIANTS (default "TBOTH TLIB TGX")
set -euo pipefail
STAGE=${1:?}; KIND=${2:?}; CPUS=${3:?}; MEM=${4:?}; PANEL=${5:?}; R=${6:?}; JOBS=${7:?}; LIMIT=${8:-2}; SHARDS=${9:-1}
run() {  # CPULIST ENTRY args...
  local cp=$1; shift
  if [ "$KIND" = bwrap ]; then bash "$STAGE/code/run_h420_local.sh" "$STAGE" "$cp" "$@"
  else bash "$STAGE/code/run_apex_xbabe0.sh" "$STAGE" "$cp" "$MEM" "$@"; fi
}
mapfile -t ALL < <(echo "$CPUS" | tr ',' '\n' | gawk -F- '{ if (NF == 2) for (i = $1; i <= $2; i++) print i; else print $1 }')
per=$(( ${#ALL[@]} / SHARDS )); [ "$per" -ge 1 ] || { echo "fewer CPUs than shards" >&2; exit 2; }
slice() { local IFS=,; echo "${ALL[*]:$(( $1 * per )):$per}"; }
run "$CPUS" train_h420.py replaycheck --run-dir "$R" --panel "$PANEL" --cores "$CPUS" --limit "$LIMIT" --jobs 12
for v in ${H420_VARIANTS:-TBOTH TLIB TGX}; do
  echo "$(date -u +%FT%TZ) variant $v ($SHARDS shards x $per cpus)"
  pids=()
  for k in $(seq 0 $(( SHARDS - 1 ))); do
    cp=$(slice "$k")
    run "$cp" train_h420.py deploy --run-dir "$R" --panel "$PANEL" --variant "$v" --cores "$cp" --jobs "$JOBS" \
      --shard "$k/$SHARDS" &
    pids+=($!)
    [ "$k" != 0 ] || sleep 45   # the first shard creates the run config
  done
  bad=0; for p in "${pids[@]}"; do wait "$p" || bad=1; done
  [ "$bad" = 0 ] || { echo "$(date -u +%FT%TZ) variant $v: a shard failed" >&2; exit 5; }
done
echo "$(date -u +%FT%TZ) chain done"
