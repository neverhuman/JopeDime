#!/bin/bash
# Run one H421 step as SHARDS context-disjoint sandbox processes, each on its own slice of the CPU list (a fill or
# deploy process is bound by its Python thread). Every process gets --cores <slice> --shard k/SHARDS appended.
#   h421_shards.sh STAGE CPUSET MEM SHARDS HEAD_START_S ENTRY.py <args...>
# HEAD_START_S: seconds the first shard runs alone (a deploy's first shard creates the run config; 0 for the fill).
set -euo pipefail
STAGE=${1:?}; CPUS=${2:?}; MEM=${3:?}; SHARDS=${4:?}; HEAD=${5:?}; ENTRY=${6:?}; shift 6
mapfile -t ALL < <(echo "$CPUS" | tr ',' '\n' | gawk -F- '{ if (NF == 2) for (i = $1; i <= $2; i++) print i; else print $1 }')
per=$(( ${#ALL[@]} / SHARDS )); [ "$per" -ge 1 ] || { echo "fewer CPUs than shards" >&2; exit 2; }
slice() { local IFS=,; echo "${ALL[*]:$(( $1 * per )):$per}"; }
echo "$(date -u +%FT%TZ) $ENTRY: $SHARDS shards x $per cpus"
pids=()
for k in $(seq 0 $(( SHARDS - 1 ))); do
  cp=$(slice "$k")
  bash "$STAGE/code/run_apex_xbabe0.sh" "$STAGE" "$cp" "$MEM" "$ENTRY" "$@" --cores "$cp" --shard "$k/$SHARDS" &
  pids+=($!)
  [ "$k" != 0 ] || [ "$HEAD" = 0 ] || sleep "$HEAD"
done
bad=0; for p in "${pids[@]}"; do wait "$p" || bad=1; done
[ "$bad" = 0 ] || { echo "$(date -u +%FT%TZ) $ENTRY: a shard failed" >&2; exit 5; }
echo "$(date -u +%FT%TZ) $ENTRY: all shards done"
